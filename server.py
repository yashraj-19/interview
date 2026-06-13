"""SViam voice interviewer — browser, full-duplex (WebRTC).

FastAPI + Pipecat SmallWebRTCTransport. One pipeline per browser connection.
The browser's WebRTC stack does acoustic echo cancellation, so the candidate can
interrupt the bot cleanly (true full-duplex) — no half-duplex gate needed.

All the interview logic is REUSED unchanged from voice_agent.py (TranscriptLogger,
DynamicsTracker, InterruptJudge, SilenceMonitor, ConversationState, the conversation
LLM factory, SYSTEM_PROMPT) and judge.py. Only the transport + UI bridge are new.

Run:  venv\\Scripts\\python.exe server.py   then open http://localhost:8000 in Chrome.

NOTE (Pipecat 1.3.0, per CLAUDE.md rule 5): there is no `allow_interruptions` field
on PipelineParams in 1.3.0. Interruptions/barge-in are produced by the user-turn-START
strategy (VADUserTurnStartStrategy) firing while the bot speaks — which works precisely
because we do NOT add the half-duplex InputGate here. So "interruptions on" == "mic stays
live + VAD turn strategy", not a parameter.
"""

import asyncio
import os
import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# --- Pipecat ---
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

# --- Human-like barge-in + turn-end (custom strategies + frame-layer manager) ---
from barge_in import (
    BACKCHANNEL_WORDS,
    BargeInManager,
    EmergencyTurnEnd,
    HumanBargeInStartStrategy,
)

# --- Reused interview logic (do NOT duplicate — import from the working module) ---
import voice_agent as va
from voice_agent import (
    ConversationState,
    DynamicsTracker,
    InterruptJudge,
    SilenceMonitor,
    TranscriptLogger,
)

logger.remove()
logger.add(sys.stderr, level="INFO")

STUN = ["stun:stun.l.google.com:19302"]

app = FastAPI()
_connections: dict[str, SmallWebRTCConnection] = {}

# ---------------------------------------------------------------------------
# UI event bus — push transcript + status to the browser over a small WebSocket
# (the WebRTC data channel could also carry these, but a sidecar ws is simpler
# and reliable for a custom client; single-session dev use).
# ---------------------------------------------------------------------------
_ui_clients: set[WebSocket] = set()


async def ui_broadcast(event: dict):
    for ws in list(_ui_clients):
        try:
            await ws.send_json(event)
        except Exception:
            _ui_clients.discard(ws)


class EventBridge(FrameProcessor):
    """Forwards transcript + speaking status to the browser UI. role='in' (after STT)
    emits candidate transcripts + status from VAD/bot-speaking frames; role='out'
    (after the LLM) emits the bot's full reply text."""

    def __init__(self, role: str = "in", **kwargs):
        super().__init__(**kwargs)
        self._role = role
        self._bot_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if self._role == "in":
                if isinstance(frame, BotStartedSpeakingFrame):
                    await ui_broadcast({"type": "status", "status": "speaking"})
                elif isinstance(frame, BotStoppedSpeakingFrame):
                    await ui_broadcast({"type": "status", "status": "listening"})
                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    await ui_broadcast({"type": "status", "status": "thinking"})
                elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
                    await ui_broadcast(
                        {"type": "transcript", "role": "candidate", "text": frame.text}
                    )
            else:  # "out"
                if isinstance(frame, LLMTextFrame):
                    self._bot_text += frame.text
                elif isinstance(frame, LLMFullResponseEndFrame):
                    if self._bot_text.strip():
                        await ui_broadcast(
                            {"type": "transcript", "role": "bot", "text": self._bot_text.strip()}
                        )
                    self._bot_text = ""
        except Exception as e:
            logger.debug(f"EventBridge error: {e}")
        await self.push_frame(frame, direction)


class LoggingSmartTurn(LocalSmartTurnAnalyzerV3):
    """Logs each smart-turn decision and tags how the turn ended (smart prediction
    vs stop_secs silence fallback) on ConversationState for the per-turn summary."""

    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def append_audio(self, buffer, is_speech):
        st = super().append_audio(buffer, is_speech)
        if st == EndOfTurnState.COMPLETE:
            self._state.last_turn_trigger = "fallback"
            logger.info("SMART_TURN append_audio -> COMPLETE (stop_secs silence fallback)")
        return st

    async def analyze_end_of_turn(self):
        st, metrics = await super().analyze_end_of_turn()
        if st == EndOfTurnState.COMPLETE:
            self._state.last_turn_trigger = "smart"
        logger.info(f"SMART_TURN analyze -> {st.name}")
        return st, metrics


# ---------------------------------------------------------------------------
# Bot pipeline — same as voice_agent.main() but WebRTC transport, full-duplex
# (no InputGate, no SmoothLocalAudioTransport).
# ---------------------------------------------------------------------------
async def run_bot(connection: SmallWebRTCConnection):
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    state = ConversationState()

    # interim_results=True: the barge-in strategy counts words in real time.
    # utterance_end_ms/vad_events + a longer endpointing keep Deepgram from
    # FINALIZING on short mid-thought pauses (it previously used aggressive
    # defaults: endpointing=None~10ms, utterance_end_ms=None). The actual
    # turn-END decision is owned by the smart-turn analyzer, not Deepgram.
    stt = DeepgramSTTService(
        api_key=va.DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="en-US",
            interim_results=True,
            smart_format=True,
            vad_events=True,
            # Low endpointing -> finalized transcripts arrive fast, so the smart-turn
            # strategy (which waits for the final after predicting complete) can end the
            # turn in ~300-500ms instead of being gated by Deepgram. utterance_end_ms is
            # only a finalization backstop. Turn-END is owned by smart-turn, so fast
            # finals do NOT cause premature ends on mid-thought pauses.
            utterance_end_ms=1000,
            endpointing=100,
        ),
    )
    tts = DeepgramTTSService(
        api_key=va.DEEPGRAM_API_KEY,
        settings=DeepgramTTSService.Settings(voice=va.TTS_VOICE),
    )
    llm = va.build_conversation_llm()

    context = LLMContext(
        [
            {"role": "system", "content": va.SYSTEM_PROMPT},
            {"role": "user", "content": "Please start the interview now."},
        ]
    )
    # Turn START (barge-in): immediate normal turns; interrupt the bot only on sustained
    # voice (>=0.7s) OR >=3 non-backchannel words. enable_interruptions=False: the user's
    # speech never cuts the bot off (full-duplex is step 4), so the turn machine can't get
    # stuck after a mid-sentence interruption. Named ref so the TTS resync can clear its
    # phantom 'bot speaking' flag.
    start_strat = HumanBargeInStartStrategy(
        min_words=3, sustained_secs=0.7, enable_interruptions=False
    )
    # Turn END (STEP 1 reliability): transcript-driven end-of-turn ~2.5s after the last
    # finalized transcript, plus a 4s WATCHDOG that force-resyncs if a TTS reconnect leaves
    # the turn state stuck. Smart-turn (semantic pause handling) is re-introduced in step 2.
    turn_strategies = UserTurnStrategies(
        start=[start_strat],
        stop=[EmergencyTurnEnd(state, silence_secs=2.5, watchdog_secs=4.0)],
    )
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies),
    )

    barge = BargeInManager(state)

    # STEP 1 reliability resync: a Deepgram TTS websocket drop (e.g. 1011) can die
    # mid-utterance WITHOUT emitting BotStoppedSpeakingFrame, leaving bot_speaking/bot_pending
    # stuck True — which blocks the turn-end gate so the bot goes permanently silent.
    # on_disconnected fires the moment the drop is detected (before reconnect), so clear the
    # stuck state immediately; EmergencyTurnEnd's watchdog then forces the overdue turn-end.
    @tts.event_handler("on_disconnected")
    async def _on_tts_disconnected(_service):
        logger.warning("TTS_RESYNC: Deepgram TTS dropped — clearing stuck bot/turn state")
        barge.resync()
        start_strat.resync()

    _tts_connects = {"n": 0}

    @tts.event_handler("on_connected")
    async def _on_tts_connected(_service):
        _tts_connects["n"] += 1
        if _tts_connects["n"] > 1:
            logger.info(f"TTS reconnected (connection #{_tts_connects['n']})")

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            TranscriptLogger(),       # CANDIDATE> logging
            EventBridge(role="in"),   # candidate transcript + status -> UI
            DynamicsTracker(state),   # ask-back detection
            # InterruptJudge(state),  # STEP 2 — disabled for now (was over-interrupting)
            SilenceMonitor(),         # silence auto-prompt
            aggregator.user(),
            llm,
            TranscriptLogger(state),  # BOT> logging + records last question
            EventBridge(role="out"),  # bot reply text -> UI
            tts,
            barge,                    # barge latency, dropped-thought/resume note, tie guard
            transport.output(),
            aggregator.assistant(),
        ]
    )

    task = PipelineWorker(
        pipeline,
        enable_rtvi=False,  # custom client doesn't use RTVI; avoids the data-channel flood
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = WorkerRunner(handle_sigint=False)  # background task under uvicorn — no signal handling

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("client connected — greeting")
        await ui_broadcast({"type": "status", "status": "listening"})
        await task.queue_frames([LLMRunFrame()])  # bot greets first

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("client disconnected — ending pipeline")
        await runner.cancel()

    logger.info(
        f"bot ready — barge-in needs >=3 words OR >=0.7s voice; "
        f"backchannels never interrupt ({sorted(BACKCHANNEL_WORDS)})"
    )
    await runner.add_workers(task)
    await runner.run()
    logger.info("bot pipeline finished")


# ---------------------------------------------------------------------------
# Signaling
# ---------------------------------------------------------------------------
@app.post("/api/offer")
async def offer(request: dict):
    pc_id = request.get("pc_id")
    if pc_id and pc_id in _connections:
        conn = _connections[pc_id]
        await conn.renegotiate(sdp=request["sdp"], type=request["type"])
    else:
        conn = SmallWebRTCConnection(STUN)
        await conn.initialize(sdp=request["sdp"], type=request["type"])

        @conn.event_handler("closed")
        async def _on_closed(c):
            _connections.pop(c.pc_id, None)

        asyncio.create_task(run_bot(conn))

    answer = conn.get_answer()
    _connections[answer["pc_id"]] = conn
    return JSONResponse(answer)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ui_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client sends nothing meaningful
    except WebSocketDisconnect:
        pass
    finally:
        _ui_clients.discard(websocket)


# Static client (mounted last so /api/offer and /ws take precedence).
app.mount("/", StaticFiles(directory="client", html=True), name="client")


if __name__ == "__main__":
    logger.info("SViam WebRTC server on http://localhost:8000  (open in Chrome)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
