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
import time

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
    """Logs each smart-turn decision and tags how the turn ended (smart ML prediction
    vs stop_secs silence fallback) on ConversationState for the per-turn summary.

    _fallback_pending keeps the label honest: when append_audio reports COMPLETE on the
    silence backstop, _handle_input_audio immediately also calls analyze_end_of_turn — we
    must not let that overwrite "fallback" with "smart"."""

    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._fallback_pending = False

    def append_audio(self, buffer, is_speech):
        st = super().append_audio(buffer, is_speech)
        if st == EndOfTurnState.COMPLETE:
            self._state.last_turn_trigger = "fallback"
            self._fallback_pending = True
            logger.info("SMART_TURN append_audio -> COMPLETE (stop_secs silence fallback)")
        return st

    async def analyze_end_of_turn(self):
        st, metrics = await super().analyze_end_of_turn()
        if st == EndOfTurnState.COMPLETE and not self._fallback_pending:
            self._state.last_turn_trigger = "smart"
        self._fallback_pending = False
        logger.info(f"SMART_TURN analyze -> {st.name}")
        return st, metrics


class GuardedSmartTurnStop(TurnAnalyzerUserTurnStopStrategy):
    """Smart-turn stop strategy + double-reply DE-DUP guard.

    A delayed/duplicate finalized transcript can fire a turn-end AFTER the bot has already
    started replying (seen live: two questions back-to-back). If a reply is already speaking
    or in flight (bot_speaking/bot_pending), suppress the turn-end so no second reply starts.
    Safe in step 2 because user-interrupts-bot is OFF (no legitimate new turn during bot
    speech) — revisit when full-duplex barge-in lands in step 4."""

    def __init__(self, state, *, turn_analyzer, **kwargs):
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._state = state

    async def trigger_user_turn_stopped(self):
        # De-dup, but ONLY for a genuinely fresh in-flight reply. A bot_pending/bot_speaking
        # flag stuck True far longer than any real generation (~6s) / spoken reply (~15s) is a
        # PHANTOM (lost BotStoppedSpeaking on a bad link). Trusting it blindly blocks every
        # turn-end -> "transcript arrives but she never replies". So clear stale flags and let
        # the reply through; only a recent flag means a real reply is actually in flight.
        now = time.monotonic()
        busy_real = (
            (self._state.bot_pending and now - self._state.t_bot_pending <= 6.0)
            or (self._state.bot_speaking and now - self._state.t_bot_speaking <= 15.0)
        )
        if busy_real:
            logger.info("TURN_END suppressed (smart): reply genuinely in-flight — de-dup")
            return
        if self._state.bot_pending or self._state.bot_speaking:
            logger.info("TURN_END: clearing stale phantom bot flag, proceeding with reply")
            self._state.bot_pending = False
            self._state.bot_speaking = False
        await super().trigger_user_turn_stopped()


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
    # The actual turn-END decision is owned by the smart-turn analyzer, not Deepgram.
    stt = DeepgramSTTService(
        api_key=va.DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="en-US",
            interim_results=True,
            smart_format=True,
            vad_events=True,
            # endpointing=250 (spec §3): finals arrive fast WITHOUT over-fragmenting.
            # endpointing=100 split utterances into "use" / "order of" / "log n." finals
            # (observed live). Smart-turn and the watchdog act on ACCUMULATED text across
            # finals, never on one fragment, so 250 just yields cleaner, larger finals.
            # utterance_end_ms is only a finalization backstop.
            utterance_end_ms=1000,
            endpointing=250,
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
    # voice (>=0.7s) OR >=3 non-backchannel words (backchannels never cut in). STEP 4 FULL-
    # DUPLEX: enable_interruptions=True lets the candidate cut the bot off mid-sentence. The
    # strategy's trigger_user_turn_started override only actually broadcasts an interruption
    # when the bot is SPEAKING, so normal turns and the thinking-gap don't fire spurious cuts.
    # The step1-3.5 turn-state hardening (heartbeat reconciler, watchdog, full-resync, de-dup)
    # is what makes flipping this on safe — it was the bug that kept the bot silent before.
    start_strat = HumanBargeInStartStrategy(
        min_words=3, sustained_secs=0.7, enable_interruptions=True
    )
    # Turn END (STEP 2): smart-turn-v3 ML analyzer OWNS the normal decision — it waits
    # through mid-sentence pauses (predicts INCOMPLETE) and ends the turn once you're
    # actually done (COMPLETE), with a 3.0s silence backstop (> ~2.5s natural pauses, spec
    # §3). GuardedSmartTurnStop adds the double-reply de-dup guard.
    #
    # EmergencyTurnEnd from STEP 1 stays UNDERNEATH as the reliability backstop. Its
    # transcript-driven timer is raised to 5.0s (> smart-turn's 3.0s) so it NEVER preempts
    # smart-turn — it only fires if smart-turn never ends the turn (e.g. degraded audio
    # starves smart-turn's audio-frame silence backstop). Its phantom-flag watchdog
    # (bot_pending stuck >6s / bot_speaking >15s) still catches lost-reply stuck states.
    # Layering: smart ML (fast) -> smart 3s silence -> watchdog 5s -> phantom guard.
    # Both strategies run in parallel (first to fire ends the turn); reset() cancels the
    # loser and the de-dup guard prevents a double reply.
    # on_bot_idle: when the playback-heartbeat reconciler detects the bot actually stopped
    # (transport skipped BotStoppedSpeaking), also clear the start strategy's private flag.
    barge = BargeInManager(state, on_bot_idle=start_strat.resync)

    def _full_resync():
        """Clear EVERY 'bot speaking' tracker (shared state + barge manager + start
        strategy) so a lost BotStoppedSpeaking can't desync the turn machine and swallow
        the next reply. Invoked by the watchdog on every forced turn-end."""
        barge.resync()
        start_strat.resync()

    smart_stop = GuardedSmartTurnStop(
        state,
        turn_analyzer=LoggingSmartTurn(state, params=SmartTurnParams(stop_secs=3.0)),
    )
    turn_strategies = UserTurnStrategies(
        start=[start_strat],
        stop=[
            smart_stop,
            EmergencyTurnEnd(
                state, silence_secs=5.0, watchdog_secs=4.0, on_resync=_full_resync
            ),
        ],
    )
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies),
    )

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
            InterruptJudge(state),    # STEP 3: wrong-answer correction — cuts in via TTS on a clear error
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
