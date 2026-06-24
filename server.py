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
import re
import sys
import time

import httpx
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
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
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
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
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
    TranscriptRuleTurnEnd,
    reply_already_handled,
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
from judge import confirms_or_denies, reveals_answer

logger.remove()
logger.add(sys.stderr, level="INFO")

STUN = ["stun:stun.l.google.com:19302"]

# --- ICE / TURN (deploy infra; interview logic unchanged) -------------------
# Local dev: STUN only (browser<->localhost needs no relay). On Render the
# browser and server cannot reach each other directly, so a TURN RELAY is
# required for audio. We use Metered's DYNAMIC credentials — METERED_DOMAIN +
# METERED_API_KEY (NOT static URLs) — fetched per connection, exposed to the
# browser via /api/ice and given to the server-side aiortc connection.
# TURN is a FALLBACK: default ICE still tries host/srflx first (we do NOT set
# iceTransportPolicy='relay').
METERED_DOMAIN = os.getenv("METERED_DOMAIN", "").strip()
METERED_API_KEY = os.getenv("METERED_API_KEY", "").strip()


async def _fetch_ice_config() -> list[dict]:
    """ICE servers as plain dicts [{urls, username?, credential?}]: STUN always,
    Metered TURN (fresh dynamic creds) appended when configured. Never raises —
    on failure we fall back to STUN-only (audio may then fail to connect on Render,
    which the logs will show)."""
    servers: list[dict] = [{"urls": u} for u in STUN]
    if METERED_DOMAIN and METERED_API_KEY:
        url = f"https://{METERED_DOMAIN}/api/v1/turn/credentials?apiKey={METERED_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                fetched = resp.json()
            if isinstance(fetched, list):
                servers.extend(s for s in fetched if isinstance(s, dict) and s.get("urls"))
        except Exception as e:
            logger.warning(
                f"Metered TURN fetch failed ({e}) — STUN only; audio may not connect on Render"
            )
    return servers


def _to_ice_servers(dicts: list[dict]) -> list[IceServer]:
    """Plain ICE dicts -> aiortc IceServer objects for the server-side connection."""
    return [
        IceServer(urls=d["urls"], username=d.get("username"), credential=d.get("credential"))
        for d in dicts
    ]


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
                elif isinstance(frame, TTSSpeakFrame) and frame.text.strip():
                    # Silence prompts + judge corrections speak via TTSSpeakFrame (not the LLM
                    # text path), so forward their text to the UI the same way LLM replies are.
                    await ui_broadcast(
                        {"type": "transcript", "role": "bot", "text": frame.text.strip()}
                    )
        except Exception as e:
            logger.debug(f"EventBridge error: {e}")
        await self.push_frame(frame, direction)


class LoggingSmartTurn(LocalSmartTurnAnalyzerV3):
    """smart-turn analyzer kept DELAY-ONLY (STEP 5). It records its latest INCOMPLETE/COMPLETE
    prediction on ConversationState — consumed by TranscriptRuleTurnEnd to push a turn-end
    LATER when it says INCOMPLETE — but it never triggers a turn-end itself (see
    SmartTurnObserver). Its COMPLETE predictions, which mis-fired turn-ends on fragmented
    speech, are recorded for the log but cannot end a turn."""

    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def append_audio(self, buffer, is_speech):
        st = super().append_audio(buffer, is_speech)
        if st == EndOfTurnState.COMPLETE:
            logger.info("SMART_TURN append_audio -> COMPLETE (silence fallback) [delay-only, ignored]")
        return st

    async def analyze_end_of_turn(self):
        st, metrics = await super().analyze_end_of_turn()
        self._state.smart_turn_incomplete = st == EndOfTurnState.INCOMPLETE
        self._state.t_smart_turn = time.monotonic()
        logger.info(f"SMART_TURN analyze -> {st.name} [delay-only]")
        return st, metrics


class SmartTurnObserver(TurnAnalyzerUserTurnStopStrategy):
    """Keeps smart-turn RUNNING (feeds it audio + VAD so LoggingSmartTurn records predictions
    on state) but NEVER triggers a turn-end. smart-turn is delay-only: TranscriptRuleTurnEnd
    consults state.smart_turn_incomplete to push a turn-end later. Its early-COMPLETE
    mispredictions — the source of premature replies + stitch breaks — can no longer end a turn."""

    def __init__(self, state, *, turn_analyzer, **kwargs):
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._state = state

    async def trigger_user_turn_stopped(self):
        return  # neutered: observe/predict only, never end the turn


class RevealGuard(FrameProcessor):
    """Interviewer NEVER reveals — corrections are the judge's lane (architecture rule). The
    conversation LLM still leaks the answer in its OPENING clause ("a stack is actually LIFO…")
    despite the prompt. Scout-era safety net: hold ONLY the first clause of each reply and scan
    it; if it reveals an answer term, REPLACE the whole reply with a neutral probe (keep the
    Start/End frames, swap the text). Clean replies stream normally — the first clause is held
    only ~as long as the TTS aggregates a sentence anyway, so ~no added latency. Placed right
    after the LLM so a blocked reveal is never logged or shown either."""

    # NEUTRAL on purpose: it neither confirms nor denies, so it's safe whether the candidate
    # was right or wrong. The old "that's worth reconsidering" implied WRONG — itself a
    # correctness signal the interviewer isn't allowed to give (and flat-out misleading when the
    # candidate was actually right, e.g. confirming an O(1) answer).
    _PROBE = "Walk me through your reasoning step by step — I want to follow exactly how you got there."
    _MAX_CLAUSE_CHARS = 90  # decide by here even without a sentence end

    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._buf = ""
        self._held: list[Frame] = []
        self._decided = False
        self._blocked = False
        self._drop = False  # drop the whole in-flight response (judge already handled this turn)

    def _reset(self):
        self._buf = ""
        self._held = []
        self._decided = False
        self._blocked = False

    async def _decide(self, direction: FrameDirection):
        self._decided = True
        # Block on EITHER an answer-term leak OR an explicit correctness verdict — both are the
        # interviewer overstepping into the judge's lane. Same handling: swap for a neutral probe.
        leaked = reveals_answer(self._buf)
        verdicted = confirms_or_denies(self._buf)
        if leaked or verdicted:
            self._blocked = True
            why = "revealed an answer" if leaked else "stated correctness"
            logger.warning(
                f"REVEAL_BLOCKED (interviewer): opener {why} — replaced reply with a neutral probe. "
                f"was: {self._buf!r}"
            )
            # Keep the Start (preserve Start/End pairing); swap the whole reply for a probe.
            for f in self._held:
                if isinstance(f, LLMFullResponseStartFrame):
                    await self.push_frame(f, direction)
            await self.push_frame(LLMTextFrame(text=self._PROBE), direction)
        else:
            for f in self._held:
                await self.push_frame(f, direction)
        self._held = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            # ANTI-DOUBLE OUTPUT GATE. If the judge already corrected this turn, the interviewer
            # must NOT also reply. The strategy-level latch suppresses the turn-end, but the
            # aggregator's user_turn_stop_timeout (5s) then FORCE-FIRES the conversation LLM past
            # it — that's the judge-correction-then-interviewer-reply double seen live. This is the
            # last chokepoint before the reply is logged/spoken/recorded: drop the whole response.
            if self._state.judge_corrected_this_turn:
                self._drop = True
                logger.info(
                    "REPLY_DROPPED (interviewer): judge already corrected this turn — "
                    "dropping the duplicate conversation reply"
                )
                return
            self._drop = False
            self._reset()
            self._held.append(frame)  # hold until the first clause is scanned
            return
        if self._drop:
            # Swallow the dropped response's body (text + end marker) so nothing reaches TTS /
            # the assistant context. Leave non-LLM frames (judge TTSSpeakFrame, control) untouched.
            if isinstance(frame, LLMFullResponseEndFrame):
                self._drop = False
                return
            if isinstance(frame, LLMTextFrame):
                return
        if isinstance(frame, LLMTextFrame):
            if not self._decided:
                self._buf += frame.text or ""
                self._held.append(frame)
                if re.search(r"[.?!]", self._buf) or len(self._buf) >= self._MAX_CLAUSE_CHARS:
                    await self._decide(direction)
                return
            if self._blocked:
                return  # drop the rest of the original (revealing) reply; probe already sent
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseEndFrame) and not self._decided:
            await self._decide(direction)  # short reply ended before the first clause completed
        await self.push_frame(frame, direction)


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
    # Turn END (STEP 5): deterministic TranscriptRuleTurnEnd is PRIMARY — smart-turn's ML
    # mispredicted BOTH ways on this speech (false COMPLETE on fragments -> premature replies
    # + stitch breaks; false INCOMPLETE on complete sentences -> 30s hangs). Rule: terminal
    # punctuation + ~1s confirm = done; continuation token (comma/conj/prep/article/aux/-ing)
    # = wait; 2.5s silence backstop otherwise. smart-turn is kept as a DELAY-ONLY observer
    # (SmartTurnObserver never triggers; LoggingSmartTurn records INCOMPLETE so the rule can
    # push a turn-end later). EmergencyTurnEnd (5s) + phantom guard + reply watchdog = floor.
    # on_bot_idle: when the playback-heartbeat reconciler detects the bot actually stopped
    # (transport skipped BotStoppedSpeaking), also clear the start strategy's private flag.
    barge = BargeInManager(state, on_bot_idle=start_strat.resync)

    def _full_resync():
        """Clear EVERY 'bot speaking' tracker (shared state + barge manager + start
        strategy) so a lost BotStoppedSpeaking can't desync the turn machine and swallow
        the next reply. Invoked by the watchdog on every forced turn-end."""
        barge.resync()
        start_strat.resync()

    # Latency trim. confirm 0.35s fires the COMMON path (26/28 turns end on terminal punct) a bit
    # faster; smart-turn INCOMPLETE delay 0.6s. silence_secs stays 2.5s ON PURPOSE — this speaker
    # pauses ~2s between fragments, so a shorter neutral-ending backstop would cut them off mid-
    # thought. If it ever cuts in while you're still talking, raise confirm_secs back toward 0.5.
    rule_stop = TranscriptRuleTurnEnd(state, confirm_secs=0.35, smart_delay_secs=0.6)
    smart_observer = SmartTurnObserver(
        state,
        turn_analyzer=LoggingSmartTurn(state, params=SmartTurnParams(stop_secs=3.0)),
    )
    turn_strategies = UserTurnStrategies(
        start=[start_strat],
        stop=[
            rule_stop,        # PRIMARY: deterministic transcript rule
            smart_observer,   # delay-only: feeds smart-turn predictions, never triggers
            EmergencyTurnEnd(
                state, silence_secs=5.0, watchdog_secs=4.0, pending_max=3.5,
                on_resync=_full_resync,
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
            RevealGuard(state),       # interviewer never reveals: scan opening clause, swap to probe
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

    greeted = {"done": False}

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("client connected — greeting")
        await ui_broadcast({"type": "status", "status": "listening"})
        # Idempotent: greet exactly once per pipeline, even if on_client_connected re-fires
        # (renegotiation / flaky reconnect on the same pipeline must not double-greet).
        if not greeted["done"]:
            greeted["done"] = True
            state.t_reply_claimed = time.monotonic()   # arm stall watchdog for the greeting too
            await task.queue_frames([LLMRunFrame()])  # bot greets first

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("client disconnected — ending pipeline")
        await runner.cancel()

    async def _reply_watchdog():
        # Demo-critical safety, TRANSCRIPT-DRIVEN (independent of the turn-state flags). A
        # smart-turn mispredict + stitch can break the turn so the forced turn-end produces no
        # reply, no UserStoppedSpeaking, and no bot_pending — so a flag-based watchdog never
        # arms and the bot sits silent until the 30s prompt. Instead: if the user has spoken
        # MORE RECENTLY than the bot last produced reply text, the bot is not speaking, and
        # reply_secs pass with still no reply -> force one via LLMRunFrame. (t_last_bot_text
        # updates on the first reply token, so a normal/slow-but-working reply clears this
        # before reply_secs; multi-fragment answers keep pushing t_last_user_text forward, so
        # it only fires reply_secs after you've truly stopped.)
        reply_secs = 7.0
        stall_secs = 12.0
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()

            # STALL RECOVERY (provider-agnostic: any LLM/TTS provider, or a network blip). A reply
            # was CLAIMED (turn-end fired / judge corrected / re-prompt) but produced ZERO output —
            # no LLM token, no TTS text, no bot audio — for > stall_secs. The LLM/TTS/network is
            # hung. turn_replied (correctly) blocks the reply-watchdog below while a reply is
            # pending, so without this the bot stays silent for as long as the provider hangs
            # (a ~74s reply stall from a network blip was seen live). Fix: cancel the stuck in-flight
            # generation FIRST (an InterruptionFrame is a SystemFrame -> the LLM closes its stream
            # on cancel) so its late frames can't double, then resync + re-prompt. The 12s window
            # is far longer than any real reply (~1s), so it never fires on a normal turn.
            if (
                state.t_reply_claimed > 0
                and now - state.t_reply_claimed > stall_secs
                and state.t_reply_text < state.t_reply_claimed
            ):
                held = now - state.t_reply_claimed
                logger.warning(
                    f"STALL_RECOVERY: reply claimed {held:.0f}s ago with no output — "
                    "cancelling stuck generation, resync + re-prompt"
                )
                await task.queue_frames([InterruptionFrame()])  # kill the hung generation
                await asyncio.sleep(0.15)                        # let the cancel land before re-prompt
                _full_resync()                                   # clear every 'bot busy' tracker
                state.turn_replied = False
                state.judge_corrected_this_turn = False
                state.bot_pending = False
                state.bot_speaking = False
                state.t_last_bot_text = now      # don't let the reply-watchdog also fire this tick
                state.t_reply_claimed = now       # re-arm: if the re-prompt ALSO hangs, recover again
                # leave t_reply_text untouched (stale) so a re-prompt that also stalls is re-detected
                await task.queue_frames([LLMRunFrame()])
                continue

            # Same single anti-double gate: only force a reply if the bot has NOT already replied
            # to (or isn't mid-replying to) the latest user content, and it's overdue.
            if (
                not reply_already_handled(state)
                and state.t_last_user_text > 0
                and now - state.t_last_user_text > reply_secs
            ):
                logger.warning(
                    f"REPLY_WATCHDOG: user spoke, bot owes a reply >{reply_secs}s — forcing via LLMRunFrame"
                )
                state.turn_replied = True      # claim the turn (latch) so we don't re-fire it
                state.t_last_bot_text = now    # backstop for the transcript-driven gate
                state.t_reply_claimed = now    # arm the stall watchdog for this forced reply
                await task.queue_frames([LLMRunFrame()])

    logger.info(
        f"bot ready — barge-in needs >=3 words OR >=0.7s voice; "
        f"backchannels never interrupt ({sorted(BACKCHANNEL_WORDS)})"
    )
    reply_wd = asyncio.create_task(_reply_watchdog())
    await runner.add_workers(task)
    await runner.run()
    reply_wd.cancel()
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
        conn = SmallWebRTCConnection(_to_ice_servers(await _fetch_ice_config()))
        await conn.initialize(sdp=request["sdp"], type=request["type"])

        @conn.event_handler("closed")
        async def _on_closed(c):
            _connections.pop(c.pc_id, None)

        asyncio.create_task(run_bot(conn))

    answer = conn.get_answer()
    _connections[answer["pc_id"]] = conn
    return JSONResponse(answer)


@app.get("/api/ice")
async def ice():
    """ICE servers for the browser: STUN + Metered TURN fallback (creds from env)."""
    return JSONResponse({"iceServers": await _fetch_ice_config()})


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
    port = int(os.getenv("PORT", "8000"))  # Render injects $PORT; 8000 for local dev
    logger.info(f"SViam WebRTC server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
