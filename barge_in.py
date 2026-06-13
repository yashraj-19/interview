"""Human-like barge-in for the WebRTC interview (CLAUDE.md rules 1, 2, 4, 5).

All timing here is in the FRAME layer — no LLM call is part of the stop/start
decision path (rule 1). The only LLM touch is the resumption *note* appended for
the NEXT turn (rule 4), which is not a timing decision.

Verified against Pipecat 1.3.0:
- Interruptions are produced by a user-turn-START strategy firing while the bot
  speaks (it calls trigger_user_turn_started(), and enable_interruptions=True ->
  the user aggregator broadcasts an InterruptionFrame).
- The assistant context aggregator already commits ONLY spoken text on interrupt
  (TTS sets the LLM text frame append_to_context=False and emits spoken
  TTSTextFrame with append_to_context=True), and TTS/output flush on interrupt.
  So truncation + stale-audio cancellation are native; we only verify + log them.

Two pieces:
- HumanBargeInStartStrategy: backchannel-immune turn-start (sub-task 1).
- BargeInManager: latency logging (2), dropped_thought + resume note (4),
  double-talk tie guard (5), and truncation verification logging (3).
"""

import asyncio
import os
import re
import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

_DEFAULT_BACKCHANNELS = "hmm,yeah,okay,ok,right,uh-huh,mhm,i see,got it"
BACKCHANNEL_WORDS = {
    w.strip().lower()
    for w in os.getenv("BACKCHANNEL_WORDS", _DEFAULT_BACKCHANNELS).split(",")
    if w.strip()
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s'-]", "", text.lower()).strip()


def nonbackchannel_word_count(text: str) -> int:
    """Words spoken that are NOT backchannels. A whole utterance that is just a
    backchannel phrase ('i see', 'got it') counts as 0."""
    norm = _normalize(text)
    if not norm or norm in BACKCHANNEL_WORDS:
        return 0
    return len([w for w in norm.split() if w not in BACKCHANNEL_WORDS])


class HumanBargeInStartStrategy(BaseUserTurnStartStrategy):
    """Turn-start that makes barge-in feel intentional, not hair-trigger.

    While the bot is NOT speaking: start the turn immediately on voice/words
    (responsive normal turns). While the bot IS speaking (barge-in candidate):
    only start (== interrupt) when the user's speech is SUSTAINED
    (>= sustained_secs continuous voice) OR has >= min_words non-backchannel
    words. Pure backchannels never interrupt.
    """

    def __init__(self, *, min_words: int = 3, sustained_secs: float = 0.7, use_interim: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._min_words = min_words
        self._sustained_secs = sustained_secs
        self._use_interim = use_interim
        self._bot_speaking = False
        self._triggered = False
        self._voice_task: asyncio.Task | None = None

    async def reset(self):
        await super().reset()
        self._triggered = False
        self._cancel_voice_task()

    def resync(self):
        """Clear stuck bot-speaking state after a TTS connection drop, so a single word
        can start the next turn again (not gated to >= min_words by a phantom 'bot speaking')."""
        self._bot_speaking = False
        self._triggered = False
        self._cancel_voice_task()

    def _cancel_voice_task(self):
        if self._voice_task and not self._voice_task.done():
            self._voice_task.cancel()
        self._voice_task = None

    async def _fire(self):
        if self._triggered:
            return
        self._triggered = True
        self._cancel_voice_task()
        await self.trigger_user_turn_started()

    async def _sustained_voice(self):
        try:
            await asyncio.sleep(self._sustained_secs)
        except asyncio.CancelledError:
            return
        if self._bot_speaking and not self._triggered:
            logger.debug(f"barge-in: sustained voice >= {self._sustained_secs}s")
            await self._fire()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._triggered = False
            self._cancel_voice_task()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._cancel_voice_task()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                await self._fire()  # normal turn: start immediately
                return ProcessFrameResult.STOP
            # barge-in candidate: arm the sustained-voice timer
            self._triggered = False
            self._cancel_voice_task()
            self._voice_task = asyncio.create_task(self._sustained_voice())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._cancel_voice_task()  # voice was not sustained
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if isinstance(frame, InterimTranscriptionFrame) and not self._use_interim:
                return ProcessFrameResult.CONTINUE
            if not self._bot_speaking:
                await self._fire()  # normal turn: any words start it
                return ProcessFrameResult.STOP
            if nonbackchannel_word_count(frame.text) >= self._min_words:
                await self._fire()  # barge-in: enough real words
                return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


RESUME_NOTE = (
    "(You were interrupted mid-answer by the candidate. React to what they just said FIRST. "
    "You may resume your earlier point only if it is still relevant, and rephrase it — "
    "never say 'as I was saying'.{dropped})"
)


class BargeInManager(FrameProcessor):
    """Frame-layer barge-in support. Place AFTER the TTS service (so it sees both
    LLMTextFrame=generated and TTSTextFrame=spoken) and before transport.output().

    - Logs BARGE_LATENCY: gate (vad-start -> interrupt) and cut (interrupt -> silence).
    - Verifies native truncation (spoken vs generated word counts).
    - Stores the unspoken remainder in state.dropped_thought and appends a system
      resume-note for the next turn (run_llm=False).
    - Double-talk tie guard: if the bot starts within tie_window_s of the user
      starting, the bot yields (user wins ties).
    """

    def __init__(self, state, tie_window_s: float = 0.3, stitch_window_s: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._tie_window_s = tie_window_s
        self._stitch_window_s = stitch_window_s
        self._generated: list[str] = []
        self._spoken: list[str] = []
        self._t_vad = 0.0
        self._t_interrupt = 0.0
        self._t_user_stopped = 0.0   # last VAD user-stop (for TURN_END_LATENCY)
        self._t_pending = 0.0        # when the turn ended and the bot started "thinking"
        self._user_speaking = False
        self._awaiting_cut = False
        self._bot_pending = False    # turn ended, LLM running, bot not speaking yet
        self._stitch_locked = False  # stitch at most once per cycle (until bot speaks)

    def resync(self):
        """Recover from a TTS connection drop (e.g. Deepgram 1011). The bot's audio died
        mid-stream, so BotStoppedSpeakingFrame may never arrive and bot_speaking/bot_pending
        would stay True forever — the turn-end gate then blocks and the bot goes permanently
        silent. Clear all half-open interrupt/turn state so the turn machine re-arms."""
        self._awaiting_cut = False
        self._bot_pending = False
        self._stitch_locked = False
        self._generated.clear()
        self._spoken.clear()
        self._state.bot_speaking = False
        self._state.bot_pending = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._t_vad = time.monotonic()
            self._user_speaking = True
            # Fragment stitching: the user resumed AFTER the turn ended but BEFORE
            # the bot started speaking -> cancel the pending reply so the new speech
            # stitches onto the answer (bot replies once, when the user is truly done).
            if (
                self._bot_pending
                and not self._stitch_locked
                and (self._t_vad - self._t_pending) < self._stitch_window_s
            ):
                logger.info(
                    "TURN_EXTENDED stitch: user resumed before bot spoke — cancelling pending reply"
                )
                self._bot_pending = False
                self._state.bot_pending = False   # clear the SHARED flag too — else the
                self._state.bot_speaking = False  # turn-end watchdog waits forever (silence)
                self._stitch_locked = True  # once per cycle; re-arms when the bot speaks
                await self.broadcast_interruption()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._t_user_stopped = time.monotonic()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # The user TURN ended (stop strategy fired) -> bot is "thinking" now.
            self._bot_pending = True
            self._t_pending = time.monotonic()
            self._state.bot_pending = True
            self._state.t_bot_pending = self._t_pending
        elif isinstance(frame, TTSTextFrame):
            if frame.text:
                self._spoken.append(frame.text)
        elif isinstance(frame, LLMTextFrame):
            if frame.text:
                self._generated.append(frame.text)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_pending = False
            self._stitch_locked = False  # bot is speaking -> stitching may re-arm
            self._state.bot_speaking = True
            self._state.bot_pending = False
            self._state.t_bot_speaking = time.monotonic()
            if self._t_user_stopped:
                latency = (time.monotonic() - self._t_user_stopped) * 1000
                trigger = self._state.last_turn_trigger or "?"
                logger.info(f"TURN_END trigger={trigger} latency={latency:.0f}ms")
                self._state.last_turn_trigger = ""
            # Double-talk tie guard — user wins ties.
            if (
                self._user_speaking
                and self._t_vad
                and (time.monotonic() - self._t_vad) < self._tie_window_s
            ):
                logger.info(
                    f"BARGE_LATENCY tie: user started within {self._tie_window_s * 1000:.0f}ms "
                    "of bot — bot yields"
                )
                await self.push_frame(frame, direction)
                await self.broadcast_interruption()
                return
        elif isinstance(frame, InterruptionFrame):
            # Forward FIRST so the assistant aggregator commits the spoken text,
            # THEN append the resume note (correct context order).
            await self.push_frame(frame, direction)
            await self._on_interrupt()
            return
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._state.bot_speaking = False
            if self._awaiting_cut:
                now = time.monotonic()
                cut = (now - self._t_interrupt) * 1000
                total = (now - self._t_vad) * 1000 if self._t_vad else cut
                logger.info(
                    f"BARGE_LATENCY cut(interrupt->silence)={cut:.0f}ms total(vad->silence)={total:.0f}ms"
                )
                self._awaiting_cut = False
            else:
                self._generated.clear()  # bot finished cleanly — nothing dropped
                self._spoken.clear()

        await self.push_frame(frame, direction)

    async def _on_interrupt(self):
        now = time.monotonic()
        self._t_interrupt = now
        gate = (now - self._t_vad) * 1000 if self._t_vad else 0.0
        logger.info(f"BARGE_LATENCY gate(vad->interrupt)={gate:.0f}ms")
        self._awaiting_cut = True

        generated = "".join(self._generated).strip()
        spoken = "".join(self._spoken).strip()
        if spoken:
            # Bot spoke part of a real answer -> the unspoken remainder is resumable.
            gen_words = generated.split()
            spoken_n = len(spoken.split())
            dropped = " ".join(gen_words[spoken_n:]).strip()
            logger.info(
                f"INTERRUPT truncation: spoken_words={spoken_n} generated_words={len(gen_words)} "
                f"dropped={'yes' if dropped else 'no'}"
            )
            self._state.dropped_thought = dropped
            if dropped:
                note = RESUME_NOTE.format(dropped=f' Your dropped point was: "{dropped}"')
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "system", "content": note}], run_llm=False
                    )
                )
        else:
            # Interrupted before any audio (fragment stitch) -> nothing to resume.
            self._state.dropped_thought = ""

        self._generated.clear()
        self._spoken.clear()


class EmergencyTurnEnd(BaseUserTurnStopStrategy):
    """Transcript-driven turn-end + reliability watchdog (CLAUDE.md rule 1: frame-layer,
    no LLM). Immune to WebRTC audio gaps/DTX — it's a real wall-clock timer, not audio-frame
    counting. Runs in PARALLEL with any other stop strategy (first to trigger wins).

    Phase 1 — normal turn-end: fire `silence_secs` after the LAST finalized transcript,
    as long as the bot isn't already speaking/thinking.

    Phase 2 — WATCHDOG: if the bot merely *looked* busy at the Phase-1 fire time, do NOT
    disarm. Keep watching. If a finalized transcript stays unanswered while the bot is
    neither speaking nor generating for `watchdog_secs`, the turn state is stuck (e.g. a
    Deepgram TTS reconnect left bot_speaking True) -> force the turn end and log
    WATCHDOG_RESYNC. This is what guarantees the bot never goes permanently silent after a
    TTS hiccup. It only fires once the resync (see BargeInManager.resync) has cleared the
    phantom bot_speaking/bot_pending flags.
    """

    def __init__(
        self,
        state,
        *,
        silence_secs: float = 2.5,
        watchdog_secs: float = 4.0,
        pending_max: float = 6.0,
        speaking_max: float = 15.0,
        on_resync=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._silence_secs = silence_secs
        self._watchdog_secs = watchdog_secs
        # Phantom-flag guard: a bot_pending/bot_speaking flag stuck True far longer than any
        # real LLM generation (~1-2s) or spoken reply (~<15s) means the flag is phantom — the
        # reply was lost (e.g. a stitch cancelled it, an LLM error, a TTS drop). The watchdog
        # then clears the flag and forces the turn, so it can never wait on it forever.
        self._pending_max = pending_max
        self._speaking_max = speaking_max
        # on_resync: a full resync of EVERY 'bot speaking' tracker (start strategy etc.), not
        # just state.bot_speaking. On a laggy link BotStoppedSpeaking can be lost, leaving the
        # start strategy's private copy stuck True -> it then refuses to start normal turns and
        # the watchdog's forced turn-end produces NO reply (seen live). Calling this on every
        # watchdog intervention keeps all trackers in sync.
        self._on_resync = on_resync
        self._has_transcript = False
        self._t_last_transcript = 0.0
        self._monitor: asyncio.Task | None = None

    async def reset(self):
        await super().reset()
        self._has_transcript = False
        self._cancel()

    async def cleanup(self):
        self._cancel()
        await super().cleanup()

    def _cancel(self):
        if self._monitor and not self._monitor.done():
            self._monitor.cancel()
        self._monitor = None

    def _restart(self):
        self._cancel()
        self._monitor = asyncio.create_task(self._monitor_turn())

    async def _end_turn(self, trigger: str, message: str):
        # Full resync FIRST: clear any stuck 'bot speaking' state across ALL trackers so the
        # forced turn-end actually produces a reply (a phantom start-strategy flag otherwise
        # blocks the turn from starting). Safe here — _end_turn only runs on watchdog/backstop
        # paths, where the bot is not legitimately speaking.
        if self._on_resync:
            self._on_resync()
        self._state.last_turn_trigger = trigger
        self._has_transcript = False
        logger.warning(message)
        await self.trigger_user_turn_stopped()

    async def _monitor_turn(self):
        try:
            # Phase 1 — normal turn-end after the user goes silent.
            await asyncio.sleep(self._silence_secs)
            if not self._has_transcript:
                return
            if not (self._state.bot_speaking or self._state.bot_pending):
                await self._end_turn(
                    "watchdog",
                    f"WATCHDOG backstop: no turn-end {self._silence_secs}s after last "
                    "transcript (smart-turn never fired) — forcing turn end",
                )
                return
            # Phase 2 — bot looked busy; keep watching until it's genuinely idle, OR until a
            # busy flag proves to be a PHANTOM (stuck True with no real progress).
            while True:
                await asyncio.sleep(0.5)
                if not self._has_transcript:
                    return  # turn was handled cleanly (reply happened, reset() ran)
                now = time.monotonic()
                speaking, pending = self._state.bot_speaking, self._state.bot_pending
                if not (speaking or pending):
                    # bot genuinely idle with an unanswered transcript -> force the turn.
                    idle = now - self._t_last_transcript
                    if idle >= self._watchdog_secs:
                        await self._end_turn(
                            "watchdog",
                            f"WATCHDOG_RESYNC: unanswered transcript idle {idle:.1f}s, bot "
                            "neither speaking nor generating — forcing turn end",
                        )
                        return
                    continue
                # bot claims busy — but a flag stuck True too long is a phantom (lost reply).
                pending_stuck = pending and self._state.t_bot_pending and (
                    now - self._state.t_bot_pending > self._pending_max
                )
                speaking_stuck = speaking and self._state.t_bot_speaking and (
                    now - self._state.t_bot_speaking > self._speaking_max
                )
                if pending_stuck or speaking_stuck:
                    which = "bot_pending" if pending_stuck else "bot_speaking"
                    held = now - (
                        self._state.t_bot_pending if pending_stuck else self._state.t_bot_speaking
                    )
                    self._state.bot_pending = False
                    self._state.bot_speaking = False
                    await self._end_turn(
                        "watchdog",
                        f"WATCHDOG_RESYNC: phantom {which} stuck True {held:.1f}s — clearing "
                        "and forcing turn end",
                    )
                    return
                # legitimately busy (speaking/generating within limits) — keep waiting.
        except asyncio.CancelledError:
            return

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        # Transcript-DRIVEN (not VAD/audio-driven): each finalized transcript restarts the
        # clock, so it waits through brief pauses; when you truly stop, it ends the turn.
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._has_transcript = True
            self._t_last_transcript = time.monotonic()
            self._restart()
        return ProcessFrameResult.CONTINUE
