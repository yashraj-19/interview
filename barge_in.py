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
    BotSpeakingFrame,
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
from pipecat.turns.user_start.base_user_turn_start_strategy import (
    BaseUserTurnStartStrategy,
    UserTurnStartedParams,
)
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


# Explicit interrupt words: "stop, let me speak". These cut the bot off even on ONE word,
# bypassing the >=3-word barge-in gate (a clipped single "wait" should stop her).
_CUT_IN_WORDS = {
    "wait", "stop", "hold", "hang", "sorry", "actually", "no", "hey", "pause", "excuse",
}


def is_cut_in(text: str) -> bool:
    """True if the utterance OPENS with an explicit interrupt word (first word only, so a
    mid-sentence 'no'/'actually' in normal speech doesn't trip it)."""
    norm = _normalize(text)
    return bool(norm) and norm.split()[0] in _CUT_IN_WORDS


def reply_already_handled(state) -> bool:
    """THE single anti-double gate — consulted by EVERY turn-end path (transcript rule,
    EmergencyTurnEnd watchdog, reply watchdog) and the judge. Suppress a reply when the bot has
    ALREADY replied to the CURRENT user turn, so a second/third reply can't fire. True when any:
      - PRIMARY: the per-turn latch is set (state.turn_replied) — a responder already answered
        this turn. This is immune to the multi-fragment STT race: late finals of the SAME
        utterance (this user pauses between words) used to bump t_last_user_text past the claim
        and re-open the timestamp gate, letting EmergencyTurnEnd fire a duplicate ~5s later. The
        latch is cleared ONLY when a genuinely NEW user turn begins (BargeInManager), so trailing
        fragments cannot re-open it.
      - the bot produced reply text at/after the user's last transcript (kept as a backstop); OR
      - a reply is genuinely in flight (recent bot_pending 'thinking' or bot_speaking)."""
    if state.turn_replied:
        return True
    if state.t_last_user_text > 0 and state.t_last_bot_text >= state.t_last_user_text:
        return True
    now = time.monotonic()
    # bot_pending window trimmed 6.0 -> 2.5s. turn_replied (above) is the RELIABLE primary guard;
    # bot_pending is only a secondary in-flight catch, and it's PHANTOM-prone (a turn-end can fire
    # without a reply on a VAD/transcript race, leaving it stuck True). A real Groq reply clears it
    # via BotStartedSpeaking in ~1s, so 2.5s covers it — while a phantom stops suppressing the next
    # legit reply ~3.5s sooner (this caused an 8s hang before the network drop, observed live).
    return (state.bot_pending and now - state.t_bot_pending <= 2.5) or (
        state.bot_speaking and now - state.t_bot_speaking <= 15.0
    )


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

    async def trigger_user_turn_started(self):
        # STEP 4 full-duplex: broadcast an INTERRUPTION only when the bot is ACTUALLY speaking
        # (a real barge-in). Normal turn starts and the thinking-gap (handled by the stitch)
        # start the turn WITHOUT interrupting — so we never cancel a reply that isn't playing.
        # When constructed with enable_interruptions=False (steps 1-3) this never interrupts.
        await self._call_event_handler(
            "on_user_turn_started",
            UserTurnStartedParams(
                enable_interruptions=self._enable_interruptions and self._bot_speaking,
                enable_user_speaking_frames=self._enable_user_speaking_frames,
            ),
        )

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
            if is_cut_in(frame.text):
                logger.debug("barge-in: explicit interrupt word (cut in on 1 word)")
                await self._fire()  # "wait"/"stop"/"hold on" -> cut in immediately
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

    def __init__(
        self,
        state,
        tie_window_s: float = 0.3,
        stitch_window_s: float = 1.5,
        on_bot_idle=None,
        bot_idle_secs: float = 1.5,
        **kwargs,
    ):
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
        # Per-turn anti-double latch bookkeeping. _turn_ended gates clearing state.turn_replied:
        # we only treat a VAD-start as a NEW turn (and re-open the gate) if the previous turn has
        # actually ended. Starts True so the very first user turn is treated as fresh. This is
        # what makes the latch immune to mid-utterance VAD flicker and to a judge correction's
        # BotStartedSpeaking landing while the user is still talking the same wrong claim.
        self._turn_ended = True
        # Bot-playback heartbeat reconciler (SOURCE fix for the BotStoppedSpeaking desync).
        # The output transport emits BotSpeakingFrame every 0.2s ONLY while audio actually
        # plays. If state.bot_speaking is True but that heartbeat has stopped for bot_idle_secs,
        # the transport's BotStoppedSpeaking was skipped (it races on _tts_audio_received when
        # an extra mid-turn utterance — e.g. a judge correction — overlaps the reply). We then
        # flip bot_speaking back to False at the source, so the watchdog never has to.
        self._on_bot_idle = on_bot_idle
        self._bot_idle_secs = bot_idle_secs
        self._t_last_bot_heartbeat = 0.0
        self._reconciler: asyncio.Task | None = None

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

        # Lazily start the bot-playback reconciler once we're inside the running loop.
        if self._reconciler is None:
            self._reconciler = asyncio.create_task(self._reconcile_loop())

        if isinstance(frame, BotSpeakingFrame):
            # Playback heartbeat (~every 0.2s while the bot's audio actually plays).
            self._t_last_bot_heartbeat = time.monotonic()
            self._state.t_reply_text = time.monotonic()  # audio playing -> reply PROGRESS
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._t_vad = time.monotonic()
            self._user_speaking = True
            # NEW TURN -> re-open the anti-double gate, but ONLY if the previous turn actually
            # ended. A mid-utterance VAD restart (this user pauses between words) or a judge
            # correction's BotStartedSpeaking does NOT end the turn, so the latch holds and a
            # trailing fragment can't trigger a duplicate reply.
            if self._turn_ended:
                self._state.turn_replied = False
                self._state.judge_corrected_this_turn = False
                self._state.t_reply_claimed = 0.0   # disarm the stall watchdog for the fresh turn
                self._turn_ended = False
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
            # The turn has now genuinely ended: the NEXT VAD-start is a new turn that may re-open
            # the anti-double gate. (Trailing STT finals of THIS utterance arrive without a new
            # VAD-start, so they still can't re-open it.)
            self._turn_ended = True
        elif isinstance(frame, TTSTextFrame):
            if frame.text:
                self._spoken.append(frame.text)
                # The bot SPOKE something (LLM reply OR judge correction OR silence prompt — all
                # speak via TTS). That counts as a reply for the reply-watchdog gate, so it won't
                # force a duplicate after a judge cut-in (which never produces an LLMTextFrame).
                self._state.t_last_bot_text = time.monotonic()
                self._state.t_reply_text = time.monotonic()  # reply PROGRESS (un-stalls the watchdog)
        elif isinstance(frame, LLMTextFrame):
            if frame.text:
                self._generated.append(frame.text)
                self._state.t_last_bot_text = time.monotonic()  # reply text produced (reply-watchdog gate)
                self._state.t_reply_text = time.monotonic()     # reply PROGRESS (un-stalls the watchdog)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_pending = False
            self._stitch_locked = False  # bot is speaking -> stitching may re-arm
            self._state.bot_speaking = True
            self._state.bot_pending = False
            self._state.t_bot_speaking = time.monotonic()
            self._state.t_reply_text = time.monotonic()  # bot audio started -> reply PROGRESS
            self._t_last_bot_heartbeat = time.monotonic()
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

    async def _reconcile_loop(self):
        """SOURCE fix for the BotStoppedSpeaking desync (root cause, not watchdog-masked).
        The output transport emits BotSpeakingFrame ~every 0.2s ONLY while audio actually
        plays; its BotStoppedSpeaking can be SKIPPED (it gates on _tts_audio_received and a
        single _bot_speaking bool, which an extra mid-turn utterance — e.g. a judge correction
        — overlapping the reply can race), leaving bot_speaking stuck True. If the playback
        heartbeat has gone quiet for bot_idle_secs while bot_speaking is still True, the bot
        has actually stopped -> flip it False here so the 15s phantom watchdog never has to."""
        try:
            while True:
                await asyncio.sleep(0.3)
                if not self._state.bot_speaking:
                    continue
                idle = time.monotonic() - self._t_last_bot_heartbeat
                if idle <= self._bot_idle_secs:
                    continue
                logger.info(
                    f"BOT_SPEAKING reconciled -> False (playback heartbeat quiet {idle:.1f}s; "
                    "transport skipped BotStoppedSpeaking)"
                )
                self._state.bot_speaking = False
                self._awaiting_cut = False
                if self._on_bot_idle:
                    self._on_bot_idle()  # clear the start-strategy's private bot-speaking flag too
        except asyncio.CancelledError:
            return

    async def cleanup(self):
        if self._reconciler and not self._reconciler.done():
            self._reconciler.cancel()
        await super().cleanup()


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
        # THE single anti-double gate (rule + watchdog + reply-watchdog all consult it). If the
        # bot already replied to the latest user content (incl. a judge correction), this forced
        # reply would be a duplicate -> suppress. This is the backstop path the doubles slipped
        # through before (it fired a 2nd reply ~5s after the rule already answered).
        if reply_already_handled(self._state):
            logger.info("TURN_REPLY_SUPPRESSED path=watchdog: bot already replied to this turn")
            self._has_transcript = False
            return
        # Full resync FIRST: clear any stuck 'bot speaking' state across ALL trackers so the
        # forced turn-end actually produces a reply (a phantom start-strategy flag otherwise
        # blocks the turn from starting). Safe here — _end_turn only runs on watchdog/backstop
        # paths, where the bot is not legitimately speaking.
        if self._on_resync:
            self._on_resync()
        now = time.monotonic()
        self._state.turn_replied = True                 # CLAIM the turn (latch + timestamp)
        self._state.t_last_bot_text = now               # anti-race; backstop for reply watchdog
        self._state.t_reply_claimed = now               # arm the stall watchdog for this reply
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
            self._state.t_last_user_text = self._t_last_transcript  # for the transcript-driven reply watchdog
            self._restart()
        return ProcessFrameResult.CONTINUE


# ---------------------------------------------------------------------------
# Transcript-rule turn-end (PRIMARY). smart-turn's ML mispredicts on this speech in BOTH
# directions (false COMPLETE on fragments -> premature replies + stitch break; false
# INCOMPLETE on complete sentences -> 30s hangs). Deepgram smart_format punctuation is a far
# more honest, inspectable end-of-utterance signal here. Rule: terminal punctuation (after a
# short confirm-silence) = done; a continuation token at the end = wait; else a 2.5s silence
# backstop = done. smart-turn is kept ONLY as a delay-only secondary.
# ---------------------------------------------------------------------------
_CONJUNCTIONS = {
    "and", "but", "or", "so", "because", "nor", "yet", "while", "if", "when", "since",
    "although", "though", "unless", "as", "whereas", "plus", "also", "then", "that", "which",
}
_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "of", "from", "by", "about", "into", "onto", "over",
    "under", "between", "through", "during", "before", "after", "above", "below", "near", "off",
    "without", "within", "upon", "per", "like", "than", "toward", "towards", "around",
}
_ARTICLES_DET = {
    "a", "an", "the", "my", "your", "his", "her", "their", "its", "our", "this", "these", "those",
}
_AUX_BARE_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i'm", "i've", "i'd", "it's", "that's", "there's", "we're", "they're", "you're", "i'll",
}
_CONTINUATION_TOKENS = _CONJUNCTIONS | _PREPOSITIONS | _ARTICLES_DET | _AUX_BARE_VERBS


def ends_in_terminal_punct(text: str) -> str:
    """Terminal punctuation char if the utterance ends in . ? ! else ''."""
    t = text.rstrip()
    return t[-1] if t.endswith((".", "?", "!")) else ""


def ends_in_continuation(text: str) -> str:
    """Trailing continuation token if the utterance ends mid-thought (-> WAIT), else ''.
    Covers a trailing comma (mid-clause/list), and a last word that is a conjunction,
    preposition, article/determiner, auxiliary/contraction verb, or gerund/dangling verb (-ing)."""
    t = text.rstrip()
    if t.endswith(","):
        return ","  # trailing comma -> mid-clause, keep waiting
    norm = re.sub(r"[^\w\s'-]", "", t.lower()).strip()
    if not norm:
        return ""
    last = norm.split()[-1]
    if last in _CONTINUATION_TOKENS:
        return last
    if last.endswith("ing") and len(last) > 4:  # gerund / dangling verb: "building", "brushing"
        return last
    return ""


class TranscriptRuleTurnEnd(BaseUserTurnStopStrategy):
    """PRIMARY turn-end: deterministic, inspectable, transcript-driven.

    - terminal punctuation (. ? !) + `confirm_secs` of silence -> DONE (the confirm window
      absorbs multi-sentence answers and spurious mid-thought periods).
    - a continuation token at the end -> WAIT (never fire; defer to a later transcript or the
      EmergencyTurnEnd floor).
    - otherwise -> DONE on `silence_secs` of silence (neutral-ending backstop).

    smart-turn is DELAY-ONLY: if it currently predicts INCOMPLETE we push the turn-end one
    `smart_delay_secs` window later (once), never earlier — its COMPLETE predictions are
    ignored. De-dup guard + EmergencyTurnEnd/reply watchdog remain the floors.
    """

    def __init__(
        self,
        state,
        *,
        confirm_secs: float = 0.7,
        silence_secs: float = 2.5,
        smart_delay_secs: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._confirm_secs = confirm_secs
        self._silence_secs = silence_secs
        self._smart_delay_secs = smart_delay_secs
        self._accum = ""
        self._timer: asyncio.Task | None = None
        self._delayed = False

    async def reset(self):
        await super().reset()
        self._accum = ""
        self._delayed = False
        self._cancel()

    async def cleanup(self):
        self._cancel()
        await super().cleanup()

    def _cancel(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _arm(self, delay: float):
        self._cancel()
        self._timer = asyncio.create_task(self._fire_after(delay))

    async def _fire_after(self, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        await self._decide()

    async def _decide(self):
        text = self._accum.strip()
        punct = ends_in_terminal_punct(text)
        token = ends_in_continuation(text)
        # smart-turn DELAY-ONLY: if it just said INCOMPLETE, push the turn-end one window later
        # (once). Its COMPLETE predictions are ignored. Floors below cap any hang.
        smart_incomplete = (
            self._state.smart_turn_incomplete
            and (time.monotonic() - self._state.t_smart_turn) < 3.0
        )
        if smart_incomplete and not self._delayed:
            self._delayed = True
            logger.info(
                f"TURN_RULE delay=smart-turn-INCOMPLETE wait={self._smart_delay_secs}s "
                f"end='{token or 'neutral'}'"
            )
            self._arm(self._smart_delay_secs)
            return
        trigger = "terminal-punct" if punct else "silence-backstop"
        end_desc = f"punct:{punct}" if punct else (token or "neutral")
        logger.info(f"TURN_RULE fire trigger={trigger} end='{end_desc}' smart_delayed={self._delayed}")
        self._state.last_turn_trigger = trigger
        self._delayed = False
        await self.trigger_user_turn_stopped()

    async def trigger_user_turn_stopped(self):
        # THE single anti-double gate (rule + watchdog + reply-watchdog all consult it).
        if reply_already_handled(self._state):
            logger.info("TURN_REPLY_SUPPRESSED path=rule: bot already replied to this turn")
            return
        # CLAIM the turn SYNCHRONOUSLY (no await before this) so the judge / watchdog see it and
        # don't also respond. The latch is the real guard (survives late STT fragments); the
        # timestamp stamp is a backstop for the transcript-driven reply watchdog.
        now = time.monotonic()
        self._state.turn_replied = True
        self._state.t_last_bot_text = now
        self._state.t_reply_claimed = now   # arm the stall watchdog for this reply
        self._state.bot_pending = False
        self._state.bot_speaking = False
        await super().trigger_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._accum = f"{self._accum} {frame.text}".strip()
            self._delayed = False  # new content -> smart-turn gets a fresh single delay budget
            token = ends_in_continuation(self._accum)
            punct = ends_in_terminal_punct(self._accum)
            if token:
                self._cancel()  # WAIT — never fire on a continuation token
                logger.info(f"TURN_RULE wait end='{token}' (continuation) — not firing")
            elif punct:
                self._arm(self._confirm_secs)  # terminal punct -> short confirm window
            else:
                self._arm(self._silence_secs)  # neutral ending -> silence backstop
        return ProcessFrameResult.CONTINUE
