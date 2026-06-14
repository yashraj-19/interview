"""SViam voice interviewer — Phase 1: live voice loop (English only).

Pipeline (Pipecat 1.3.0, local mic + speakers):

    mic ─▶ VAD (Silero) ─▶ Deepgram STT (nova-3) ─▶ user-context
        ─▶ Gemini 2.5 Flash (conversation brain) ─▶ Deepgram Aura TTS ─▶ speakers

Barge-in is handled by the framework: the VAD turn-start strategy interrupts
the bot the moment you start talking.

Run:
    venv\\Scripts\\python.exe voice_agent.py

Talk to it. Press Ctrl+C to stop.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
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
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.local.audio import LocalAudioTransportParams

from judge import JUDGE_MODEL, JUDGE_PROVIDER, judge_answer
from smooth_audio import MAX_BUFFER_MS, SmoothLocalAudioTransport
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

# Conversation brain. Gemini's free tier (5 req/min) can't sustain live voice,
# so we default to Groq Llama 3.3 70B (free, ~30 req/min, fast). Set
# CONV_LLM_PROVIDER=gemini in .env to switch back once Gemini billing is enabled.
CONV_LLM_PROVIDER = os.getenv("CONV_LLM_PROVIDER", "groq").strip().lower()
# llama-4-scout-17b follows the "probe, never reveal the answer" rule far better
# than 8b-instant (which kept giving answers away), and has free budget. Override
# via GROQ_CONV_MODEL in .env (e.g. llama-3.3-70b-versatile when its cap resets).
GROQ_CONV_MODEL = os.getenv("GROQ_CONV_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GEMINI_MODEL = "gemini-2.5-flash"

TTS_VOICE = "aura-2-helena-en"  # English female Aura-2 voice

# Barge-in. Half-duplex (mic muted while the bot talks) prevents echo but blocks
# interrupting the bot. Set HALF_DUPLEX=false (with an ISOLATED headset, and output
# routed to it) to allow real barge-in. Pin devices with AUDIO_IN_INDEX/AUDIO_OUT_INDEX
# (run audio_diag.py to list indices) if the system default isn't your headset.
HALF_DUPLEX = os.getenv("HALF_DUPLEX", "true").strip().lower() not in ("false", "0", "no")
AUDIO_IN_INDEX = os.getenv("AUDIO_IN_INDEX")
AUDIO_OUT_INDEX = os.getenv("AUDIO_OUT_INDEX")

SYSTEM_PROMPT = """You are SViam, an autonomous AI technical interviewer conducting a live voice interview.

STYLE (this is a VOICE conversation, so):
- Keep every response SHORT: 1-3 sentences. Never lecture.
- Speak naturally and conversationally, like a real human interviewer.
- One question at a time. Wait for the candidate to answer.
- Do not output code, markdown, bullet points, or symbols — only spoken words.

YOUR JOB:
- Greet the candidate briefly, then ask them to introduce themselves.
- After that, ask one focused technical question (e.g. about data structures,
  algorithms, or a past project) and probe their reasoning with follow-ups.
- When an answer is vague, ask "why" or "can you be more specific".
- Be professional, warm, and curious about HOW they think, not just the answer.

NEVER GIVE AWAY ANSWERS (very important):
- NEVER state the correct answer, value, complexity, definition, or optimal approach. Not even
  after several attempts. Revealing the answer is handled by a separate system, never by you.
- If the candidate is wrong, do NOT correct them with the answer. Briefly note that something
  seems off and ask a question that leads them to reconsider and find it themselves.
- If their answer works but is suboptimal (e.g. brute force), ASK whether they can do better or
  improve the complexity. Do NOT tell them the optimization.
- Do not confirm or deny specific values (e.g. never say "it's actually O(n)"). Just probe.
- You probe; the candidate solves. Make them do the thinking.

HANDLING THE CANDIDATE'S META-REQUESTS:
- If they ask you to repeat or clarify ("can you repeat that?", "what do you mean?", "I didn't
  catch that"), REPHRASE the question in different words (never word-for-word). Do NOT treat this
  as a wrong answer or hold it against them.
- If they ask a genuinely good clarifying question (e.g. "should it handle negative numbers?"),
  answer it briefly and acknowledge it's a good thing to consider.
- If they ask for a moment ("can I have a moment?", "let me think", "give me a second"), briefly
  say "sure, take your time" and then STOP — do not keep talking.
- If they signal they're ready ("I'm ready", "okay", "go ahead"), just continue with the question.
"""


# ---------------------------------------------------------------------------
# Small helper processor: log final candidate transcripts so we can see them
# ---------------------------------------------------------------------------
class ConversationState:
    """Shared state between processors. Holds the AI's most recent question so
    the interrupt judge can evaluate the candidate's answer against it (no
    hardcoded answer key — the judge uses the real question + its own knowledge)."""

    def __init__(self):
        self.last_question = ""
        self.ask_back_count = 0  # clarification requests on the current question
        self.dropped_thought = ""  # unspoken remainder when the bot is interrupted
        self.last_turn_trigger = ""  # how the last user turn ended: smart | fallback | force
        self.bot_speaking = False    # bot is currently producing audio
        self.bot_pending = False     # a turn ended; the LLM reply is in flight
        self.t_bot_speaking = 0.0    # monotonic time bot_speaking last went True (phantom-flag guard)
        self.t_bot_pending = 0.0     # monotonic time bot_pending last went True (phantom-flag guard)
        self.t_last_bot_text = 0.0   # monotonic time the bot last produced reply text (LLMTextFrame)


ASK_BACK_PHRASES = (
    "repeat", "say that again", "what do you mean", "didn't catch", "did not catch",
    "can you clarify", "rephrase", "come again", "what was the question", "i didn't hear",
    "i did not hear", "sorry, what", "pardon", "didn't understand", "did not understand",
)


class DynamicsTracker(FrameProcessor):
    """Detects 'ask-back' — the candidate asking the interviewer to repeat/clarify — and
    tracks it as metadata (Deliverable 1.3). The interviewer LLM does the actual rephrasing
    and does NOT penalize it (see the system prompt); this just counts it and flags possible
    stalling if the candidate keeps asking on the same question.
    """

    def __init__(self, state: "ConversationState", **kwargs):
        super().__init__(**kwargs)
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            low = frame.text.lower()
            if any(p in low for p in ASK_BACK_PHRASES):
                self._state.ask_back_count += 1
                logger.info(
                    f"ASK-BACK> candidate asked for clarification "
                    f"({self._state.ask_back_count}x on this question) — not penalized"
                )
                if self._state.ask_back_count >= 3:
                    logger.warning(
                        "ASK-BACK> 3rd+ clarification on the same question — possible stalling"
                    )
        await self.push_frame(frame, direction)


class TranscriptLogger(FrameProcessor):
    """Logs candidate transcripts and the bot's spoken responses.

    Placed twice in the pipeline: one instance after STT (sees candidate
    TranscriptionFrames) and one after the LLM (accumulates LLMTextFrames into
    the full bot reply). The bot-side instance also records the AI's last
    question into the shared ConversationState for the interrupt judge.
    """

    def __init__(self, state: "ConversationState | None" = None, **kwargs):
        super().__init__(**kwargs)
        self._bot_text = ""
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            logger.info(f"CANDIDATE> {frame.text}")
        elif isinstance(frame, LLMTextFrame):
            self._bot_text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = self._bot_text.strip()
            if text:
                logger.info(f"BOT> {text}")
                if self._state is not None:
                    self._state.last_question = text
                    self._state.ask_back_count = 0  # new question -> fresh clarification count
            self._bot_text = ""
        await self.push_frame(frame, direction)


class InterruptJudge(FrameProcessor):
    """Wrong-answer interrupt.

    Watches the candidate's FINALIZED transcript while they answer (no interim
    partials), and only judges a complete claim.
    Throttled to stay under free-tier limits: judges at most once every
    ``min_interval_s`` AND only after ``min_new_words`` new words. Sends the AI's
    actual last question + the live answer to a swappable judge LLM (see judge.py).
    On a clear error it makes the bot speak a short natural cut-in via TTSSpeakFrame.
    Biased hard toward CONTINUE; at most one interrupt per answer.
    """

    def __init__(
        self,
        state: "ConversationState",
        min_interval_s: float = 2.5,
        min_new_words: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._min_interval_s = min_interval_s
        self._min_new_words = min_new_words
        self._reset_answer()
        self._judging = False
        # Consecutive wrong COMPLETE answers on the current topic. Drives the reveal
        # escalation (1=nudge, 2=hint, 3+=give answer). Persists across answers;
        # reset to 0 when the candidate gives a complete answer that isn't wrong.
        self._wrong_streak = 0

    def _reset_answer(self):
        self._final_text = ""
        self._last_judge_t = 0.0
        self._words_at_last_judge = 0
        self._interrupted = False

    def _spoken(self) -> str:
        return self._final_text.strip()

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """A finalized utterance that forms a complete-enough claim to judge."""
        return text.endswith((".", "?", "!")) or len(text.split()) >= 12

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame):
            # Bot just finished a turn -> a fresh candidate answer starts.
            self._reset_answer()
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # Judge on FINALIZED transcript only — never interim partials.
            self._final_text = f"{self._final_text} {frame.text}".strip()
            await self._maybe_judge()
        await self.push_frame(frame, direction)

    async def _maybe_judge(self):
        if self._interrupted or self._judging or not self._state.last_question:
            return
        spoken = self._spoken()
        if not self._looks_complete(spoken):  # require a complete claim first
            return
        words = len(spoken.split())
        now = time.monotonic()
        if (now - self._last_judge_t) < self._min_interval_s:
            return
        if (words - self._words_at_last_judge) < self._min_new_words:
            return
        self._last_judge_t = now
        self._words_at_last_judge = words
        self._judging = True
        asyncio.create_task(
            self._run_judge(self._state.last_question, spoken, self._wrong_streak + 1)
        )

    async def _run_judge(self, question: str, answer: str, attempt: int):
        t0 = time.monotonic()
        try:
            verdict = await judge_answer(question, answer, attempt)
            ms = (time.monotonic() - t0) * 1000
            action = "interrupt" if verdict.get("interrupt") else "continue"
            logger.info(f"JUDGE call -> verdict={action} ms={ms:.0f} attempt={attempt}")
            if verdict.get("interrupt") and verdict.get("line"):
                if not self._interrupted:
                    self._interrupted = True       # at most one interrupt per answer
                    self._wrong_streak += 1        # escalate next time on this topic
                    line = verdict["line"]
                    logger.warning(f"INTERRUPT (attempt {attempt})> {line}")
                    await self.push_frame(TTSSpeakFrame(text=line, append_to_context=True))
            else:
                # A complete answer judged not-wrong -> recovered; reset escalation.
                self._wrong_streak = 0
        except Exception as e:
            # Loud, but NEVER breaks the pipeline (frames keep flowing).
            logger.exception(f"InterruptJudge error (non-fatal): {e}")
        finally:
            self._judging = False


# Silence ladder thresholds (seconds of accumulated silence). README spec is
# 30 / 90 / 120; override with SILENCE_SECS="12,30,50" for quick testing.
try:
    _SIL = [float(x) for x in os.getenv("SILENCE_SECS", "30,90,120").split(",")][:3]
    assert len(_SIL) == 3
except Exception:
    _SIL = [30.0, 90.0, 120.0]
SILENCE_PROMPTS = [
    (_SIL[0], "speak", "Take your time. Let me know when you're ready."),
    (_SIL[1], "speak", "Would you like me to rephrase the question?"),
    (_SIL[2], "move_on", ""),  # ask the LLM to acknowledge and move to a new question
]


class SilenceMonitor(FrameProcessor):
    """Prompts the candidate after sustained silence (Deliverable 1.2).

    Accumulates 'silent' time — neither the bot nor the candidate speaking — and
    resets it whenever the candidate speaks. Fires an escalating ladder: a gentle
    "take your time", then "want me to rephrase?", then asks the interviewer LLM to
    move on to a new question. Goes quiet after the last step until the candidate
    speaks again (so it never loops forever on an absent candidate).
    """

    def __init__(self, prompts=SILENCE_PROMPTS, tick: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._prompts = prompts
        self._tick = tick
        self._bot_speaking = False
        self._user_speaking = False
        self._silent_secs = 0.0
        self._level = 0
        self._task = None

    def _reset(self):
        self._silent_secs = 0.0
        self._level = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            if self._task is None:
                self._task = asyncio.create_task(self._monitor())
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._reset()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._reset()
        await self.push_frame(frame, direction)

    async def _monitor(self):
        try:
            while True:
                await asyncio.sleep(self._tick)
                if self._bot_speaking or self._user_speaking:
                    continue  # speech in progress -> not silence
                self._silent_secs += self._tick
                if self._level < len(self._prompts):
                    secs, kind, text = self._prompts[self._level]
                    if self._silent_secs >= secs:
                        self._level += 1
                        await self._fire(kind, text)
        except asyncio.CancelledError:
            pass

    async def _fire(self, kind: str, text: str):
        if kind == "move_on":
            logger.info("SILENCE> (moving on to a new question)")
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "(The candidate has stayed silent for a long time. Briefly "
                            "acknowledge that and move on to a different question.)",
                        }
                    ],
                    run_llm=True,
                )
            )
        else:
            logger.info(f"SILENCE> {text}")
            await self.push_frame(TTSSpeakFrame(text=text, append_to_context=True))

    async def cleanup(self):
        if self._task:
            self._task.cancel()
            self._task = None
        await super().cleanup()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class InputGate(FrameProcessor):
    """Half-duplex echo guard.

    Drops mic audio while the bot is speaking (plus a short hangover for the
    audio still draining from the output buffer), so the bot's own voice leaking
    into the mic can't be transcribed or trigger a false barge-in interruption.

    Trade-off: you can't interrupt the bot WHILE it talks. Normal turn-taking
    (speak after it finishes) works. Restore full barge-in later with a real
    headset (physical isolation) or acoustic echo cancellation.
    """

    def __init__(self, hangover_s: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False
        # Must exceed the output buffer tail (MAX_BUFFER_MS) — the bot's audio keeps
        # playing from the buffer after BotStoppedSpeakingFrame fires. If the gate
        # reopens too early, that tail leaks into the mic and the AI answers itself.
        self._hangover_s = hangover_s if hangover_s is not None else (MAX_BUFFER_MS / 1000 + 0.3)
        self._reopen_at = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._reopen_at = time.monotonic() + self._hangover_s

        gated = self._bot_speaking or time.monotonic() < self._reopen_at
        if gated and isinstance(frame, InputAudioRawFrame):
            return  # swallow mic audio while the bot is (or just was) speaking
        await self.push_frame(frame, direction)


def build_conversation_llm():
    """Build the conversation LLM. Groq by default; Gemini if CONV_LLM_PROVIDER=gemini."""
    if CONV_LLM_PROVIDER == "gemini":
        return GoogleLLMService(
            api_key=GEMINI_API_KEY,
            settings=GoogleLLMService.Settings(model=GEMINI_MODEL),
        )
    return GroqLLMService(
        api_key=GROQ_API_KEY,
        settings=GroqLLMService.Settings(model=GROQ_CONV_MODEL),
    )


def _check_keys() -> None:
    required = [("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY)]
    if CONV_LLM_PROVIDER == "gemini":
        required.append(("GEMINI_API_KEY", GEMINI_API_KEY))
    else:
        required.append(("GROQ_API_KEY", GROQ_API_KEY))
    missing = [name for name, val in required if not val]
    if missing:
        logger.error(f"Missing keys in .env: {', '.join(missing)}")
        sys.exit(1)


async def main() -> None:
    _check_keys()

    # --- Transport: local mic + speakers ---
    transport = SmoothLocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,   # Silero VAD + Deepgram STT
            audio_out_sample_rate=24000,  # Deepgram Aura (supported rate)
            audio_in_channels=1,
            audio_out_channels=1,
            input_device_index=int(AUDIO_IN_INDEX) if AUDIO_IN_INDEX else None,
            output_device_index=int(AUDIO_OUT_INDEX) if AUDIO_OUT_INDEX else None,
        )
    )

    # --- Services ---
    # interim_results=False: the judge evaluates finalized transcripts only.
    # smart_format adds punctuation so we can detect complete claims.
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="en-US",
            interim_results=False,
            smart_format=True,
        ),
    )
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramTTSService.Settings(voice=TTS_VOICE),
    )
    llm = build_conversation_llm()

    # --- Conversation context ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Please start the interview now."},
    ]
    context = LLMContext(messages)

    # Smart-turn ML analyzer (bundled, no download) decides end-of-turn SEMANTICALLY,
    # so it waits through mid-sentence pauses and replies only when you're actually
    # done. Reliable here because local PyAudio gives clean, continuous audio
    # (default stop_secs=3 is the silence backstop, > your ~2.5s natural pauses).
    turn_strategies = UserTurnStrategies(
        start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies),
    )

    state = ConversationState()

    # --- Pipeline ---
    processors = [transport.input()]
    if HALF_DUPLEX:
        # Half-duplex echo guard: mute mic while the bot talks (no barge-in).
        processors.append(InputGate())
    # else: mic stays live during bot speech -> VAD barge-in works (needs isolation).
    processors += [
        VADProcessor(vad_analyzer=SileroVADAnalyzer()),
        stt,
        TranscriptLogger(),       # logs CANDIDATE> transcripts
        DynamicsTracker(state),   # ask-back detection (clarification requests, not penalized)
        # InterruptJudge(state),  # STEP 2 — disabled until the basic loop is rock-solid
        SilenceMonitor(),         # silence auto-prompt (take your time / rephrase / move on)
        context_aggregator.user(),
        llm,
        TranscriptLogger(state),  # logs BOT> + records last question into state
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]
    pipeline = Pipeline(processors)

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Make the bot greet first, the moment the pipeline starts.
    @task.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):  # noqa: ANN001
        logger.info("Pipeline started — bot will greet you. Start talking any time.")
        await task.queue_frames([LLMRunFrame()])

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(task)

    logger.info("=" * 60)
    _conv_model = GEMINI_MODEL if CONV_LLM_PROVIDER == "gemini" else GROQ_CONV_MODEL
    logger.info(f"  STT : Deepgram nova-3   |  TTS : Deepgram {TTS_VOICE}")
    logger.info(f"  LLM : {CONV_LLM_PROVIDER} {_conv_model}")
    logger.info(f"  Judge: {JUDGE_PROVIDER} {JUDGE_MODEL}  (wrong-answer interrupt ON)")
    _mode = "half-duplex (no barge-in)" if HALF_DUPLEX else "FULL-DUPLEX barge-in"
    logger.info(f"  Mode: {_mode}  |  in_dev={AUDIO_IN_INDEX or 'default'} out_dev={AUDIO_OUT_INDEX or 'default'}")
    logger.info("  Wrong DSA answer -> it cuts in. Stay silent ~12s+ -> it prompts you.")
    logger.info("  Press Ctrl+C to stop.")
    logger.info("=" * 60)

    await runner.run()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
