# SViam Voice Interviewer

Real-time AI technical interviewer with full-duplex, human-like turn-taking.
Pipecat 1.3.0, Python 3.11+, Windows dev machine.

## Architecture (do not change without asking)
- STT: Deepgram nova-3 streaming. TTS: Deepgram Aura-2.
- Conversation LLM: provider-swappable via .env (Groq llama-4-scout now,
  OpenAI gpt-4o-mini once OPENAI_API_KEY exists). Speaks to the candidate.
- Judge LLM: Groq, JUDGE_MODEL from .env (llama-4-scout on free tier; switch to
  llama-3.3-70b-versatile when paid Groq tier exists). Separate from
  conversation. Decides interrupt/defer/continue. Never speaks long text.
- Transport: browser client over SmallWebRTCTransport (browser AEC gives
  echo-free full duplex). Local PyAudio transport is deprecated.

## Hard rules
1. Timing logic (barge-in, turn-end, interruption) lives in the audio and
   frame layer. NEVER make an LLM call part of the stop/start decision path.
2. Latency budget: user barge-in to TTS silence < 200ms. End of user turn
   to first bot audio < 800ms. Judge verdict < 1.5s after a wrong claim.
3. The judge defaults to CONTINUE. Existing biases in judge.py (one
   interrupt per answer, escalation ladder, never judge garbled text) must
   survive every refactor.
4. The interviewer NEVER reveals answers. Only the judge ladder reveals,
   on 3rd wrong attempt.
5. Check installed pipecat module paths before writing imports
   (venv/Lib/site-packages/pipecat). Do not guess APIs. If an API does
   not exist in 1.3.0, say so and propose the closest real one.
6. Preserve working code: TranscriptLogger, DynamicsTracker, SilenceMonitor,
   ConversationState, judge.py escalation. Refactor, do not rewrite.
7. After each task, give me a voice test checklist before moving on.
8. Keep frontend minimal: functional, clean, no component libraries beyond
   Tailwind. The conversation engine is the product.

## Target turn-taking state machine
IDLE -> AI_SPEAKING -> (user sustained speech) -> USER_BARGE_IN -> USER_SPEAKING
USER_SPEAKING -> (turn-end detected) -> AI_THINKING -> AI_SPEAKING
USER_SPEAKING -> (judge fires wrong-claim) -> AI_BARGE_IN -> AI_SPEAKING
Backchannels ("hmm", "yeah", short sounds) never change state.
