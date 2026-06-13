# SViam Week-1 — Status vs. README

**Setup:** Standalone Pipecat (1.3.0) local-audio app, English. STT = Deepgram nova-3,
TTS = Deepgram Aura, conversation + interrupt-judge = Groq Llama-4-Scout (both swappable
via `.env`). Files: `voice_agent.py` (orchestrator), `judge.py` (swappable wrong-answer
judge), `smooth_audio.py` (glitch-free callback audio), `audio_diag.py`, `test_judge.py`.

Legend: ✅ done · 🟡 partial · ⏸ built but blocked · ❌ not done

---

## Our differentiators (the "brain" things) — DONE
These are what put us ahead of a basic voice-Q&A bot built from the same free resources:

| Capability | Status | Notes |
|---|---|---|
| **Dual-brain split** (separate conversation LLM + dedicated interrupt-judge LLM) | ✅ | Independent judge watches every answer in real time |
| **Wrong-answer interrupt** — AI cuts in when you state a clearly wrong claim | ✅ | README "Interruption & Pushback" |
| **Escalation ladder** — nudge → hint → reveal across repeated wrong attempts | ✅ | One interrupt per answer, tracks wrong-streak |
| **Never gives the answer away** — interviewer probes, only the judge ladder ever reveals | ✅ | Beats bots that just lecture the solution |
| **Hard bias to stay quiet** — never interrupts vague / correct / partial / garbled / meta | ✅ | Tuned + verified offline |
| **Glitch-free streaming audio** on Windows (callback ring buffer + jitter cushion) | ✅ | Beyond README; fixed real choppiness |
| **Half-duplex echo guard** | ✅ | Makes single-device audio usable |
| **Provider-swappable** (judge/conversation → Claude/OpenAI via one `.env` line) | ✅ | Future-proof |

---

## Deliverable 1 — Conversation Dynamics

| README item | Status | What we did | Why / premium path |
|---|---|---|---|
| 1.1 Barge-in: candidate interrupts AI, TTS cut <200ms | ⏸ | Implemented (VAD + buffer-flush, `HALF_DUPLEX=false` toggle) | **Blocked by hardware** — your single USB device's mic hears its own output (echo). Needs an isolated headset (free) or **Krisp AEC (paid)**, or a browser/WebRTC client (free AEC). |
| 1.1 AI adapts to interruption, doesn't repeat | 🟡 | LLM handles it via context | Untestable until barge-in works on isolated audio |
| 1.1 "circle back, never say 'As I was saying'" | ❌ | — | Small prompt addition; deferred (depends on barge-in) |
| 1.2 Silence state machine (30/90/120s prompts) | ✅ | `SilenceMonitor`: take-your-time → rephrase → move-on | Done (spec timings, env-tunable) |
| 1.2 Explicit pause requests ("can I have a moment / let me think") | ✅ | Detected; interviewer says "take your time" and waits | Done |
| 1.2 Resume on "I'm ready / okay / go ahead" | ✅ | Handled in interviewer prompt | Done |
| 1.2 Thresholds vary by strictness | ❌ | — | Tied to strictness (deferred — see D3) |
| 1.3 Ask-back detection (repeat / what do you mean / didn't catch) | ✅ | `DynamicsTracker` detects + counts | Done |
| 1.3 Rephrase (not verbatim), don't penalize | ✅ | Interviewer prompt | Done |
| 1.3 Track metadata ("asked 2x"), flag stalling after 3rd | ✅ | Logged (no scorecard yet to store into) | Done (logging) |
| 1.3 Credit good clarifying questions | ✅ | Interviewer prompt | Done |
| 1.4 Resumption module (remember thread, transition) | 🟡 | Implicit via LLM context | No explicit module; LLM handles most of it |

## Deliverable 2 — Multi-language (en-US / en-IN / Hindi / Telugu)

| README item | Status | Why / premium path |
|---|---|---|
| `language` field on session config | ❌ | **You deferred multi-language at kickoff** ("English only for now") |
| STT routes per language | ❌ | Free Telugu *streaming* STT doesn't exist → **Azure AI Speech (paid)** does hi-IN/te-IN/en-IN streaming |
| Hindi/Telugu TTS | ❌ | **Deepgram Aura is English-only** → needs **Azure TTS / ElevenLabs / Sarvam (paid)** |
| Code terms stay English | ❌ | Prompt-only once a multilingual LLM (Claude/GPT-4o) is in place |
| `language_config.py` | ❌ | ~half a day once a paid speech key exists |

## Deliverable 3 — Strictness levels (Friendly / Standard / Tough / Adversarial)

| README item | Status | Why |
|---|---|---|
| `strictness` field, 4 system-prompt templates, threshold/follow-up/hint policy per level | ❌ | **You deferred strictness at kickoff.** This is **FREE** — pure prompt engineering, ~half a day. No premium needed. |
| `strictness_profiles.py` | ❌ | Same |
| Scoring identical across strictness | ❌ | No scoring component built yet (see below) |

## File structure & Definition-of-Done

| README item | Status | Note |
|---|---|---|
| `voice_agent.py` | ✅ | Orchestrator |
| `conversation_dynamics.py` (barge-in / pause / ask-back) | 🟡 | Logic exists, split across `voice_agent.py` + `judge.py` + `smooth_audio.py` (not that filename) |
| `language_config.py` / `strictness_profiles.py` | ❌ | Deferred features |
| Unit tests (`test_*`) | ❌ | **You said skip unit tests**; we have `test_judge.py` + `smoke_test.py` |
| Code on a feature branch | ❌ | Folder isn't a git repo yet (can `git init` + branch anytime) |

---

## Beyond Week-1 scope (the actual product, not asked in this assignment)
- **Monaco code editor + speech↔code correlation** (the product's core signal) — that's the existing frontend/backend; we built the voice interviewer half.
- **FastAPI + Supabase + Redis + WebSocket backend** — **you said skip it**; FREE to build (free tiers). Moving here also fixes barge-in for free (browser WebRTC AEC).
- **Scorecard** (hire verdict, evidence, timestamps) — FREE (LLM analyzes the transcript); not requested in Week 1.
- Roadmap items (Hard Mode, Round Memory, Calibration, audit trail) — out of scope.

---

## Why each gap exists (one line each)
- **Multi-language, strictness, FastAPI backend, unit tests:** you explicitly deferred/skipped them at kickoff to ship a working voice loop + interrupt first.
- **Barge-in:** hardware echo on your single USB device; you asked to skip software AEC.
- **Gemini conversation brain:** free tier = 5 req/min, unusable for live voice → switched to Groq.
- **Scorecard / code-editor correlation:** not in the Week-1 voice-interviewer scope.

## What needs PAID vs FREE to finish
- **PAID:** multi-language (Azure Speech + Claude ≈ $1/hr STT + ~$16/1M-char TTS), clean barge-in without a headset (Krisp), unthrottled/sharper LLM (Claude/GPT-4o/Groq Dev).
- **FREE (just build time):** strictness levels, scorecard, FastAPI/Supabase backend, resumption polish, "feature branch".

## Highest-leverage things still left (to extend the lead)
1. **Strictness levels** — free, ~½ day, makes the interviewer adaptive (a real differentiator).
2. **Scorecard with evidence + timestamps** — free, this is the actual hiring-manager deliverable.
3. **Multi-language** — paid, the India-market differentiator.
4. **Browser/WebSocket client** — free, unlocks real barge-in (WebRTC AEC) + matches the product architecture.
