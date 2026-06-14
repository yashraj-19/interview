# SViam Voice Interviewer — Deployment Guide

A real-time, full-duplex AI technical interviewer. The candidate speaks to it in the
browser; it asks data-structures/algorithms questions, pushes back on wrong claims
**without ever revealing the answer**, and handles natural turn-taking — barge-in,
mid-thought pauses, and self-corrections — like a human interviewer.

---

## 1. Stack

| Layer | Technology |
|-------|-----------|
| Speech-to-text | Deepgram **nova-3** (streaming) |
| Text-to-speech | Deepgram **Aura-2** |
| Interviewer LLM | Groq **llama-4-scout** (provider-swappable) |
| Judge LLM (separate corrector) | Groq **llama-4-scout** |
| Transport | Browser **WebRTC** (`SmallWebRTCTransport`) — browser AEC gives echo-free full duplex |
| Server | **FastAPI + Uvicorn**, one Pipecat pipeline per browser connection |
| Turn-taking / reliability | Custom frame-layer engine (`barge_in.py`) |

---

## 2. Prerequisites

- **Python 3.11+** (3.11 matches the dev environment)
- A **Chromium browser** (Chrome / Edge) for the client — microphone permission required
- **Two API keys**, both with free tiers:
  - [Deepgram](https://console.deepgram.com) — STT + TTS (one key, both)
  - [Groq](https://console.groq.com/keys) — interviewer + judge LLM

---

## 3. Setup

```bash
# 1. Get the code
git clone <repo-url>
cd Voice_Assist

# 2. Virtual environment + dependencies
#    Windows (PowerShell):
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
#    macOS / Linux:
python3.11 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 3. Configure keys
copy .env.example .env        # Windows  (macOS/Linux: cp .env.example .env)
#    Edit .env and fill DEEPGRAM_API_KEY and GROQ_API_KEY. The rest can stay default.
```

> The first install pulls Pipecat plus `aiortc` / `av` / `torch` (for Silero VAD) — allow a few minutes.

---

## 4. Run

```bash
venv\Scripts\python.exe server.py        # Windows
./venv/bin/python server.py              # macOS / Linux
```

Then open **http://localhost:8000** in Chrome, click to start, and **allow the microphone**.
The bot greets first; start talking any time.

---

## 5. How it works (architecture)

The product is the **conversation engine**, not the UI. Three concerns are kept strictly
separate:

1. **Turn-taking (`barge_in.py`)** — owns *when* to listen, reply, or stop. Pure frame/timing
   logic; **no LLM call is ever on the stop/start decision path** (that would add latency and
   non-determinism). A deterministic transcript rule decides turn-end (terminal punctuation +
   short confirm, or a silence backstop); smart-turn ML is delay-only.
2. **Interviewer LLM** — asks questions and probes. **Never** confirms, denies, or reveals an
   answer. A code-level guard (`RevealGuard` in `server.py` + `judge.confirms_or_denies` /
   `reveals_answer`) replaces any leaked answer/verdict with a neutral probe — a safety net
   under the prompt.
3. **Judge LLM (`judge.py`)** — a *separate* model that watches for a clearly wrong claim and
   cuts in to correct it, on an escalation ladder (nudge → hint → reveal-only-on-3rd-attempt).
   Defaults to CONTINUE; one interrupt per answer.

**Reliability layer** (built for live free-tier voice):
- **Anti-double engine** — a per-turn latch + an output gate guarantee **exactly one reply per
  turn**, even across the judge + interviewer + watchdog paths and Pipecat's internal turn timeout.
- **Stall recovery** — if a claimed reply produces *zero* output for >12s (LLM/TTS/network hang),
  it cancels the stuck generation, resyncs, and re-prompts (`STALL_RECOVERY` in the log).
- **Phantom-flag + TTS-reconnect watchdogs** — recover from stuck state or a Deepgram TTS drop
  (`1011`) without going silent.

### What the candidate hears on a WRONG answer (by design)

The interviewer must **never reveal or confirm the answer**, so wrong claims are met with
**fixed, content-free lines** — not the LLM improvising. These will *recur across questions
on purpose*; they are the safety net and the judge's escalation ladder, not the model
repeating itself:

| When | Source | Line |
|------|--------|------|
| Interviewer's reply was about to reveal/confirm an answer | `RevealGuard` (safety net) | *"Walk me through your reasoning step by step — I want to follow exactly how you got there."* |
| Judge catches a wrong claim — **1st** attempt (nudge) | `judge.py` ladder | *"Are you sure about that? Walk me through your reasoning."* |
| Judge — **2nd** attempt (hint) | `judge.py` ladder | *"That doesn't sound right to me. Think carefully, step by step, about how it actually works."* |
| Judge — **3rd** attempt (reveal) | `judge.py` ladder | The actual correction — **the only time the answer is given.** |

So if you give a wrong answer and hear *"Walk me through your reasoning…"* or *"Are you sure
about that?…"* repeatedly, that is the system working as intended: it is probing without
handing you the answer.

See `DESIGN_TURNTAKING.md` and `CLAUDE.md` for the full design and hard rules.

---

## 6. Configuration reference

All config is in `.env` (see `.env.example`). Required: `DEEPGRAM_API_KEY`, `GROQ_API_KEY`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPGRAM_API_KEY` | — | **Required.** STT + TTS. |
| `GROQ_API_KEY` | — | **Required.** Interviewer + judge LLM. |
| `CONV_LLM_PROVIDER` | `groq` | Interviewer provider: `groq` or `gemini`. |
| `GROQ_CONV_MODEL` | `llama-4-scout-17b…` | Interviewer model. |
| `JUDGE_PROVIDER` | `groq` | Judge provider: `groq` or `anthropic`. |
| `JUDGE_MODEL` | `llama-4-scout-17b…` | Judge model. |
| `GEMINI_API_KEY` | — | Only if `CONV_LLM_PROVIDER=gemini`. |
| `ANTHROPIC_API_KEY` | — | Only if `JUDGE_PROVIDER=anthropic` (`pip install anthropic`). |
| `SILENCE_SECS` | `30,90,120` | Candidate-silence nudge / reprompt / move-on thresholds. |

---

## 7. Latency

Measured end of candidate's turn → first bot audio, on Groq:

| Path | Typical | Made of |
|------|---------|---------|
| Sentence-ending turn (most turns) | **~1.0–1.5 s** | STT finalize ~0.4s + turn-end confirm 0.35s + LLM first sentence ~0.4s + TTS first audio ~0.3s |
| Turn that doesn't end on punctuation | ~2.5 s | 2.5s silence backstop (kept high **on purpose** so the bot never cuts off a speaker who pauses mid-thought) |

This is near the practical floor for this STT + turn-detection stack — the LLM and TTS are
already fast. The remaining latency is dominated by the (deliberate) safety margin that keeps
the bot from interrupting a thoughtful candidate.

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing keys in .env` on start | Fill `DEEPGRAM_API_KEY` and `GROQ_API_KEY` in `.env`. |
| Browser won't grant microphone | Use Chrome/Edge; the client must be on **`localhost`** or **HTTPS** (browsers block mic on plain-HTTP remote IPs). For remote demos, tunnel with HTTPS (e.g. `ngrok http 8000`). |
| Bot pauses then recovers | Normal: a Deepgram free-tier TTS reconnect (`1011` → `TTS_RESYNC`) or a network blip (`STALL_RECOVERY`). The reliability layer self-heals. |
| Port 8000 already in use | Change the port at the bottom of `server.py` (`uvicorn.run(... port=8000)`). |
| Replies feel slow | Free Deepgram/Groq tiers throttle under load; a paid tier removes the jitter. |

---

## 9. Notes

- **`voice_agent.py` is the deprecated local-microphone path** (PyAudio). The supported,
  full-duplex transport is the browser server (`server.py`). `voice_agent.py` is still imported
  by `server.py` for the shared LLM factory, prompt, and turn-taking — don't delete it.
- Helper scripts (`audio_diag.py`, `smoke_test.py`, `test_judge.py`) are dev tools, not required
  to run the app.
- `.env` and `venv/` are git-ignored and must never be committed.
