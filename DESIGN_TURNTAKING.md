# SViam Turn-Taking & Barge-In Engine — Design Spec

The conversation engine is the product. This is the north-star design for full-duplex,
human-like turn-taking with **correction barge-in**. Understanding/answer quality can be
"fine, not optimized"; the *timing* — where to start, where to stop, when to listen, when
to correct, and how to survive collisions — must be up to the mark.

All timing logic lives in the **audio + frame layer**. No LLM call is ever in the
stop/start decision path (CLAUDE.md rule 1). The only ML in the turn path is the local
smart-turn ONNX model (fast, audio-frame-driven — not an LLM call). The correction judge
is an LLM, but it runs as a **parallel side-channel** that can *request* an interrupt; it
never blocks normal turn-taking.

---

## 0. Latency budget (hard targets, CLAUDE.md rule 2)

| Event | Budget |
|---|---|
| User barge-in → TTS silence | < 200 ms |
| End of user turn → first bot audio | < 800 ms |
| Judge verdict after a wrong claim | < 1.5 s |
| Stuck-state auto-recovery (watchdog) | < 4 s |

"Human-like reaction time" = hit the 200 ms cut and the 800 ms reply. Everything else
(LLM quality, model size) can lag without breaking the feel.

---

## 1. State machine (extended from CLAUDE.md)

```
IDLE
 └─(pipeline start)──────────────► AI_SPEAKING        (greeting)

AI_SPEAKING
 ├─(user backchannel)────────────► AI_SPEAKING        (ignore — no state change)
 ├─(user sustained speech)───────► USER_BARGE_IN ─────► USER_SPEAKING   (cut TTS <200ms)
 └─(TTS finished)────────────────► LISTENING

LISTENING / USER_SPEAKING
 ├─(turn-end detected)───────────► AI_THINKING ──────► AI_SPEAKING
 ├─(judge: wrong-claim, interrupt)► AI_BARGE_IN ─────► AI_SPEAKING      (correction)
 └─(silence ≥ tier)──────────────► AI_SPEAKING        (silence auto-prompt)

AI_BARGE_IN / USER_BARGE_IN (collision window)
 └─(resolve via §5 matrix)───────► whoever wins holds the floor

ANY
 └─(watchdog: stuck)─────────────► force-resync → LISTENING            (§6)
```

Backchannels ("hmm", "yeah", short sounds) **never** change state.

---

## 2. Layer 1 — Voice activity & segmentation (no LLM)

- **VAD:** Silero (`VADProcessor(SileroVADAnalyzer())`). Emits
  `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`.
- **STT:** Deepgram nova-3 streaming, `interim_results=True` (barge-in word-counting needs
  real-time words), `endpointing=250ms` (fast finals without over-fragmentation),
  `utterance_end_ms=1000ms` (finalization backstop only). Deepgram does **not** own the
  turn-end decision.

  > `endpointing=100ms` **over-fragments**: observed live, it emits finals like "use" /
  > "order of" / "log n." as separate transcripts. **The judge and turn-end logic act on the
  > ACCUMULATED answer text, never on an individual fragment final** — fragments are only
  > appended to the running answer; no decision keys off one fragment.

These produce the raw signals every higher layer consumes: *is voice present*, *interim
words so far*, *finalized transcript*, *VAD stopped*.

---

## 3. Layer 2 — Turn-END: pause vs done (the core "when to stop listening")

A human keeps the floor through "My name is Yashraj… *(pause)* …and I study at SRM." The
machine must not reply to the fragment. Two signals decide end-of-turn:

1. **smart-turn-v3 (ML, local ONNX, bundled):** on each `VADUserStoppedSpeakingFrame` it
   predicts `COMPLETE` (semantically done) vs `INCOMPLETE` (trailing off / will continue).
   This is the primary, human-like signal. Proven: on stable local audio it held the floor
   through ~2.5 s mid-sentence pauses for 5 clean turns.
2. **Silence backstop:** `stop_secs = 3.0` (default). If the user goes fully silent for 3 s,
   end the turn regardless of the ML. **Rule: `stop_secs` must exceed the longest natural
   mid-sentence pause** (~2.5 s observed) so the backstop never cuts a thinking pause.

Owner: `TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())` inside
the user aggregator. After it predicts COMPLETE it waits for one finalized transcript, then
ends the turn → `AI_THINKING`.

**Thresholds**

| Param | Value | Why |
|---|---|---|
| smart-turn `stop_secs` | 3.0 s | silence backstop > max natural pause |
| Deepgram `endpointing` | 250 ms | fast finals without over-fragmentation (100 ms split "use" / "order of" / "log n.") |
| Deepgram `utterance_end_ms` | 1000 ms | finalization backstop only |
| min turn audio | ≥ 0.4 s voiced | ignore stray clicks/coughs as "turns" |

> Audio quality caveat: smart-turn mispredicts INCOMPLETE on **degraded** audio (the old
> WebRTC-DTX problem). Keep input continuous: browser `usedtx=0` + a live data channel, or
> local PyAudio. If audio degrades, the 3 s backstop is the safety net (sluggish but never
> silent).

---

## 4. Layer 3 — Barge-IN gate: user interrupts the bot (the "when to listen" override)

While the bot is speaking, **not every sound is a turn**. The gate distinguishes a real
interruption from a backchannel.

**Backchannel immunity (never interrupt):**
- Lexicon (env `BACKCHANNEL_WORDS`): `hmm, yeah, okay, ok, right, uh-huh, mhm, i see, got it`.
- A user utterance is a backchannel iff: ≤ 2 words **and** every word ∈ lexicon, **or**
  voiced < 600 ms. → bot keeps talking, no state change.

**Real barge-in (cut the bot):** fires when, *while the bot speaks*, the user produces
- ≥ **3 non-backchannel words** (counted on interim transcripts), **OR**
- ≥ **0.7 s sustained voiced speech**.

On fire (`USER_BARGE_IN`):
1. Cut TTS output **< 200 ms** (flush TTS buffer, cancel in-flight LLM/TTS — Pipecat native
   on `InterruptionFrame`).
2. **Truncate context to spoken text only.** The assistant aggregator commits *only the
   words actually played* (TTS marks unspoken `LLMTextFrame.append_to_context=False`); the
   bot never "remembers" saying something the user never heard.
3. Save the cut-off sentence as `dropped_thought`. On regaining the floor, **react to what
   the user just said FIRST** — never replay with "As I was saying". Resume the dropped point
   **only if it's still relevant** after their interjection, and **rephrase it fresh** (not
   verbatim). If it's no longer relevant, drop it silently.

Owner: `HumanBargeInStartStrategy` (start strategy) + `BargeInManager` (frame processor).
When bot is *silent*, a single word is enough to start a normal turn (min_words drops to 1).

---

## 5. Layer 4 — Correction engine: bot interrupts the user (the USP — "when to correct")

A separate **judge LLM** (Groq, `JUDGE_MODEL`) watches finalized candidate transcripts and
decides one of: **CONTINUE** (default), **DEFER** (vague/shallow/evasive answer — probe at
the natural turn-end), or **INTERRUPT** (cut in *now* — a clear factual error; this is the
product's money moment). It runs in parallel; the verdict must land < 1.5 s.

**Hard biases (must survive every refactor — CLAUDE.md rule 3):**
- Default **CONTINUE**. Silence is the right answer unless clearly warranted.
- **One interrupt per answer.** Escalation ladder across attempts, not within one.
- **Never judge garbled / low-confidence transcripts.**
- The **interviewer never reveals answers**; only the judge ladder reveals, on the 3rd wrong
  attempt (CLAUDE.md rule 4).

**Interrupt-vs-defer rule (severity gate):**

| Situation | Action |
|---|---|
| **Clear factual error** (any wrong claim) — *the product's money moment* | **INTERRUPT now** |
| **Compounding error** (rest of the answer builds on it) — most urgent case | **INTERRUPT now** |
| Vague / shallow / evasive, no clear error yet | DEFER to turn-end (probe then) |
| Ambiguous / possibly-right / garbled | CONTINUE |
| Already interrupted once this answer | CONTINUE (ladder advances next answer) |

> A clear factual error is **INTERRUPT**, not DEFER — catching it live is the whole point.
> Compounding errors are the *most* urgent, not the *only* interruptible case. DEFER is
> reserved for answers that are merely vague/shallow/evasive (nothing concrete to correct yet).

**Escalation ladder (push-back, never hand the answer over):**
1. **Attempt 1 — Nudge:** probe, don't reveal. *"Are you sure about that complexity? Walk me
   through it."*
2. **Attempt 2 — Hint:** point at the flaw. *"Think about what happens at the worst case."*
3. **Attempt 3 — Reveal:** only here may the answer be stated, then move on.

On INTERRUPT → `AI_BARGE_IN`: cut nothing of the *user's* audio (we don't control their mic),
but the bot starts speaking over the gap, TTS plays the nudge, and the turn flips to the bot.
Owner: `InterruptJudge` (parallel, side-channel) + `judge.py` (`judge_answer`).

---

## 6. Layer 5 — Collision resolution (graceful double-talk)

Both parties can hold the floor at once. The resolver runs in the frame layer and applies a
fixed policy — **no LLM in this path.**

**Collision matrix:**

| Bot state | User signal | Resolution |
|---|---|---|
| AI_SPEAKING | backchannel | bot continues (ignore) |
| AI_SPEAKING | sustained / ≥3 words | **bot yields** → cut TTS, USER_SPEAKING (§4) |
| USER_SPEAKING | judge INTERRUPT (high-sev) | **bot takes floor** → AI_BARGE_IN (§5) |
| USER_SPEAKING | judge DEFER / CONTINUE | bot stays silent, keeps listening |
| Simultaneous onset (< 300 ms apart) | both start together | **bot yields to user** (human politeness) *unless* bot is mid-3rd-attempt reveal |
| Bot mid-correction + user resumes | overlap | bot finishes the *current short* correction phrase, then yields |

**Tie-break principles:**
- **Default yield = the bot.** A человек expects to be able to cut off the interviewer.
- The **only** time the bot wins a tie is an active, high-severity correction (so a wrong
  claim can't be "talked over" into the record).
- **Hangover / debounce:** after either party yields, a 300 ms guard prevents immediate
  re-trigger ping-pong (no rapid cut/uncut oscillation).
- **Resume the floor:** whoever was cut keeps their partial context (`dropped_thought` for
  the bot; the user's interim transcript is preserved and re-fed so a barge-in doesn't lose
  what they'd already said).

---

## 7. Layer 6 — Recovery watchdog (so it NEVER dies after a few turns)

Root cause of the recurring "works 4-5 turns then nothing": Deepgram TTS websocket drops
(`1011 internal error`), pipecat reconnects, but bot-speaking / turn state is left desynced
and never recovers. Independent of transport.

**Watchdog rules (frame layer):**
- If a finalized candidate transcript exists **and** the bot is *neither speaking nor
  producing a reply* for **> 4 s** → force turn-end / resync (re-arm the turn machine).
- On TTS reconnect event → reset `bot_speaking=False`, clear any half-open interrupt state,
  re-arm input.
- The silence auto-prompt (30 / 90 / 120 s tiers) remains a *separate* higher-level nudge,
  not a substitute for this fast resync.

This is the single most important reliability fix and gates everything else feeling good.

---

## 8. Pipecat 1.3.0 ownership map

| Concern | Component |
|---|---|
| VAD / segmentation | `VADProcessor(SileroVADAnalyzer())` + Deepgram STT |
| Turn-END (pause vs done) | `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())` |
| Turn-START / barge-in gate | `HumanBargeInStartStrategy` (start strategy) |
| Cut/resume/dropped-thought/tie-guard | `BargeInManager` (FrameProcessor) |
| Correction judge | `InterruptJudge` + `judge.py` (parallel side-channel) |
| Collision policy | resolver inside `BargeInManager` (matrix §6) |
| Recovery watchdog | new processor (or extend `BargeInManager`) §7 |
| Transport | `SmallWebRTCTransport` (browser AEC = clean full-duplex) |

Full-duplex requires **no half-duplex `InputGate`** (mic stays live; browser AEC removes the
bot's voice from the mic). The local PyAudio path keeps `InputGate` and is half-duplex only.

---

## 9. Heuristics / data used

- **smart-turn-v3.2** ONNX (bundled, offline) — semantic end-of-turn. No training needed.
- **Backchannel lexicon** (rule-based, env-tunable) — interruption immunity.
- **Fixed thresholds** (this doc) — durations and word counts, tuned by voice testing.
- **Severity rules** (judge prompt) — interrupt vs defer.
- No external dataset required; if endpointing needs tuning later, log real pause
  distributions from `TranscriptLogger` and fit `stop_secs` to the 95th percentile pause.

---

## 10. Build order (each step ends with a voice-test checklist — CLAUDE.md rule 7)

1. **Reliability first:** recovery watchdog (§7) + clean TTS-reconnect handling. *Goal: never
   goes silent past N turns.*
2. **Turn-end:** smart-turn on the chosen transport, verified through real pauses (§3).
3. **Correction:** re-enable `InterruptJudge`, tune interrupt-vs-defer + ladder (§5).
4. **Full-duplex barge-in:** user-interrupts-bot with backchannel immunity + cut <200 ms (§4).
5. **Collision polish:** tie-breaks, hangover, resume (§6).
6. **Latency:** trim the 800 ms reply path last (§0).

Reliability (1) and turn-end (2) are the foundation; correction (3) is the USP; full-duplex
+ collisions (4-5) are the human feel; latency (6) is the final polish.
