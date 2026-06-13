"""Wrong-answer interrupt judge — provider-swappable (Groq now, Claude later).

The judge is given the interviewer's ACTUAL last question and the candidate's
LIVE (finalized) spoken answer. It decides, from its own knowledge, whether the
candidate has stated a clear error — there is NO stored answer key.

Two SEPARATE concerns, kept apart so a small model stays reliable:
  1. DECISION  — interrupt or continue (depends only on correctness).
  2. REVEAL    — if interrupting, how much to give away (depends on attempt #):
                 attempt 1 = gentle nudge, 2 = hint, 3+ = give the answer.

Swap providers with .env (no pipeline changes):
    JUDGE_PROVIDER=groq        JUDGE_MODEL=llama-3.1-8b-instant
    JUDGE_PROVIDER=groq        JUDGE_MODEL=llama-3.3-70b-versatile
    JUDGE_PROVIDER=anthropic   JUDGE_MODEL=claude-sonnet-4-6   (set ANTHROPIC_API_KEY, pip install anthropic)
"""

import json
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()  # ensure .env is read regardless of import order

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq").strip().lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant").strip()

# --- STEP 1: the interrupt/continue DECISION (independent of attempt #) ---
JUDGE_DECISION = """You are the real-time judgment module of an AI technical interviewer.

You receive the interviewer's last question and the candidate's FINALIZED spoken answer so far
(from speech-to-text; judge the intended meaning, ignore minor transcription errors).

FIRST decide INTERRUPT or CONTINUE. This decision depends ONLY on correctness, never on the
attempt number. Your DEFAULT is CONTINUE — stay silent unless you must correct a clear mistake.

INTERRUPT only if ALL THREE are true:
1. The candidate has stated an EXPLICIT, COMPLETE claim (not a fragment or a thought in progress).
2. That claim is CLEARLY and FACTUALLY WRONG or self-contradictory, and relevant to the question.
3. You can name the exact wrong statement and the correct fact.

Otherwise CONTINUE. In particular, CONTINUE (do NOT interrupt) when the answer is:
- vague, partial, incomplete, or still being formed;
- hesitation, filler, or thinking out loud;
- correct, or correct with a reasonable qualifier (e.g. "O(1) on average") — do NOT add caveats;
- a reasonable partial idea or direction toward a solution;
- garbled, unclear, or you are not sure what the candidate actually said;
- small talk, a meta comment, or the candidate wanting to stop, pause, or move on;
- something you only have a minor quibble with, or are not fully sure is wrong.

NEVER interrupt to agree, encourage, praise, add nuance, ask a follow-up, ask for clarification,
or because the transcript is garbled or unclear. The ONLY reason to interrupt is an explicit,
complete, clearly WRONG claim. An interrupt that agrees with, clarifies, follows up on, or merely
builds on the candidate is a BUG. If speech is garbled or you are unsure what was said, CONTINUE.
When in doubt, CONTINUE.

Decision examples (interrupt ONLY the clearly-wrong, complete claims):
Q: "Hash map lookup complexity?"  A: "On average O(1), it hashes the key to a bucket."  -> continue
Q: "Hash map lookup complexity?"  A: "O(n), you scan every element."                    -> interrupt
Q: "Which structure is LIFO?"     A: "A queue, it's last in first out."                  -> interrupt
Q: "Time complexity of merge sort?" A: "Merge sort is O(n squared)."                     -> interrupt
Q: "Explain binary search."       A: "Um, you have a sorted array and, let me think..."  -> continue
Q: "Find the closest pair of points?" A: "Maybe I could sort the points first..."        -> continue
Q: "How do you balance a BST?"    A: "You balance it by like adjusting the nodes I guess." -> continue
Q: "Walk me through two-sum."     A: "I'd do the take loop and then the uh map thing for the..." -> continue (garbled / unclear)"""

# --- STEP 2: the REVEAL level, chosen in code by attempt # (one clear instruction) ---
REVEAL = {
    1: "This is the candidate's FIRST wrong attempt on this topic: ONLY gently flag that it does "
    "not seem right and nudge them to reconsider, phrased as a question. Reveal NOTHING — do not "
    "state the correct answer, the value, or the approach.",
    2: "The candidate has now been wrong more than once: give a pointed HINT that pushes toward the "
    "fix, but still do NOT state the correct answer or value outright.",
    3: "The candidate keeps getting it wrong: it is now okay to briefly give the correct answer or "
    "the key idea, then let them move on.",
}

# --- output / style (literal JSON, so no str.format here) ---
JUDGE_OUTPUT = """If you interrupt, write a SHORT spoken interjection (1-2 sentences) like a sharp but fair human
interviewer cutting in. Be specific about what is off. Never call a clearly wrong answer
"partially correct." Never lecture. Vary phrasing. Conversational, no markdown, no lists.

Respond with ONLY a JSON object and nothing else:
{"interrupt": false, "line": ""}
or
{"interrupt": true, "line": "<your spoken interjection>"}"""

_groq_client = None
_anthropic_client = None


async def _call_groq(system: str, user: str) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq

        _groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    resp = await _groq_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=120,
    )
    return resp.choices[0].message.content or ""


async def _call_anthropic(system: str, user: str) -> str:
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic

        _anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = await _anthropic_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=120,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _parse(raw: str) -> dict:
    """Extract {'interrupt': bool, 'line': str}; safe-default to continue."""
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"interrupt": False, "line": ""}
        obj = json.loads(raw[start : end + 1])
        return {"interrupt": bool(obj.get("interrupt")), "line": str(obj.get("line") or "").strip()}
    except Exception:
        return {"interrupt": False, "line": ""}


async def judge_answer(question: str, answer: str, attempt: int = 1) -> dict:
    """Return {'interrupt': bool, 'line': str}. Never raises.

    attempt: consecutive wrong attempts on the current topic + 1. Controls only how
    much the interjection reveals (1=nudge, 2=hint, 3+=answer), not the decision.
    """
    level = min(max(attempt, 1), 3)
    system = f"{JUDGE_DECISION}\n\nIF YOU INTERRUPT: {REVEAL[level]}\n\n{JUDGE_OUTPUT}"
    user = (
        f"Interviewer's last question:\n{question}\n\n"
        f"Candidate's finalized answer so far:\n{answer}"
    )
    try:
        if JUDGE_PROVIDER == "anthropic":
            raw = await _call_anthropic(system, user)
        else:
            raw = await _call_groq(system, user)
    except Exception as e:
        logger.error(f"judge LLM call failed: {e}")
        return {"interrupt": False, "line": ""}
    return _parse(raw)
