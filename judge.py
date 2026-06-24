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
import re

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

A plain textbook error — a wrong complexity, a wrong definition, or a swapped concept (e.g. calling
a stack FIFO, or a hash map O(log n)) — clears all three: INTERRUPT it confidently. Do not talk
yourself out of an obvious mistake. (Note: speech-to-text garbles a FEW DSA terms. Only these
SPECIFIC, high-confidence substitutions are safe to read as the intended term, and ONLY when the
rest of the sentence already forms that DSA claim: in a complexity context "login"/"order of login"
= O(log n) (never a literal login system); of a stack/queue, "FIFA"/"five-four" = FIFO and "lie-fo"/
"life-o" = LIFO. Do NOT broadly "fix" any other similar-sounding word, and do NOT infer a wrong
claim from a garble alone — if you are unsure what the candidate meant, CONTINUE.)

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
    1: "FIRST wrong attempt on this topic: be DIRECT and confident — plainly say the claim is wrong "
    "and name the SPECIFIC thing that's wrong (e.g. 'a stack isn't FIFO'). But do NOT give the "
    "correct answer or value — end with a sharp question that makes them find it themselves.",
    2: "Still wrong: stay direct, and give a pointed HINT toward the fix — but still do NOT state the "
    "correct answer or value outright.",
    3: "Keeps getting it wrong: state the correct answer or key idea directly and briefly, then move on.",
}

# --- output / style (literal JSON, so no str.format here) ---
JUDGE_OUTPUT = """If you interrupt, write a SHORT, DIRECT spoken cut-in (1 sentence, max 2) — like a sharp senior
interviewer who won't let a wrong claim slide. LEAD by plainly naming the error. Do NOT hedge:
never use "I think", "there might be some confusion", "might be", "not quite", "it seems", "sort
of", or "are you sure". Be firm and confident, never rude. Never call a clearly wrong answer
"partially correct." No markdown, no lists — spoken words only.

Respond with ONLY a JSON object and nothing else:
{"interrupt": false, "line": ""}
or
{"interrupt": true, "line": "<direct cut-in that plainly names the specific wrong claim>"}"""

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
    """Extract {'interrupt': bool, 'line': str, 'defer': bool}; safe-default to continue."""
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"interrupt": False, "line": "", "defer": False}
        obj = json.loads(raw[start : end + 1])
        return {
            "interrupt": bool(obj.get("interrupt")),
            "line": str(obj.get("line") or "").strip(),
            "defer": False,
        }
    except Exception:
        return {"interrupt": False, "line": "", "defer": False}


# --- Post-verdict discipline (caveat 2): a clarification/follow-up is the INTERVIEWER's job
# at turn-end, not a judge interrupt. Only a CLEAR FACTUAL ERROR earns a hard interrupt; an
# interjection that merely asks the candidate to clarify/elaborate — naming no specific
# incorrect fact — is downgraded interrupt -> DEFER. ---
_CLARIFY_MARKERS = (
    "clarify", "can you explain", "could you explain", "tell me more", "walk me through",
    "elaborate", "what do you mean", "how did you", "how exactly", "what was", "what made",
    "describe", "help me understand", "more about", "what specifically", "in what way",
    "can you tell me more", "go into", "what kind of", "what type of",
)
_CORRECTION_MARKERS = (
    # "are you sure" (the nudge), NOT bare "sure" — "I'm not sure I follow" is a clarification.
    "are you sure", "actually", "isn't", "aren't", "wasn't", "doesn't", "don't think",
    "not correct", "incorrect", "not right", "not quite", "not the", "wrong", "mistake",
    "reconsider", "that's not", "thats not", "confusion", "i think there", "double-check",
    "double check", "rethink", "not a ", "not an ", "really o", "should be",
)


def _reads_as_clarification(line: str) -> bool:
    """True if the interjection is a clarification/follow-up request that names no specific
    incorrect fact — the interviewer's job at turn-end (DEFER), not a judge interrupt."""
    t = line.lower()
    asks_clarification = any(m in t for m in _CLARIFY_MARKERS)
    names_error = any(m in t for m in _CORRECTION_MARKERS)
    return asks_clarification and not names_error


# --- REVEAL_BLOCKED guard (Fix #2): a CODE safety net over the small judge model, which
# ignores the "reveal nothing on attempt 1/2" prompt instruction. If an attempt-1/2 interjection
# contains an answer-bearing term or value, we DO NOT speak it — we swap in a content-free
# nudge/hint. Erring toward catching reveals: this matches the common DSA giveaways, and any
# over-match just becomes a generic (still valid) nudge. Only attempt 3 may reveal. ---
_REVEAL_RE = re.compile(
    r"\blifo\b|\bfifo\b"
    r"|last[\s,-]*in[\s,-]*first[\s,-]*out|first[\s,-]*in[\s,-]*first[\s,-]*out"
    r"|\bo\s*\(\s*1\s*\)|\bo\s*\(\s*log\s*n\s*\)|\bo\s*\(\s*n\s*log\s*n\s*\)"
    r"|\bo\s*\(\s*n\s*\)|\bo\s*\(\s*n\s*(?:\^|\*\*|²|2)\s*\)|\bo\s*\(\s*n\s*squared\s*\)"
    r"|constant[\s-]*time|logarithmic|linear[\s-]*time|quadratic|amortized",
    re.IGNORECASE,
)
# Direct, assertive fallbacks (used only when the model tried to state the actual ANSWER on
# attempt 1/2). Lead by plainly saying it's wrong — no "are you sure" hedging — but still make
# the candidate find the fix (Socratic). Generic because they fire after the specific line is stripped.
_NUDGE_LVL1 = "That's not right. Think it through again — how does it actually work?"
_HINT_LVL2 = "Still not right. Slow down and reason it out step by step — what's really happening there?"


def reveals_answer(text: str) -> bool:
    """True if the text contains an answer-bearing DSA term/value (LIFO/FIFO, 'last in first
    out', big-O values, complexity words). Shared by the judge's REVEAL_BLOCKED ladder and the
    interviewer's first-clause guard (server.RevealGuard)."""
    return bool(_REVEAL_RE.search(text or ""))


# Confirming/denying correctness is the judge's lane ALONE (architecture rule 4: the interviewer
# NEVER states correctness). Scout still opens with "That's generally true…" / "that's not right…"
# despite the prompt. This catches an explicit correctness verdict in the interviewer's opener so
# RevealGuard can swap it for a neutral redirect. Deliberately narrow — it must NOT fire on normal
# interviewer phrasing ("that's a good start", "that's one way", "good question").
_CONFIRM_DENY_RE = re.compile(
    r"^\W*(?:yes|yep|yeah|no|nope|correct|incorrect|exactly|precisely|absolutely)\b[\s,.!:]"
    r"|that(?:'s| is)\s+(?:generally\s+|basically\s+|essentially\s+|mostly\s+|absolutely\s+"
    r"|partially\s+|not\s+quite\s+|not\s+exactly\s+|not\s+)?(?:right|correct|true|accurate|wrong|incorrect|false)"
    r"|you(?:'re| are)\s+(?:absolutely\s+|basically\s+|partially\s+|not\s+quite\s+)?(?:right|correct|wrong|incorrect|mistaken)"
    r"|spot[\s-]*on"
    r"|you(?:'ve| have)?\s*got\s+it"
    r"|that(?:'s| is)\s+(?:the\s+)?(?:right|correct)\s+(?:answer|idea|approach)",
    re.IGNORECASE,
)


def confirms_or_denies(text: str) -> bool:
    """True if the text states a correctness verdict (confirm OR deny). Used ONLY by the
    interviewer guard (server.RevealGuard) — the judge is exempt; pushing back is its job."""
    return bool(_CONFIRM_DENY_RE.search(text or ""))


def _block_reveal(line: str, level: int, answer: str = "") -> tuple[str, bool]:
    """On attempt level 1 or 2, block ONLY answer terms the candidate did NOT already say (i.e. the
    actual ANSWER). Naming/negating the candidate's OWN wrong term ('a stack isn't FIFO') gives
    nothing away and lets the cut-in stay direct/assertive, so it's allowed. If a NEW answer term
    is present, swap the line for a direct nudge (1) / hint (2). Level 3+ may reveal anything.
    Returns (new_line, blocked)."""
    if level >= 3:
        return line, False
    line_terms = {m.group(0).lower() for m in _REVEAL_RE.finditer(line)}
    if not line_terms:
        return line, False
    said = {m.group(0).lower() for m in _REVEAL_RE.finditer(answer or "")}
    if not (line_terms - said):  # only repeats the candidate's own term(s) -> reveals nothing new
        return line, False
    return (_NUDGE_LVL1 if level == 1 else _HINT_LVL2), True


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
        return {"interrupt": False, "line": "", "defer": False}
    verdict = _parse(raw)
    # Discipline downgrade: a clarification/follow-up interjection (no incorrect fact named)
    # is the interviewer's job -> DEFER, not a hard interrupt.
    if verdict["interrupt"] and _reads_as_clarification(verdict["line"]):
        logger.info(
            f"JUDGE_DOWNGRADED interrupt->defer (clarification, no fact named): {verdict['line']}"
        )
        verdict["interrupt"] = False
        verdict["defer"] = True
    # REVEAL_BLOCKED (Fix #2): never let attempt 1/2 give away the answer, even if the model tried.
    if verdict["interrupt"]:
        new_line, blocked = _block_reveal(verdict["line"], level, answer)
        if blocked:
            logger.warning(
                f"REVEAL_BLOCKED (attempt {level}): model tried to reveal — swapped to nudge/hint. "
                f"was: {verdict['line']!r}"
            )
            verdict["line"] = new_line
    return verdict
