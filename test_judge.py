"""Quick offline check of the interrupt judge (no audio).

These are ONLY test prompts — the judge has no answer key; it reasons from the
question + answer. Run: python test_judge.py
"""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from judge import JUDGE_MODEL, JUDGE_PROVIDER, judge_answer  # noqa: E402

CASES = [
    # (question, answer, expectation)
    (
        "What's the time complexity of looking up a value in a hash map?",
        "Hash map lookup is O of n because you have to scan through every element to find the key.",
        "EXPECT INTERRUPT (wrong: lookup is O(1) average)",
    ),
    (
        "What's the time complexity of looking up a value in a hash map?",
        "On average it's O of 1 because the hash function maps the key directly to a bucket.",
        "EXPECT CONTINUE (correct)",
    ),
    (
        "Can you explain how a binary search works?",
        "Um, so binary search, let me think... you have a sorted array and...",
        "EXPECT CONTINUE (incomplete / thinking out loud)",
    ),
    (
        "What data structure would you use for a LIFO order?",
        "I would use a queue because a queue is last in first out.",
        "EXPECT INTERRUPT (wrong: queue is FIFO; LIFO is a stack)",
    ),
    (
        "How would you find the closest pair of points in 2D?",
        "Maybe I could sort the points first and then check neighboring points.",
        "EXPECT CONTINUE (reasonable partial idea — must NOT interrupt to agree)",
    ),
    (
        "How do you keep a binary search tree balanced?",
        "Um, you balance it by like adjusting the nodes I guess.",
        "EXPECT CONTINUE (vague / incomplete)",
    ),
    (
        "What's the time complexity of merge sort?",
        "Merge sort is O of n squared because it compares a lot of elements.",
        "EXPECT INTERRUPT (wrong: O(n log n)) — line must NUDGE, not reveal 'n log n'",
    ),
    (
        "Can you walk me through the two-sum problem?",
        "I'd do the take loop and then the uh map thing for the the values.",
        "EXPECT CONTINUE (garbled / unclear — must NOT interrupt to clarify)",
    ),
]


async def main():
    print(f"Judge: {JUDGE_PROVIDER} / {JUDGE_MODEL}\n" + "=" * 70)
    for q, a, exp in CASES:
        verdict = await judge_answer(q, a)
        mark = "INTERRUPT" if verdict["interrupt"] else "continue"
        print(f"\nQ: {q}\nA: {a}\n{exp}\n-> {mark}")
        if verdict["interrupt"]:
            print(f"   line: {verdict['line']}")

    print("\n" + "=" * 70 + "\nESCALATION (same wrong answer, attempts 1->2->3):")
    q = "What's the time complexity of merge sort?"
    a = "Merge sort is O of n squared because it compares a lot of elements."
    for attempt in (1, 2, 3):
        v = await judge_answer(q, a, attempt)
        print(f"\nattempt {attempt}: {'INTERRUPT' if v['interrupt'] else 'continue'}")
        if v["interrupt"]:
            print(f"   line: {v['line']}")


if __name__ == "__main__":
    asyncio.run(main())
