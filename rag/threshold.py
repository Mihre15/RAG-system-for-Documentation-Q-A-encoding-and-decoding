"""Choose the cosine cutoff with a rule fixed before seeing scores.

The gate is a yes/no decision: do we have a grounded passage to show
the language model, or should we say "I don't know"?

`hit` must mean "the expected fact is somewhere in the top-k passages",
because those are the passages that would go into the prompt.

The rule, written before the Northstar scores were measured:

- Count a question as correct when the gate opens only if this question
  is answerable AND the fact was retrieved.
- Try cutoffs that sit between the observed top scores.
- Keep the cutoff with the most correct decisions.
- If two cutoffs tie, keep the higher one, so a tie prefers
  "I don't know" over risking an ungrounded answer.

This rule looks at the same question list we report on. It is not an
unbiased estimate for brand-new questions. REPORT.md says that plainly.
"""

from __future__ import annotations


def gate_successes(rows: list[dict], cutoff: float) -> int:
    """How many questions this cutoff would gate correctly."""
    correct = 0
    for row in rows:
        should_answer = bool(row["answerable"]) and bool(row["hit"])
        will_answer = float(row["top_score"]) >= cutoff
        if should_answer == will_answer:
            correct += 1
    return correct


def recommend_cutoff(rows: list[dict]) -> tuple[float, int, int]:
    """Return (cutoff, successes, question_count)."""
    if not rows:
        raise ValueError("recommend_cutoff needs at least one row")

    scores = sorted({round(float(row["top_score"]), 6) for row in rows})
    candidates = [scores[0] - 0.05]
    for left, right in zip(scores, scores[1:]):
        candidates.append((left + right) / 2.0)
    candidates.append(scores[-1] + 0.05)

    best_cutoff = candidates[0]
    best_n = -1
    for cutoff in candidates:
        n = gate_successes(rows, cutoff)
        # `>=` and increasing candidates: a tie keeps the higher cutoff.
        if n >= best_n:
            best_n = n
            best_cutoff = cutoff
    return best_cutoff, best_n, len(rows)
