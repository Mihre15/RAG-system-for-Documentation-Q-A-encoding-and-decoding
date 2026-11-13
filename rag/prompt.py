"""Build the text prompt that the language model reads.

The retrieval vectors are not turned back into words. We kept the
original passage text and we paste that text in. The model never sees
the 384 numbers.

The wording below is the second prompt tried. The first one numbered
the passages [1], [2], [3]. On the 16-question run saved in
eval/numbered_prompt_results.json, flan-t5-small often answered with
the number instead of the fact. This prompt has no numbers, asks for
one short sentence, and puts the nearest passage in as plain text.
"""

from __future__ import annotations

from rag.types import Chunk


def build_prompt(question: str, passages: list[Chunk]) -> str:
    """One short-answer instruction, the passage text, then the question."""
    blocks = [chunk.text.strip() for chunk in passages if chunk.text.strip()]
    context = "\n".join(blocks) if blocks else "(none)"
    return (
        "Using only the context, answer the question in one short sentence. "
        "If the context does not say, answer: I don't know.\n"
        f"Context: {context}\n"
        f"Question: {question.strip()}"
    )


def passages_that_fit(
    question: str,
    passages: list[Chunk],
    count_tokens,
    limit: int,
) -> tuple[str, list[Chunk], int]:
    """Drop the lowest-ranked passages until the prompt fits.

    `passages` must be best-first. `list.pop()` removes the last item,
    which is the weakest match, so the best passage is the last one
    we would give up.
    """
    shown = list(passages)
    prompt = build_prompt(question, shown)
    n_tokens = int(count_tokens(prompt))
    while shown and n_tokens > limit:
        shown.pop()
        prompt = build_prompt(question, shown)
        n_tokens = int(count_tokens(prompt))
    return prompt, shown, n_tokens
