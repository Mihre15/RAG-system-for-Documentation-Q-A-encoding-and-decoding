"""Cut documents into passages that can be embedded and retrieved.

Analogy: a card catalog works better with one idea per card. A whole
chapter is too much to compare, and a single word is too little to
mean anything. We start from paragraphs, staple a short heading onto
the paragraph it introduces, and only then cut a card that is still
too long.

Why paragraphs first: a blank line is a boundary the author already
chose. Why a character cap: a few very long paragraphs would crowd
the language model's input limit. Why overlap: if we must cut, the
sentence on the cut should still appear with its neighbors.
"""

from __future__ import annotations

import re

from rag.config import MAX_CHARS, MIN_BLOCK_CHARS, OVERLAP
from rag.types import Chunk, Document


def chunk_text(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    min_chars: int = MIN_BLOCK_CHARS,
) -> list[str]:
    """Split one document's text into passage strings."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    blocks = _merge_short_blocks(blocks, min_chars)
    parts: list[str] = []
    for block in blocks:
        parts.extend(_split_long_paragraph(block, max_chars, overlap))
    return parts


def chunk_documents(
    documents: list[Document],
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    min_chars: int = MIN_BLOCK_CHARS,
) -> list[Chunk]:
    """Chunk each document and number the passages inside that file."""
    chunks: list[Chunk] = []
    for doc in documents:
        parts = chunk_text(doc.text, max_chars, overlap, min_chars)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}#{index}",
                    source=doc.source,
                    index=index,
                    text=part,
                )
            )
    return chunks


def _merge_short_blocks(blocks: list[str], min_chars: int) -> list[str]:
    """Glue a short block, usually a heading, to the block that follows."""
    merged: list[str] = []
    pending = ""
    for block in blocks:
        if pending:
            block = pending + "\n\n" + block
            pending = ""
        if len(block) < min_chars:
            pending = block
        else:
            merged.append(block)
    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + pending
        else:
            merged.append(pending)
    return merged


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """Window a long paragraph. Short text is returned unchanged."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    # Safety rail: each step must move forward, so this cannot loop forever.
    for _ in range(len(text) + 1):
        if start >= len(text):
            break
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + 1, end)
            # Break on a space when that still leaves a substantial piece,
            # so we don't slice a word in half just to be exact.
            if space != -1 and space - start >= max_chars // 2:
                end = space
        if end <= start:
            end = min(start + 1, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        # The overlap rewind can land in the middle of a word.
        # Move forward to the next word so a card does not start with a fragment.
        # The word we skip here is already on the previous card.
        next_start = _snap_to_word_start(text, next_start)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _snap_to_word_start(text: str, start: int) -> int:
    """If `start` is inside a word, move to the following word."""
    if start <= 0 or start >= len(text):
        return start
    if text[start].isspace() or text[start - 1].isspace():
        while start < len(text) and text[start].isspace():
            start += 1
        return start
    space = text.find(" ", start)
    if space == -1:
        return start
    return min(space + 1, len(text))
