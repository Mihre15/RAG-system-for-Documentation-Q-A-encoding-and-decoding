"""Plain data objects passed between the pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One source file after the bytes have been turned into text."""

    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One card in the catalog: a piece of a document we can search."""

    chunk_id: str
    source: str
    index: int
    text: str


@dataclass(frozen=True)
class Citation:
    """A passage that was actually placed in the model prompt."""

    source: str
    index: int
    chunk_id: str


@dataclass
class Generation:
    """What the language model returned, and which passages it was shown."""

    text: str
    shown: list[Chunk]
    prompt_tokens: int


@dataclass
class Answer:
    """A user-facing reply.

    retrieved is filled even when we abstain, so evaluation can score
    the search step separately from the answer step.
    prompt_tokens is None when the language model was not called.
    """

    text: str
    citations: list[Citation]
    abstained: bool
    retrieved: list[tuple[Chunk, float]]
    shown: list[Chunk]
    prompt_tokens: int | None = None
