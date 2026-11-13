"""Tie indexing, search, the abstention gate, and generation together."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag.chunking import chunk_documents
from rag.config import (
    EMBEDDING_MODEL,
    MAX_CHARS,
    MIN_BLOCK_CHARS,
    MIN_SIMILARITY,
    OVERLAP,
    PROMPT_PASSAGES,
    TOP_K,
)
from rag.embeddings import Embedder
from rag.loaders import fingerprint_documents, load_directory
from rag.store import Index
from rag.types import Answer, Chunk, Citation, Document, Generation

# Settings that change the vectors. A mismatch means the index is stale.
_INDEX_SETTINGS = {
    "embedding_model": EMBEDDING_MODEL,
    "max_chars": MAX_CHARS,
    "overlap": OVERLAP,
    "min_block_chars": MIN_BLOCK_CHARS,
}


def is_abstention(text: str) -> bool:
    """True when the model declined, or returned nothing.

    A long reply that merely starts with "I don't know" and then states
    a fact is kept. We only treat short refusals as abstention.
    """
    cleaned = " ".join(text.strip().lower().split())
    cleaned = cleaned.strip(" .")
    if cleaned == "":
        return True
    exact = {
        "i don't know",
        "i do not know",
        "i dont know",
        "i don't know the answer",
        "i do not know the answer",
    }
    if cleaned in exact:
        return True
    if cleaned.startswith("i don't know") and len(cleaned) < 40:
        return True
    if cleaned.startswith("i do not know") and len(cleaned) < 40:
        return True
    return False


class RagPipeline:
    """Ask a question against an already built index."""

    def __init__(
        self,
        index: Index,
        embedder: Embedder,
        generator,
        min_similarity: float = MIN_SIMILARITY,
        top_k: int = TOP_K,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.min_similarity = min_similarity
        self.top_k = top_k

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[object, list[tuple[Chunk, float]]]:
        """Encode the question and return (vector, ranked passages)."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        k = self.top_k if top_k is None else top_k
        vector = self.embedder.encode([question])[0]
        hits = self.index.search(vector, k)
        return vector, hits

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        use_gate: bool = True,
    ) -> Answer:
        """Full path: encode, search, maybe abstain, otherwise decode."""
        _vector, hits = self.retrieve(question, top_k)
        best_score = hits[0][1] if hits else 0.0
        if use_gate and best_score < self.min_similarity:
            return Answer(
                text="I don't know.",
                citations=[],
                abstained=True,
                retrieved=hits,
                shown=[],
                prompt_tokens=None,
            )
        if self.generator is None:
            raise RuntimeError("No generator loaded.")

        passages = [chunk for chunk, _score in hits][:PROMPT_PASSAGES]
        generation: Generation = self.generator.generate(question.strip(), passages)
        # No passage left in the prompt means the model would be guessing.
        if not generation.shown or is_abstention(generation.text):
            return Answer(
                text="I don't know.",
                citations=[],
                abstained=True,
                retrieved=hits,
                shown=list(generation.shown),
                prompt_tokens=generation.prompt_tokens,
            )
        citations = [
            Citation(source=chunk.source, index=chunk.index, chunk_id=chunk.chunk_id)
            for chunk in generation.shown
        ]
        return Answer(
            text=generation.text.strip(),
            citations=citations,
            abstained=False,
            retrieved=hits,
            shown=list(generation.shown),
            prompt_tokens=generation.prompt_tokens,
        )


def build_index(
    documents: list[Document],
    index_dir: Path,
    embedder: Embedder,
) -> Index:
    """Chunk, encode, and save. Empty files are skipped with a warning."""
    usable = []
    for doc in documents:
        if doc.text.strip():
            usable.append(doc)
        else:
            print(f"Warning: no text extracted from {doc.source}", file=sys.stderr)
    if not usable:
        raise SystemExit("No documentation text found.")
    chunks = chunk_documents(usable)
    if not chunks:
        raise SystemExit("Chunking produced no passages.")
    vectors = embedder.encode([chunk.text for chunk in chunks])
    meta = dict(_INDEX_SETTINGS)
    # Hash the files we were given, including any that had no text,
    # so this matches the hash load_or_build computed.
    meta["fingerprint"] = fingerprint_documents(documents)
    meta["chunk_count"] = len(chunks)
    index = Index(chunks, vectors, meta)
    index.save(index_dir)
    return index


def load_or_build(docs_dir: Path, index_dir: Path, embedder: Embedder) -> Index:
    """Reuse the index when the files and the settings still match."""
    documents = load_directory(docs_dir)
    fingerprint = fingerprint_documents(documents)
    index_dir = Path(index_dir)
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        settings_ok = all(meta.get(key) == value for key, value in _INDEX_SETTINGS.items())
        if settings_ok and meta.get("fingerprint") == fingerprint:
            return Index.load(index_dir)
        print("Rebuilding the index because the documents or settings changed.")
    else:
        print("Building the index.")
    return build_index(documents, index_dir, embedder)
