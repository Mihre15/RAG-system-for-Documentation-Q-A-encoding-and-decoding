"""Save passage vectors and find the nearest ones.

Analogy: each passage is an arrow drawn from a shared origin. The
question is another arrow. We want the arrows that point nearly the
same way. Cosine similarity measures the angle and ignores the length
of the arrow. Rescaling every arrow to length 1 turns that angle
comparison into a dot product, which is one matrix multiply.

We normalize again inside the index. Doing it twice does not change
the direction of a non-zero vector. Tests can pass raw numbers and
search still means cosine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.types import Chunk


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors, after rescaling them."""
    return float(np.dot(_normalize_vector(a), _normalize_vector(b)))


class Index:
    """In-memory vectors plus the passage text they came from."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if matrix.shape[0] != len(chunks):
            raise ValueError("one embedding per chunk is required")
        self.chunks = list(chunks)
        self.embeddings = _normalize_rows(matrix)
        self.meta = dict(meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, cosine) pairs, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self.chunks:
            return []
        query = _normalize_vector(query_vector)
        scores = self.embeddings @ query
        k = min(k, len(self.chunks))
        # argsort is enough: this catalog is tiny. Read it as
        # "sort every passage by score and keep the first k".
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in order]

    def save(self, folder: Path) -> None:
        """Write vectors, passage text, and build settings.

        meta.json is written last. A crash before that leaves an
        incomplete folder, and load() will refuse it.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.embeddings)
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "index": chunk.index,
                "text": chunk.text,
            }
            for chunk in self.chunks
        ]
        (folder / "chunks.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (folder / "meta.json").write_text(
            json.dumps(self.meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path) -> Index:
        folder = Path(folder)
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            raw_chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
            embeddings = np.load(folder / "embeddings.npy")
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Index in {folder} is incomplete. Run: python main.py index"
            ) from exc
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                index=int(item["index"]),
                text=item["text"],
            )
            for item in raw_chunks
        ]
        return cls(chunks, embeddings, meta)
