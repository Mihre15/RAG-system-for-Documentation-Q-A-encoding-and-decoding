"""Tests for cosine search and the abstention gate. No neural nets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.pipeline import RagPipeline, is_abstention
from rag.store import Index
from rag.types import Chunk, Generation


def _chunk(text: str) -> Chunk:
    return Chunk(f"{text}#0", "docs/notes.md", 0, text)


class StoreTests(unittest.TestCase):
    def test_unnormalized_vectors_still_rank_by_cosine(self) -> None:
        chunks = [_chunk("notes"), _chunk("photos")]
        vectors = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        index = Index(chunks, vectors, meta={})
        hits = index.search(np.array([0.2, 0.0], dtype=np.float32), k=1)
        self.assertEqual(hits[0][0].text, "notes")
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)

    def test_k_larger_than_the_catalog_returns_everything(self) -> None:
        chunks = [_chunk("only")]
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        index = Index(chunks, vectors, meta={})
        hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=5)
        self.assertEqual(len(hits), 1)


class _ScriptedEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.vector for _ in texts])


class _BoomGenerator:
    def generate(self, question: str, passages: list[Chunk]) -> Generation:
        raise AssertionError("generator should not be called")


class _FixedGenerator:
    def __init__(self, text: str, keep: int) -> None:
        self.text = text
        self.keep = keep

    def generate(self, question: str, passages: list[Chunk]) -> Generation:
        shown = list(passages[: self.keep])
        return Generation(text=self.text, shown=shown, prompt_tokens=12)


class PipelineTests(unittest.TestCase):
    def _index(self) -> Index:
        chunks = [
            Chunk("docs/storage.md#0", "docs/storage.md", 0, "home directory"),
            Chunk("docs/commands.md#0", "docs/commands.md", 0, "northstar list"),
        ]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return Index(chunks, vectors, meta={})

    def test_low_score_abstains_without_calling_the_model(self) -> None:
        # The only passage points along x. The question points along y.
        # The angle is 90 degrees, so cosine is 0, which is under the cutoff.
        only = Index(
            [Chunk("docs/a.md#0", "docs/a.md", 0, "notes")],
            np.array([[1.0, 0.0]], dtype=np.float32),
            meta={},
        )
        pipeline = RagPipeline(
            only,
            _ScriptedEmbedder([0.0, 1.0]),
            _BoomGenerator(),
            min_similarity=0.5,
        )
        answer = pipeline.ask("What is the price?")
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.text, "I don't know.")
        self.assertEqual(answer.citations, [])
        self.assertIsNone(answer.prompt_tokens)
        self.assertEqual(len(answer.retrieved), 1)

    def test_high_score_cites_only_passages_shown_to_the_model(self) -> None:
        pipeline = RagPipeline(
            self._index(),
            _ScriptedEmbedder([1.0, 0.0]),
            _FixedGenerator("Notes live in the home directory.", keep=1),
            min_similarity=0.2,
            top_k=2,
        )
        answer = pipeline.ask("Where are notebooks stored?")
        self.assertFalse(answer.abstained)
        self.assertEqual(len(answer.retrieved), 2)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].chunk_id, "docs/storage.md#0")

    def test_model_refusal_hides_citations(self) -> None:
        pipeline = RagPipeline(
            self._index(),
            _ScriptedEmbedder([1.0, 0.0]),
            _FixedGenerator("I don't know.", keep=2),
            min_similarity=0.2,
            top_k=1,
        )
        answer = pipeline.ask("Where are notebooks stored?")
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.citations, [])
        self.assertIsNotNone(answer.prompt_tokens)

    def test_abstention_detector(self) -> None:
        self.assertTrue(is_abstention("I don't know."))
        self.assertTrue(is_abstention(""))
        self.assertFalse(
            is_abstention(
                "I don't know. The maximum note body is 20000 characters."
            )
        )
        self.assertFalse(is_abstention("Notes are stored as UTF-8 text."))


if __name__ == "__main__":
    unittest.main()
