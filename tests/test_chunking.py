"""Tests for paragraph chunking. No models required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_two_long_paragraphs_stay_split(self) -> None:
        text = (
            "The maximum note body is 20000 characters in this documentation file.\n\n"
            "A notebook may contain at most 500 notes according to the limit page."
        )
        parts = chunk_text(text)
        self.assertEqual(len(parts), 2)
        self.assertIn("20000", parts[0])
        self.assertIn("500", parts[1])

    def test_short_heading_is_glued_to_the_next_paragraph(self) -> None:
        text = (
            "## Install\n\n"
            "pip install northstar-notes is the command to use here."
        )
        parts = chunk_text(text)
        self.assertEqual(len(parts), 1)
        self.assertIn("Install", parts[0])
        self.assertIn("pip install", parts[0])

    def test_long_paragraph_covers_every_word_and_respects_the_cap(self) -> None:
        text = " ".join(["alpha"] * 400)
        parts = chunk_text(text, max_chars=100, overlap=20, min_chars=1)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 100 for part in parts))
        covered = {word for part in parts for word in part.split()}
        self.assertEqual(covered, {"alpha"})

    def test_long_token_is_split_and_overlap_adds_characters(self) -> None:
        text = "A" * 300
        parts = chunk_text(text, max_chars=50, overlap=10, min_chars=1)
        self.assertTrue(parts)
        self.assertTrue(all(len(part) <= 50 for part in parts))
        self.assertGreater(sum(len(part) for part in parts), 300)

    def test_overlap_must_be_smaller_than_the_window(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("hello world " * 40, max_chars=20, overlap=20, min_chars=1)


if __name__ == "__main__":
    unittest.main()
