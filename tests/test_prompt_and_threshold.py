"""Tests for the prompt layout and the precommitted cutoff rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.prompt import build_prompt, passages_that_fit
from rag.threshold import gate_successes, recommend_cutoff
from rag.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, chunk_id.split("#")[0], 0, text)


class PromptTests(unittest.TestCase):
    def test_prompt_names_the_passage_and_then_the_question(self) -> None:
        chunk = _chunk("docs/storage.md#0", "folder named Northstar")
        prompt = build_prompt("Where are notebooks stored?", [chunk])
        self.assertLess(
            prompt.index("folder named Northstar"),
            prompt.index("Where are notebooks stored?"),
        )
        self.assertIn("I don't know", prompt)
        self.assertNotIn("[1]", prompt)
        self.assertNotIn("Source:", prompt)

    def test_lowest_ranked_passage_is_dropped_first(self) -> None:
        passages = [
            _chunk("a.md#0", "ALPHA " * 20),
            _chunk("b.md#0", "BETA " * 20),
        ]

        def count(prompt: str) -> int:
            return 10000 if "BETA" in prompt else 10

        prompt, shown, n_tokens = passages_that_fit(
            "Question about alpha?", passages, count, limit=100
        )
        self.assertEqual([chunk.chunk_id for chunk in shown], ["a.md#0"])
        self.assertIn("Question about alpha?", prompt)
        self.assertNotIn("BETA", prompt)
        self.assertEqual(n_tokens, 10)

    def test_nothing_fits_leaves_the_question_and_no_passages(self) -> None:
        passages = [_chunk("a.md#0", "ALPHA")]
        prompt, shown, _n = passages_that_fit(
            "A long question?", passages, lambda _prompt: 9999, limit=10
        )
        self.assertEqual(shown, [])
        self.assertIn("A long question?", prompt)


class ThresholdRuleTests(unittest.TestCase):
    """Locks the rule that was written before the corpus was scored.

    Do not edit the expected numbers to flatter a later eval run.
    If the corpus scores are awkward, report them. Do not move this rule.
    """

    def test_separable_scores_land_between_the_groups(self) -> None:
        rows = [
            {"answerable": False, "hit": False, "top_score": 0.2},
            {"answerable": True, "hit": True, "top_score": 0.8},
        ]
        cutoff, successes, total = recommend_cutoff(rows)
        self.assertAlmostEqual(cutoff, 0.5, places=6)
        self.assertEqual(successes, 2)
        self.assertEqual(total, 2)

    def test_overlap_example_from_the_module_notes(self) -> None:
        # Scores: unanswerable 0.30 and 0.65, miss 0.40, hits 0.70 and 0.80.
        # The midpoint between 0.65 and 0.70 is 0.675, which keeps both hits
        # and rejects both unanswerable questions and the miss.
        rows = [
            {"answerable": True, "hit": True, "top_score": 0.80},
            {"answerable": True, "hit": True, "top_score": 0.70},
            {"answerable": True, "hit": False, "top_score": 0.40},
            {"answerable": False, "hit": False, "top_score": 0.30},
            {"answerable": False, "hit": False, "top_score": 0.65},
        ]
        cutoff, successes, total = recommend_cutoff(rows)
        self.assertAlmostEqual(cutoff, 0.675, places=6)
        self.assertEqual(successes, 5)
        self.assertEqual(total, 5)
        self.assertEqual(gate_successes(rows, cutoff), 5)

    def test_tie_keeps_the_higher_cutoff(self) -> None:
        # Both 0.50 and 0.85 get 2 of 3. The rule must keep 0.85.
        rows = [
            {"answerable": True, "hit": True, "top_score": 0.6},
            {"answerable": False, "hit": False, "top_score": 0.4},
            {"answerable": False, "hit": False, "top_score": 0.8},
        ]
        cutoff, successes, total = recommend_cutoff(rows)
        self.assertAlmostEqual(cutoff, 0.85, places=6)
        self.assertEqual(successes, 2)
        self.assertEqual(total, 3)


if __name__ == "__main__":
    unittest.main()
