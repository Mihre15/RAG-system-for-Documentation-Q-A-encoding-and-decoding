#!/usr/bin/env python3
"""Score retrieval, and optionally the abstention gate plus generation.

Retrieval is scored on its own, before any language model is involved.

Hit@k: for an answerable question, the expected snippet appears in at
least one of the top-k passages. The snippet is a fact that is actually
written in the docs. A hit means we retrieved that fact, not that the
model phrased the answer well.

The cutoff recommendation uses rag/threshold.py. This script does not
overwrite rag/config.py. If the configured cutoff differs, it prints
a warning and still uses the configured value.

--retrieval-only skips the language model entirely.

Full results are written to eval/last_results.json. Quote that file.
Do not invent numbers from memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from rag.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CHARS,
    MIN_BLOCK_CHARS,
    MIN_SIMILARITY,
    OVERLAP,
    PROMPT_PASSAGES,
    TOP_K,
)
from rag.embeddings import Embedder
from rag.pipeline import RagPipeline, load_or_build
from rag.store import cosine
from rag.threshold import gate_successes, recommend_cutoff


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"No questions in {path}")
    seen = set()
    for item in data:
        for key in ("id", "question", "answerable", "group"):
            if key not in item:
                raise SystemExit(f"Question missing {key}: {item}")
        if item["id"] in seen:
            raise SystemExit(f"Duplicate question id: {item['id']}")
        seen.add(item["id"])
        if item["answerable"] and not item.get("expected_snippet"):
            raise SystemExit(f"{item['id']} is answerable but has no expected_snippet")
    return data


def snippet_rank(hits, snippet: str | None) -> int | None:
    if not snippet:
        return None
    needle = snippet.lower()
    for rank, (chunk, _score) in enumerate(hits, start=1):
        if needle in chunk.text.lower():
            return rank
    return None


def forbidden_in_hits(hits, forbidden: str | None) -> bool:
    if not forbidden:
        return False
    needle = forbidden.lower()
    return any(needle in chunk.text.lower() for chunk, _score in hits)


def _round(value: float | None, places: int = 6):
    if value is None:
        return None
    return round(float(value), places)


def summarize(rows: list[dict], cutoff: float) -> dict:
    """Rates from the rows. Denominators are counts, so small N is obvious."""

    def rate(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    answerable = [row for row in rows if row["answerable"]]
    groups = {}
    for group in sorted({row["group"] for row in answerable}):
        subset = [row for row in answerable if row["group"] == group]
        groups[group] = {
            "n": len(subset),
            "hit_at_1": sum(1 for row in subset if row["hit_at_1"]),
            "hit_at_k": sum(1 for row in subset if row["hit_at_k"]),
            "hit_at_1_rate": rate(sum(1 for row in subset if row["hit_at_1"]), len(subset)),
            "hit_at_k_rate": rate(sum(1 for row in subset if row["hit_at_k"]), len(subset)),
        }

    unanswerable = [row for row in rows if not row["answerable"]]
    # Gate prediction uses the top score only. It does not look at the model.
    gate_abstain = [row for row in unanswerable if row["top_score"] < cutoff]
    answerable_and_hit = [row for row in answerable if row["hit_at_k"]]
    false_abstain_gate = [row for row in answerable_and_hit if row["top_score"] < cutoff]

    summary = {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "top_k": TOP_K,
        "cutoff": cutoff,
        "hit_at_1": sum(1 for row in answerable if row["hit_at_1"]),
        "hit_at_k": sum(1 for row in answerable if row["hit_at_k"]),
        "hit_at_1_rate": rate(
            sum(1 for row in answerable if row["hit_at_1"]), len(answerable)
        ),
        "hit_at_k_rate": rate(
            sum(1 for row in answerable if row["hit_at_k"]), len(answerable)
        ),
        "by_group": groups,
        "gate_abstain_on_unanswerable": len(gate_abstain),
        "gate_false_abstain_on_hits": len(false_abstain_gate),
        "gate_successes": gate_successes(
            [
                {
                    "answerable": row["answerable"],
                    "hit": row["hit_at_k"],
                    "top_score": row["top_score"],
                }
                for row in rows
            ],
            cutoff,
        ),
    }

    generated = [row for row in rows if row.get("generator_called") is not None]
    if generated and any(row.get("answer_text") is not None or row.get("generator_called") for row in rows):
        # Filled only on a full run, where ask() was used.
        if any("abstained" in row and row["abstained"] is not None for row in rows):
            summary["system_abstain_on_unanswerable"] = sum(
                1 for row in unanswerable if row.get("abstained")
            )
            summary["system_false_abstain_on_answerable"] = sum(
                1 for row in answerable if row.get("abstained")
            )
            summary["system_answered_unanswerable"] = sum(
                1 for row in unanswerable if row.get("abstained") is False
            )
    ungated = [row for row in unanswerable if row.get("ungated_answer") is not None]
    if ungated:
        summary["ungated_model_abstain_on_unanswerable"] = sum(
            1 for row in ungated if row["ungated_abstained"]
        )
        summary["ungated_model_n"] = len(ungated)
    return summary


def encoding_check(embedder: Embedder) -> dict:
    """Two calls on the same string, and one call on an unrelated string."""
    same_a = embedder.encode(["backup notes"])[0]
    same_b = embedder.encode(["backup notes"])[0]
    other = embedder.encode(["photosynthesis in algae"])[0]
    return {
        "left": "backup notes",
        "right_same": "backup notes",
        "right_unrelated": "photosynthesis in algae",
        "identical_cosine": cosine(same_a, same_b),
        "unrelated_cosine": cosine(same_a, other),
        "vector_dim": int(same_a.shape[0]),
    }


def package_versions(include_generation: bool) -> dict:
    import numpy
    import sentence_transformers

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }
    try:
        import pypdf

        versions["pypdf"] = pypdf.__version__
    except ImportError:
        versions["pypdf"] = None
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = None
    if include_generation:
        import transformers

        versions["transformers"] = transformers.__version__
    return versions


def chunk_stats(index) -> dict:
    from collections import Counter

    lengths = [len(chunk.text) for chunk in index.chunks]
    return {
        "chunk_count": len(lengths),
        "chunk_chars_min": min(lengths) if lengths else 0,
        "chunk_chars_max": max(lengths) if lengths else 0,
        "chunk_chars_mean": (sum(lengths) / len(lengths)) if lengths else 0,
        "chunks_per_source": dict(Counter(chunk.source for chunk in index.chunks)),
        "decoy_in_index": any(
            "3 megabytes" in chunk.text.lower() for chunk in index.chunks
        ),
    }


def evaluate(args: argparse.Namespace) -> dict:
    questions = load_questions(Path(args.questions))
    started_embedder = time.perf_counter()
    embedder = Embedder()
    embedder_seconds = time.perf_counter() - started_embedder
    index_started = time.perf_counter()
    index = load_or_build(Path(args.docs), Path(args.index), embedder)
    index_seconds = time.perf_counter() - index_started

    check = encoding_check(embedder)
    preview = embedder.encode([questions[0]["question"]])[0]
    first_vector_preview = [float(value) for value in preview[:8]]
    generator = None
    generator_seconds = None
    if not args.retrieval_only:
        from rag.generator import FlanGenerator

        started_generator = time.perf_counter()
        generator = FlanGenerator()
        generator_seconds = time.perf_counter() - started_generator
    pipeline = RagPipeline(index, embedder, generator, top_k=args.top_k)

    rows = []
    for item in questions:
        started = time.perf_counter()
        ungated = None
        if args.retrieval_only:
            _vector, hits = pipeline.retrieve(item["question"], top_k=args.top_k)
            answer = None
        else:
            answer = pipeline.ask(item["question"], top_k=args.top_k, use_gate=True)
            hits = answer.retrieved
            if not item["answerable"]:
                ungated = pipeline.ask(
                    item["question"], top_k=args.top_k, use_gate=False
                )
        elapsed = time.perf_counter() - started

        rank = snippet_rank(hits, item.get("expected_snippet"))
        top_score = hits[0][1] if hits else 0.0
        row = {
            "id": item["id"],
            "group": item["group"],
            "question": item["question"],
            "answerable": bool(item["answerable"]),
            "expected_snippet": item.get("expected_snippet"),
            "why": item.get("why"),
            "top_score": top_score,
            "top_source": hits[0][0].source if hits else None,
            "top_chunk_id": hits[0][0].chunk_id if hits else None,
            "retrieved": [
                {
                    "rank": rank_number,
                    "score": score,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text": chunk.text,
                }
                for rank_number, (chunk, score) in enumerate(hits, start=1)
            ],
            "hit_rank": rank,
            "hit_at_1": rank == 1,
            "hit_at_k": rank is not None,
            "forbidden_in_top_k": forbidden_in_hits(hits, item.get("forbidden_snippet")),
            "seconds": elapsed,
        }
        if answer is not None:
            row["abstained"] = answer.abstained
            row["answer_text"] = answer.text
            row["generator_called"] = answer.prompt_tokens is not None
            row["prompt_tokens"] = answer.prompt_tokens
            row["passages_used"] = [citation.chunk_id for citation in answer.citations]
        else:
            row["abstained"] = None
            row["answer_text"] = None
            row["generator_called"] = None
            row["prompt_tokens"] = None
            row["passages_used"] = None
        if ungated is not None:
            row["ungated_abstained"] = ungated.abstained
            row["ungated_answer"] = ungated.text
            row["ungated_generator_called"] = ungated.prompt_tokens is not None
        rows.append(row)
        hit_mark = "hit" if row["hit_at_k"] else "miss"
        if not row["answerable"]:
            hit_mark = "n/a"
        answer_preview = ""
        if row["answer_text"]:
            answer_preview = " | " + " ".join(row["answer_text"].split())[:70]
        print(
            f"{row['id']:4} {row['group']:13} score={row['top_score']:.4f} "
            f"{hit_mark:4} rank={row['hit_rank']}{answer_preview}",
            flush=True,
        )

    rule_rows = [
        {
            "answerable": row["answerable"],
            "hit": row["hit_at_k"],
            "top_score": row["top_score"],
        }
        for row in rows
    ]
    recommended, successes, total = recommend_cutoff(rule_rows)
    summary = summarize(rows, MIN_SIMILARITY)
    results = {
        "retrieval_only": bool(args.retrieval_only),
        "versions": package_versions(include_generation=not args.retrieval_only),
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "max_chars": MAX_CHARS,
            "overlap": OVERLAP,
            "min_block_chars": MIN_BLOCK_CHARS,
            "top_k": args.top_k,
            "prompt_passages": PROMPT_PASSAGES,
            "min_similarity": MIN_SIMILARITY,
        },
        "embedder_seconds": embedder_seconds,
        "generator_seconds": generator_seconds,
        "index_seconds": index_seconds,
        "encoding_check": check,
        "first_question_id": questions[0]["id"],
        "first_question_vector_first8": first_vector_preview,
        "chunks": chunk_stats(index),
        "recommended_cutoff": recommended,
        "recommended_gate_successes": successes,
        "recommended_gate_total": total,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Wrote {out}")
    print(
        f"Encoding check: identical cosine={check['identical_cosine']:.6f}, "
        f"unrelated cosine={check['unrelated_cosine']:.6f}, "
        f"dim={check['vector_dim']}"
    )
    print(
        f"Chunks: {results['chunks']['chunk_count']} "
        f"(chars min/mean/max "
        f"{results['chunks']['chunk_chars_min']}/"
        f"{results['chunks']['chunk_chars_mean']:.1f}/"
        f"{results['chunks']['chunk_chars_max']})"
    )
    print(f"Decoy '3 megabytes' present in the index: {results['chunks']['decoy_in_index']}")
    hit_k = summary["hit_at_k"]
    hit_1 = summary["hit_at_1"]
    n_ans = summary["n_answerable"]
    print(f"Hit@1: {hit_1}/{n_ans}    Hit@{args.top_k}: {hit_k}/{n_ans}")
    for group, stats in summary["by_group"].items():
        print(
            f"  {group}: hit@1 {stats['hit_at_1']}/{stats['n']}, "
            f"hit@{args.top_k} {stats['hit_at_k']}/{stats['n']}"
        )
    print(
        f"Recommended cutoff: {recommended:.6f} "
        f"({successes}/{total} gate decisions on this question list)"
    )
    print(f"Configured cutoff: {MIN_SIMILARITY:.6f}")
    if abs(recommended - MIN_SIMILARITY) > 1e-6:
        print(
            "WARNING: rag/config.py MIN_SIMILARITY differs from the "
            "recommended cutoff. Generation used the configured value. "
            "Update MIN_SIMILARITY if you want them to match, then re-run."
        )
    else:
        print("Configured cutoff matches the recommended cutoff.")
    if not args.retrieval_only:
        print(
            "System abstentions on unanswerable: "
            f"{summary.get('system_abstain_on_unanswerable')}/"
            f"{summary['n_unanswerable']}"
        )
        print(
            "System abstentions on answerable (false): "
            f"{summary.get('system_false_abstain_on_answerable')}/"
            f"{summary['n_answerable']}"
        )
        print(
            "Model abstentions with the gate forced open, unanswerable only: "
            f"{summary.get('ungated_model_abstain_on_unanswerable')}/"
            f"{summary.get('ungated_model_n')}"
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--index", default="index")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--output", default="eval/last_results.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score search and the cutoff rule, and do not load the language model",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
