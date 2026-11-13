#!/usr/bin/env python3
"""Ask questions about the local documentation.

Examples, from the project root:

    python main.py chunks
    python main.py index
    python main.py search "Where are notebooks stored by default?"
    python main.py ask "Where are notebooks stored by default?"
    python main.py ask "What is the monthly price of Northstar Notes?" --verbose
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from rag.config import EMBEDDING_MODEL, MIN_SIMILARITY, PROMPT_PASSAGES, TOP_K
from rag.embeddings import Embedder
from rag.generator import FlanGenerator
from rag.loaders import load_directory
from rag.chunking import chunk_documents
from rag.pipeline import RagPipeline, load_or_build


def _preview(text: str, limit: int = 180) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def cmd_chunks(args: argparse.Namespace) -> None:
    documents = load_directory(args.docs)
    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks from {args.docs}")
    for chunk in chunks:
        print()
        print(f"{chunk.chunk_id}  ({len(chunk.text)} chars)")
        print(_preview(chunk.text, limit=400))


def cmd_index(args: argparse.Namespace) -> None:
    embedder = Embedder()
    index = load_or_build(args.docs, args.index, embedder)
    shape = index.embeddings.shape
    print(f"Indexed {len(index.chunks)} chunks.")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector matrix shape: {shape[0]} chunks x {shape[1]} dimensions")
    print(f"Saved to: {args.index}")


def cmd_search(args: argparse.Namespace) -> None:
    embedder = Embedder()
    index = load_or_build(args.docs, args.index, embedder)
    pipeline = RagPipeline(index, embedder, generator=None, top_k=args.top_k)
    vector, hits = pipeline.retrieve(args.question)
    print(f"Encoded the question into {int(vector.shape[0])} numbers.")
    first = " ".join(f"{float(value):.4f}" for value in vector[:8])
    print(f"First 8 values: {first}")
    print()
    if not hits:
        print("No passages in the index.")
        return
    for rank, (chunk, score) in enumerate(hits, start=1):
        print(f"{rank}. score={score:.4f}  {chunk.source} (chunk {chunk.index})")
        print(f"   {_preview(chunk.text)}")
    best = hits[0][1]
    print()
    if best < MIN_SIMILARITY:
        print(
            f"Gate: best score {best:.4f} is below the cutoff "
            f"{MIN_SIMILARITY:.4f}, so ask would say \"I don't know\" "
            "without calling the language model."
        )
    else:
        print(
            f"Gate: best score {best:.4f} clears the cutoff "
            f"{MIN_SIMILARITY:.4f}, so ask would call the language model "
            f"with the nearest {min(PROMPT_PASSAGES, len(hits))} passage"
            f"{'' if min(PROMPT_PASSAGES, len(hits)) == 1 else 's'}."
        )


def cmd_ask(args: argparse.Namespace) -> None:
    embedder = Embedder()
    index = load_or_build(args.docs, args.index, embedder)
    generator = FlanGenerator()
    pipeline = RagPipeline(index, embedder, generator, top_k=args.top_k)
    answer = pipeline.ask(args.question)
    if answer.abstained:
        print("I don't know.")
    else:
        print(answer.text)
        if answer.citations:
            print()
            print("Passages used:")
            for citation in answer.citations:
                print(f"- {citation.source} (chunk {citation.index})")
    if not args.verbose:
        return
    print()
    if answer.prompt_tokens is None:
        print("Language model: not called (similarity gate).")
    else:
        print(f"Language model: called. Prompt tokens: {answer.prompt_tokens}.")
    print("Retrieved passages:")
    for rank, (chunk, score) in enumerate(answer.retrieved, start=1):
        print(f"{rank}. {score:.4f}  {chunk.source} (chunk {chunk.index})")
        print(f"   {_preview(chunk.text)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    chunks = sub.add_parser("chunks", help="Print chunks without embedding them")
    chunks.add_argument("--docs", default="docs")
    chunks.set_defaults(func=cmd_chunks)

    index = sub.add_parser("index", help="Embed the documents and save the index")
    index.add_argument("--docs", default="docs")
    index.add_argument("--index", default="index")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="Encode a question and show nearest chunks")
    search.add_argument("question")
    search.add_argument("--docs", default="docs")
    search.add_argument("--index", default="index")
    search.add_argument("--top-k", type=int, default=TOP_K)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="Retrieve passages and generate an answer")
    ask.add_argument("question")
    ask.add_argument("--docs", default="docs")
    ask.add_argument("--index", default="index")
    ask.add_argument("--top-k", type=int, default=TOP_K)
    ask.add_argument("--verbose", action="store_true")
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
