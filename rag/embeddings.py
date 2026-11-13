"""Encode passages and questions with the same embedding model.

Analogy: the card catalog and the question slip are written in the
same handwriting. If they were not, comparing them would be meaningless.

This is the ENCODING step of the project. It is a different encoder
from the one inside FLAN-T5. That second encoder belongs to decoding
and only runs when we ask the language model for an answer.
"""

from __future__ import annotations

import numpy as np

from rag.config import EMBEDDING_MODEL


class Embedder:
    """A thin wrapper around sentence-transformers.

    The wrapper exists so the rest of the code calls one method,
    encode(list of strings) -> matrix of floats, and so tests can
    substitute a fake with the same method.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency sentence-transformers. From the project "
                "folder run: pip install -r requirements.txt"
            ) from exc
        try:
            # CPU on purpose: same path on a lab machine with no GPU.
            self.model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            raise SystemExit(
                f"Could not load the embedding model {model_name}. "
                "The first run needs a network connection so the weights "
                f"can be downloaded. Underlying error: {exc}"
            ) from exc
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Map each string to a float32 vector.

        Normalization happens later, in the index, so cosine search
        has one implementation.
        """
        if not texts:
            raise ValueError("encode() needs at least one string")
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
