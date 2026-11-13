"""DECODING: generate an answer from the retrieved passage text.

FLAN-T5 is an encoder-decoder. Calling it does two things that are
easy to mix up with retrieval:

1. T5's own encoder reads the prompt tokens.
2. T5's decoder then writes the answer one token at a time.

That encoder is not the MiniLM encoder we used to search. MiniLM
built the card catalog. T5 reads the cards we already selected.

do_sample=False is greedy decoding: at each step the model takes the
single most likely next token. The demo is repeatable. It is not a
claim that greedy decoding is the best sampling method.
"""

from __future__ import annotations

import os

from rag.config import GENERATION_MODEL, MAX_NEW_TOKENS
from rag.prompt import passages_that_fit
from rag.types import Chunk, Generation

# A warning from the tokenizer, and a request that the hub library
# skip usage telemetry. I did not audit the network traffic, so the
# telemetry flag is best-effort. The first run still downloads weights.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class FlanGenerator:
    """Local google/flan-t5-small, on CPU, greedy."""

    def __init__(self, model_name: str = GENERATION_MODEL) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency transformers. From the project folder "
                "run: pip install -r requirements.txt"
            ) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            raise SystemExit(
                f"Could not load the generation model {model_name}. "
                "The first run needs a network connection so the weights "
                f"can be downloaded. Underlying error: {exc}"
            ) from exc
        self.model.eval()
        self.model_name = model_name
        configured = int(getattr(self.tokenizer, "model_max_length", 512))
        # Some tokenizers report a huge sentinel instead of a real limit.
        if configured > 10000:
            configured = 512
        # Leave a small margin under the model maximum.
        self.prompt_limit = min(configured, 512) - 8

    def generate(self, question: str, passages: list[Chunk]) -> Generation:
        import torch

        prompt, shown, n_tokens = passages_that_fit(
            question,
            passages,
            count_tokens=self._count_tokens,
            limit=self.prompt_limit,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_limit,
        )
        with torch.no_grad():
            # We are not training, so there is no need to store gradients.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return Generation(text=text.strip(), shown=list(shown), prompt_tokens=n_tokens)

    def _count_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))
