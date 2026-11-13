"""Knobs for chunking, retrieval, and the two models.

Each value is a design choice. REPORT.md explains why.

MIN_SIMILARITY is applied when a question is asked. Changing it does
not require a new index. Changing the embedding model, MAX_CHARS,
OVERLAP, or MIN_BLOCK_CHARS does, because those change the vectors.
"""

# Small, free, CPU-friendly embedding model. 384-dimensional vectors.
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Small, free, CPU-friendly encoder-decoder. Used only for the answer.
# https://huggingface.co/google/flan-t5-small
GENERATION_MODEL = "google/flan-t5-small"

# Paragraphs longer than this are cut into windows.
# 500 characters is roughly 100 to 150 tokens. Three passages of that
# size, plus the question and instructions, fit in FLAN-T5's 512-token
# input. This is a fitting choice, not a claim that 500 is optimal.
MAX_CHARS = 500

# Characters repeated between neighboring windows of a long paragraph,
# so a sentence that lands on the cut still appears in one window whole
# enough to match. Must be smaller than MAX_CHARS.
OVERLAP = 80

# A block shorter than this is glued to the next block.
# A heading alone is a poor card in the catalog.
MIN_BLOCK_CHARS = 40

# How many passages to retrieve. Hit@k in the evaluation uses this k.
# Three is enough to see whether the right fact is just behind a
# distractor, without pretending we have a large corpus.
TOP_K = 3

# How many of those passages are pasted into the prompt.
# On a probe, extra passages and labels like [1] made flan-t5-small
# copy a label or a heading instead of the fact. Retrieval is still
# scored at TOP_K. See REPORT.md.
PROMPT_PASSAGES = 1

# Cosine cutoff for "I don't know".
# This is the number `recommend_cutoff` in rag/threshold.py printed for
# eval/questions.json on the measured run: 13 of 16 gate decisions.
# It sits between the unanswerable question U1 (0.527536) and the
# answerable hit P2 (0.533547). It was not adjusted by hand after that.
# Changing it does not require a new index. REPORT.md records the tradeoff.
MIN_SIMILARITY = 0.5305415

# Greedy decoding length cap for the answer.
MAX_NEW_TOKENS = 96
