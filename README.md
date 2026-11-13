# Documentation question answering (RAG)

This project answers questions about a small set of documentation files.
It has two steps, and they are kept separate on purpose.

**Encoding.** A free embedding model turns each passage, and later your
question, into a list of numbers. Think of a card catalog: the same
handwriting is used on every card and on the question slip, so the
cards can be compared.

**Decoding.** A free local language model reads the original words of
the closest cards and writes an answer. The numbers are not turned
back into words. We kept the words. If nothing in the catalog is close
enough, the program says **I don't know** and does not ask the model
to guess.

The documents in `docs/` describe **Northstar Notes**, a fictional
command-line notebook. They exist so the demo runs offline. Northstar
Notes is not a real product.

The course brief in the request was still the placeholder text (no
rubric, deadline, or required extras), so this repository contains the
four deliverables that were named: the program, an evaluation script,
this README, and `REPORT.md`.

## Setup

Tested commands are for macOS or Linux, from this folder. On Windows,
create the virtual environment with `py -m venv .venv` and activate it
with `.venv\Scripts\activate`. The Windows path was not run here.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first real question downloads two models into the Hugging Face
cache (not into this folder):

- `sentence-transformers/all-MiniLM-L6-v2` encodes passages and questions
- `google/flan-t5-small` writes the answer

Both are open weights. No API key is required. The run behind
`REPORT.md` used Python 3.9.6 on macOS arm64, sentence-transformers
3.4.1, transformers 4.57.6, torch 2.8.0, numpy 2.0.2, and pypdf 5.9.0.
Those versions are also in `eval/last_results.json`. On that Python,
urllib3 prints a LibreSSL warning before the answer. The warning did
not stop the run.

`docs/backup.pdf` is already in the folder. To rebuild that sample
file (this is not part of answering questions):

```bash
python scripts/make_backup_pdf.py
```

## Run

```bash
python main.py chunks
python main.py index
python main.py search "Where are notebooks stored by default?"
python main.py ask "What pip command installs Northstar Notes?"
python main.py ask "What is the monthly price of Northstar Notes?" --verbose
```

`chunks` only shows how the files were cut. It does not load a model.

`search` is encoding plus retrieval. It prints the question vector and
the nearest passages. It does not call the language model. If the best
score clears the cutoff, it says the language model would be called
with the nearest passage only. `ask` is what actually calls it.

`ask` is the full system. The passage placed in the prompt is listed
under **Passages used**. An abstention does not list a passage, because
that passage did not support an answer. Add `--verbose` to see scores
anyway.

You only need to run `index` yourself if you want to inspect the saved
files in `index/`. `search` and `ask` build or reuse that index. They
rebuild it when the documents or the chunking settings change.

## Evaluate

```bash
python -m unittest discover -s tests
python evaluate.py --retrieval-only
python evaluate.py
```

`tests/` checks chunking, HTML and PDF loading, cosine search, the
abstention gate, and the cutoff rule. Those tests do not download models.

`evaluate.py` scores the questions in `eval/questions.json`.

- **Hit@k** means the expected fact (a short quote from the docs)
  appears in one of the top k passages. It does not grade the sentence
  the model wrote.
- **Hit@1** means that fact is the single nearest passage.
- Questions are grouped as `direct`, `paraphrase`, or `unanswerable`.
- Unanswerable questions have no supporting fact. The script counts how
  often the system says "I don't know".
- With the gate forced open, the script also records what the small
  language model says on unanswerable questions. That is a diagnostic,
  not the normal path.

The script prints a cutoff recommended by the rule in `rag/threshold.py`.
It does **not** edit `rag/config.py`. If the printed recommendation and
`MIN_SIMILARITY` differ, generation still used `MIN_SIMILARITY`, and the
script says so.

The full write-up of the measured run is `REPORT.md`. The raw file for
the prompt the code uses now is `eval/last_results.json`. An earlier
prompt that often answered with `[1]` or `[3]` is kept in
`eval/numbered_prompt_results.json`.

On this machine, after the models were already downloaded, these two
commands printed the answers below. A LibreSSL warning from urllib3
appeared first and is left out here.

```text
$ python main.py ask "What pip command installs Northstar Notes?"
pip install northstar-notes

Passages used:
- docs/getting-started.md (chunk 1)

$ python main.py ask "What is the monthly price of Northstar Notes?"
I don't know.
```

The price is not in the documents. `--verbose` on that second command
showed the best cosine at 0.5275, under the cutoff, and the language
model was not called.

## Layout

| Path | Role |
| --- | --- |
| `rag/loaders.py` | Markdown, HTML, and PDF to text |
| `rag/chunking.py` | Paragraphs, then a character cap |
| `rag/embeddings.py` | Encoding with MiniLM |
| `rag/store.py` | Cosine search |
| `rag/threshold.py` | Precommitted cutoff rule |
| `rag/prompt.py` | The words shown to the language model |
| `rag/generator.py` | Decoding with FLAN-T5 |
| `rag/pipeline.py` | Gate, then generate, then citations |
| `rag/config.py` | The knobs, in one place |
| `main.py` | `chunks`, `index`, `search`, `ask` |
| `evaluate.py` | Hit rate and abstention counts |
| `docs/` | Fictional documentation, including HTML and PDF |
| `REPORT.md` | Why the design looks like this, and what the run did |

## Design in one minute

Passages are cards. MiniLM writes a vector for each card and for the
question. Cosine similarity compares the angles of those vectors.
The top 3 cards are retrieved. If the best score is below
`MIN_SIMILARITY` in `rag/config.py` (0.5305415 on the measured run),
the answer is "I don't know." Otherwise FLAN-T5 reads the nearest
passage only and writes a short answer. The program cites that
passage. It does not ask the model to invent the citation.

Why these choices, and what the measured scores were, is the subject
of `REPORT.md`. Read that before a viva. The code comments are the
short version of the same explanations.
