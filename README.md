# Exam Paper Analyzer

Give it a textbook (or an index/syllabus you already have), a folder of past
exam papers, and your own [OpenRouter](https://openrouter.ai) API key. It
tells you which topics actually get examined, how often, and which
questions repeat.

```bash
python cli.py analyze --textbook "textbook.pdf" --question-papers "exam_papers/"
```

## What it does

You give it:

- a textbook (with its own table of contents) **or** an index/syllabus/topic
  list you already have
- a folder of past exam-paper PDFs
- your OpenRouter API key

It produces:

- questions grouped by chapter/topic
- how many times each question (or a reworded equivalent of it) has been
  asked
- which years/papers each question appeared in
- a breakdown by question type (MCQ, short answer, long answer, essay, ...)
- the most-repeated questions across all papers
- the most-tested chapters, and the chapters that have never been tested
- questions that couldn't be confidently matched to a topic, flagged rather
  than silently dropped

## Features

- Textbook table-of-contents discovery, or bring your own index/syllabus
- Question extraction from exam-paper PDFs
- Topic classification of every extracted question
- Semantic deduplication (catches reworded repeats of the same question, not
  just exact text matches)
- Repetition/frequency analysis across papers and years
- Chapter-level tested/untested breakdown
- Resumable: re-running against a folder that only gained a few new papers
  reprocesses just what's new, not everything from scratch
- Automatic fallback across a pool of OpenRouter models if one is
  unavailable or rate-limited

## Requirements

- **Python 3.14** (the only version this has actually been tested on; the
  underlying dependencies declare support back to Python 3.10, but that
  hasn't been validated here)
- An [OpenRouter](https://openrouter.ai/keys) API key (free tier works --
  the default model pool is entirely `:free` models)
- ~3-4 GB of disk space for dependencies (`docling`'s document-conversion
  stack pulls in `torch`, which is the bulk of it) plus a few hundred MB for
  a local embedding model, downloaded automatically on first use

Ollama is **not required**. It's supported as an optional local backend for
development/testing (`--provider ollama`), but the normal workflow above
uses OpenRouter exclusively and needs no local model install.

## Installation

```bash
git clone <this-repository-url>
cd exampapersorter
pip install -r requirements.txt
```

The first `docling`/`torch` install is large and can take a while depending
on your connection. This is a one-time cost.

## API key setup

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). Supply
it one of three ways, in this order of precedence:

```bash
# 1. Command-line flag
python cli.py analyze --openrouter-api-key "sk-or-v1-..." --textbook "..." --question-papers "..."

# 2. Environment variable
export OPENROUTER_API_KEY="sk-or-v1-..."          # macOS/Linux
$env:OPENROUTER_API_KEY = "sk-or-v1-..."          # Windows PowerShell

# 3. Saved to your OS credential store (Windows Credential Manager / macOS
#    Keychain / Linux Secret Service), so you don't have to pass it again
python cli.py analyze --openrouter-api-key "sk-or-v1-..." --save-key --textbook "..." --question-papers "..."
```

Never commit a real API key to this repository (README, tests, config
files, logs, etc.). The key is only ever read from the CLI flag, the
environment variable, or your OS credential store -- it is never written to
the database, to any output file, or anywhere in plain text by this
application.

## Usage

Textbook mode (the textbook has its own table of contents that gets
discovered automatically):

```bash
python cli.py analyze --textbook "textbook.pdf" --question-papers "exam_papers/"
```

Index mode (you already have a topic list -- skips textbook TOC discovery;
accepts `.pdf`, `.txt`, `.md`, or `.json`):

```bash
python cli.py analyze --index "syllabus.txt" --question-papers "exam_papers/"
```

Re-run either command later against the same folder plus new papers, and
only the new papers get processed -- prior work is reused, not repeated.

## Output

```text
output/final_analysis/report.md     human-readable report
output/final_analysis/analysis.json same data, structured (for further processing)
```

Intermediate per-stage output (extracted questions, topic classifications,
deduplication results) is also written under `output/`, if you want to
inspect the pipeline's work at each step.

## Architecture

Internally, `analyze` runs a five-stage pipeline against a local SQLite
database (`data/pipeline.db`), each stage resumable independently:

1. **Topic authority** -- discover the textbook's table of contents, or load
   a supplied index/syllabus
2. **Question extraction** -- pull individual questions out of each
   exam-paper PDF
3. **Topic classification** -- match each question to a topic
4. **Semantic deduplication** -- group reworded repeats of the same question
   together
5. **Frequency analysis / report generation** -- aggregate everything into
   the final report

Every LLM call goes through OpenRouter by default, against a configurable,
ordered pool of free-tier models (`config/openrouter_models.json`) with
automatic fallback on failure/rate-limit. PDF parsing uses
[Docling](https://github.com/docling-project/docling); semantic
deduplication uses a small local embedding model via
[fastembed](https://github.com/qdrant/fastembed) (no external embedding
API).

## Limitations

- OCR/layout quality depends on the source PDF -- scanned, low-quality, or
  unusually formatted exam papers extract less reliably than clean digital
  PDFs
- Not every question can be confidently matched to a topic or deduplicated
  against another; these are reported explicitly (`no_match`, `ambiguous`,
  `unclassified`) rather than guessed
- OpenRouter's free-tier model availability changes over time and is
  outside this project's control -- if every model in the pool is
  rate-limited or unavailable at once, a run can fail and need a retry
  later. This application does not provide unlimited usage or guarantee
  availability
- Processing time is dominated by LLM calls and is roughly proportional to
  the number of questions across your exam papers -- expect several minutes
  for a real exam-paper set, not seconds
- The first run downloads Docling's conversion models and the local
  embedding model, which takes noticeably longer than subsequent runs

## Testing

```bash
pytest
```

The test suite (currently 400 tests) is fully offline: no OpenRouter API
key, no internet access, and no real textbook/exam-paper PDFs are required
to run it.

## A note on your own PDFs

This repository does not include any textbook or exam-paper PDFs. Bring
your own -- the application never redistributes or requires any specific
copyrighted material.
