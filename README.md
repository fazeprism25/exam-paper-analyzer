# Exam Paper Analyzer

Give it a textbook (or an index/syllabus you already have), a folder of past
exam papers, and your own [OpenRouter](https://openrouter.ai) API key. It
tells you which topics actually get examined, how often, and which
questions repeat.

## Getting an OpenRouter API key

Every analysis (desktop app or command line) needs an
[OpenRouter](https://openrouter.ai) API key -- this is what lets the app
call an LLM to read and classify questions. It takes about a minute and the
default model pool is entirely free:

1. Go to [openrouter.ai](https://openrouter.ai) and sign up (email, or
   continue with Google/GitHub).
2. Once logged in, go to [openrouter.ai/keys](https://openrouter.ai/keys)
   (or **Keys** in the left sidebar).
3. Click **Create Key**, give it any name (e.g. "exam-paper-analyzer"), and
   click **Create**.
4. Copy the key (it starts with `sk-or-v1-...`) -- OpenRouter only shows it
   once, so copy it before closing the dialog. If you lose it, just create a
   new one.
5. Paste that key into the desktop app's **API key** field (see
   [First launch](#first-launch) below), or use it on the command line (see
   [API key setup](#api-key-setup)).

No payment method is required to use the free-tier models this project
defaults to.

## Desktop installation

Most people should use the desktop app -- no Python, no `pip`, no command
line. There are two different downloads on the
[releases page](https://github.com/fazeprism25/exam-paper-analyzer/releases)
-- one for Windows, one for macOS. Their filenames say which is which, so
grab the one matching your OS below; don't download both.

### Windows

1. Download `ExamPaperAnalyzer-Windows-Setup.exe` from the
   [latest release](https://github.com/fazeprism25/exam-paper-analyzer/releases).
2. Run it. Windows SmartScreen may warn that the app is from an unrecognized
   publisher -- this build isn't code-signed yet (see
   [Windows signing](#windows-signing) below). Click **More info** ->
   **Run anyway**.
3. Launch **Exam Paper Analyzer** from the Start Menu (or the desktop
   shortcut, if you opted into one during setup).

### macOS

Apple Silicon (M1/M2/M3/M4) only -- see [Supported architectures](#supported-architectures)
for why Intel Macs aren't supported.

1. Download `ExamPaperAnalyzer-macOS-arm64.dmg` from the
   [latest release](https://github.com/fazeprism25/exam-paper-analyzer/releases).
2. Open the DMG and drag **Exam Paper Analyzer** into **Applications**.
3. The app is unsigned and not notarized (no Apple Developer account is
   configured for this project -- see [macOS Gatekeeper](#macos-gatekeeper)
   below), so a plain double-click on first launch will be blocked. Instead,
   **right-click (or Control-click) the app -> Open -> Open** -- you only
   need to do this once.

### First launch

- **Topic source**: pick your textbook PDF (its table of contents is found
  automatically) or an index/syllabus file you already have.
- **Exam papers**: pick the folder of exam-paper PDFs to analyze.
- **API key**: paste your [OpenRouter](#getting-an-openrouter-api-key) key (free
  tier works). Check "Save API key securely on this computer" to store it in
  your OS credential store (Windows Credential Manager / macOS Keychain) so
  you don't have to paste it again next time.
- Click **Analyze** and watch the progress log. The very first analysis on a
  machine downloads Docling's document-conversion models and the local
  embedding model (a few hundred MB total) -- this needs internet access and
  only happens once; every later run reuses the cached copy.
- When it finishes, use **Open Report** / **Open Output Folder** to see the
  results.

### Where your data lives

The installed app is read-only program files; your database, reports, and
model cache live in a per-user folder instead, so they survive
reinstalls/updates and an uninstall never deletes them:

- Windows: `%LOCALAPPDATA%\ExamPaperAnalyzer\`
- macOS: `~/Library/Application Support/ExamPaperAnalyzer/`

A log file for troubleshooting is written to `logs\app.log` inside that
folder.

### Internet requirements

- **Installing** the app: no internet required (the installer/DMG is
  self-contained).
- **First run**: internet required, to download the document-conversion and
  embedding models (one-time).
- **Every analysis**: internet required, to call OpenRouter -- this
  application is not fully offline.

### Windows signing

This build is not code-signed (no code-signing certificate is available for
this project). SmartScreen will show an "unrecognized publisher" warning on
first run; this is expected, not a sign of a corrupted download. Signing is
a possible future improvement, not implemented here.

### macOS Gatekeeper

This build is not notarized (no Apple Developer Program account is
configured for this project). Gatekeeper blocks a plain double-click with an
"unidentified developer" message; use right-click -> Open as described
above. This is Apple's standard behavior for unsigned apps, not a bug.
Signing/notarization is a possible future improvement, not implemented here.

### Supported architectures

- Windows: x86_64 only.
- macOS: Apple Silicon (arm64) only, built and regression-tested via GitHub
  Actions on a native `macos-15` runner. Intel (x86_64) Macs are **not
  supported**: PyTorch (a docling dependency) stopped publishing macOS
  x86_64 wheels after version 2.2.2 (deprecated since January 2024), and
  2.2.2 itself predates the Python 3.14 this project targets -- there is no
  torch version that satisfies both "Python 3.14" and "Intel macOS" at the
  same time. Confirmed directly: `pip install -r requirements.txt` fails on
  a `macos-15-intel` GitHub Actions runner with "Could not find a version
  that satisfies the requirement torch<3.0.0,>=2.2.2 ... (from versions:
  none)". This is an upstream PyTorch limitation, not something this
  project can fix without either dropping to a years-old Python version or
  waiting for PyTorch to resume Intel Mac wheel builds (not expected).
- Runtime GUI testing on real hardware has only been done on Windows -- see
  the release notes for current validation status.

### Advanced / developer: command line

The desktop app is a thin GUI wrapper around the same pipeline described
below -- everything from here on is the underlying CLI, useful for
scripting, automation, or running from source.

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
- The desktop app is not fully offline -- every analysis still needs
  internet access to reach OpenRouter

## Building the installers yourself

Both build scripts run the exact steps the GitHub Actions workflows
(`.github/workflows/build-windows.yml`, `build-macos.yml`) run, so a
release can always be reproduced from source rather than depending on any
one machine:

```powershell
# Windows (requires Inno Setup 6 -- https://jrsoftware.org/isinfo.php)
.\scripts\build_windows.ps1
```

```bash
# macOS only -- PyInstaller does not cross-compile, and .icns/hdiutil
# packaging needs macOS's own tools
./scripts/build_macos.sh
```

Both scripts call the shared `packaging/exampapersorter.spec` (PyInstaller,
onedir mode). See that file's docstring for why onedir over onefile, and
why `collect_all` is used for Docling/FastEmbed/ONNX Runtime/torch instead
of hand-picked hidden imports.

## Testing

```bash
pytest
```

The test suite (currently 440 tests) is fully offline: no OpenRouter API
key, no internet access, and no real textbook/exam-paper PDFs are required
to run it.

## A note on your own PDFs

This repository does not include any textbook or exam-paper PDFs. Bring
your own -- the application never redistributes or requires any specific
copyrighted material.
