# Exam Paper Analyzer — n8n Implementation

This is a working n8n re-implementation of the Python Exam Paper Analyzer
(the reference implementation — see [README.md](README.md)), built as a
demonstration of how the same AI-agent pipeline could be delivered as a
company-hosted n8n workflow rather than a desktop app. It runs in a local
n8n instance, uses OpenRouter for LLM calls, and persists all state in n8n
Data Tables so a long analysis survives quota pauses, restarts, or the
browser being closed.

The Python implementation is the validated behavioral reference and was
**not modified**. Everything below is new, additive n8n content.

## Architecture

Eight workflows, one shared utility plus one workflow per pipeline stage,
matching the Python reference's six stages (Stage 5+6 are combined, exactly
as the Python `final_analysis` module already does internally):

| # | Workflow | Purpose |
|---|----------|---------|
| 0 | **EPA - LLM Structured Call** | Shared utility. Every other workflow calls this instead of hitting OpenRouter directly. Checks `epa_llm_cache` first, then rotates through a pool of free OpenRouter models, classifying each failure as `success` / `quota` (stop, bubble up) / `retry` (try next model). |
| 1 | **EPA - Stage 1: Topic Authority** | Builds the topic hierarchy from a textbook PDF (LLM, bounded excerpt) or an index/syllabus (txt/md outline or JSON — fully deterministic, no LLM). Deterministic validation (duplicate/empty/cyclic/unknown-parent checks) before anything is persisted. |
| 2 | **EPA - Stage 2: Question Extraction** | Called once per uploaded exam-paper PDF. Detects paper boundaries within the file (an upload may contain several distinct papers), resolves overlaps/gaps deterministically, then extracts metadata + questions per detected paper. Resumable at paper granularity. |
| 3 | **EPA - Stage 3: Topic Classification** | Called once per paper. Classifies all its questions against the topic hierarchy in one LLM call, then deterministically reconciles the response (unknown question ids discarded, missing verdicts → `unclassified`, invented topic ids downgraded to `unclassified` — never trusted). Low-confidence/ambiguous verdicts are routed to Stage 3b. |
| 3b | **EPA - Stage 3b: Textbook Retrieval Adjudication** | Only invoked for weak verdicts. Re-adjudicates using the candidate topic's sibling topics as evidence, constrained to never pick a topic outside that evidence set. |
| 4 | **EPA - Stage 4: Deduplication** | Corpus-wide (all questions in the job). Deterministic exact-duplicate detection, deterministic near-duplicate candidate generation, an LLM semantic judge that distinguishes "same question" from "same topic", a confidence floor before trusting a merge, and deterministic union-find grouping into stable canonical questions. |
| 5+6 | **EPA - Stage 5+6: Frequency Analysis and Report** | Pure aggregation, zero LLM calls. Produces the executive summary, most-repeated-questions ranking, chapter-level breakdowns (tested/untested), question-type distribution, no-match/ambiguous lists, and data-integrity checks — as both structured JSON and a Markdown report. |
| — | **EPA - Main: Analyze Exam Papers** | The single entry point ("Analyze Exam Papers"). A form collects the textbook/index + exam papers, creates or resumes a job by Job ID, and sequences the stages above, pausing safely on quota exhaustion at any point. |

Workflows communicate via **Execute Sub-workflow** calls, passing `job_id`
(and stage-specific ids) as workflow inputs; binary files pass through
automatically. Nothing is inlined — each stage is independently viewable,
testable, and re-runnable in the n8n editor.

## Persistent storage

Six n8n Data Tables (self-hosted, no external DB needed), mirroring the
Python reference's SQLite schema:

- **epa_jobs** — one row per analysis job: `status` (`running` / `paused` /
  `completed`), `pause_reason`, `current_stage`, `total_papers`, timestamps.
- **epa_topics** — the topic hierarchy for a job (`topic_id`, `name`,
  `level`, `parent_id`).
- **epa_papers** — one row per detected exam paper: `status`
  (`success`/`failed`), `classification_done`, metadata, page range.
- **epa_questions** — one row per extracted question occurrence, plus its
  classification (`topic_id`, `classification_status`, confidence) and its
  `canonical_question_id` once deduplicated.
- **epa_canonical_questions** — one row per canonical (deduplicated)
  question group: representative text, `dedup_status`, `occurrence_count`,
  `source_question_ids_json`.
- **epa_llm_cache** — every OpenRouter response, keyed by a deterministic
  `cache_key` (e.g. `classify:{job_id}:{paper_id}`), so re-running a job
  never re-spends quota on work already done.

**Verified 2026-08-13**: after the n8n restart, Data Table reads are
reliable — see "Status as of 2026-08-13" below for the direct test that
confirms this.

## Required credentials

One credential is needed by the `Call OpenRouter` node in **EPA - LLM
Structured Call**: a **Header Auth** credential (n8n type `httpHeaderAuth`)
with `Name: Authorization`, `Value: Bearer sk-or-v1-...` (your OpenRouter
key), attached to that node (`authentication: genericCredentialType`,
`genericAuthType: httpHeaderAuth`).

**Confirmed working with a real live OpenRouter call on 2026-08-13.** The
user created a **Header Auth** credential ("Header Auth account", n8n type
`httpHeaderAuth`) and attached it to the `Call OpenRouter` node. A live
smoke test (see "Live validation pass" below) sent real HTTP requests to
`openrouter.ai` and received real authenticated responses — confirmed by
inspecting the raw HTTP response headers/cookies/`cf-ray` id, and later by a
fully successful structured JSON completion. No further setup is needed;
this is the working, current state.

The key is never written into any workflow, file, or log — it only ever
flows through this credential.

## How to run

1. Open n8n and find **EPA - Main: Analyze Exam Papers**. Copy its Form
   Trigger URL (Test or Production).
2. Fill in the form:
   - **Job ID** — leave blank for a new analysis, or paste a previous Job
     ID to resume a paused run.
   - **Textbook or Index Source** — "Textbook PDF" or "Index or Syllabus
     File".
   - **Textbook PDF, or Index/Syllabus File** — the single file for
     whichever mode you chose (pdf / txt / md / json).
   - **Exam Paper PDFs** — one or more PDFs; each may itself contain
     several distinct exam papers (they're detected automatically).
3. Submit. The page shows the final JSON response when the run finishes,
   pauses, or (if already completed) returns the existing report.

Do **not** run individual stage workflows by hand for a real analysis —
always go through the Main orchestrator, which handles job creation,
resumability, and quota pausing correctly. The stage workflows are exposed
separately for **testing** and **transparency** (so each part of the
pipeline can be inspected/re-run on its own), matching the "clear, testable
sub-workflows" goal of this demo.

## Input format

- Textbook: PDF (any size; a bounded excerpt around the "Contents"/"Index"
  heading is sent to the LLM, not the whole book — see Limitations).
- Index/syllabus: PDF, or a plain-text/Markdown outline (heading depth or
  indentation = hierarchy level), or JSON (nested `{name, children}` tree,
  or a flat list of `{name, level, parent_id}`).
- Exam papers: one or more PDFs, each possibly containing multiple distinct
  exam papers concatenated together.

## Output format

The orchestrator's final response (and the Stage 5+6 workflow's own output)
is JSON with:
- `summary` — papers/questions/canonical-question counts, tested/untested
  topic counts, classification-status counts, data-integrity flag.
- `most_repeated_questions` — ranked, with occurrence count, years, source
  filenames, question types, and chapter.
- `most_tested_topics_by_occurrences` / `..._by_unique_questions` — two
  separate rankings (they can disagree, same as the Python reference).
- `topic_analysis` / `untested_topics` — full chapter tree with per-chapter
  occurrence/unique-question counts, and topics whose entire subtree had
  zero questions.
- `no_match_questions` / `ambiguous_questions` — full text, not just ids.
- `data_quality_notes`, `data_integrity`.
- `report_markdown` — the same data rendered as a readable report (executive
  summary → most repeated questions → most tested chapters → question-type
  distribution → chapter-by-chapter breakdown → untested chapters →
  no-match/ambiguous sections → data-quality notes).

## Persistence and resume

Every stage workflow checks its own completion state before doing
LLM-costing work:

- Stage 1 is skipped entirely if `epa_topics` already has rows for the job.
- Stage 2 checks, per detected paper span, whether `epa_papers` already has
  a `success` row for that exact file + page range.
- Stage 3 and Stage 4's actual LLM calls are cached in `epa_llm_cache` by a
  deterministic key, so even a full re-run of an already-classified paper
  is a cache hit, not a new API call.

The Main orchestrator loops through **every** uploaded file and **every**
paper needing classification on **every** run — this looks wasteful but is
cheap and correct: each stage's own cache/status checks make already-done
work a near-instant no-op, so re-running after a pause simply "fast
forwards" through completed work rather than needing separate bespoke
resume logic in the orchestrator itself.

To resume a paused job: resubmit the form with the same **Job ID**
(returned in the paused response, and visible on the `epa_jobs` row).
Papers already marked `success`, and papers already classified
(`classification_done = true`), are skipped; processing continues from the
first incomplete item.

## How OpenRouter rate limits are handled

The shared **LLM Structured Call** workflow:
1. Checks `epa_llm_cache` for this exact call — instant return if hit.
2. Otherwise rotates through a small pool of free OpenRouter models. The
   original pool (picked without live verification) turned out to be
   **entirely stale** — a live test on 2026-08-13 got HTTP 404
   ("unavailable"/"no endpoints") from all 5 models. Replaced with 5 models
   confirmed live via `GET https://openrouter.ai/api/v1/models` (filtered to
   `id` ending `:free`) on that date: `openai/gpt-oss-20b:free`,
   `nvidia/nemotron-3-super-120b-a12b:free`, `google/gemma-4-31b-it:free`,
   `nvidia/nemotron-3-nano-30b-a3b:free`, `liquid/lfm-2.5-2.6b:free`.
   OpenRouter's free catalog turns over — re-check periodically by hitting
   that endpoint directly and editing the "Build Model Pool" Code node; a
   stale slug just fails fast with 404 and the pool moves on to the next
   model, so a partially-stale pool degrades gracefully rather than breaking.
3. Classifies every response:
   - **HTTP 402**, or **429** whose error message mentions daily/credit
     exhaustion (`"per-day"`, `"add credits"`, etc.) → `quota` — retrying
     is pointless, so the workflow stops immediately and bubbles up
     `quota_exhausted: true` to its caller.
   - Any other failure (plain rate limit, malformed JSON, missing required
     keys) → `retry` — try the next model in the pool, after a brief 2s
     wait, without ever re-hitting the same exhausted quota repeatedly.
   - All models in the pool failing → `success: false`, `quota_exhausted:
     false` (a genuine content/model problem, not a quota problem).
4. On success, the raw response is written to `epa_llm_cache` before
   returning, so the exact same call is never repeated.

Every stage workflow checks `quota_exhausted` on its own LLM call(s); if
true, the Main orchestrator marks the job `paused` (with a stage-specific
`pause_reason`) and ends the run — nothing already persisted is touched.
This is what makes the required scenario work: **20 papers → quota
exhausted after paper 12 → papers 1–12 stay saved → job pauses → user
resumes later with the same Job ID → papers 1–12 are skipped (cache hits) →
processing continues from paper 13.** This exact scenario (scaled down) was
directly tested end-to-end on 2026-08-13 — see below.

## How textbook retrieval works (Stage 3b)

The Python reference retrieves textbook headings/passages using
Docling-derived page provenance. This n8n version doesn't retain
page-level provenance for LLM-extracted topics (there's no local
equivalent of Docling's block-typed PDF parsing available to n8n), so
Stage 3b instead grounds re-adjudication in the **topic hierarchy itself**:
the low-confidence candidate topic plus its sibling topics under the same
parent, with the LLM constrained to pick only from that set (or say
`no_match`/`ambiguous`) — never inventing a topic id. This preserves the
two-stage "classify, then give the LLM more context to disambiguate
confusable nearby topics" architecture, just with a different evidence
source. See [Known limitations](#known-limitations) below.

## Known limitations / deviations from the Python reference

- **Topic Authority (Stage 1) PDF path** is a bounded excerpt around the
  first "Contents"/"Index" heading match, sent to one LLM call — not
  Docling's adaptive, block-typed windowed TOC search with name-recovery.
  For a textbook whose TOC isn't found in that excerpt (or spans more than
  ~9000 characters), Stage 1 will under-extract. Index/syllabus mode
  (txt/md/json) is unaffected — that path is fully deterministic in both
  implementations. The real `Satyanarayan Biochemistry.pdf` textbook does
  have a genuine PDF text layer (confirmed 2026-08-13), so this path should
  work against it once the credential is configured.
- **Question Extraction (Stage 2) has no in-workflow OCR.** It does plain
  PDF text-layer extraction (`extractFromFile`), unlike the Python
  reference's Docling OCR pipeline. The real
  `question_papers/BIOCHEM 3RD SESSIONALS.pdf` fixture is scanned/image-based
  with no embedded text layer. **Investigated 2026-08-13 whether n8n itself
  could run OCR natively** (Tesseract + `ocrmypdf` are both installed and
  working on this machine): it cannot, in this n8n installation specifically
  — there is no `Execute Command` node type installed (`n8n-nodes-base.executeCommand`
  is listed by the node catalog but rejected as "Unrecognized node type" when
  actually added), and the Code node's sandbox explicitly disallows
  `require('child_process')` (confirmed directly: `Module 'child_process' is
  disallowed`). No OCR-capable node or npm package is available either. The
  practical fix, given those two constraints ruled out, is that OCR runs as a
  **local pre-processing step outside n8n**, using the same already-installed
  `ocrmypdf`/Tesseract, before the file is uploaded to the form:
  `ocrmypdf --force-ocr --pdf-renderer sandwich --optimize 0 --output-type pdf
  in.pdf out.pdf`. This is not a workaround inside the workflow — Stage 2's
  extraction node itself is completely unmodified; it just receives a PDF
  that already has a text layer, exactly as it would for a naturally
  text-based exam paper. **Verified 2026-08-13**: OCR'd the real fixture this
  way in ~35s, producing a PDF pypdf could pull ~26,000 characters of
  accurate exam text from. To make this fully self-serve in n8n (not
  required for the demo to work, but the concrete next step if wanted): add
  the `Execute Command` node type to this n8n instance, or set
  `NODE_FUNCTION_ALLOW_BUILTIN=child_process` and restart n8n so a Code node
  can shell out — either lets Stage 2 call `ocrmypdf` itself instead of it
  being a manual pre-step.
- **Retrieval-assisted classification (Stage 3b)** uses topic-hierarchy
  context (siblings) instead of real textbook passages — see above.
- **Deduplication candidate generation (Stage 4)** uses deterministic
  token-Jaccard similarity (same question_type only, top 120 pairs) instead
  of the Python reference's local embeddings (fastembed/BGE), since running
  a local embedding model isn't practical inside n8n and this demo
  deliberately avoids adding a paid external embeddings API. The rest of
  the architecture — deterministic exact-dup detection, LLM semantic judge,
  confidence-gated union-find grouping — is unchanged.
- **File hashing** uses a fast non-cryptographic hash of the filename
  (mirroring the Python reference's role for cache-keying, not for
  security), not a true SHA-256 content hash.
- **Resumability is Job-ID-based**, not auto-derived from input file
  identity. You must save/paste the Job ID yourself to resume — there's no
  equivalent of the Python app silently recognizing "the same inputs as
  before" from re-uploaded files.
- **Binary field naming for the multi-file exam-paper upload** was built
  against my best understanding of n8n Form Trigger conventions
  (`exam_papers_0`, `exam_papers_1`, ...); pause/resume testing on
  2026-08-13 used synthetic binaries with Stage 2 mocked at the boundary, so
  this exact naming convention has still not been independently confirmed
  against a real multi-file form submission — verify this first if papers
  aren't being picked up.
- **A minor per-file statistics quirk** was observed during testing: when a
  paper fails (e.g. quota-exhausted) and is later retried successfully,
  `epa_papers` ends up with two rows for that paper_id (one `failed`, one
  `success`) since old failed-attempt rows aren't cleaned up on retry. This
  does not affect final report correctness (Stage 3/5+6 correctly filter to
  `status=success` rows), but the per-file `papers_failed` count returned
  by Stage 2 immediately after a successful retry can over-count historical
  failed attempts. Not fixed in this pass — noted as a known, non-corrupting
  discrepancy.
- **Credential wiring required one manual fix** — see "Required
  credentials" above.

## Status as of 2026-08-13 (post-restart verification pass)

The n8n restart requested in the previous session has been applied, and a
full autonomous verification/debugging pass was completed against the live
n8n instance via the n8n MCP tools. Summary: **the Data Table stale-read
issue is gone**, but that pass surfaced **two real, previously-undetected
bugs that were pipeline-breaking** (both fixed and retested) and **two
genuine blockers** that stop a real end-to-end run today (credential
wiring, and OCR — both described above). Nothing below required rebuilding
any workflow from scratch — every fix is a targeted node-level correction.

### Data Tables: confirmed fixed by the restart

Direct test: ran **EPA - LLM Structured Call** twice with an identical
`cache_key` via `test_workflow`. First run: real cache miss, full
model-pool path, row written. Second run: `Check LLM Cache` correctly
returned the row (previously it returned empty even when the row existed)
in ~14ms, no model-pool retry. **Data Table reads are reliable now.**

### Bug found #1 — `Cache Hit?` switch never matched a real cache hit

While confirming the above, the *read* worked but `Cache Hit?` (a Switch
node using `operator: isNotEmpty` with `typeValidation: loose`) still
routed every call to the "miss" branch, re-spending a model call on every
request even with a cached row present. Isolated in a disposable scratch
workflow: `isEmpty`/`isNotEmpty` string operators combined with
`typeValidation: loose` never match in this n8n version, regardless of the
actual value — a real operator-level bug, not a data problem. **Fixed** by
replacing with an unambiguous `notEquals ''` check (`typeValidation:
strict`). Verified: a repeat call now returns the cached value in ~50ms and
never re-invokes OpenRouter.

The same `isEmpty`/`isNotEmpty` pattern was also found and fixed in three
gates inside **EPA - Main**: `Job Status?` (new-vs-resume detection — every
*new* job would have been misrouted into the resume path, which updates a
non-existent `epa_jobs` row instead of inserting one), `Topics Already
Exist?` (Stage 1 skip-if-done — would never detect existing topics, causing
duplicate rows and wasted LLM calls on every resume), and `Has Papers To
Classify?` (would never detect papers needing classification). All three
replaced with string-length comparisons and verified via direct execution
(a synthetic new-job run correctly inserted one `epa_jobs` row and
correctly skipped Stage 1 when a topic already existed, with zero errors).

### Bug found #2 — every "Quota Exhausted?" gate crashed on first real use

More serious: once bug #1 was fixed and a real quota-boolean reached these
gates for the first time (previously untested, since no run had gotten
this far), **every single "Quota Exhausted?" IF node in the entire
pipeline threw a hard `NodeOperationError`** — `Wrong type: '' is a string
but was expecting a boolean` — even when the field was a genuine boolean.
Root cause, confirmed in an isolated scratch workflow: `{type: 'boolean',
operation: 'true'}` combined with `typeValidation: 'strict'` fails on the
unused `rightValue: ''` placeholder under strict coercion. This pattern was
used identically in **9 gates across 4 workflows**: `Boundary Call Quota
Exhausted?` and `Extraction Call Quota Exhausted?` (Stage 2),
`Classification Call Quota Exhausted?` and `Needs Retrieval Adjudication?`
(Stage 3), `Judge Call Quota Exhausted?` (Stage 4), and all four `Stage N
Quota Exhausted?` gates in Main. Every one of them would have crashed the
pipeline (not paused it cleanly) the moment any stage actually reported a
quota result. **Fixed** by switching all 9 to `typeValidation: 'loose'`,
verified correct in isolation (true routes true, false routes false, no
error) and then end-to-end in the pause/resume tests below.

### Pause/resume — fully verified (mocked at the LLM boundary, real everything else)

Per the user's own suggested approach ("temporarily simulate the quota
response at the appropriate n8n boundary"), pause/resume was tested with
**real Data Table reads/writes and real Stage 2 logic** end-to-end, mocking
only the LLM call nodes directly via `test_workflow` pin data (confirmed
pinning works cleanly on Execute-Workflow sub-calls and Code nodes, not
just HTTP nodes) — no real OpenRouter quota was spent.

**At Stage 2 (per-paper) granularity**, five sequential real executions
against a live test job:
1. Paper A: mocked extraction succeeds → persisted `status=success` +
   1 question row. Confirmed.
2. Paper B: same, different file → persisted independently. Confirmed.
3. Paper C: mocked extraction reports `quota_exhausted: true` → persisted
   as `status=failed` with the reason noted; `Finalize - Quota Aborted
   Mid-File` correctly reports `quota_exhausted: true`,
   `papers_processed: 0`, `papers_failed: 1`. Papers A and B untouched.
   Confirmed.
4. **Resume** Paper C (same file/pages): `Check Existing Paper` found the
   old `failed` row; `Paper Already Done?` correctly evaluated **false**
   (only `success` counts as done) → retried extraction → this time
   succeeded → new `success` row inserted. Confirmed.
5. **Re-submit** Paper A (already succeeded): `Paper Already Done?`
   evaluated **true** → looped straight back without ever calling `Build
   Paper Text` or `Call LLM - Extract Paper Structure` (absent from the
   execution entirely) → zero LLM calls, no duplicate row. Confirmed.

**At the Main-orchestrator (per-job) level**, two sequential real
executions against a live test job (Stage 1 skipped via a pre-seeded
topic; Stage 2/4 LLM calls mocked at the Execute-Workflow boundary):
1. Submit: `Job Status?` correctly detects a new job → `Insert New Job
   Row`. Stage 2 reports `quota_exhausted: true` → `Stage 2 Quota
   Exhausted?` (now fixed) correctly routes true → job marked
   `status: paused`, `pause_reason` and `current_stage` set correctly →
   response `status: "paused"`. Confirmed.
2. **Resume** with the same Job ID: `Job Status?` correctly detects
   `status: "paused"` → `Update Job - Resuming` resets it to `running`
   **without inserting a duplicate `epa_jobs` row** (same row `id`,
   `createdAt` unchanged) → processing continues from Stage 2 → Stage 4 →
   Stage 5+6 → job marked `status: "completed"` with a correctly-generated
   report. Confirmed.

This directly demonstrates the required scenario (scaled down for a fast,
credential-free test): papers before the interruption stay persisted, the
job pauses safely with no data loss, resuming with the same Job ID does not
duplicate work or records, and the interrupted item is retried (not
skipped) on resume while already-completed items are skipped (not
re-processed).

All diagnostic job/topic/paper/question/cache rows created during this
testing pass were deleted afterward via disposable scratch workflows
(archived, not left in the workspace); the real `epa_*` tables are back to
their pre-testing (empty) state.

## Status as of 2026-08-13 (live validation pass, session 2)

A second autonomous pass took the credential/OCR blockers described above
off the table and ran the pipeline for real: real OpenRouter calls, the real
Biochemistry textbook, and the real (OCR'd) `BIOCHEM 3RD SESSIONALS.pdf`
fixture. This surfaced and fixed several genuine bugs that no amount of
mocked-boundary testing could have caught, because they only manifest with
real file uploads and real binary data.

### Bug found #3 — real Form Trigger file uploads never reached `json`, only `binary`

`Normalize Input` and `Prep Exam Papers List` (both in **EPA - Main**) read
file metadata (filename, count) from `$json.textbook_file` /
`$json.exam_papers`, matching the shape the MCP `execute_workflow` tool's
simulated form-data uses. A **real** multipart submission to the form's
production URL behaves differently: file fields land **only** in `$binary`
(`binary.textbook_file`, `binary.exam_papers` for a single file, or
`binary.exam_papers_0`, `binary.exam_papers_1`, ... for multiple) — the
matching `$json` fields are always `null`. Consequence: `exam_paper_count`
was always computed as `0` and `Prep Exam Papers List` always returned an
empty array, so **every real form submission silently processed zero exam
papers**, no matter how many were uploaded. This was invisible to all prior
testing because that testing mocked the LLM/Execute-Workflow boundary using
synthetic binaries injected below this exact bug. **Fixed** by rewriting
both nodes to derive filename/extension/count from `$binary` instead of
`$json` (confirmed the `exam_papers_N` vs plain `exam_papers` naming
convention empirically with 1-file and 2-file real submissions — this also
resolves the "not independently confirmed" flag from the previous pass).

### Bug found #4 — `extractFromFile(pdf)` with `options.maxPages: 0` silently extracts zero pages

The headline finding of this pass. Both PDF-extraction nodes
(`Extract Textbook PDF Text` in Stage 1, `Extract Exam PDF Pages` in Stage 2)
were configured with `maxPages: 0`, which the field's own default/docs imply
means "no limit." **It does not** — empirically, `maxPages: 0` extracts
**zero pages of text**, while `numpages`/`numrender` metadata still report
the correct page count, making a real PDF look exactly like a genuinely
empty/scanned one. This was **not** an OCR problem: it silently broke
extraction for every PDF regardless of whether it had a real text layer,
including the textbook (confirmed independently via three different control
PDFs — a hand-built single-`Tj` file with a bare base-14 font, a `pypdf`-
rewritten textbook excerpt, and a `pikepdf`-preserved textbook excerpt — all
three returned `text: []`/`text: ""` with `maxPages: 0` and **all three**
extracted correctly once `maxPages` was set to an explicit large number).
**Fixed** in both nodes (`maxPages: 5000` for the textbook, `2000` for exam
papers). This means the "no OCR" limitation described above was real but
incomplete — even a perfectly OCR'd or naturally text-based PDF would have
failed to extract until this fix, in every session before this one.

### Real live-LLM results

With both bugs fixed and the model pool refreshed:

- **Stage 1 (Topic Authority)**: fully succeeded against the real textbook.
  Sent a real ~41,500-character TOC excerpt (containing the actual "Contents"
  page) to `nvidia/nemotron-3-super-120b-a12b:free`, which returned a real,
  accurate 50-topic hierarchy (7 sections + 43 chapters) matching the
  textbook's real table of contents. The 233MB original textbook exceeds
  n8n's form-upload size limit (see below), so this used a 20-page
  front-matter excerpt (pages 1-20, containing the real TOC on page 10) —
  legitimate given Stage 1 only ever reads a bounded excerpt around the TOC
  regardless of total book size.
- **Stage 2 boundary detection**: fully succeeded against the real OCR'd
  15-page exam-paper fixture — correctly identified a real paper span
  (pages 1-4) from real extracted OCR text.
- **Stage 2 per-paper extraction**: succeeded for multiple real papers in
  sequence. Real OCR'd question text (verbatim MCQs, long-essay questions,
  etc. from the actual exam paper) reached the extraction LLM's prompt and
  came back as accurate, correctly structured JSON — e.g. paper 1 (pages
  1-4) returned 10 real MCQs plus a long-essay question, each with the
  right `question_type`, verbatim `question_text`/`options`, correct
  `source_pages`, and `extraction_confidence: 0.99`, via
  `openai/gpt-oss-20b:free`. Each real call took several minutes (observed
  ~4-6 min per paper, consistent with free-tier model latency, not a bug —
  see the timeout math above for why this is still within the retry logic's
  own worst-case bound once you account for a model actually using its full
  90s allowance rather than failing fast), so processing the fixture's ~9
  detected papers end-to-end through every remaining stage (3, 3b, 4, 5+6)
  takes a genuinely long time on free models — plan for the better part of
  an hour for a fixture this size, not minutes. This is real, unmocked,
  end-to-end LLM pipeline execution against the real fixture.

### Two operational gotchas found while driving n8n from the command line

Neither is a workflow bug, but both cost significant time to diagnose and
are worth recording:

- **n8n's Form Trigger has a real upload-size ceiling around 200-210MB**
  (`formidable`-based parser; HTTP 413 `"The submitted form data exceeds the
  allowed size"`). The real textbook is 233MB, so it cannot be uploaded
  as-is — use a smaller excerpt (see Stage 1 above) or raise
  `N8N_FORMDATA_FILE_SIZE_MAX`/restart n8n if a full-book upload is ever
  needed.
- **curl on this machine is the native Windows build**, so mingw-style paths
  (`/c/Users/...`, as auto-translated by Git Bash for most arguments) are
  **not** understood inside a compound `-F "field=@/c/Users/...;type=..."`
  value — curl reports `(26) Failed to open/read local data from file`, which
  looks identical to a genuinely missing/unreadable file. Fix: pass
  Windows-style paths (`C:\Users\...`) to `-F` on this machine.

## Testing performed

- **Structural validation**: every workflow re-validated after edits (all
  passed).
- **LLM retry/quota logic**: previously tested with mocked HTTP responses
  (timing-based signal — see below); this pass additionally found and
  fixed the two bugs above, which were not exercised by the earlier
  mocked-timing tests since those never reached a real cache-hit check or
  a real boolean quota gate.
  - Valid JSON success → resolved in ~350ms (no retries). Confirmed.
  - HTTP 402 → resolved in ~80ms (immediate quota stop, no retries). Confirmed.
  - HTTP 429 with daily/credit-exhaustion wording → ~75ms (correctly
    classified as quota, not rate-limit). Confirmed.
  - HTTP 429 plain rate-limit wording → ~10.2s (correctly retried through
    the full model pool instead of stopping). Confirmed.
  - Malformed/non-JSON response → ~10.2s (retried through the full pool,
    then cleanly reported as failed — not silently fabricated). Confirmed.
  - Real cache hit → ~50ms, no model call, correct cached value returned.
    Confirmed.
- **Data persistence and resumability**: fully verified against real Data
  Tables — see "Pause/resume — fully verified" above.
- **Python reference suite**: `pytest -q` → **440 passed**, 0 failed (only
  pre-existing deprecation warnings from `docling`/`rapid_ocr`, unrelated
  to this work). Confirms the Python reference is untouched and healthy.
- **Real live OpenRouter calls** (session 2, 2026-08-13): a minimal
  synthetic-fixture smoke test and the real Biochemistry/BIOCHEM fixture
  both produced genuine authenticated HTTP requests to `openrouter.ai` with
  real responses (401→ruled out, 404 stale-model responses observed and
  correctly classified as retryable, then a real 200 with valid structured
  JSON parsed and cached). This is real network I/O, not a mock.
- **Real end-to-end run against the real fixture** (session 2): Stage 1
  (real 50-topic hierarchy from the real textbook), Stage 2 boundary
  detection (real paper span from real OCR'd text), and Stage 2 per-paper
  extraction (real, accurate question JSON for multiple papers in a row —
  see "Real live-LLM results" above) all succeeded end-to-end against real
  data with real OpenRouter calls. The full run (9 detected papers through
  Stage 3/3b/4/5+6) takes on the order of an hour on free-tier models given
  the ~4-6 minutes/call observed, so this document was finalized before the
  run reached its own terminal `completed` state — the mechanics of every
  stage exercised so far are proven with real data, not mocked, but a
  from-scratch fresh run should be expected to take a while and not judged
  against Python's much faster local pipeline.
- **Real multi-file form-upload binary handling** (session 2): the
  `exam_papers`/`exam_papers_N` binary-field naming convention flagged as
  unverified in the previous pass is now confirmed directly against real
  1-file and 2-file multipart submissions (see Bug #3 above).

## How to reproduce the demo

1. In n8n, open **Credentials** and confirm a **Header Auth** credential
   (`Name: Authorization`, `Value: Bearer <your OpenRouter key>`) is attached
   to the `Call OpenRouter` node in **EPA - LLM Structured Call** — see
   "Required credentials" above. This is already done and confirmed working
   in this instance.
2. If the exam papers you're using are scanned/image-based (no PDF text
   layer), OCR them first — this n8n installation cannot shell out to a
   local OCR tool itself (see "Known limitations" → OCR above), so this is a
   one-time manual step per file:
   `ocrmypdf --force-ocr --pdf-renderer sandwich --optimize 0 --output-type pdf in.pdf out.pdf`.
   Naturally text-based PDFs need no such step.
3. Open **EPA - Main: Analyze Exam Papers**, get its form URL, and submit it
   with a textbook (or a bounded excerpt containing its table of contents —
   n8n's form upload has a real ~200-210MB ceiling, see above) and your exam
   paper PDF(s). Every stage (topic extraction, boundary detection, per-paper
   question extraction) is confirmed working end-to-end against real data as
   of 2026-08-13. Be patient: each real LLM call on the free-tier model pool
   takes several minutes, so a fixture with several detected papers can take
   the better part of an hour to fully complete — this is expected, not a
   hang (see "Real live-LLM results" above).
4. Compare whatever the final report produces against the validated Python
   baseline (`output/final_analysis/analysis.json`): 9 papers, 215 question
   occurrences, 191 canonical questions, 64 topics (28 tested / 36
   untested), 16 repeated canonical-question groups, 206 classified / 7
   no_match / 2 ambiguous / 0 unclassified — semantic comparison only, not
   exact equality (see Limitations). Expect the n8n version's topic count to
   be lower (a 2-level section/chapter hierarchy from a bounded TOC excerpt,
   not Docling's deeper full-book parse).
5. To test pause/resume against a real run: temporarily point the
   OpenRouter credential at an invalid key (or exhaust the free daily
   quota) — the job should pause cleanly with a Job ID you can resume once
   the credential is restored. (Verified via mocked-LLM-boundary testing in
   the first pass; not re-verified live in the second pass — no real quota
   exhaustion occurred during session 2's testing.)
