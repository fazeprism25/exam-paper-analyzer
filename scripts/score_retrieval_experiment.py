#!/usr/bin/env python
"""Combine baseline_scored.json + retrieval_adjudication_results.json into
the final experiment comparison report (see scripts/retrieval_experiment.py
for phases 1/2). Pure local scoring, no LLM/API calls.

For each adjudicated pair, re-derives an exact/defensible/wrong verdict for
the ADJUDICATED topic the same way baseline was scored (reusing
score_baseline.py's per-gt_index judgment table, since the ground-truth
chapter and defensibility reasoning don't change -- only which topic ended
up assigned does).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BASELINE_SCORED_PATH = REPO / "output" / "baseline_scored.json"
ADJUDICATION_PATH = REPO / "output" / "retrieval_adjudication_results.json"
FINAL_REPORT_PATH = REPO / "output" / "retrieval_experiment_report.json"

# topic_id -> set of topic_ids that would count as "exact" or "defensible"
# for that gt_index, derived from score_baseline.py's GENUINELY_WRONG/
# DEFENSIBLE tables + the ground-truth chapter's own canonical topic_id.
# Rebuilt here explicitly (not imported) because the adjudicated topic_id
# for a given gt_index is restricted by construction to exactly the two
# candidates offered (current baseline pick, rival pick) -- see
# retrieval_experiment.ERROR_PAIRS/CONTROL_PAIRS -- so we only need to know,
# for each gt_index, which of {current, rival} is the ground-truth-correct
# ("exact") one. Every ERROR_PAIRS rival IS the ground-truth-correct topic
# (that's how the pairs were constructed); every CONTROL_PAIRS current pick
# already was correct.
CORRECT_TOPIC_BY_INDEX = {
    # errors: rival is correct
    14: "section_two_chapter_2_3", 19: "section_three_chapter_3_3", 20: "section_three_chapter_3_3",
    22: "section_three_chapter_3_3", 26: "section_three_chapter_3_3", 27: "section_three_chapter_3_4",
    36: "section_three_chapter_3_4", 37: "section_three_chapter_3_7", 42: "section_four_chapter_4_2",
    # controls: current is correct
    17: "section_three_chapter_3_3", 25: "section_three_chapter_3_3", 43: "section_four_chapter_4_2",
    6: "section_one_chapter_1_6", 39: "section_three_chapter_3_7", 29: "section_three_chapter_3_4",
}


def main() -> None:
    baseline = json.loads(BASELINE_SCORED_PATH.read_text(encoding="utf-8"))
    baseline_by_index = {r["gt_index"]: r for r in baseline}
    adjudications = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))

    rows = []
    for adj in adjudications:
        idx = adj["gt_index"]
        base = baseline_by_index[idx]
        correct_topic_id = CORRECT_TOPIC_BY_INDEX[idx]

        if adj.get("call_failed"):
            rows.append({
                "gt_index": idx, "kind": adj["kind"], "gt_chapter": base["gt_chapter"],
                "question_text": base["gt_question_text"],
                "baseline_verdict": base["baseline_verdict"], "baseline_topic": base["topic_name"],
                "adjudicated_status": "CALL_FAILED", "adjudicated_topic": None,
                "outcome": "call_failed",
            })
            continue

        adj_status = adj["adjudicated_status"]
        adj_topic_id = adj["adjudicated_topic_id"]
        adj_correct = adj_status == "classified" and adj_topic_id == correct_topic_id

        was_wrong = base["baseline_verdict"] == "wrong"
        if was_wrong:
            if adj_correct:
                outcome = "corrected"
            elif adj_status == "ambiguous":
                outcome = "became_ambiguous"
            else:
                outcome = "still_wrong"
        else:
            # control: baseline was exact/defensible (correct)
            if adj_correct:
                outcome = "unchanged_correct"
            elif adj_status == "ambiguous":
                outcome = "regressed_to_ambiguous"
            else:
                outcome = "regressed_wrong"

        rows.append({
            "gt_index": idx, "kind": adj["kind"], "gt_chapter": base["gt_chapter"],
            "question_text": base["gt_question_text"],
            "baseline_verdict": base["baseline_verdict"], "baseline_topic": base["topic_name"],
            "rival_topic_id": adj["rival_topic_id"], "correct_topic_id": correct_topic_id,
            "adjudicated_status": adj_status, "adjudicated_topic": adj_topic_id,
            "adjudicated_confidence": adj["adjudicated_confidence"], "adjudicated_reason": adj["adjudicated_reason"],
            "outcome": outcome,
        })

    errors = [r for r in rows if r["kind"] == "error"]
    controls = [r for r in rows if r["kind"] == "control"]

    error_outcomes = {"corrected": 0, "still_wrong": 0, "became_ambiguous": 0, "call_failed": 0}
    for r in errors:
        error_outcomes[r["outcome"]] = error_outcomes.get(r["outcome"], 0) + 1

    control_outcomes = {"unchanged_correct": 0, "regressed_wrong": 0, "regressed_to_ambiguous": 0, "call_failed": 0}
    for r in controls:
        control_outcomes[r["outcome"]] = control_outcomes.get(r["outcome"], 0) + 1

    report = {
        "n_errors_tested": len(errors),
        "n_controls_tested": len(controls),
        "error_outcomes": error_outcomes,
        "control_outcomes": control_outcomes,
        "rows": rows,
    }
    FINAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== {len(errors)} known errors (retrieval-assisted) ===")
    for k, v in error_outcomes.items():
        print(f"  {k}: {v}")
    print(f"\n=== {len(controls)} controls (already-correct baseline) ===")
    for k, v in control_outcomes.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {FINAL_REPORT_PATH}")

    print("\n--- per-question detail ---")
    for r in rows:
        print(f"[{r['kind']:7}] gt{r['gt_index']:>3}  baseline={r['baseline_verdict']:11} -> "
              f"outcome={r['outcome']:20}  ({r['gt_chapter']})")


if __name__ == "__main__":
    main()
