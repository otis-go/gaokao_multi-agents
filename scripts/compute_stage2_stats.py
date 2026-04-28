#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Stage2 evaluation statistics after an interrupted run.

Reads all stage2/unit_*/evaluation_state.json files under an experiment
directory, extracts ai_eval and pedagogical_eval overall_score values, and
computes aggregate statistics.

Usage:
    python scripts/compute_stage2_stats.py <experiment_dir>

Example:
    python scripts/compute_stage2_stats.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def extract_scores_from_evaluation_state(eval_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key scoring fields from evaluation_state.json.

    Returns:
    {
        "unit_id": str,
        "current_stage": str,
        "final_decision": str,
        "ai_overall_score": float or None,
        "ped_overall_score": float or None,
        "question_type": str or None,
        "material_type": str or None,
    }
    """
    result = {
        "unit_id": eval_state.get("unit_id"),
        "current_stage": eval_state.get("current_stage"),
        "final_decision": eval_state.get("final_decision"),
        "ai_overall_score": None,
        "ped_overall_score": None,
        "question_type": None,
        "material_type": None,
    }

    # Extract AI evaluation score.
    ai_eval = eval_state.get("ai_eval", {})
    if ai_eval and ai_eval.get("success"):
        ai_result = ai_eval.get("result", {})
        if ai_result:
            result["ai_overall_score"] = ai_result.get("overall_score")

    # Backward-compatible legacy ai_eval_result structure.
    if result["ai_overall_score"] is None:
        ai_eval_result = eval_state.get("ai_eval_result", {})
        if isinstance(ai_eval_result, dict):
            result["ai_overall_score"] = ai_eval_result.get("overall_score")

    # Extract pedagogical evaluation score.
    ped_eval = eval_state.get("pedagogical_eval", {})
    if ped_eval and ped_eval.get("success"):
        ped_result = ped_eval.get("result", {})
        if ped_result:
            result["ped_overall_score"] = ped_result.get("overall_score")

    # Backward-compatible legacy pedagogical_eval_result structure.
    if result["ped_overall_score"] is None:
        ped_eval_result = eval_state.get("pedagogical_eval_result", {})
        if isinstance(ped_eval_result, dict):
            result["ped_overall_score"] = ped_eval_result.get("overall_score")
        elif hasattr(ped_eval_result, "overall_score"):
            result["ped_overall_score"] = ped_eval_result.overall_score

    # Extract question metadata.
    input_data = eval_state.get("input", {})
    if input_data:
        result["question_type"] = input_data.get("question_type")
        result["material_type"] = input_data.get("material_type")

    return result


def compute_stage2_stats(exp_dir: Path) -> Dict[str, Any]:
    """
    Compute statistics for all Stage2 evaluations in an experiment directory.
    """
    stage2_dir = exp_dir / "stage2"
    if not stage2_dir.exists():
        raise FileNotFoundError(f"stage2 directory does not exist: {stage2_dir}")

    # Collect all evaluation results.
    all_results: List[Dict[str, Any]] = []
    ai_scores: List[float] = []
    ped_scores: List[float] = []

    # Scores grouped by question type.
    score_by_qtype = {
        "single-choice": {"ai": [], "ped": []},
        "essay": {"ai": [], "ped": []},
        "other": {"ai": [], "ped": []},
    }

    # Status counters.
    total_units = 0
    completed_count = 0
    pass_count = 0
    fail_count = 0
    error_count = 0

    # Iterate through all unit_* directories.
    for unit_dir in sorted(stage2_dir.iterdir(), key=lambda x: int(x.name.replace("unit_", "")) if x.name.startswith("unit_") else 0):
        if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
            continue

        eval_file = unit_dir / "evaluation_state.json"
        if not eval_file.exists():
            continue

        total_units += 1

        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_state = json.load(f)

            scores = extract_scores_from_evaluation_state(eval_state)
            all_results.append(scores)

            # Count completion status.
            if scores["current_stage"] == "completed":
                completed_count += 1

                # Count pass/fail decisions.
                if scores["final_decision"] == "pass":
                    pass_count += 1
                elif scores["final_decision"] == "fail":
                    fail_count += 1

                # Collect scores.
                if scores["ai_overall_score"] is not None:
                    ai_scores.append(float(scores["ai_overall_score"]))
                    qt = scores.get("question_type") or "other"
                    if qt not in score_by_qtype:
                        qt = "other"
                    score_by_qtype[qt]["ai"].append(float(scores["ai_overall_score"]))

                if scores["ped_overall_score"] is not None:
                    ped_scores.append(float(scores["ped_overall_score"]))
                    qt = scores.get("question_type") or "other"
                    if qt not in score_by_qtype:
                        qt = "other"
                    score_by_qtype[qt]["ped"].append(float(scores["ped_overall_score"]))
            else:
                error_count += 1

        except Exception as e:
            print(f"[WARN] Failed to parse {eval_file}: {e}")
            error_count += 1

    # Aggregate helpers.
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        avg = _avg(xs)
        return (sum((x - avg) ** 2 for x in xs) / len(xs)) ** 0.5

    def _min_max(xs: List[float]) -> tuple:
        return (min(xs), max(xs)) if xs else (0.0, 0.0)

    # Build statistics payload.
    stats = {
        "experiment_id": exp_dir.name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),

        # Basic counts.
        "total_units": total_units,
        "completed_count": completed_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "pass_rate": pass_count / completed_count if completed_count > 0 else 0.0,

        # AI score statistics.
        "ai_score_count": len(ai_scores),
        "avg_ai_score": _avg(ai_scores),
        "std_ai_score": _std(ai_scores),
        "min_ai_score": _min_max(ai_scores)[0],
        "max_ai_score": _min_max(ai_scores)[1],

        # Pedagogical score statistics.
        "ped_score_count": len(ped_scores),
        "avg_ped_score": _avg(ped_scores),
        "std_ped_score": _std(ped_scores),
        "min_ped_score": _min_max(ped_scores)[0],
        "max_ped_score": _min_max(ped_scores)[1],

        # Per-question-type statistics.
        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_by_qtype.items()},
        "avg_ped_score_by_question_type": {k: _avg(v["ped"]) for k, v in score_by_qtype.items()},
        "count_by_question_type": {k: len(v["ai"]) for k, v in score_by_qtype.items()},

        # Detailed results.
        "results": all_results,
    }

    return stats


def print_summary(stats: Dict[str, Any]):
    """Print a statistics summary."""
    print("\n" + "=" * 80)
    print("Stage2 Evaluation Statistics")
    print("=" * 80)
    print(f"  Experiment ID: {stats['experiment_id']}")
    print(f"  Timestamp: {stats['timestamp']}")
    print("-" * 80)
    print(f"  Total units: {stats['total_units']}")
    print(f"  Completed: {stats['completed_count']}")
    print(f"  Passed: {stats['pass_count']}")
    print(f"  Failed: {stats['fail_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Pass rate: {stats['pass_rate']*100:.1f}%")
    print("-" * 80)
    print("  AI score statistics:")
    print(f"    Valid score count: {stats['ai_score_count']}")
    print(f"    Average: {stats['avg_ai_score']:.2f}")
    print(f"    Standard deviation: {stats['std_ai_score']:.2f}")
    print(f"    Minimum: {stats['min_ai_score']:.2f}")
    print(f"    Maximum: {stats['max_ai_score']:.2f}")
    print("-" * 80)
    print("  Pedagogical score statistics:")
    print(f"    Valid score count: {stats['ped_score_count']}")
    print(f"    Average: {stats['avg_ped_score']:.2f}")
    print(f"    Standard deviation: {stats['std_ped_score']:.2f}")
    print(f"    Minimum: {stats['min_ped_score']:.2f}")
    print(f"    Maximum: {stats['max_ped_score']:.2f}")
    print("-" * 80)
    print("  Per-question-type statistics:")
    for qt in ["single-choice", "essay", "other"]:
        count = stats['count_by_question_type'].get(qt, 0)
        if count > 0:
            ai_avg = stats['avg_ai_score_by_question_type'].get(qt, 0)
            ped_avg = stats['avg_ped_score_by_question_type'].get(qt, 0)
            print(f"    {qt}: count={count}, avg_ai={ai_avg:.2f}, avg_ped={ped_avg:.2f}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_stage2_stats.py <experiment_dir>")
        print("Example: python scripts/compute_stage2_stats.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"Error: directory does not exist - {exp_dir}")
        sys.exit(1)

    print(f"[INFO] Computing statistics for experiment directory: {exp_dir}")

    try:
        stats = compute_stage2_stats(exp_dir)

        # Print summary.
        print_summary(stats)

        # Save summary statistics to JSON.
        output_file = exp_dir / "stage2_stats_summary.json"
        with open(output_file, "w", encoding="utf-8") as f:
            # Exclude detailed results from the summary file to keep it compact.
            stats_to_save = {k: v for k, v in stats.items() if k != "results"}
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] Summary statistics saved to: {output_file}")

        # Save detailed per-unit results separately.
        results_file = exp_dir / "stage2_results_detail.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stats["results"], f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Detailed results saved to: {results_file}")

    except Exception as e:
        print(f"[ERROR] Statistics computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
