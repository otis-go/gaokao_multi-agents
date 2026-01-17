# src/shared/analysis.py
"""
Stage2 Results Analysis Module

This module provides a unified function for analyzing Stage2 evaluation results.
It is decoupled from CLI and can be reused by other tools.

Design Principles:
1. Only reads structured fields from EvaluationPipelineState / evaluation_state.json
2. Does NOT read Stage1 artifacts for score inference
3. Provides reusable summary statistics
4. Supports both in-memory states and JSON files

Usage:
    from src.shared.analysis import analyze_stage2_results

    # From in-memory states
    summary = analyze_stage2_results(evaluation_states)

    # From JSON files
    summary = analyze_stage2_results_from_dir(stage2_output_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Stage2ResultSummary:
    """
    Summary of Stage2 evaluation results.

    All fields are computed from Stage2 structured outputs only.
    """
    # Counts
    total_evaluated: int = 0
    ai_eval_success: int = 0
    ped_eval_success: int = 0
    both_success: int = 0

    # Decision counts
    pass_count: int = 0
    reject_count: int = 0
    error_count: int = 0
    pending_count: int = 0

    # Scores (only from successful evaluations)
    ai_scores: List[float] = field(default_factory=list)
    ped_scores: List[float] = field(default_factory=list)

    # Aggregated scores
    avg_ai_score: float = 0.0
    avg_ped_score: float = 0.0

    # By question type
    scores_by_question_type: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Model info
    eval_models: List[str] = field(default_factory=list)

    # Per-unit details (optional)
    unit_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_evaluated": self.total_evaluated,
            "ai_eval_success": self.ai_eval_success,
            "ped_eval_success": self.ped_eval_success,
            "both_success": self.both_success,
            "pass_count": self.pass_count,
            "reject_count": self.reject_count,
            "error_count": self.error_count,
            "pending_count": self.pending_count,
            "pass_rate": self.pass_count / self.total_evaluated if self.total_evaluated > 0 else 0.0,
            "avg_ai_score": self.avg_ai_score,
            "avg_ped_score": self.avg_ped_score,
            "scores_by_question_type": {
                qt: {
                    "avg_ai": sum(scores["ai"]) / len(scores["ai"]) if scores["ai"] else 0.0,
                    "avg_ped": sum(scores["ped"]) / len(scores["ped"]) if scores["ped"] else 0.0,
                    "count": len(scores["ai"]) if scores["ai"] else len(scores["ped"]),
                }
                for qt, scores in self.scores_by_question_type.items()
            },
            "eval_models": self.eval_models,
        }


def _extract_overall_score(result_obj: Any) -> Optional[float]:
    """
    Extract overall_score from an evaluation result object/dict.

    Handles:
    - object with .overall_score attribute
    - dict with 'overall_score', 'total_score', or 'score' key
    """
    if result_obj is None:
        return None

    # Try attribute access first
    if hasattr(result_obj, "overall_score"):
        try:
            return float(getattr(result_obj, "overall_score"))
        except (TypeError, ValueError):
            pass

    # Try dict access
    if isinstance(result_obj, dict):
        for key in ("overall_score", "total_score", "score"):
            if key in result_obj:
                try:
                    return float(result_obj[key])
                except (TypeError, ValueError):
                    pass

    return None


def _extract_decision(result_obj: Any) -> Optional[str]:
    """Extract decision from result object/dict."""
    if result_obj is None:
        return None

    if hasattr(result_obj, "decision"):
        return getattr(result_obj, "decision")

    if isinstance(result_obj, dict):
        return result_obj.get("decision")

    return None


def analyze_stage2_results(
    evaluation_states: List[Any],
    *,
    include_unit_details: bool = False,
) -> Stage2ResultSummary:
    """
    Analyze a list of Stage2 evaluation states.

    Args:
        evaluation_states: List of EvaluationPipelineState objects or dicts
        include_unit_details: Whether to include per-unit details in summary

    Returns:
        Stage2ResultSummary with aggregated statistics

    Note:
        This function only reads structured fields from Stage2 outputs.
        It does NOT infer scores from Stage1 artifacts.
    """
    summary = Stage2ResultSummary()
    summary.total_evaluated = len(evaluation_states)

    seen_models: set = set()

    for state in evaluation_states:
        unit_result: Dict[str, Any] = {}

        # Extract unit_id (for tracking)
        unit_id = None
        if hasattr(state, "unit_id"):
            unit_id = getattr(state, "unit_id")
        elif isinstance(state, dict):
            unit_id = state.get("unit_id")
        unit_result["unit_id"] = unit_id

        # Extract AI eval result
        ai_result = None
        ai_success = False
        if hasattr(state, "ai_eval_result"):
            ai_result = getattr(state, "ai_eval_result")
            ai_success = getattr(state, "ai_eval_success", False)
        elif isinstance(state, dict):
            ai_result = state.get("ai_eval_result") or state.get("ai_eval", {}).get("result")
            ai_success = state.get("ai_eval_success", False) or state.get("ai_eval", {}).get("success", False)

        ai_score = _extract_overall_score(ai_result)
        if ai_success and ai_score is not None:
            summary.ai_eval_success += 1
            summary.ai_scores.append(ai_score)
        unit_result["ai_score"] = ai_score
        unit_result["ai_success"] = ai_success

        # Extract pedagogical eval result
        ped_result = None
        ped_success = False
        if hasattr(state, "pedagogical_eval_result"):
            ped_result = getattr(state, "pedagogical_eval_result")
            ped_success = getattr(state, "pedagogical_eval_success", False)
        elif isinstance(state, dict):
            ped_result = state.get("pedagogical_eval_result") or state.get("pedagogical_eval", {}).get("result")
            ped_success = state.get("pedagogical_eval_success", False) or state.get("pedagogical_eval", {}).get("success", False)

        ped_score = _extract_overall_score(ped_result)
        if ped_success and ped_score is not None:
            summary.ped_eval_success += 1
            summary.ped_scores.append(ped_score)
        unit_result["ped_score"] = ped_score
        unit_result["ped_success"] = ped_success

        # Track both success
        if ai_success and ped_success:
            summary.both_success += 1

        # Extract final decision
        final_decision = None
        if hasattr(state, "final_decision"):
            final_decision = getattr(state, "final_decision")
        elif isinstance(state, dict):
            final_decision = state.get("final_decision")

        if final_decision == "pass":
            summary.pass_count += 1
        elif final_decision == "reject":
            summary.reject_count += 1
        elif final_decision == "error":
            summary.error_count += 1
        else:
            summary.pending_count += 1
        unit_result["final_decision"] = final_decision

        # Extract question type (from input if available)
        question_type = None
        input_core = None
        if hasattr(state, "input_core"):
            input_core = getattr(state, "input_core")
        elif isinstance(state, dict):
            input_core = state.get("input_core") or state.get("input")

        if isinstance(input_core, dict):
            question_type = input_core.get("question_type")
        elif hasattr(input_core, "question_type"):
            question_type = getattr(input_core, "question_type")

        if question_type:
            if question_type not in summary.scores_by_question_type:
                summary.scores_by_question_type[question_type] = {"ai": [], "ped": []}
            if ai_score is not None:
                summary.scores_by_question_type[question_type]["ai"].append(ai_score)
            if ped_score is not None:
                summary.scores_by_question_type[question_type]["ped"].append(ped_score)
        unit_result["question_type"] = question_type

        # Extract model names
        models = None
        if hasattr(state, "eval_models"):
            models = getattr(state, "eval_models")
        elif isinstance(state, dict):
            models = state.get("eval_models") or state.get("models", {}).get("eval_models")

        if isinstance(models, list):
            for m in models:
                if m:
                    seen_models.add(m)

        if include_unit_details:
            summary.unit_results.append(unit_result)

    # Compute averages
    if summary.ai_scores:
        summary.avg_ai_score = sum(summary.ai_scores) / len(summary.ai_scores)
    if summary.ped_scores:
        summary.avg_ped_score = sum(summary.ped_scores) / len(summary.ped_scores)

    summary.eval_models = sorted(seen_models)

    return summary


def analyze_stage2_results_from_dir(
    stage2_dir: Union[str, Path],
    *,
    include_unit_details: bool = False,
) -> Stage2ResultSummary:
    """
    Analyze Stage2 results from a directory of evaluation_state.json files.

    Args:
        stage2_dir: Path to stage2 output directory
        include_unit_details: Whether to include per-unit details

    Returns:
        Stage2ResultSummary

    Expected directory structure:
        stage2_dir/
        ├── unit_1/evaluation_state.json
        ├── unit_2/evaluation_state.json
        └── ...
    """
    stage2_path = Path(stage2_dir)
    states = []

    # Find all evaluation_state.json files
    for eval_file in stage2_path.glob("**/evaluation_state.json"):
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            states.append(data)
        except Exception as e:
            print(f"[Warning] Failed to load {eval_file}: {e}")

    return analyze_stage2_results(states, include_unit_details=include_unit_details)


__all__ = [
    "Stage2ResultSummary",
    "analyze_stage2_results",
    "analyze_stage2_results_from_dir",
]
