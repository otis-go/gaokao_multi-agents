#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/evaluate_original_baseline.py

Baseline Evaluation Script for Original Questions

This script evaluates original questions (from raw_material.json) through Stage2 only,
bypassing Stage1 entirely. The results serve as baseline scores for comparison with
generated questions.

Design Principles:
1. Uses the SAME EvaluationOrchestrator and prompts as generated question evaluation
2. Does NOT introduce baseline-only prompts or judge models
3. Supports caching - skips already-evaluated units unless --force is set
4. Batch-capable with proper error handling (continues on failure)

Usage:
    # Evaluate all 183 original questions (full baseline)
    python tools/evaluate_original_baseline.py --all --out-dir outputs/baseline_original

    # Evaluate specific unit_ids
    python tools/evaluate_original_baseline.py --unit-ids 1,2,3,4,5 --out-dir outputs/baseline_original

    # Evaluate unit_ids from a file
    python tools/evaluate_original_baseline.py --unit-ids-from outputs/exp1/subset_unit_ids.json

    # Sample N random units
    python tools/evaluate_original_baseline.py --max-n 10 --out-dir outputs/baseline_original

    # Force re-evaluation (overwrite existing)
    python tools/evaluate_original_baseline.py --unit-ids 1,2,3 --force

Output Structure:
    outputs/baseline_original/
    ├── unit_1/evaluation_state_original.json
    ├── unit_2/evaluation_state_original.json
    ├── ...
    ├── baseline_summary.json
    └── logs/prompts/...

Note on Cost:
    Full baseline (183 units) adds significant token cost.
    Recommend running baseline only for the unit_id subset involved in your experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows console UTF-8 support
if sys.platform == "win32":
    import io
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass
class BaselineSummary:
    """Summary of baseline evaluation run."""
    baseline_tag: str = "original"
    generated_at: str = ""
    raw_material_path: str = ""
    out_dir: str = ""

    # Counts
    total_units: int = 0
    evaluated_count: int = 0
    cached_count: int = 0
    error_count: int = 0

    # Judge model info (for comparability verification)
    eval_model_names: List[str] = field(default_factory=list)
    eval_model_weights: Dict[str, float] = field(default_factory=dict)

    # Score statistics
    ai_scores: List[float] = field(default_factory=list)
    ped_scores: List[float] = field(default_factory=list)
    avg_ai_score: float = 0.0
    avg_ped_score: float = 0.0

    # By question type
    scores_by_question_type: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Unit lists
    evaluated_unit_ids: List[str] = field(default_factory=list)
    cached_unit_ids: List[str] = field(default_factory=list)
    error_unit_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_tag": self.baseline_tag,
            "generated_at": self.generated_at,
            "raw_material_path": self.raw_material_path,
            "out_dir": self.out_dir,
            "total_units": self.total_units,
            "evaluated_count": self.evaluated_count,
            "cached_count": self.cached_count,
            "error_count": self.error_count,
            "eval_model_names": self.eval_model_names,
            "eval_model_weights": self.eval_model_weights,
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
            "errors": self.errors,
            "evaluated_unit_ids": self.evaluated_unit_ids,
            "cached_unit_ids": self.cached_unit_ids,
            "error_unit_ids": self.error_unit_ids,
        }


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_overall_score(result: Any) -> Optional[float]:
    """Extract overall_score from evaluation result."""
    if result is None:
        return None
    if hasattr(result, "overall_score"):
        return _safe_float(getattr(result, "overall_score"))
    if isinstance(result, dict):
        for key in ("overall_score", "total_score", "score"):
            if key in result:
                return _safe_float(result[key])
    return None


def load_unit_ids_from_file(file_path: str) -> List[str]:
    """Load unit_ids from a JSON file (list or dict with unit_ids key)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(uid) for uid in data]
    if isinstance(data, dict):
        # Try common keys
        for key in ("unit_ids", "subset_unit_ids", "units"):
            if key in data:
                return [str(uid) for uid in data[key]]
    raise ValueError(f"Cannot extract unit_ids from {file_path}")


def evaluate_single_baseline(
    raw_item: Dict[str, Any],
    evaluation_orchestrator: Any,
    out_dir: Path,
    force: bool = False,
    baseline_tag: str = "original",
) -> Dict[str, Any]:
    """
    Evaluate a single original question and persist results.

    Returns:
        Dict with keys: unit_id, status, ai_score, ped_score, question_type, error
    """
    from src.shared.adapters.original_to_stage2 import build_stage2_record_from_original

    unit_id = str(raw_item.get("unit_id", "unknown"))
    result = {
        "unit_id": unit_id,
        "status": "pending",
        "ai_score": None,
        "ped_score": None,
        "question_type": "unknown",
        "error": None,
    }

    # Check cache
    unit_dir = out_dir / f"unit_{unit_id}"
    eval_state_path = unit_dir / "evaluation_state_original.json"

    if eval_state_path.exists() and not force:
        # Load cached result
        try:
            with open(eval_state_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            result["status"] = "cached"
            result["ai_score"] = _safe_float(
                cached_data.get("ai_eval", {}).get("result", {}).get("overall_score")
            )
            result["ped_score"] = _safe_float(
                cached_data.get("pedagogical_eval", {}).get("result", {}).get("overall_score")
            )
            result["question_type"] = cached_data.get("input", {}).get("question_type", "unknown")
            return result
        except Exception as e:
            # Cache corrupted, re-evaluate
            print(f"  [WARN] Cache corrupted for unit_{unit_id}, re-evaluating: {e}")

    # Build Stage2Record from original
    build_result = build_stage2_record_from_original(
        raw_item,
        experiment_id=f"BASELINE_{baseline_tag.upper()}",
        baseline_tag=baseline_tag,
    )

    if not build_result.success:
        result["status"] = "build_failed"
        result["error"] = "; ".join(build_result.skip_reasons)
        return result

    stage2_record = build_result.record
    result["question_type"] = build_result.inferred_question_type

    # Run Stage2 evaluation
    try:
        evaluation_state = evaluation_orchestrator.run(stage2_record)

        # Extract scores
        ai_result = getattr(evaluation_state, "ai_eval_result", None)
        ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)

        result["ai_score"] = _get_overall_score(ai_result)
        result["ped_score"] = _get_overall_score(ped_result)
        result["status"] = "evaluated"

        # Persist evaluation state
        unit_dir.mkdir(parents=True, exist_ok=True)

        # Build serializable state
        state_dict = _build_baseline_state_dict(
            evaluation_state,
            raw_item,
            stage2_record,
            build_result,
            baseline_tag,
        )

        with open(eval_state_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    except Exception as e:
        result["status"] = "eval_failed"
        result["error"] = str(e)
        traceback.print_exc()

    return result


def _build_baseline_state_dict(
    evaluation_state: Any,
    raw_item: Dict[str, Any],
    stage2_record: Any,
    build_result: Any,
    baseline_tag: str,
) -> Dict[str, Any]:
    """Build serializable state dict for baseline evaluation."""
    import dataclasses

    def _to_serializable(obj):
        if obj is None:
            return None
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(x) for x in obj]
        return obj

    # Build state dict similar to regular evaluation_state.json
    state_dict = {
        "experiment_id": f"BASELINE_{baseline_tag.upper()}",
        "unit_id": str(raw_item.get("unit_id")),
        "baseline_tag": baseline_tag,
        "is_baseline": True,
        "final_decision": getattr(evaluation_state, "final_decision", None),
        "current_stage": getattr(evaluation_state, "current_stage", None),

        # Input (from original question)
        "input": _to_serializable(stage2_record.core_input) if stage2_record else None,

        # Stage1 meta (placeholder for baseline)
        "stage1_meta": _to_serializable(stage2_record.stage1_meta) if stage2_record else None,

        # Original question metadata
        "original_question": {
            "source": raw_item.get("source"),
            "question_id": raw_item.get("question_id"),
            "inferred_question_type": build_result.inferred_question_type,
        },

        # AI eval
        "ai_eval": {
            "enabled": True,
            "success": getattr(evaluation_state, "ai_eval_success", False),
            "result": _to_serializable(getattr(evaluation_state, "ai_eval_result", None)),
        },

        # Pedagogical eval
        "pedagogical_eval": {
            "enabled": True,
            "success": getattr(evaluation_state, "pedagogical_eval_success", False),
            "result": _to_serializable(getattr(evaluation_state, "pedagogical_eval_result", None)),
        },

        # Models
        "models": {
            "eval_models": getattr(evaluation_state, "eval_models", []),
            "ai_eval_models": getattr(evaluation_state, "ai_eval_model_names", []),
            "ped_eval_models": getattr(evaluation_state, "ped_eval_model_names", []),
        },

        "skipped_modules": getattr(evaluation_state, "skipped_modules", []),
        "notes": getattr(evaluation_state, "notes", ""),
        "errors": getattr(evaluation_state, "errors", []),
    }

    return state_dict


def run_baseline_evaluation(
    raw_material_path: str,
    out_dir: str,
    unit_ids: Optional[List[str]] = None,
    max_n: Optional[int] = None,
    force: bool = False,
    baseline_tag: str = "original",
    verbose: bool = False,
) -> BaselineSummary:
    """
    Run baseline evaluation for original questions.

    Args:
        raw_material_path: Path to raw_material.json
        out_dir: Output directory for baseline results
        unit_ids: Specific unit_ids to evaluate (None = all)
        max_n: Maximum number of units to evaluate (for sampling)
        force: Force re-evaluation (overwrite cache)
        baseline_tag: Tag for this baseline run
        verbose: Show verbose output

    Returns:
        BaselineSummary with aggregated statistics
    """
    from src.shared.adapters.original_to_stage2 import load_raw_materials
    from src.shared.config import create_default_config
    from src.shared.llm_router import LLMRouter
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator

    summary = BaselineSummary(
        baseline_tag=baseline_tag,
        generated_at=datetime.now().isoformat(),
        raw_material_path=raw_material_path,
        out_dir=out_dir,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load raw materials
    print(f"\n[Baseline] Loading raw materials from: {raw_material_path}")
    raw_items = load_raw_materials(raw_material_path, unit_ids)
    print(f"[Baseline] Loaded {len(raw_items)} items")

    # Apply max_n sampling
    if max_n and len(raw_items) > max_n:
        print(f"[Baseline] Sampling {max_n} items from {len(raw_items)}")
        raw_items = random.sample(raw_items, max_n)

    summary.total_units = len(raw_items)

    if not raw_items:
        print("[Baseline] No items to evaluate")
        return summary

    # Create config and orchestrator
    print("\n[Baseline] Initializing evaluation orchestrator...")
    config = create_default_config(f"BASELINE_{baseline_tag.upper()}")
    config.output_dir = out_path
    config.llm.verbose = verbose

    router = LLMRouter.from_config(config)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router)

    # Record model info for comparability
    summary.eval_model_names = router.get_eval_model_names()
    summary.eval_model_weights = router.get_eval_model_weights()

    print(f"[Baseline] Using evaluation models: {summary.eval_model_names}")
    print(f"[Baseline] Force re-evaluation: {force}")
    print()

    # Evaluate each unit
    for i, raw_item in enumerate(raw_items):
        unit_id = str(raw_item.get("unit_id", "unknown"))
        print(f"[{i+1}/{summary.total_units}] Evaluating unit_{unit_id}...", end=" ")

        result = evaluate_single_baseline(
            raw_item,
            evaluation_orchestrator,
            out_path,
            force=force,
            baseline_tag=baseline_tag,
        )

        # Update summary
        if result["status"] == "cached":
            summary.cached_count += 1
            summary.cached_unit_ids.append(unit_id)
            print("[CACHED]")
        elif result["status"] == "evaluated":
            summary.evaluated_count += 1
            summary.evaluated_unit_ids.append(unit_id)
            print("[OK]")
        else:
            summary.error_count += 1
            summary.error_unit_ids.append(unit_id)
            summary.errors.append({
                "unit_id": unit_id,
                "status": result["status"],
                "error": result["error"],
            })
            print(f"[ERROR: {result['status']}]")
            continue

        # Collect scores
        if result["ai_score"] is not None:
            summary.ai_scores.append(result["ai_score"])
        if result["ped_score"] is not None:
            summary.ped_scores.append(result["ped_score"])

        # By question type
        qt = result["question_type"] or "unknown"
        if qt not in summary.scores_by_question_type:
            summary.scores_by_question_type[qt] = {"ai": [], "ped": []}
        if result["ai_score"] is not None:
            summary.scores_by_question_type[qt]["ai"].append(result["ai_score"])
        if result["ped_score"] is not None:
            summary.scores_by_question_type[qt]["ped"].append(result["ped_score"])

    # Compute averages
    if summary.ai_scores:
        summary.avg_ai_score = sum(summary.ai_scores) / len(summary.ai_scores)
    if summary.ped_scores:
        summary.avg_ped_score = sum(summary.ped_scores) / len(summary.ped_scores)

    # Save summary
    summary_path = out_path / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n[Baseline] Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for original questions (Stage2 only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all original questions
  python tools/evaluate_original_baseline.py --all --out-dir outputs/baseline_original

  # Evaluate specific unit_ids
  python tools/evaluate_original_baseline.py --unit-ids 1,2,3,4,5

  # Load unit_ids from experiment subset file
  python tools/evaluate_original_baseline.py --unit-ids-from outputs/exp1/subset_unit_ids.json

  # Sample 10 random units
  python tools/evaluate_original_baseline.py --max-n 10

  # Force re-evaluation
  python tools/evaluate_original_baseline.py --unit-ids 1,2,3 --force

Note on Cost:
  Full baseline (183 units) adds significant token cost.
  Recommend using --unit-ids-from to match your experiment subset.
        """,
    )

    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all units in raw_material.json",
    )
    input_group.add_argument(
        "--unit-ids",
        type=str,
        default=None,
        help="Comma-separated list of unit_ids to evaluate",
    )
    input_group.add_argument(
        "--unit-ids-from",
        type=str,
        default=None,
        help="Load unit_ids from JSON file (list or dict with unit_ids key)",
    )

    # Paths
    parser.add_argument(
        "--raw-material",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw_material.json"),
        help="Path to raw_material.json (default: data/raw_material.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "baseline_original"),
        help="Output directory (default: outputs/baseline_original)",
    )

    # Options
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Maximum number of units to evaluate (for sampling)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation (overwrite cached results)",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default="original",
        help="Tag for this baseline run (default: original)",
    )
    parser.add_argument(
        "--subset-filter",
        type=str,
        default=None,
        help="Filter by source field (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Determine unit_ids
    unit_ids = None

    if args.all:
        unit_ids = None  # Will load all
    elif args.unit_ids:
        unit_ids = [uid.strip() for uid in args.unit_ids.split(",")]
    elif args.unit_ids_from:
        if not Path(args.unit_ids_from).exists():
            print(f"Error: unit_ids file not found: {args.unit_ids_from}")
            sys.exit(1)
        unit_ids = load_unit_ids_from_file(args.unit_ids_from)
        print(f"[Baseline] Loaded {len(unit_ids)} unit_ids from {args.unit_ids_from}")
    else:
        # Default: require explicit selection
        print("Error: Must specify --all, --unit-ids, or --unit-ids-from")
        parser.print_help()
        sys.exit(1)

    # Run evaluation
    print("\n" + "=" * 70)
    print("Baseline Evaluation for Original Questions")
    print("=" * 70)

    summary = run_baseline_evaluation(
        raw_material_path=args.raw_material,
        out_dir=args.out_dir,
        unit_ids=unit_ids,
        max_n=args.max_n,
        force=args.force,
        baseline_tag=args.baseline_tag,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Baseline Evaluation Summary")
    print("=" * 70)
    print(f"  Total units:     {summary.total_units}")
    print(f"  Evaluated:       {summary.evaluated_count}")
    print(f"  Cached:          {summary.cached_count}")
    print(f"  Errors:          {summary.error_count}")
    print(f"  Avg AI score:    {summary.avg_ai_score:.2f}")
    print(f"  Avg Ped score:   {summary.avg_ped_score:.2f}")
    print(f"  Eval models:     {summary.eval_model_names}")
    print()

    if summary.scores_by_question_type:
        print("By Question Type:")
        for qt, scores in summary.scores_by_question_type.items():
            ai_avg = sum(scores["ai"]) / len(scores["ai"]) if scores["ai"] else 0
            ped_avg = sum(scores["ped"]) / len(scores["ped"]) if scores["ped"] else 0
            count = len(scores["ai"]) or len(scores["ped"])
            print(f"  {qt}: count={count}, AI={ai_avg:.2f}, Ped={ped_avg:.2f}")

    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for err in summary.errors[:5]:
            print(f"  - unit_{err['unit_id']}: {err['error']}")
        if len(summary.errors) > 5:
            print(f"  ... and {len(summary.errors) - 5} more")

    print(f"\nOutput directory: {args.out_dir}")

    sys.exit(0 if summary.error_count == 0 else 1)


if __name__ == "__main__":
    main()
