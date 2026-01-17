#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/analytics_demo.py

Automatic statistics/analysis demo for batch experiment results.

This module reads Stage2-persisted evaluation_state.json files and produces
aggregated analytics including:
- Core metrics: AI mean, ped mean, sample size, quantiles
- Grouped metrics by question_type
- Rule-based inference for missing question_type
- Extensible METRICS_CATALOG

Usage:
    python tools/analytics_demo.py --batch-dir outputs/selfcheck
    python tools/analytics_demo.py --batch-dir outputs/batch_run --glob "**/evaluation_state.json"
    python tools/analytics_demo.py --batch-dir outputs/batch_run --out-dir outputs/analytics --format all
    python tools/analytics_demo.py --batch-dir outputs/batch_run --group-by question_type,ablation_skip_agent
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Windows console UTF-8 support
if sys.platform == "win32":
    import io
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Already wrapped or unavailable


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# =============================================================================
# METRICS_CATALOG - Extensible metrics registry
# =============================================================================

@dataclass
class MetricDefinition:
    """Definition of a single metric."""
    name: str
    description: str
    extractor: Callable[[Any], Optional[float]]
    aggregator: str = "mean"  # mean, sum, count, min, max, median
    category: str = "core"    # core, dimension, ablation, custom


def _extract_ai_score(item: "EvalItem") -> Optional[float]:
    """Extract AI evaluation overall score."""
    return item.ai_score


def _extract_ped_score(item: "EvalItem") -> Optional[float]:
    """Extract pedagogical evaluation overall score."""
    return item.ped_score


def _extract_agent5_score(item: "EvalItem") -> Optional[float]:
    """Extract Stage1 Agent5 overall score."""
    return item.agent5_score


def _extract_pass_flag(item: "EvalItem") -> Optional[float]:
    """Extract pass flag (1.0 for pass, 0.0 otherwise)."""
    return 1.0 if item.final_decision == "pass" else 0.0


def _extract_reject_flag(item: "EvalItem") -> Optional[float]:
    """Extract reject flag (1.0 for reject, 0.0 otherwise)."""
    return 1.0 if item.final_decision == "reject" else 0.0


def _extract_error_flag(item: "EvalItem") -> Optional[float]:
    """Extract error flag (1.0 for error, 0.0 otherwise)."""
    return 1.0 if item.final_decision == "error" else 0.0


# ===== Baseline comparison metric extractors =====

def _extract_delta_ai(item: "EvalItem") -> Optional[float]:
    """Extract delta AI score (generated - original)."""
    return item.delta_ai


def _extract_delta_ped(item: "EvalItem") -> Optional[float]:
    """Extract delta pedagogical score (generated - original)."""
    return item.delta_ped


def _extract_ge_original_ai(item: "EvalItem") -> Optional[float]:
    """Extract AI closeness flag (1.0 if generated >= original, else 0.0)."""
    if item.ge_original_ai is None:
        return None
    return 1.0 if item.ge_original_ai else 0.0


def _extract_ge_original_ped(item: "EvalItem") -> Optional[float]:
    """Extract ped closeness flag (1.0 if generated >= original, else 0.0)."""
    if item.ge_original_ped is None:
        return None
    return 1.0 if item.ge_original_ped else 0.0


def _extract_ai_score_original(item: "EvalItem") -> Optional[float]:
    """Extract original (baseline) AI score."""
    return item.ai_score_original


def _extract_ped_score_original(item: "EvalItem") -> Optional[float]:
    """Extract original (baseline) pedagogical score."""
    return item.ped_score_original


def _extract_has_baseline(item: "EvalItem") -> Optional[float]:
    """Extract baseline availability flag (1.0 if has baseline, else 0.0)."""
    return 1.0 if item.has_baseline else 0.0


# METRICS_CATALOG: Registry of all available metrics
METRICS_CATALOG: Dict[str, MetricDefinition] = {
    "ai_score": MetricDefinition(
        name="ai_score",
        description="AI-centric evaluation overall score (0-100)",
        extractor=_extract_ai_score,
        aggregator="mean",
        category="core",
    ),
    "ped_score": MetricDefinition(
        name="ped_score",
        description="Pedagogical evaluation overall score (0-100)",
        extractor=_extract_ped_score,
        aggregator="mean",
        category="core",
    ),
    "agent5_score": MetricDefinition(
        name="agent5_score",
        description="Stage1 Agent5 quality score (0-1)",
        extractor=_extract_agent5_score,
        aggregator="mean",
        category="core",
    ),
    "pass_rate": MetricDefinition(
        name="pass_rate",
        description="Proportion of items with pass decision",
        extractor=_extract_pass_flag,
        aggregator="mean",
        category="core",
    ),
    "reject_rate": MetricDefinition(
        name="reject_rate",
        description="Proportion of items with reject decision",
        extractor=_extract_reject_flag,
        aggregator="mean",
        category="core",
    ),
    "error_rate": MetricDefinition(
        name="error_rate",
        description="Proportion of items with error decision",
        extractor=_extract_error_flag,
        aggregator="mean",
        category="core",
    ),
    # ===== Baseline comparison metrics =====
    "delta_ai": MetricDefinition(
        name="delta_ai",
        description="Delta AI score (generated - original)",
        extractor=_extract_delta_ai,
        aggregator="mean",
        category="baseline",
    ),
    "delta_ped": MetricDefinition(
        name="delta_ped",
        description="Delta pedagogical score (generated - original)",
        extractor=_extract_delta_ped,
        aggregator="mean",
        category="baseline",
    ),
    "ge_original_ai_rate": MetricDefinition(
        name="ge_original_ai_rate",
        description="Proportion of items where generated AI score >= original",
        extractor=_extract_ge_original_ai,
        aggregator="mean",
        category="baseline",
    ),
    "ge_original_ped_rate": MetricDefinition(
        name="ge_original_ped_rate",
        description="Proportion of items where generated ped score >= original",
        extractor=_extract_ge_original_ped,
        aggregator="mean",
        category="baseline",
    ),
    "ai_score_original": MetricDefinition(
        name="ai_score_original",
        description="Original (baseline) AI evaluation score",
        extractor=_extract_ai_score_original,
        aggregator="mean",
        category="baseline",
    ),
    "ped_score_original": MetricDefinition(
        name="ped_score_original",
        description="Original (baseline) pedagogical evaluation score",
        extractor=_extract_ped_score_original,
        aggregator="mean",
        category="baseline",
    ),
    "baseline_coverage": MetricDefinition(
        name="baseline_coverage",
        description="Proportion of items with baseline data available",
        extractor=_extract_has_baseline,
        aggregator="mean",
        category="baseline",
    ),
}


def register_metric(metric: MetricDefinition) -> None:
    """Register a custom metric to the catalog."""
    METRICS_CATALOG[metric.name] = metric


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalItem:
    """
    Normalized evaluation item from evaluation_state.json.

    All fields are extracted robustly with graceful degradation.
    """
    # Identifiers
    experiment_id: str = ""
    unit_id: str = ""
    file_path: str = ""

    # Core scores
    ai_score: Optional[float] = None
    ped_score: Optional[float] = None
    agent5_score: Optional[float] = None

    # Decision
    final_decision: str = "unknown"

    # AI eval details
    ai_eval_success: bool = False
    ai_dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Ped eval details
    ped_eval_success: bool = False
    ped_dimension_results: List[Dict[str, Any]] = field(default_factory=list)

    # Question metadata
    question_type: str = "unknown"
    question_type_inferred: bool = False
    stem: str = ""

    # Stage1 meta
    ablation_skip_agent: str = "none"

    # Models used
    eval_models: List[str] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Raw data (for debugging)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    # ===== Baseline comparison fields (populated when baseline_dir is provided) =====
    # Original (baseline) scores
    ai_score_original: Optional[float] = None
    ped_score_original: Optional[float] = None

    # Delta metrics (generated - original)
    delta_ai: Optional[float] = None
    delta_ped: Optional[float] = None

    # Closeness flags (generated >= original)
    ge_original_ai: Optional[bool] = None
    ge_original_ped: Optional[bool] = None

    # Baseline metadata
    has_baseline: bool = False
    baseline_file_path: str = ""


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    name: str
    count: int = 0
    valid_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "valid_count": self.valid_count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "q25": self.q25,
            "q75": self.q75,
        }


@dataclass
class GroupedMetrics:
    """Metrics grouped by a specific field."""
    group_by: str
    groups: Dict[str, Dict[str, MetricStats]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_by": self.group_by,
            "groups": {
                k: {m: s.to_dict() for m, s in v.items()}
                for k, v in self.groups.items()
            }
        }


@dataclass
class BatchReport:
    """
    Complete analytics report for a batch of evaluation results.
    """
    # Metadata
    generated_at: str = ""
    batch_dir: str = ""
    glob_pattern: str = ""
    total_files_found: int = 0
    total_items_loaded: int = 0

    # Core metrics
    core_metrics: Dict[str, MetricStats] = field(default_factory=dict)

    # Grouped metrics
    grouped_metrics: List[GroupedMetrics] = field(default_factory=list)

    # Decision distribution
    decision_distribution: Dict[str, int] = field(default_factory=dict)

    # Question type distribution
    question_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Ablation distribution
    ablation_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-item details (optional)
    items: List[EvalItem] = field(default_factory=list)

    # Warnings/errors during loading
    load_warnings: List[str] = field(default_factory=list)

    # ===== Baseline comparison section =====
    baseline_dir: str = ""
    baseline_enabled: bool = False
    baseline_metrics: Dict[str, MetricStats] = field(default_factory=dict)
    baseline_coverage: float = 0.0  # Proportion of items with baseline
    baseline_items_matched: int = 0
    missing_baseline_unit_ids: List[str] = field(default_factory=list)

    def to_dict(self, include_items: bool = False) -> Dict[str, Any]:
        result = {
            "generated_at": self.generated_at,
            "batch_dir": self.batch_dir,
            "glob_pattern": self.glob_pattern,
            "total_files_found": self.total_files_found,
            "total_items_loaded": self.total_items_loaded,
            "core_metrics": {k: v.to_dict() for k, v in self.core_metrics.items()},
            "grouped_metrics": [g.to_dict() for g in self.grouped_metrics],
            "decision_distribution": self.decision_distribution,
            "question_type_distribution": self.question_type_distribution,
            "ablation_distribution": self.ablation_distribution,
            "load_warnings": self.load_warnings,
        }
        # Include baseline section if enabled
        if self.baseline_enabled:
            result["baseline"] = {
                "baseline_dir": self.baseline_dir,
                "baseline_coverage": self.baseline_coverage,
                "baseline_items_matched": self.baseline_items_matched,
                "missing_baseline_unit_ids": self.missing_baseline_unit_ids[:50],  # Limit
                "metrics": {k: v.to_dict() for k, v in self.baseline_metrics.items()},
            }
        if include_items:
            result["items"] = [asdict(item) for item in self.items]
        return result


# =============================================================================
# Question Type Inference
# =============================================================================

# Rule-based patterns for inferring question_type
QUESTION_TYPE_PATTERNS: List[Tuple[str, List[str]]] = [
    # Single-choice indicators (Chinese)
    ("single-choice", [
        r"选择.*正确",
        r"选择.*错误",
        r"下列.*正确.*是",
        r"下列.*错误.*是",
        r"以下.*说法.*正确",
        r"以下.*说法.*错误",
        r"符合.*的是",
        r"不符合.*的是",
        r"^[ABCD][\.\、]",
        # English patterns
        r"which.*following.*correct",
        r"which.*following.*incorrect",
        r"which.*statement.*true",
        r"which.*statement.*false",
        r"select.*correct",
        r"choose.*correct",
        r"according to.*which",
        r"best describes",
    ]),
    # Essay / subjective indicators (Chinese)
    ("essay", [
        r"简要.*分析",
        r"请.*概括",
        r"请.*说明",
        r"请.*阐述",
        r"请.*解释",
        r"谈谈.*看法",
        r"你.*认为",
        r"结合.*分析",
        r"分析.*原因",
        r"概括.*特点",
        r"归纳.*要点",
        # English patterns
        r"please.*analyze",
        r"please.*explain",
        r"please.*describe",
        r"briefly.*explain",
        r"discuss.*how",
        r"analyze.*cause",
        r"summarize.*main",
        r"propose.*solution",
    ]),
    # Objective indicators (broader category)
    ("objective", [
        r"判断.*对错",
        r"是否.*正确",
        r"正确.*是",
        r"错误.*是",
        r"true or false",
        r"is.*correct",
    ]),
    # Subjective indicators
    ("subjective", [
        r"论述",
        r"评价",
        r"写.*作文",
        r"请.*写",
        r"write.*essay",
        r"compose",
        r"evaluate",
    ]),
]


def infer_question_type(stem: str, options: Optional[List[Any]] = None) -> Tuple[str, bool]:
    """
    Infer question type from stem text and options.

    Returns:
        Tuple of (question_type, was_inferred)
    """
    if not stem:
        return ("unknown", True)

    # If options exist, likely single-choice
    if options and len(options) >= 2:
        return ("single-choice", True)

    # Pattern matching
    for qtype, patterns in QUESTION_TYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, stem, re.IGNORECASE):
                return (qtype, True)

    # Default fallback
    return ("unknown", True)


# =============================================================================
# Normalization - Robust field extraction
# =============================================================================

def _safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely get nested value from dict or object.

    Supports both dict access and attribute access.
    """
    current = data
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, None)
        elif hasattr(current, key):
            current = getattr(current, key, None)
        else:
            return default
    return current if current is not None else default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_evaluation_state(
    data: Dict[str, Any],
    file_path: str = "",
) -> EvalItem:
    """
    Normalize evaluation_state.json data to EvalItem.

    Handles multiple JSON structures and provides graceful degradation.

    Args:
        data: Raw dict from evaluation_state.json
        file_path: Source file path for debugging

    Returns:
        Normalized EvalItem
    """
    item = EvalItem(file_path=file_path, raw_data=data)

    # Basic identifiers
    item.experiment_id = _safe_get(data, "experiment_id", default="")
    item.unit_id = str(_safe_get(data, "unit_id", default=""))

    # Final decision
    item.final_decision = _safe_get(data, "final_decision", default="unknown")

    # AI evaluation score - try multiple paths
    ai_score = None
    # Path 1: ai_eval.result.overall_score
    ai_score = _safe_float(_safe_get(data, "ai_eval", "result", "overall_score"))
    # Path 2: ai_eval.result.total_score
    if ai_score is None:
        ai_score = _safe_float(_safe_get(data, "ai_eval", "result", "total_score"))
    # Path 3: ai_eval_result.overall_score (flat structure)
    if ai_score is None:
        ai_score = _safe_float(_safe_get(data, "ai_eval_result", "overall_score"))
    item.ai_score = ai_score
    item.ai_eval_success = _safe_get(data, "ai_eval", "success", default=False) or ai_score is not None

    # AI dimension scores
    ai_dims = _safe_get(data, "ai_eval", "result", "dimensions", default={})
    if isinstance(ai_dims, dict):
        for dim_name, dim_data in ai_dims.items():
            if isinstance(dim_data, dict) and "score" in dim_data:
                item.ai_dimension_scores[dim_name] = _safe_float(dim_data["score"], 0.0)

    # Pedagogical evaluation score - try multiple paths
    ped_score = None
    # Path 1: pedagogical_eval.result.overall_score
    ped_score = _safe_float(_safe_get(data, "pedagogical_eval", "result", "overall_score"))
    # Path 2: pedagogical_eval_result.overall_score (flat structure)
    if ped_score is None:
        ped_score = _safe_float(_safe_get(data, "pedagogical_eval_result", "overall_score"))
    item.ped_score = ped_score
    item.ped_eval_success = _safe_get(data, "pedagogical_eval", "success", default=False) or ped_score is not None

    # Ped dimension results
    ped_dims = _safe_get(data, "pedagogical_eval", "result", "dimension_results", default=[])
    if isinstance(ped_dims, list):
        item.ped_dimension_results = ped_dims

    # Agent5 score from stage1_meta
    item.agent5_score = _safe_float(_safe_get(data, "stage1_meta", "agent5_overall_score"))

    # Question type - first try explicit field
    question_type = _safe_get(data, "input", "question_type", default="")
    if not question_type:
        question_type = _safe_get(data, "question_type", default="")

    # If still missing, try inference
    stem = _safe_get(data, "input", "stem", default="")
    options = _safe_get(data, "input", "options", default=None)

    if not question_type or question_type == "unknown":
        inferred_type, was_inferred = infer_question_type(stem, options)
        item.question_type = inferred_type
        item.question_type_inferred = was_inferred
    else:
        item.question_type = question_type
        item.question_type_inferred = False

    item.stem = stem[:200] if stem else ""  # Truncate for summary

    # Ablation info
    item.ablation_skip_agent = _safe_get(data, "stage1_meta", "ablation_skip_agent", default="none")

    # Models used
    item.eval_models = _safe_get(data, "models", "eval_models", default=[])
    if not item.eval_models:
        item.eval_models = _safe_get(data, "eval_models", default=[])

    # Errors
    item.errors = _safe_get(data, "errors", default=[])

    return item


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metric_stats(
    items: List[EvalItem],
    metric_def: MetricDefinition,
) -> MetricStats:
    """
    Compute statistics for a single metric across items.
    """
    stats = MetricStats(name=metric_def.name, count=len(items))

    # Extract values
    values = []
    for item in items:
        val = metric_def.extractor(item)
        if val is not None:
            values.append(val)

    stats.values = values
    stats.valid_count = len(values)

    if not values:
        return stats

    # Compute statistics
    stats.mean = statistics.mean(values)
    stats.median = statistics.median(values)
    stats.min_val = min(values)
    stats.max_val = max(values)

    if len(values) >= 2:
        stats.std = statistics.stdev(values)

    # Quantiles
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n >= 4:
        stats.q25 = sorted_vals[n // 4]
        stats.q75 = sorted_vals[(3 * n) // 4]

    return stats


def compute_grouped_metrics(
    items: List[EvalItem],
    group_by: str,
    metric_names: List[str],
) -> GroupedMetrics:
    """
    Compute metrics grouped by a specific field.
    """
    result = GroupedMetrics(group_by=group_by)

    # Group items
    groups: Dict[str, List[EvalItem]] = {}
    for item in items:
        key = getattr(item, group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    # Compute metrics per group
    for group_key, group_items in groups.items():
        result.groups[group_key] = {}
        for metric_name in metric_names:
            if metric_name in METRICS_CATALOG:
                metric_def = METRICS_CATALOG[metric_name]
                stats = compute_metric_stats(group_items, metric_def)
                result.groups[group_key][metric_name] = stats

    return result


# =============================================================================
# File Loading
# =============================================================================

def load_evaluation_files(
    batch_dir: Union[str, Path],
    glob_pattern: str = "**/evaluation_state.json",
) -> Tuple[List[EvalItem], List[str]]:
    """
    Load all evaluation_state.json files from batch directory.

    Returns:
        Tuple of (items, warnings)
    """
    batch_path = Path(batch_dir)
    items = []
    warnings = []

    if not batch_path.exists():
        warnings.append(f"Batch directory does not exist: {batch_dir}")
        return items, warnings

    # Find all matching files
    files = list(batch_path.glob(glob_pattern))

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            item = normalize_evaluation_state(data, str(file_path))
            items.append(item)
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error in {file_path}: {e}")
        except Exception as e:
            warnings.append(f"Failed to load {file_path}: {e}")

    return items, warnings


# =============================================================================
# Baseline Loading and Alignment
# =============================================================================

def load_baseline_scores(
    baseline_dir: Union[str, Path],
    glob_pattern: str = "**/evaluation_state_original.json",
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Load baseline evaluation scores from baseline directory.

    Looks for evaluation_state_original.json files and extracts scores keyed by unit_id.

    Args:
        baseline_dir: Directory containing baseline evaluation results
        glob_pattern: Glob pattern to find baseline files

    Returns:
        Tuple of (baseline_scores_by_unit_id, warnings)
        baseline_scores_by_unit_id: {unit_id: {"ai_score": float, "ped_score": float, "file_path": str}}
    """
    baseline_path = Path(baseline_dir)
    scores = {}
    warnings = []

    if not baseline_path.exists():
        warnings.append(f"Baseline directory does not exist: {baseline_dir}")
        return scores, warnings

    # Find all matching baseline files
    files = list(baseline_path.glob(glob_pattern))

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract unit_id
            unit_id = str(_safe_get(data, "unit_id", default=""))
            if not unit_id:
                warnings.append(f"Missing unit_id in baseline file: {file_path}")
                continue

            # Extract AI score (try multiple paths)
            ai_score = _safe_float(_safe_get(data, "ai_eval", "result", "overall_score"))
            if ai_score is None:
                ai_score = _safe_float(_safe_get(data, "ai_eval", "result", "total_score"))
            if ai_score is None:
                ai_score = _safe_float(_safe_get(data, "ai_eval_result", "overall_score"))

            # Extract ped score (try multiple paths)
            ped_score = _safe_float(_safe_get(data, "pedagogical_eval", "result", "overall_score"))
            if ped_score is None:
                ped_score = _safe_float(_safe_get(data, "pedagogical_eval_result", "overall_score"))

            scores[unit_id] = {
                "ai_score": ai_score,
                "ped_score": ped_score,
                "file_path": str(file_path),
            }

        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error in baseline {file_path}: {e}")
        except Exception as e:
            warnings.append(f"Failed to load baseline {file_path}: {e}")

    return scores, warnings


def align_items_with_baseline(
    items: List[EvalItem],
    baseline_scores: Dict[str, Dict[str, Any]],
) -> Tuple[List[EvalItem], List[str]]:
    """
    Populate baseline comparison fields in EvalItem list.

    For each item, if a baseline with matching unit_id exists:
    - Set ai_score_original, ped_score_original
    - Compute delta_ai = ai_score - ai_score_original
    - Compute delta_ped = ped_score - ped_score_original
    - Set ge_original_ai = (ai_score >= ai_score_original)
    - Set ge_original_ped = (ped_score >= ped_score_original)
    - Set has_baseline = True

    Args:
        items: List of EvalItem to align
        baseline_scores: Dict of baseline scores keyed by unit_id

    Returns:
        Tuple of (aligned_items, missing_unit_ids)
    """
    missing_unit_ids = []

    for item in items:
        unit_id = item.unit_id

        if unit_id not in baseline_scores:
            missing_unit_ids.append(unit_id)
            continue

        baseline = baseline_scores[unit_id]

        # Populate baseline fields
        item.ai_score_original = baseline.get("ai_score")
        item.ped_score_original = baseline.get("ped_score")
        item.baseline_file_path = baseline.get("file_path", "")
        item.has_baseline = True

        # Compute delta metrics
        if item.ai_score is not None and item.ai_score_original is not None:
            item.delta_ai = item.ai_score - item.ai_score_original
            item.ge_original_ai = item.ai_score >= item.ai_score_original

        if item.ped_score is not None and item.ped_score_original is not None:
            item.delta_ped = item.ped_score - item.ped_score_original
            item.ge_original_ped = item.ped_score >= item.ped_score_original

    return items, missing_unit_ids


# =============================================================================
# Report Generation
# =============================================================================

def generate_batch_report(
    batch_dir: Union[str, Path],
    glob_pattern: str = "**/evaluation_state.json",
    group_by: Optional[List[str]] = None,
    include_items: bool = False,
    baseline_dir: Optional[Union[str, Path]] = None,
    baseline_glob: str = "**/evaluation_state_original.json",
) -> BatchReport:
    """
    Generate complete batch analytics report.

    Args:
        batch_dir: Directory containing evaluation results
        glob_pattern: Glob pattern to find evaluation files
        group_by: List of fields to group by (e.g., ["question_type", "ablation_skip_agent"])
        include_items: Whether to include per-item details in report
        baseline_dir: Optional directory containing baseline evaluation results for delta metrics
        baseline_glob: Glob pattern for baseline files (default: **/evaluation_state_original.json)

    Returns:
        BatchReport with aggregated analytics
    """
    report = BatchReport(
        generated_at=datetime.now().isoformat(),
        batch_dir=str(batch_dir),
        glob_pattern=glob_pattern,
    )

    # Load files
    items, warnings = load_evaluation_files(batch_dir, glob_pattern)
    report.total_files_found = len(list(Path(batch_dir).glob(glob_pattern))) if Path(batch_dir).exists() else 0
    report.total_items_loaded = len(items)
    report.load_warnings = warnings

    if not items:
        if include_items:
            report.items = items
        return report

    # ===== Baseline alignment (if baseline_dir provided) =====
    if baseline_dir:
        report.baseline_dir = str(baseline_dir)
        report.baseline_enabled = True

        baseline_scores, baseline_warnings = load_baseline_scores(baseline_dir, baseline_glob)
        report.load_warnings.extend(baseline_warnings)

        items, missing_unit_ids = align_items_with_baseline(items, baseline_scores)
        report.missing_baseline_unit_ids = missing_unit_ids

        # Compute baseline coverage
        items_with_baseline = sum(1 for item in items if item.has_baseline)
        report.baseline_items_matched = items_with_baseline
        report.baseline_coverage = items_with_baseline / len(items) if items else 0.0

    if include_items:
        report.items = items

    # Compute core metrics
    core_metric_names = ["ai_score", "ped_score", "agent5_score", "pass_rate", "reject_rate", "error_rate"]
    for metric_name in core_metric_names:
        if metric_name in METRICS_CATALOG:
            stats = compute_metric_stats(items, METRICS_CATALOG[metric_name])
            report.core_metrics[metric_name] = stats

    # Decision distribution
    for item in items:
        decision = item.final_decision
        report.decision_distribution[decision] = report.decision_distribution.get(decision, 0) + 1

    # Question type distribution
    for item in items:
        qtype = item.question_type
        report.question_type_distribution[qtype] = report.question_type_distribution.get(qtype, 0) + 1

    # Ablation distribution
    for item in items:
        ablation = item.ablation_skip_agent
        report.ablation_distribution[ablation] = report.ablation_distribution.get(ablation, 0) + 1

    # Grouped metrics
    if group_by:
        for group_field in group_by:
            grouped = compute_grouped_metrics(items, group_field, core_metric_names)
            report.grouped_metrics.append(grouped)
    else:
        # Default grouping by question_type
        grouped = compute_grouped_metrics(items, "question_type", core_metric_names)
        report.grouped_metrics.append(grouped)

    # ===== Baseline metrics (if enabled) =====
    if report.baseline_enabled:
        # Filter to items with baseline for delta metrics computation
        items_with_baseline = [item for item in items if item.has_baseline]

        baseline_metric_names = [
            "delta_ai", "delta_ped",
            "ge_original_ai_rate", "ge_original_ped_rate",
            "ai_score_original", "ped_score_original",
            "baseline_coverage",
        ]

        for metric_name in baseline_metric_names:
            if metric_name in METRICS_CATALOG:
                # For delta metrics, use items_with_baseline
                # For coverage, use all items
                if metric_name == "baseline_coverage":
                    stats = compute_metric_stats(items, METRICS_CATALOG[metric_name])
                else:
                    stats = compute_metric_stats(items_with_baseline, METRICS_CATALOG[metric_name])
                report.baseline_metrics[metric_name] = stats

        # Also add baseline grouped metrics by question_type
        if items_with_baseline:
            baseline_grouped = compute_grouped_metrics(
                items_with_baseline,
                "question_type",
                ["delta_ai", "delta_ped", "ge_original_ai_rate", "ge_original_ped_rate"],
            )
            baseline_grouped.group_by = "question_type (baseline)"
            report.grouped_metrics.append(baseline_grouped)

    return report


# =============================================================================
# Output Formatters
# =============================================================================

def format_report_json(report: BatchReport, include_items: bool = False) -> str:
    """Format report as JSON."""
    return json.dumps(report.to_dict(include_items), ensure_ascii=False, indent=2)


def format_report_csv(report: BatchReport) -> str:
    """Format report as CSV (detail rows)."""
    if not report.items:
        return "No items to export"

    output = []
    fieldnames = [
        "experiment_id", "unit_id", "final_decision",
        "ai_score", "ped_score", "agent5_score",
        "question_type", "question_type_inferred",
        "ablation_skip_agent", "ai_eval_success", "ped_eval_success",
        "file_path",
    ]

    # Add baseline columns if baseline is enabled
    if report.baseline_enabled:
        fieldnames.extend([
            "ai_score_original", "ped_score_original",
            "delta_ai", "delta_ped",
            "ge_original_ai", "ge_original_ped",
            "has_baseline",
        ])

    # Header
    output.append(",".join(fieldnames))

    # Rows
    for item in report.items:
        row = [
            item.experiment_id,
            item.unit_id,
            item.final_decision,
            str(item.ai_score) if item.ai_score is not None else "",
            str(item.ped_score) if item.ped_score is not None else "",
            str(item.agent5_score) if item.agent5_score is not None else "",
            item.question_type,
            str(item.question_type_inferred),
            item.ablation_skip_agent,
            str(item.ai_eval_success),
            str(item.ped_eval_success),
            item.file_path,
        ]

        # Add baseline columns if enabled
        if report.baseline_enabled:
            row.extend([
                str(item.ai_score_original) if item.ai_score_original is not None else "",
                str(item.ped_score_original) if item.ped_score_original is not None else "",
                str(item.delta_ai) if item.delta_ai is not None else "",
                str(item.delta_ped) if item.delta_ped is not None else "",
                str(item.ge_original_ai) if item.ge_original_ai is not None else "",
                str(item.ge_original_ped) if item.ge_original_ped is not None else "",
                str(item.has_baseline),
            ])

        output.append(",".join(row))

    return "\n".join(output)


def format_report_markdown(report: BatchReport) -> str:
    """Format report as human-readable Markdown."""
    lines = []
    lines.append("# Batch Analytics Report")
    lines.append("")
    lines.append(f"**Generated:** {report.generated_at}")
    lines.append(f"**Batch Directory:** {report.batch_dir}")
    lines.append(f"**Glob Pattern:** {report.glob_pattern}")
    lines.append(f"**Total Files Found:** {report.total_files_found}")
    lines.append(f"**Total Items Loaded:** {report.total_items_loaded}")
    lines.append("")

    # Core Metrics
    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Metric | Count | Valid | Mean | Median | Std | Min | Max |")
    lines.append("|--------|-------|-------|------|--------|-----|-----|-----|")

    for name, stats in report.core_metrics.items():
        mean_str = f"{stats.mean:.2f}" if stats.mean is not None else "-"
        median_str = f"{stats.median:.2f}" if stats.median is not None else "-"
        std_str = f"{stats.std:.2f}" if stats.std is not None else "-"
        min_str = f"{stats.min_val:.2f}" if stats.min_val is not None else "-"
        max_str = f"{stats.max_val:.2f}" if stats.max_val is not None else "-"
        lines.append(f"| {name} | {stats.count} | {stats.valid_count} | {mean_str} | {median_str} | {std_str} | {min_str} | {max_str} |")

    lines.append("")

    # Decision Distribution
    lines.append("## Decision Distribution")
    lines.append("")
    for decision, count in sorted(report.decision_distribution.items()):
        pct = (count / report.total_items_loaded * 100) if report.total_items_loaded > 0 else 0
        lines.append(f"- **{decision}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Question Type Distribution
    lines.append("## Question Type Distribution")
    lines.append("")
    for qtype, count in sorted(report.question_type_distribution.items()):
        pct = (count / report.total_items_loaded * 100) if report.total_items_loaded > 0 else 0
        lines.append(f"- **{qtype}**: {count} ({pct:.1f}%)")
    lines.append("")

    # Ablation Distribution
    if report.ablation_distribution:
        lines.append("## Ablation Distribution")
        lines.append("")
        for ablation, count in sorted(report.ablation_distribution.items()):
            pct = (count / report.total_items_loaded * 100) if report.total_items_loaded > 0 else 0
            lines.append(f"- **{ablation}**: {count} ({pct:.1f}%)")
        lines.append("")

    # Grouped Metrics
    for grouped in report.grouped_metrics:
        lines.append(f"## Metrics by {grouped.group_by}")
        lines.append("")

        for group_key, metrics in sorted(grouped.groups.items()):
            lines.append(f"### {group_key}")
            lines.append("")
            lines.append("| Metric | Count | Mean | Median |")
            lines.append("|--------|-------|------|--------|")

            for metric_name, stats in metrics.items():
                mean_str = f"{stats.mean:.2f}" if stats.mean is not None else "-"
                median_str = f"{stats.median:.2f}" if stats.median is not None else "-"
                lines.append(f"| {metric_name} | {stats.valid_count} | {mean_str} | {median_str} |")

            lines.append("")

    # ===== Baseline Comparison Section =====
    if report.baseline_enabled:
        lines.append("## Baseline Comparison (Original vs Generated)")
        lines.append("")
        lines.append(f"**Baseline Directory:** {report.baseline_dir}")
        lines.append(f"**Baseline Coverage:** {report.baseline_coverage:.1%} ({report.baseline_items_matched}/{report.total_items_loaded} items)")
        lines.append("")

        if report.baseline_metrics:
            lines.append("### Delta Metrics (Generated - Original)")
            lines.append("")
            lines.append("| Metric | Count | Mean | Median | Std | Min | Max |")
            lines.append("|--------|-------|------|--------|-----|-----|-----|")

            for name, stats in report.baseline_metrics.items():
                mean_str = f"{stats.mean:.2f}" if stats.mean is not None else "-"
                median_str = f"{stats.median:.2f}" if stats.median is not None else "-"
                std_str = f"{stats.std:.2f}" if stats.std is not None else "-"
                min_str = f"{stats.min_val:.2f}" if stats.min_val is not None else "-"
                max_str = f"{stats.max_val:.2f}" if stats.max_val is not None else "-"
                lines.append(f"| {name} | {stats.valid_count} | {mean_str} | {median_str} | {std_str} | {min_str} | {max_str} |")

            lines.append("")

            # Interpretation summary
            delta_ai = report.baseline_metrics.get("delta_ai")
            delta_ped = report.baseline_metrics.get("delta_ped")
            ge_ai = report.baseline_metrics.get("ge_original_ai_rate")
            ge_ped = report.baseline_metrics.get("ge_original_ped_rate")

            lines.append("### Summary Interpretation")
            lines.append("")
            if delta_ai and delta_ai.mean is not None:
                direction = "better" if delta_ai.mean > 0 else "worse" if delta_ai.mean < 0 else "same"
                lines.append(f"- **AI Score:** Generated questions are on average **{abs(delta_ai.mean):.1f} points {direction}** than originals")
            if delta_ped and delta_ped.mean is not None:
                direction = "better" if delta_ped.mean > 0 else "worse" if delta_ped.mean < 0 else "same"
                lines.append(f"- **Ped Score:** Generated questions are on average **{abs(delta_ped.mean):.1f} points {direction}** than originals")
            if ge_ai and ge_ai.mean is not None:
                lines.append(f"- **AI Closeness Rate:** {ge_ai.mean:.1%} of generated questions score >= original")
            if ge_ped and ge_ped.mean is not None:
                lines.append(f"- **Ped Closeness Rate:** {ge_ped.mean:.1%} of generated questions score >= original")
            lines.append("")

        if report.missing_baseline_unit_ids:
            lines.append("### Missing Baseline Unit IDs")
            lines.append("")
            shown_ids = report.missing_baseline_unit_ids[:20]
            lines.append(f"The following unit_ids have no baseline data: {', '.join(shown_ids)}")
            if len(report.missing_baseline_unit_ids) > 20:
                lines.append(f"... and {len(report.missing_baseline_unit_ids) - 20} more")
            lines.append("")

    # Warnings
    if report.load_warnings:
        lines.append("## Load Warnings")
        lines.append("")
        for warn in report.load_warnings[:20]:  # Limit to 20
            lines.append(f"- {warn}")
        if len(report.load_warnings) > 20:
            lines.append(f"- ... and {len(report.load_warnings) - 20} more warnings")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch analytics demo for Stage2 evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/analytics_demo.py --batch-dir outputs/selfcheck
  python tools/analytics_demo.py --batch-dir outputs/batch --out-dir outputs/analytics
  python tools/analytics_demo.py --batch-dir outputs/batch --format all --group-by question_type,ablation_skip_agent

  # With baseline comparison (delta metrics):
  python tools/analytics_demo.py --batch-dir outputs/generated --baseline-dir outputs/baseline_original
        """,
    )

    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Directory containing evaluation results (required)",
    )

    parser.add_argument(
        "--glob",
        default="**/evaluation_state.json",
        help="Glob pattern to find evaluation files (default: **/evaluation_state.json)",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for reports (default: print to stdout)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "md", "all"],
        default="md",
        help="Output format (default: md)",
    )

    parser.add_argument(
        "--group-by",
        default="question_type",
        help="Comma-separated list of fields to group by (default: question_type)",
    )

    parser.add_argument(
        "--include-items",
        action="store_true",
        help="Include per-item details in JSON output",
    )

    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory containing baseline evaluation results for delta metrics (optional)",
    )

    parser.add_argument(
        "--baseline-glob",
        default="**/evaluation_state_original.json",
        help="Glob pattern for baseline files (default: **/evaluation_state_original.json)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Parse group-by fields
    group_by = [f.strip() for f in args.group_by.split(",") if f.strip()]

    if args.verbose:
        print(f"Batch directory: {args.batch_dir}")
        print(f"Glob pattern: {args.glob}")
        print(f"Group by: {group_by}")
        if args.baseline_dir:
            print(f"Baseline directory: {args.baseline_dir}")
            print(f"Baseline glob: {args.baseline_glob}")
        print()

    # Generate report
    report = generate_batch_report(
        batch_dir=args.batch_dir,
        glob_pattern=args.glob,
        group_by=group_by,
        include_items=args.include_items or args.format in ("csv", "all"),
        baseline_dir=args.baseline_dir,
        baseline_glob=args.baseline_glob,
    )

    if args.verbose:
        print(f"Found {report.total_files_found} files, loaded {report.total_items_loaded} items")
        if report.baseline_enabled:
            print(f"Baseline coverage: {report.baseline_coverage:.1%} ({report.baseline_items_matched} items matched)")
        if report.load_warnings:
            print(f"Warnings: {len(report.load_warnings)}")
        print()

    # Output
    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        formats_to_write = ["json", "csv", "md"] if args.format == "all" else [args.format]

        for fmt in formats_to_write:
            if fmt == "json":
                content = format_report_json(report, args.include_items)
                file_path = out_path / "batch_report.json"
            elif fmt == "csv":
                content = format_report_csv(report)
                file_path = out_path / "batch_detail.csv"
            else:  # md
                content = format_report_markdown(report)
                file_path = out_path / "batch_report.md"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Written: {file_path}")
    else:
        # Print to stdout
        if args.format == "json":
            print(format_report_json(report, args.include_items))
        elif args.format == "csv":
            print(format_report_csv(report))
        else:  # md or all
            print(format_report_markdown(report))

    # Exit with appropriate code
    if report.total_items_loaded == 0:
        print("\nWarning: No evaluation items found!", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
