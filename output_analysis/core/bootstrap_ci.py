# -*- coding: utf-8 -*-
"""
bootstrap_ci.py - Bootstrap confidence interval calculation module

Supported metrics:
- Micro P/R/F1
- Macro P/R/F1 (dimension view)
- exclude_high_freq version (after excluding high-frequency dimensions)
- AI score

Parameter description:
- n_bootstrap: 1000 (number of resampling iterations)
- ci_level: 0.95 (2.5%, 97.5% percentiles)
- Resampling with replacement
- Sample size same as original data
"""

import json
import numpy as np
from typing import Dict, List, Set, Any, Optional
import pandas as pd

# Import high-frequency dimensions from config
from output_analysis.config import get_high_freq_dims

# Type aliases
BootstrapResult = Dict[str, Any]  # {mean, ci_lower, ci_upper, ci_str}


def format_ci_str(mean: float, ci_lower: float, ci_upper: float, precision: int = 3) -> str:
    """
    Format to bracket notation: 0.75 [0.72, 0.78]

    Args:
        mean: Mean value
        ci_lower: CI lower bound
        ci_upper: CI upper bound
        precision: Number of decimal places

    Returns:
        Formatted string, e.g. "0.75 [0.72, 0.78]"
    """
    if np.isnan(mean):
        return ""
    return f"{mean:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"


def _parse_dims(x) -> List[str]:
    """
    Parse dimension column (JSON string or list)

    Args:
        x: Dimension data, can be JSON string or list

    Returns:
        List of dimensions
    """
    if isinstance(x, str):
        try:
            return json.loads(x) if x else []
        except:
            return []
    elif isinstance(x, list):
        return x
    return []


def _calc_ci(values: List[float]) -> BootstrapResult:
    """
    Calculate confidence interval

    Args:
        values: List of bootstrap sample values

    Returns:
        BootstrapResult dictionary
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_str": ""}

    mean = float(np.mean(values))
    ci_lower = float(np.percentile(values, 2.5))
    ci_upper = float(np.percentile(values, 97.5))

    return {
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_str": format_ci_str(mean, ci_lower, ci_upper)
    }


def bootstrap_micro_metrics(
    per_unit_data: List[Dict],
    n_bootstrap: int = 1000,
    exclude_dims: Set[str] = None,
    seed: int = 42
) -> Dict[str, BootstrapResult]:
    """
    Calculate Bootstrap CI for Micro P/R/F1

    Micro metric calculation:
    1. Aggregate TP/FP/FN from all units
    2. Calculate P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2*P*R/(P+R)

    Bootstrap procedure:
    1. Resample per_unit_data with replacement (n_samples = len(data))
    2. Aggregate TP/FP/FN from resampled data
    3. Calculate P/R/F1
    4. Repeat n_bootstrap times
    5. Take 2.5% and 97.5% percentiles

    Args:
        per_unit_data: Per-unit data list
        n_bootstrap: Number of resampling iterations
        exclude_dims: Set of dimensions to exclude (for calculating exclude_high_freq metrics)
        seed: Random seed

    Returns:
        {
            "micro_precision": BootstrapResult,
            "micro_recall": BootstrapResult,
            "micro_f1": BootstrapResult
        }
    """
    np.random.seed(seed)
    exclude_dims = exclude_dims or set()
    n_samples = len(per_unit_data)

    if n_samples == 0:
        empty = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_str": ""}
        return {"micro_precision": empty, "micro_recall": empty, "micro_f1": empty}

    boot_p, boot_r, boot_f1 = [], [], []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        total_tp, total_fp, total_fn = 0, 0, 0
        for idx in indices:
            unit = per_unit_data[idx]
            gold = set(_parse_dims(unit.get("gold_dimensions", []))) - exclude_dims
            pred = set(_parse_dims(unit.get("predicted_dimensions", []))) - exclude_dims

            # Skip if gold is empty after excluding high-frequency (these units don't participate in calculation)
            if not gold:
                continue

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calculate metrics
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        boot_p.append(p)
        boot_r.append(r)
        boot_f1.append(f1)

    return {
        "micro_precision": _calc_ci(boot_p),
        "micro_recall": _calc_ci(boot_r),
        "micro_f1": _calc_ci(boot_f1)
    }


def bootstrap_macro_metrics_dimension_view(
    per_unit_data: List[Dict],
    n_bootstrap: int = 1000,
    exclude_dims: Set[str] = None,
    seed: int = 42
) -> Dict[str, BootstrapResult]:
    """
    Calculate Bootstrap CI for dimension-view Macro P/R/F1

    [Key] Dimension-view Macro calculation method:
    1. Calculate TP/FP/FN for each dimension separately
    2. Calculate P/R/F1 for each dimension
    3. macro_P/R/F1 = mean(P/R/F1 across all dimensions)

    Bootstrap procedure:
    1. Resample per_unit_data with replacement (n_samples = len(data))
    2. Calculate TP/FP/FN for each dimension:
       - TP: dimension in both gold and pred
       - FP: dimension in pred but not in gold
       - FN: dimension in gold but not in pred
    3. Calculate P/R/F1 for each dimension
    4. macro = mean(metrics across all dimensions)
    5. Repeat n_bootstrap times
    6. Take 2.5% and 97.5% percentiles

    Args:
        per_unit_data: Per-unit data list
        n_bootstrap: Number of resampling iterations
        exclude_dims: Set of dimensions to exclude
        seed: Random seed

    Returns:
        {
            "macro_precision": BootstrapResult,
            "macro_recall": BootstrapResult,
            "macro_f1": BootstrapResult
        }
    """
    np.random.seed(seed)
    exclude_dims = exclude_dims or set()
    n_samples = len(per_unit_data)

    if n_samples == 0:
        empty = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_str": ""}
        return {"macro_precision": empty, "macro_recall": empty, "macro_f1": empty}

    boot_p, boot_r, boot_f1 = [], [], []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Calculate TP/FP/FN for each dimension
        dim_stats: Dict[str, Dict[str, int]] = {}

        for idx in indices:
            unit = per_unit_data[idx]
            gold = set(_parse_dims(unit.get("gold_dimensions", []))) - exclude_dims
            pred = set(_parse_dims(unit.get("predicted_dimensions", []))) - exclude_dims

            # Skip if gold is empty
            if not gold:
                continue

            # Iterate over all dimensions that appear
            for d in gold | pred:
                if d not in dim_stats:
                    dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats[d]["tp"] += 1
                elif d in pred:
                    dim_stats[d]["fp"] += 1
                elif d in gold:
                    dim_stats[d]["fn"] += 1

        # Calculate P/R/F1 for each dimension
        dim_p_list, dim_r_list, dim_f1_list = [], [], []
        for d, st in dim_stats.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p_list.append(p_d)
            dim_r_list.append(r_d)
            dim_f1_list.append(f1_d)

        # Macro average
        macro_p = np.mean(dim_p_list) if dim_p_list else 0.0
        macro_r = np.mean(dim_r_list) if dim_r_list else 0.0
        macro_f1 = np.mean(dim_f1_list) if dim_f1_list else 0.0

        boot_p.append(macro_p)
        boot_r.append(macro_r)
        boot_f1.append(macro_f1)

    return {
        "macro_precision": _calc_ci(boot_p),
        "macro_recall": _calc_ci(boot_r),
        "macro_f1": _calc_ci(boot_f1)
    }


def bootstrap_ai_score(
    per_unit_data: List[Dict],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> BootstrapResult:
    """
    Calculate Bootstrap CI for AI score mean

    Args:
        per_unit_data: Per-unit data list (must contain ai_score field)
        n_bootstrap: Number of resampling iterations
        seed: Random seed

    Returns:
        BootstrapResult
    """
    np.random.seed(seed)

    # Extract valid AI scores
    scores = []
    for unit in per_unit_data:
        ai = unit.get("ai_score")
        if ai is not None and not (isinstance(ai, float) and np.isnan(ai)):
            try:
                scores.append(float(ai))
            except (ValueError, TypeError):
                pass

    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_str": ""}

    scores_arr = np.array(scores)
    n_samples = len(scores_arr)
    boot_means = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_means.append(np.mean(scores_arr[indices]))

    return _calc_ci(boot_means)


def compute_experiment_bootstrap_ci(
    per_unit_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, BootstrapResult]]:
    """
    Calculate Bootstrap CI for all experiments

    This is the main entry function that calculates CI for all metrics per experiment.

    Args:
        per_unit_df: Per-unit DataFrame (contains experiment_id, domain, gold_dimensions, etc.)
        n_bootstrap: Number of resampling iterations
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        {
            experiment_id: {
                "micro_precision": BootstrapResult,
                "micro_recall": BootstrapResult,
                "micro_f1": BootstrapResult,
                "macro_precision": BootstrapResult,
                "macro_recall": BootstrapResult,
                "macro_f1": BootstrapResult,
                "exclude_hf_micro_precision": BootstrapResult,
                "exclude_hf_micro_recall": BootstrapResult,
                "exclude_hf_micro_f1": BootstrapResult,
                "exclude_hf_macro_precision": BootstrapResult,
                "exclude_hf_macro_recall": BootstrapResult,
                "exclude_hf_macro_f1": BootstrapResult,
                "ai_score_mean": BootstrapResult
            }
        }
    """
    if per_unit_df.empty:
        return {}

    results = {}
    exp_ids = per_unit_df["experiment_id"].unique()
    total_exps = len(exp_ids)

    for i, exp_id in enumerate(exp_ids):
        if verbose and i % 10 == 0:
            print(f"       Processing experiment {i+1}/{total_exps}: {exp_id[:50]}...")

        exp_df = per_unit_df[per_unit_df["experiment_id"] == exp_id]
        per_unit_data = exp_df.to_dict("records")

        # Get domain and high-frequency dimensions
        domain = exp_df["domain"].iloc[0] if "domain" in exp_df.columns else ""
        high_freq_dims = get_high_freq_dims(domain) if domain else set()

        exp_results = {}

        # Use different seed for each experiment to increase randomness while remaining reproducible
        exp_seed = seed + i

        # 1. Normal metrics (no dimensions excluded)
        micro = bootstrap_micro_metrics(per_unit_data, n_bootstrap, exclude_dims=None, seed=exp_seed)
        macro = bootstrap_macro_metrics_dimension_view(per_unit_data, n_bootstrap, exclude_dims=None, seed=exp_seed)
        exp_results.update(micro)
        exp_results.update(macro)

        # 2. Metrics after excluding high-frequency
        if high_freq_dims:
            micro_excl = bootstrap_micro_metrics(
                per_unit_data, n_bootstrap, exclude_dims=high_freq_dims, seed=exp_seed
            )
            macro_excl = bootstrap_macro_metrics_dimension_view(
                per_unit_data, n_bootstrap, exclude_dims=high_freq_dims, seed=exp_seed
            )
            exp_results["exclude_hf_micro_precision"] = micro_excl["micro_precision"]
            exp_results["exclude_hf_micro_recall"] = micro_excl["micro_recall"]
            exp_results["exclude_hf_micro_f1"] = micro_excl["micro_f1"]
            exp_results["exclude_hf_macro_precision"] = macro_excl["macro_precision"]
            exp_results["exclude_hf_macro_recall"] = macro_excl["macro_recall"]
            exp_results["exclude_hf_macro_f1"] = macro_excl["macro_f1"]
        else:
            # If no high-frequency dimensions defined, use empty result
            empty = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_str": ""}
            for key in ["exclude_hf_micro_precision", "exclude_hf_micro_recall", "exclude_hf_micro_f1",
                       "exclude_hf_macro_precision", "exclude_hf_macro_recall", "exclude_hf_macro_f1"]:
                exp_results[key] = empty

        # 3. AI score
        exp_results["ai_score_mean"] = bootstrap_ai_score(per_unit_data, n_bootstrap, seed=exp_seed)

        results[exp_id] = exp_results

    return results


def save_bootstrap_results_csv(
    bootstrap_results: Dict[str, Dict[str, BootstrapResult]],
    output_path: str
):
    """
    Save Bootstrap results to CSV file

    Args:
        bootstrap_results: Return result from compute_experiment_bootstrap_ci
        output_path: Output file path
    """
    if not bootstrap_results:
        return

    rows = []
    for exp_id, metrics in bootstrap_results.items():
        row = {"experiment_id": exp_id}
        for metric_name, result in metrics.items():
            row[f"{metric_name}_mean"] = result.get("mean", 0.0)
            row[f"{metric_name}_ci_lower"] = result.get("ci_lower", 0.0)
            row[f"{metric_name}_ci_upper"] = result.get("ci_upper", 0.0)
            row[f"{metric_name}_ci"] = result.get("ci_str", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Bootstrap CI results saved: {output_path}")
