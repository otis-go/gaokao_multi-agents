# -*- coding: utf-8 -*-
"""
Markdown Report Generation Module

Generates human-readable Markdown format experiment reports:
- Stage1 report: Question generation results
- Stage2 report: Evaluation results summary
- Complete experiment report: Stage1 + Stage2 combined
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def _truncate_text(text: str, max_len: int = 500) -> str:
    """Truncate long text, add ellipsis marker"""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_dimensions(dimension_ids: List[str]) -> str:
    """Format dimension list as readable string"""
    if not dimension_ids:
        return "None"
    return ", ".join(dimension_ids)


def _question_type_cn(qtype: str) -> str:
    """Convert question type to readable English labels"""
    mapping = {
        "single-choice": "Single Choice",
        "multiple-choice": "Multiple Choice",
        "essay": "Subjective",
        "other": "Other",
    }
    return mapping.get(qtype, qtype or "Unknown")


def _status_cn(status: str) -> str:
    """Convert status to readable English labels"""
    mapping = {
        "success": "Success",
        "failed": "Failed",
        "skipped": "Skipped",
        "pass": "Pass",
        "error": "Error",
    }
    return mapping.get(status, status or "Unknown")


# ==================== Stage1 Report Generation ====================

def generate_stage1_md_report(
    summary_data: Dict[str, Any],
    stage1_dir: Path,
    output_path: Path,
) -> str:
    """
    Generate Stage1 Markdown report

    Args:
        summary_data: Contents of stage1_summary.json
        stage1_dir: stage1 output directory (for reading pipeline_state)
        output_path: MD report output path

    Returns:
        Generated MD file path
    """
    lines = []

    # Title
    exp_id = summary_data.get("experiment_id", "Unknown Experiment")
    lines.append(f"# Stage1 Question Generation Report")
    lines.append(f"")
    lines.append(f"**Experiment ID**: `{exp_id}`")
    lines.append(f"")
    lines.append(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")

    # Configuration information
    config = summary_data.get("config", {})
    lines.append(f"## Configuration")
    lines.append(f"")
    lines.append(f"| Configuration | Value |")
    lines.append(f"|--------|-----|")
    lines.append(f"| Generator Model | {config.get('generator_model', 'Unknown')} |")
    lines.append(f"| Dimension Mode | {config.get('dim_mode', 'Unknown')} |")
    lines.append(f"| Prompt Level | {config.get('prompt_level', 'Unknown')} |")
    if config.get("stage1_skip_agent"):
        lines.append(f"| Skip Agent | {config.get('stage1_skip_agent')} |")
    lines.append(f"")

    # Statistics summary
    lines.append(f"## Generation Statistics")
    lines.append(f"")
    gen_success = summary_data.get("generation_success", 0)
    gen_failed = summary_data.get("generation_failed", 0)
    total = gen_success + gen_failed
    lines.append(f"- **Successfully Generated**: {gen_success} / {total}")
    lines.append(f"- **Generation Failed**: {gen_failed}")
    lines.append(f"- **Stage2Record Count**: {summary_data.get('stage2_record_count', 0)}")
    lines.append(f"")

    # Question type distribution
    qt_dist = summary_data.get("question_type_distribution", {})
    if qt_dist:
        lines.append(f"### Question Type Distribution")
        lines.append(f"")
        lines.append(f"| Question Type | Count |")
        lines.append(f"|------|------|")
        for qtype, count in qt_dist.items():
            lines.append(f"| {_question_type_cn(qtype)} | {count} |")
        lines.append(f"")

    # Results list
    results = summary_data.get("results", [])
    if results:
        lines.append(f"## Question List Summary")
        lines.append(f"")
        lines.append(f"| unit_id | Type | Status | Dim Count |")
        lines.append(f"|---------|------|------|--------|")
        for r in results:
            unit_id = r.get("unit_id", "?")
            qtype = _question_type_cn(r.get("question_type", ""))
            status = _status_cn(r.get("stage1_status", ""))
            dim_count = r.get("dimension_count", 0)
            lines.append(f"| {unit_id} | {qtype} | {status} | {dim_count} |")
        lines.append(f"")

    # Question details
    lines.append(f"## Question Details")
    lines.append(f"")

    for r in results:
        unit_id = r.get("unit_id", "?")
        if r.get("stage1_status") != "success":
            lines.append(f"### Unit {unit_id} - Generation Failed")
            lines.append(f"")
            lines.append(f"Status: {_status_cn(r.get('stage1_status', ''))}")
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")
            continue

        # Try to read pipeline_state to get detailed content
        question_body = ""
        answer_text = ""
        analysis = ""
        material_text = ""
        dimension_ids = r.get("dimension_ids", [])

        # Find pipeline_state file
        unit_dir = stage1_dir / f"unit_{unit_id}"
        if unit_dir.exists():
            pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
            if pipeline_files:
                try:
                    with open(pipeline_files[0], "r", encoding="utf-8") as f:
                        ps = json.load(f)

                    # Extract stage2_record or agent4_output
                    s2r = ps.get("stage2_record", {})
                    core_input = s2r.get("core_input", {})

                    if core_input:
                        question_body = core_input.get("question_body", "")
                        answer_text = core_input.get("answer_text", "")
                        analysis = core_input.get("analysis", "")
                        material_text = core_input.get("material_text", "")
                        dimension_ids = core_input.get("dimension_ids", dimension_ids)
                    else:
                        # Extract from agent4_output
                        a4 = ps.get("agent4_output", {})
                        question_body = a4.get("question", "")
                        answer_text = a4.get("solution", "")
                        analysis = a4.get("analysis", "")

                        a1 = ps.get("agent1_output", {})
                        material_text = a1.get("material_text", "")

                except Exception:
                    pass

        lines.append(f"### Unit {unit_id} ({_question_type_cn(r.get('question_type', ''))})")
        lines.append(f"")

        # Associated dimensions
        lines.append(f"**Associated Dimensions**: {_format_dimensions(dimension_ids)}")
        lines.append(f"")

        # Material
        if material_text:
            lines.append(f"**Material**:")
            lines.append(f"")
            lines.append(f"> {_truncate_text(material_text, 800)}")
            lines.append(f"")

        # Question
        if question_body:
            lines.append(f"**Question**:")
            lines.append(f"")
            lines.append(f"```")
            lines.append(question_body.strip())
            lines.append(f"```")
            lines.append(f"")

        # Answer
        if answer_text:
            lines.append(f"**Answer**:")
            lines.append(f"")
            lines.append(f"```")
            lines.append(answer_text.strip())
            lines.append(f"```")
            lines.append(f"")

        # Analysis
        if analysis:
            lines.append(f"**Analysis**:")
            lines.append(f"")
            lines.append(f"{_truncate_text(analysis, 600)}")
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"")

    # Write to file
    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)


# ==================== Stage2/Complete Experiment Report Generation ====================

def generate_experiment_md_report(
    summary_data: Dict[str, Any],
    stage1_dir: Path,
    stage2_dir: Path,
    output_path: Path,
) -> str:
    """
    Generate complete experiment Markdown report (Stage1 + Stage2)

    Args:
        summary_data: Contents of summary.json
        stage1_dir: stage1 output directory
        stage2_dir: stage2 output directory
        output_path: MD report output path

    Returns:
        Generated MD file path
    """
    lines = []

    # Title
    exp_id = summary_data.get("experiment_id", "Unknown Experiment")
    lines.append(f"# Experiment Report")
    lines.append(f"")
    lines.append(f"**Experiment ID**: `{exp_id}`")
    lines.append(f"")
    lines.append(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")

    # Configuration information
    config = summary_data.get("config", {})
    lines.append(f"## Configuration")
    lines.append(f"")
    lines.append(f"| Configuration | Value |")
    lines.append(f"|--------|-----|")
    lines.append(f"| Generator Model | {config.get('generator_model', 'Unknown')} |")
    lines.append(f"| Dimension Mode | {config.get('dim_mode', 'Unknown')} |")
    lines.append(f"| Prompt Level | {config.get('prompt_level', 'Unknown')} |")
    if config.get("stage1_skip_agent"):
        lines.append(f"| Skip Agent | {config.get('stage1_skip_agent')} |")
    lines.append(f"")

    # Overall statistics
    lines.append(f"## Overall Statistics")
    lines.append(f"")
    gen_success = summary_data.get("generation_success", 0)
    eval_success = summary_data.get("evaluation_success", 0)
    total = summary_data.get("run_total", gen_success)

    lines.append(f"| Metric | Value |")
    lines.append(f"|------|------|")
    lines.append(f"| Total Questions | {total} |")
    lines.append(f"| Stage1 Success | {gen_success} |")
    lines.append(f"| Stage2 Success | {eval_success} |")
    ai_score = summary_data.get('avg_ai_score')
    ped_score = summary_data.get('avg_ped_score')
    ai_str = f"{ai_score:.2f}" if ai_score is not None else "N/A"
    ped_str = f"{ped_score:.2f}" if ped_score is not None else "N/A"
    lines.append(f"| AI Average Score | {ai_str} |")
    lines.append(f"| Pedagogical Average Score | {ped_str} |")
    lines.append(f"")

    # Question type distribution
    qt_dist = summary_data.get("question_type_distribution", {})
    if qt_dist:
        lines.append(f"### Question Type Distribution")
        lines.append(f"")
        lines.append(f"| Question Type | Count |")
        lines.append(f"|------|------|")
        for qtype, count in qt_dist.items():
            lines.append(f"| {_question_type_cn(qtype)} | {count} |")
        lines.append(f"")

    # Average scores by question type
    ai_by_qt = summary_data.get("avg_ai_score_by_question_type", {})
    ped_by_qt = summary_data.get("ped_round_metrics_by_question_type", {})
    if ai_by_qt or ped_by_qt:
        lines.append(f"### Average Scores by Question Type")
        lines.append(f"")
        lines.append(f"| Question Type | AI Average | Pedagogical F1 |")
        lines.append(f"|------|----------|----------|")
        all_qtypes = set(ai_by_qt.keys()) | set(ped_by_qt.keys())
        for qt in all_qtypes:
            ai_score = ai_by_qt.get(qt)
            ped_f1 = ped_by_qt.get(qt, {}).get("macro", {}).get("f1")
            ai_str = f"{ai_score:.2f}" if ai_score is not None else "N/A"
            ped_str = f"{ped_f1:.4f}" if ped_f1 is not None else "N/A"
            lines.append(f"| {_question_type_cn(qt)} | {ai_str} | {ped_str} |")
        lines.append(f"")

    # Pedagogical round metrics
    ped_metrics = summary_data.get("ped_round_metrics", {})
    if ped_metrics:
        lines.append(f"### Pedagogical Evaluation Metrics")
        lines.append(f"")
        micro = ped_metrics.get("micro", {})
        macro = ped_metrics.get("macro", {})
        lines.append(f"| Metric Type | Precision | Recall | F1 |")
        lines.append(f"|----------|-----------|--------|-----|")
        if micro:
            lines.append(f"| Micro | {micro.get('precision', 0):.3f} | {micro.get('recall', 0):.3f} | {micro.get('f1', 0):.3f} |")
        if macro:
            lines.append(f"| Macro | {macro.get('precision', 0):.3f} | {macro.get('recall', 0):.3f} | {macro.get('f1', 0):.3f} |")
        lines.append(f"")

    # Results list
    results = summary_data.get("results", [])
    if results:
        lines.append(f"## Question Evaluation Summary")
        lines.append(f"")
        lines.append(f"| unit_id | Type | AI Score | Ped Score | Decision |")
        lines.append(f"|---------|------|------|----------|------|")
        for r in results:
            unit_id = r.get("unit_id", "?")
            qtype = _question_type_cn(r.get("question_type", ""))
            ai_score = r.get("ai_overall_score", 0)
            ped_score = r.get("ped_overall_score", 0)
            decision = _status_cn(r.get("final_decision", ""))

            ai_str = f"{ai_score:.1f}" if ai_score else "-"
            ped_str = f"{ped_score:.1f}" if ped_score else "-"
            lines.append(f"| {unit_id} | {qtype} | {ai_str} | {ped_str} | {decision} |")
        lines.append(f"")

    # Question details
    lines.append(f"## Question Details")
    lines.append(f"")

    for r in results:
        unit_id = r.get("unit_id", "?")

        # Basic information
        lines.append(f"### Unit {unit_id} ({_question_type_cn(r.get('question_type', ''))})")
        lines.append(f"")

        # Scoring information
        ai_score = r.get("ai_overall_score", 0)
        ped_score = r.get("ped_overall_score", 0)
        decision = r.get("final_decision", "")

        lines.append(f"**Evaluation Results**:")
        lines.append(f"- AI Overall Score: {ai_score:.2f}" if ai_score else "- AI Overall Score: -")
        lines.append(f"- Pedagogical Overall Score: {ped_score:.2f}" if ped_score else "- Pedagogical Overall Score: -")
        lines.append(f"- Final Decision: {_status_cn(decision)}")
        lines.append(f"")

        # Pedagogical metrics
        ped_m = r.get("ped_metrics", {})
        if ped_m:
            lines.append(f"**Pedagogical Metrics**: F1={ped_m.get('f1', 0):.3f}, Precision={ped_m.get('precision', 0):.3f}, Recall={ped_m.get('recall', 0):.3f}")
            lines.append(f"")

        # Try to read detailed content
        question_body = ""
        answer_text = ""
        analysis = ""
        material_text = ""
        dimension_ids = []

        # Read from pipeline_state
        unit_dir = stage1_dir / f"unit_{unit_id}"
        if unit_dir.exists():
            pipeline_files = list(unit_dir.glob("pipeline_state_*.json"))
            if pipeline_files:
                try:
                    with open(pipeline_files[0], "r", encoding="utf-8") as f:
                        ps = json.load(f)

                    s2r = ps.get("stage2_record", {})
                    core_input = s2r.get("core_input", {})

                    if core_input:
                        question_body = core_input.get("question_body", "")
                        answer_text = core_input.get("answer_text", "")
                        analysis = core_input.get("analysis", "")
                        material_text = core_input.get("material_text", "")
                        dimension_ids = core_input.get("dimension_ids", [])
                    else:
                        a4 = ps.get("agent4_output", {})
                        question_body = a4.get("question", "")
                        answer_text = a4.get("solution", "")
                        analysis = a4.get("analysis", "")

                        a1 = ps.get("agent1_output", {})
                        material_text = a1.get("material_text", "")

                except Exception:
                    pass

        # Supplement dimension evaluation details from evaluation_state
        eval_details = {}
        eval_state_path = stage2_dir / f"unit_{unit_id}" / "evaluation_state.json"
        if eval_state_path.exists():
            try:
                with open(eval_state_path, "r", encoding="utf-8") as f:
                    es = json.load(f)

                # Pedagogical evaluation details
                ped_eval = es.get("pedagogical_eval", {})
                if ped_eval.get("success"):
                    ped_result = ped_eval.get("result", {})
                    eval_details["pedagogical"] = ped_result.get("dimension_details", {})

                # AI evaluation details
                ai_eval = es.get("ai_eval", {})
                if ai_eval.get("success"):
                    ai_result = ai_eval.get("result", {})
                    eval_details["ai_centric"] = ai_result.get("dimension_details", {})

            except Exception:
                pass

        # Associated dimensions
        if dimension_ids:
            lines.append(f"**Associated Dimensions**: {_format_dimensions(dimension_ids)}")
            lines.append(f"")

        # Material (excerpt)
        if material_text:
            lines.append(f"**Material** (excerpt):")
            lines.append(f"")
            lines.append(f"> {_truncate_text(material_text, 300)}")
            lines.append(f"")

        # Question
        if question_body:
            lines.append(f"**Question**:")
            lines.append(f"")
            lines.append(f"```")
            lines.append(question_body.strip())
            lines.append(f"```")
            lines.append(f"")

        # Answer
        if answer_text:
            lines.append(f"**Answer**:")
            lines.append(f"")
            lines.append(f"```")
            lines.append(_truncate_text(answer_text.strip(), 500))
            lines.append(f"```")
            lines.append(f"")

        # Analysis
        if analysis:
            lines.append(f"**Analysis**:")
            lines.append(f"")
            lines.append(f"{_truncate_text(analysis, 400)}")
            lines.append(f"")

        # Dimension evaluation details
        if eval_details.get("pedagogical"):
            lines.append(f"**Pedagogical Dimension Evaluation**:")
            lines.append(f"")
            lines.append(f"| Dimension | Score | Hit Level |")
            lines.append(f"|------|------|----------|")
            for dim_name, dim_info in eval_details["pedagogical"].items():
                score = dim_info.get("aggregated_score", 0)
                hit_level = dim_info.get("hit_level", "")
                lines.append(f"| {_truncate_text(dim_name, 30)} | {score:.1f} | {hit_level} |")
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"")

    # High variance dimension summary
    hv_summary = summary_data.get("high_variance_summary", {})
    if hv_summary.get("has_high_variance"):
        lines.append(f"## High Variance Dimension Warning")
        lines.append(f"")
        lines.append(f"The following dimensions have significant disagreement in multi-model evaluation (score difference >= 50 points):")
        lines.append(f"")
        lines.append(f"- Affected questions: {hv_summary.get('total_count', 0)}")
        lines.append(f"- AI evaluation high variance: {len(hv_summary.get('ai_centric', []))} items")
        lines.append(f"- Pedagogical high variance: {len(hv_summary.get('pedagogical', []))} items")
        lines.append(f"")

    # Write to file
    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)


# ==================== Convenience Functions ====================

def generate_reports_from_summary(
    summary_path: Path,
    output_dir: Path,
    is_stage1_only: bool = False,
) -> Dict[str, str]:
    """
    Auto-generate MD report from summary.json

    Args:
        summary_path: Path to summary.json or stage1_summary.json
        output_dir: Output directory
        is_stage1_only: Whether in stage1-only mode

    Returns:
        {"md_report": "Generated MD file path"}
    """
    if not summary_path.exists():
        return {"error": f"Summary file not found: {summary_path}"}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"

    if is_stage1_only:
        md_path = output_dir / "stage1_report.md"
        result_path = generate_stage1_md_report(
            summary_data=summary_data,
            stage1_dir=stage1_dir,
            output_path=md_path,
        )
    else:
        md_path = output_dir / "experiment_report.md"
        result_path = generate_experiment_md_report(
            summary_data=summary_data,
            stage1_dir=stage1_dir,
            stage2_dir=stage2_dir,
            output_path=md_path,
        )

    return {"md_report": result_path}


# ==================== PRF Statistics Printing (similar to recalc_prf_with_voted_gold.py style) ====================

def get_year_group(year: Optional[int]) -> str:
    """Return group label based on year"""
    if year is None:
        return "Unknown Year"
    if 2016 <= year <= 2020:
        return "2016-2020"
    elif 2021 <= year <= 2025:
        return "2021-2025"
    else:
        return f"Other({year})"


def normalize_question_type_for_stats(qtype: str) -> str:
    """Normalize question type name for statistics"""
    # Keep Chinese strings as data keys
    if qtype in ["选择题", "single-choice"]:
        return "Multiple Choice"
    elif qtype in ["简答题", "essay"]:
        return "Short Answer"
    else:
        return "Other"


def compute_group_prf_metrics(
    metrics_list: List[Dict[str, Any]],
    exclude_dims: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Compute P/R/F1 metrics for a group of records

    Args:
        metrics_list: Metrics list per question, each element contains tp, fp, fn, precision, recall, f1
        exclude_dims: High-frequency dimension set to exclude (will recalculate if provided)

    Returns:
        Dictionary with micro/macro metrics: {micro: {precision, recall, f1, ...}, macro: {...}, count: N}
    """
    if not metrics_list:
        return {
            "count": 0,
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    # If excluding high-frequency dimensions, recalculate TP/FP/FN
    if exclude_dims:
        from src.evaluation.pedagogical_eval import calculate_prf

        new_metrics_list = []
        for m in metrics_list:
            gold = m.get("gold_dimensions", [])
            pred = m.get("predicted_dimensions", [])

            # Check if any gold dimensions remain after excluding high-frequency ones
            gold_after_excl = set(gold) - exclude_dims
            if not gold_after_excl:
                continue  # Skip questions that only have high-frequency dimensions

            tp, fp, fn, p, r, f1 = calculate_prf(gold, pred, exclude_dims=exclude_dims)
            new_metrics_list.append({
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f1,
            })
        metrics_list = new_metrics_list

        if not metrics_list:
            return {
                "count": 0,
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

    # Micro average
    total_tp = sum(m.get("tp", 0) for m in metrics_list)
    total_fp = sum(m.get("fp", 0) for m in metrics_list)
    total_fn = sum(m.get("fn", 0) for m in metrics_list)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro average
    n = len(metrics_list)
    macro_p = sum(m.get("precision", 0) for m in metrics_list) / n
    macro_r = sum(m.get("recall", 0) for m in metrics_list) / n
    macro_f1 = sum(m.get("f1", 0) for m in metrics_list) / n

    return {
        "count": n,
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        },
    }


def print_prf_summary(
    metrics_list: List[Dict[str, Any]],
    dim_mode: str = "gk",
    title: str = "",
    show_by_year: bool = True,
    show_by_qtype: bool = True,
    show_cross: bool = True,
    show_exclude_high_freq: bool = True,
) -> Dict[str, Any]:
    """
    Print PRF statistics summary in recalc_prf_with_voted_gold.py style

    Args:
        metrics_list: Metrics list per question, each element should contain:
            - tp, fp, fn, precision, recall, f1
            - gold_dimensions, predicted_dimensions
            - question_type (Multiple Choice/Short Answer)
            - year (e.g., 2020)
            - unit_id
        dim_mode: Dimension mode (gk/cs), determines high-frequency dimensions
        title: Report title
        show_by_year: Whether to display year-based grouping
        show_by_qtype: Whether to display question-type grouping
        show_cross: Whether to display question type x year cross-tabulation
        show_exclude_high_freq: Whether to show statistics excluding high-frequency dimensions

    Returns:
        Complete statistics result dictionary
    """
    from collections import defaultdict
    from src.evaluation.pedagogical_eval import get_high_freq_dims_by_mode

    # Filter out questions without dimensions
    valid_metrics = []
    skipped_no_dims = 0
    for m in metrics_list:
        gold = m.get("gold_dimensions", [])
        if not gold:
            skipped_no_dims += 1
            continue
        valid_metrics.append(m)

    if not valid_metrics:
        print(f"\n{'=' * 80}")
        print(f"[{dim_mode.upper()} Dimension Evaluation Summary] No valid data")
        print(f"{'=' * 80}")
        return {"error": "no_valid_data", "skipped_no_dims": skipped_no_dims}

    # Get high-frequency dimensions
    high_freq_dims = get_high_freq_dims_by_mode(dim_mode) if show_exclude_high_freq else set()

    # ========== Grouped Statistics ==========
    by_qtype: Dict[str, List[Dict]] = defaultdict(list)
    by_year_group: Dict[str, List[Dict]] = defaultdict(list)
    by_qtype_year: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for m in valid_metrics:
        qtype = normalize_question_type_for_stats(m.get("question_type", "Other"))
        year = m.get("year")
        year_group = get_year_group(year)

        by_qtype[qtype].append(m)
        by_year_group[year_group].append(m)
        by_qtype_year[qtype][year_group].append(m)

    # ========== Calculate Group Metrics ==========
    overall_metrics = compute_group_prf_metrics(valid_metrics)
    qtype_metrics = {qt: compute_group_prf_metrics(ms) for qt, ms in by_qtype.items()}
    year_group_metrics = {yg: compute_group_prf_metrics(ms) for yg, ms in by_year_group.items()}
    cross_metrics = {
        qt: {yg: compute_group_prf_metrics(ms) for yg, ms in yg_dict.items()}
        for qt, yg_dict in by_qtype_year.items()
    }

    # Statistics after excluding high-frequency dimensions
    excl_overall = None
    excl_by_qtype = {}
    excl_by_year_group = {}
    skipped_only_high_freq = 0

    if high_freq_dims:
        # Calculate how many questions will be skipped (only contain high-frequency dimensions)
        for m in valid_metrics:
            gold = m.get("gold_dimensions", [])
            if not (set(gold) - high_freq_dims):
                skipped_only_high_freq += 1

        excl_overall = compute_group_prf_metrics(valid_metrics, exclude_dims=high_freq_dims)
        excl_by_qtype = {qt: compute_group_prf_metrics(ms, exclude_dims=high_freq_dims) for qt, ms in by_qtype.items()}
        excl_by_year_group = {yg: compute_group_prf_metrics(ms, exclude_dims=high_freq_dims) for yg, ms in by_year_group.items()}

    # ========== Print Output ==========
    mode_label = dim_mode.upper()
    total_count = len(valid_metrics)

    print(f"\n{'=' * 80}")
    if title:
        print(f"[{mode_label} Dimension Evaluation Summary] {title}")
    else:
        print(f"[{mode_label} Dimension Evaluation Summary] Total {total_count} questions")
    if skipped_no_dims > 0:
        print(f"  (Skipped questions without dimensions: {skipped_no_dims})")
    print(f"{'=' * 80}")

    # Overall metrics
    m = overall_metrics["micro"]
    mac = overall_metrics["macro"]
    print(f"\n[Overall Metrics]")
    print(f"  Micro: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    print(f"  Macro: P={mac['precision']:.4f}, R={mac['recall']:.4f}, F1={mac['f1']:.4f}")

    # Grouped by question type
    if show_by_qtype and qtype_metrics:
        print(f"\n[Grouped by Question Type]")
        print(f"{'Type':<12} {'Count':<6} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10} {'Macro-F1':<10}")
        print("-" * 70)
        for qtype in ["Multiple Choice", "Short Answer", "Other"]:
            if qtype in qtype_metrics:
                metrics = qtype_metrics[qtype]
                m = metrics["micro"]
                mac = metrics["macro"]
                cnt = metrics["count"]
                print(f"{qtype:<12} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}     {mac['f1']:.4f}")

    # Grouped by year range
    if show_by_year and year_group_metrics:
        print(f"\n[Grouped by Year Range]")
        print(f"{'Year Range':<12} {'Count':<6} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10} {'Macro-F1':<10}")
        print("-" * 70)
        for yg in ["2016-2020", "2021-2025", "Unknown Year"]:
            if yg in year_group_metrics:
                metrics = year_group_metrics[yg]
                m = metrics["micro"]
                mac = metrics["macro"]
                cnt = metrics["count"]
                print(f"{yg:<12} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}     {mac['f1']:.4f}")

    # Question type × year cross grouping
    if show_cross and cross_metrics:
        for yg in ["2016-2020", "2021-2025"]:
            has_data = any(yg in cross_metrics.get(qt, {}) for qt in ["Multiple Choice", "Short Answer"])
            if not has_data:
                continue
            print(f"\n[{yg} Year Range]")
            print(f"{'Type':<12} {'Count':<6} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10} {'Macro-F1':<10}")
            print("-" * 70)
            for qtype in ["Multiple Choice", "Short Answer"]:
                if qtype in cross_metrics and yg in cross_metrics[qtype]:
                    metrics = cross_metrics[qtype][yg]
                    m = metrics["micro"]
                    mac = metrics["macro"]
                    cnt = metrics["count"]
                    print(f"{qtype:<12} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}     {mac['f1']:.4f}")
            # Year range subtotal
            if yg in year_group_metrics:
                metrics = year_group_metrics[yg]
                m = metrics["micro"]
                mac = metrics["macro"]
                cnt = metrics["count"]
                print("-" * 70)
                print(f"{'Subtotal':<12} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}     {mac['f1']:.4f}")

    # Statistics after excluding high-frequency dimensions
    if show_exclude_high_freq and high_freq_dims and excl_overall and excl_overall["count"] > 0:
        print(f"\n{'=' * 80}")
        print(f"[After Excluding High-Frequency Dimensions] Excluded {sorted(list(high_freq_dims))}")
        print(f"  Valid questions: {excl_overall['count']}, Skipped (only high-freq): {skipped_only_high_freq}")
        m = excl_overall["micro"]
        print(f"  Micro: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

        # By question type (after excluding high-freq)
        if excl_by_qtype:
            print(f"\n  [By Question Type]")
            print(f"  {'Type':<10} {'Count':<6} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10}")
            print(f"  " + "-" * 55)
            for qtype in ["Multiple Choice", "Short Answer"]:
                if qtype in excl_by_qtype and excl_by_qtype[qtype]["count"] > 0:
                    metrics = excl_by_qtype[qtype]
                    m = metrics["micro"]
                    cnt = metrics["count"]
                    print(f"  {qtype:<10} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}")

        # By year (after excluding high-freq)
        if excl_by_year_group:
            print(f"\n  [By Year Range]")
            print(f"  {'Year Range':<10} {'Count':<6} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10}")
            print(f"  " + "-" * 55)
            for yg in ["2016-2020", "2021-2025"]:
                if yg in excl_by_year_group and excl_by_year_group[yg]["count"] > 0:
                    metrics = excl_by_year_group[yg]
                    m = metrics["micro"]
                    cnt = metrics["count"]
                    print(f"  {yg:<10} {cnt:<6} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}")

    print(f"\n{'=' * 80}")

    # Return complete results
    return {
        "total_questions": total_count,
        "skipped_no_dims": skipped_no_dims,
        "overall": overall_metrics,
        "by_question_type": qtype_metrics,
        "by_year_group": year_group_metrics,
        "cross_qtype_year": cross_metrics,
        # [2026-01 Refactoring] Rename exclude_high_freq -> metrics_rare_only
        "metrics_rare_only": {
            "excluded_dims": sorted(list(high_freq_dims)) if high_freq_dims else [],
            "note": "Statistics for rare dimensions only (after excluding high-frequency dimensions)",
            "overall": excl_overall,
            "by_question_type": excl_by_qtype,
            "by_year_group": excl_by_year_group,
            "skipped_only_high_freq": skipped_only_high_freq,
        } if high_freq_dims else None,
    }


__all__ = [
    "generate_stage1_md_report",
    "generate_experiment_md_report",
    "generate_reports_from_summary",
    "print_prf_summary",
    "get_year_group",
    "normalize_question_type_for_stats",
    "compute_group_prf_metrics",
]
