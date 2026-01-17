#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 Stage2 评估结果 - 用于中断恢复后的统计补全

读取指定实验目录下所有 stage2/unit_*/evaluation_state.json 文件，
提取 ai_eval 和 pedagogical_eval 的 overall_score，计算平均值等统计信息。

用法:
    python scripts/compute_stage2_stats.py <实验目录>

示例:
    python scripts/compute_stage2_stats.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def extract_scores_from_evaluation_state(eval_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 evaluation_state.json 提取关键评分信息

    返回:
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

    # 提取 AI 评估分数
    ai_eval = eval_state.get("ai_eval", {})
    if ai_eval and ai_eval.get("success"):
        ai_result = ai_eval.get("result", {})
        if ai_result:
            result["ai_overall_score"] = ai_result.get("overall_score")

    # 兼容旧结构 ai_eval_result
    if result["ai_overall_score"] is None:
        ai_eval_result = eval_state.get("ai_eval_result", {})
        if isinstance(ai_eval_result, dict):
            result["ai_overall_score"] = ai_eval_result.get("overall_score")

    # 提取教育学评估分数
    ped_eval = eval_state.get("pedagogical_eval", {})
    if ped_eval and ped_eval.get("success"):
        ped_result = ped_eval.get("result", {})
        if ped_result:
            result["ped_overall_score"] = ped_result.get("overall_score")

    # 兼容旧结构 pedagogical_eval_result
    if result["ped_overall_score"] is None:
        ped_eval_result = eval_state.get("pedagogical_eval_result", {})
        if isinstance(ped_eval_result, dict):
            result["ped_overall_score"] = ped_eval_result.get("overall_score")
        elif hasattr(ped_eval_result, "overall_score"):
            result["ped_overall_score"] = ped_eval_result.overall_score

    # 提取题型信息
    input_data = eval_state.get("input", {})
    if input_data:
        result["question_type"] = input_data.get("question_type")
        result["material_type"] = input_data.get("material_type")

    return result


def compute_stage2_stats(exp_dir: Path) -> Dict[str, Any]:
    """
    计算实验目录下所有 stage2 评估的统计信息
    """
    stage2_dir = exp_dir / "stage2"
    if not stage2_dir.exists():
        raise FileNotFoundError(f"stage2 目录不存在: {stage2_dir}")

    # 收集所有评估结果
    all_results: List[Dict[str, Any]] = []
    ai_scores: List[float] = []
    ped_scores: List[float] = []

    # 按题型分组的分数
    score_by_qtype = {
        "single-choice": {"ai": [], "ped": []},
        "essay": {"ai": [], "ped": []},
        "other": {"ai": [], "ped": []},
    }

    # 统计计数
    total_units = 0
    completed_count = 0
    pass_count = 0
    fail_count = 0
    error_count = 0

    # 遍历所有 unit_* 目录
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

            # 统计完成状态
            if scores["current_stage"] == "completed":
                completed_count += 1

                # 统计 pass/fail
                if scores["final_decision"] == "pass":
                    pass_count += 1
                elif scores["final_decision"] == "fail":
                    fail_count += 1

                # 收集分数
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
            print(f"[警告] 解析 {eval_file} 失败: {e}")
            error_count += 1

    # 计算平均值的辅助函数
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        avg = _avg(xs)
        return (sum((x - avg) ** 2 for x in xs) / len(xs)) ** 0.5

    def _min_max(xs: List[float]) -> tuple:
        return (min(xs), max(xs)) if xs else (0.0, 0.0)

    # 构建统计结果
    stats = {
        "experiment_id": exp_dir.name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),

        # 基础统计
        "total_units": total_units,
        "completed_count": completed_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "pass_rate": pass_count / completed_count if completed_count > 0 else 0.0,

        # AI 评分统计
        "ai_score_count": len(ai_scores),
        "avg_ai_score": _avg(ai_scores),
        "std_ai_score": _std(ai_scores),
        "min_ai_score": _min_max(ai_scores)[0],
        "max_ai_score": _min_max(ai_scores)[1],

        # 教育学评分统计
        "ped_score_count": len(ped_scores),
        "avg_ped_score": _avg(ped_scores),
        "std_ped_score": _std(ped_scores),
        "min_ped_score": _min_max(ped_scores)[0],
        "max_ped_score": _min_max(ped_scores)[1],

        # 分题型统计
        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_by_qtype.items()},
        "avg_ped_score_by_question_type": {k: _avg(v["ped"]) for k, v in score_by_qtype.items()},
        "count_by_question_type": {k: len(v["ai"]) for k, v in score_by_qtype.items()},

        # 详细结果
        "results": all_results,
    }

    return stats


def print_summary(stats: Dict[str, Any]):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("Stage2 评估统计报告")
    print("=" * 80)
    print(f"  实验 ID: {stats['experiment_id']}")
    print(f"  统计时间: {stats['timestamp']}")
    print("-" * 80)
    print(f"  总 unit 数: {stats['total_units']}")
    print(f"  评估完成数: {stats['completed_count']}")
    print(f"  评估通过数: {stats['pass_count']}")
    print(f"  评估失败数: {stats['fail_count']}")
    print(f"  评估错误数: {stats['error_count']}")
    print(f"  通过率: {stats['pass_rate']*100:.1f}%")
    print("-" * 80)
    print(f"  AI 评分统计:")
    print(f"    有效分数数量: {stats['ai_score_count']}")
    print(f"    平均分: {stats['avg_ai_score']:.2f}")
    print(f"    标准差: {stats['std_ai_score']:.2f}")
    print(f"    最小值: {stats['min_ai_score']:.2f}")
    print(f"    最大值: {stats['max_ai_score']:.2f}")
    print("-" * 80)
    print(f"  教育学评分统计:")
    print(f"    有效分数数量: {stats['ped_score_count']}")
    print(f"    平均分: {stats['avg_ped_score']:.2f}")
    print(f"    标准差: {stats['std_ped_score']:.2f}")
    print(f"    最小值: {stats['min_ped_score']:.2f}")
    print(f"    最大值: {stats['max_ped_score']:.2f}")
    print("-" * 80)
    print(f"  分题型统计:")
    for qt in ["single-choice", "essay", "other"]:
        count = stats['count_by_question_type'].get(qt, 0)
        if count > 0:
            ai_avg = stats['avg_ai_score_by_question_type'].get(qt, 0)
            ped_avg = stats['avg_ped_score_by_question_type'].get(qt, 0)
            print(f"    {qt}: 数量={count}, AI均分={ai_avg:.2f}, Ped均分={ped_avg:.2f}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/compute_stage2_stats.py <实验目录>")
        print("示例: python scripts/compute_stage2_stats.py outputs/EXP_BASELINE_gk+cs_C_20251209_213427")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"错误: 目录不存在 - {exp_dir}")
        sys.exit(1)

    print(f"[INFO] 正在统计实验目录: {exp_dir}")

    try:
        stats = compute_stage2_stats(exp_dir)

        # 打印摘要
        print_summary(stats)

        # 保存统计结果到 JSON
        output_file = exp_dir / "stage2_stats_summary.json"
        with open(output_file, "w", encoding="utf-8") as f:
            # 不保存详细结果到 JSON，避免文件过大
            stats_to_save = {k: v for k, v in stats.items() if k != "results"}
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] 统计结果已保存到: {output_file}")

        # 同时保存详细结果
        results_file = exp_dir / "stage2_results_detail.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stats["results"], f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] 详细结果已保存到: {results_file}")

    except Exception as e:
        print(f"[ERROR] 统计失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
