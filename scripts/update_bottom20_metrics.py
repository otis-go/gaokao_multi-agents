#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新所有实验的 summary.json，添加后20%低质量题目的统计指标

新增字段 bottom_20_metrics：
- overall: 总体后20%题目的 recall, precision, ai_avg
- single_choice: 选择题后20%的 recall, precision, ai_avg
- essay: 主观题后20%的 recall, precision, ai_avg

使用方法：
    python scripts/update_bottom20_metrics.py [--outputs-dir PATH]
"""

import json
import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse


def calculate_bottom_20_metrics(results: List[Dict], dim_mode: str) -> Dict[str, Any]:
    """
    计算后20%低质量题目的指标

    Args:
        results: summary.json 中的 results 列表
        dim_mode: 维度模式 (gk/gk_only/cs/cs_only/gk+cs)

    Returns:
        包含 overall, single_choice, essay 三类指标的字典

    【2026-01 更新】
    - 当 AI 评估被禁用（ai_overall_score 全为 None）时，使用 F1 作为排序依据
    - 添加 sort_by 字段说明使用了哪种排序方式
    """
    # 根据 dim_mode 选择使用的 metrics 字段
    if dim_mode in ('cs', 'cs_only'):
        metrics_field = 'cs_metrics'
    elif dim_mode in ('gk', 'gk_only'):
        metrics_field = 'gk_metrics'
    else:
        metrics_field = 'ped_metrics'

    # 分离选择题和主观题
    all_items = []
    sc_items = []  # single-choice
    essay_items = []

    # 【2026-01 新增】统计有效 AI 分数数量，判断是否需要 fallback 到 F1
    ai_score_count = 0

    for r in results:
        if r.get('stage2_status') != 'success':
            continue

        ai_score = r.get('ai_overall_score')
        metrics = r.get(metrics_field) or r.get('ped_metrics')

        # 【2026-01 修复】不再因为 ai_score 为 None 而跳过
        # 改为收集所有评估成功的题目
        if metrics is None:
            continue

        if ai_score is not None:
            ai_score_count += 1

        item = {
            'unit_id': r.get('unit_id'),
            'question_type': r.get('question_type'),
            'ai_score': ai_score,  # 可能为 None
            'precision': metrics.get('precision', 0) if metrics else 0,
            'recall': metrics.get('recall', 0) if metrics else 0,
            'f1': metrics.get('f1', 0) if metrics else 0,
        }

        all_items.append(item)

        if r.get('question_type') == 'single-choice':
            sc_items.append(item)
        elif r.get('question_type') == 'essay':
            essay_items.append(item)

    # 【2026-01 新增】判断排序方式
    # 如果有 AI 分数，按 AI 分数排序；否则按 F1 排序
    use_ai_sort = ai_score_count > len(all_items) * 0.5  # 至少一半题目有 AI 分数才用 AI 排序
    sort_by = 'ai_score' if use_ai_sort else 'f1'

    def get_bottom_20_stats(items: List[Dict], sort_key: str) -> Dict[str, Optional[float]]:
        """获取后20%题目的统计"""
        if not items:
            return {
                'count': 0,
                'total_count': 0,
                'ai_avg': None,
                'recall_avg': None,
                'precision_avg': None,
                'f1_avg': None,
            }

        n = len(items)
        # 后20%数量，至少1个
        bottom_n = max(1, math.ceil(n * 0.2))

        # 【2026-01 更新】根据 sort_key 排序
        if sort_key == 'ai_score':
            # 过滤掉 AI 分数为 None 的项目，按 AI 分数升序排序
            valid_items = [item for item in items if item['ai_score'] is not None]
            if not valid_items:
                # 如果没有有效的 AI 分数，fallback 到 F1
                sort_key = 'f1'
                sorted_items = sorted(items, key=lambda x: x['f1'])
            else:
                sorted_items = sorted(valid_items, key=lambda x: x['ai_score'])
        else:
            # 按 F1 升序排序（F1 越低表示质量越差）
            sorted_items = sorted(items, key=lambda x: x['f1'])

        bottom_items = sorted_items[:bottom_n]

        # 计算统计值
        ai_values = [item['ai_score'] for item in bottom_items if item['ai_score'] is not None]
        ai_avg = sum(ai_values) / len(ai_values) if ai_values else None

        recall_avg = sum(item['recall'] for item in bottom_items) / len(bottom_items)
        precision_avg = sum(item['precision'] for item in bottom_items) / len(bottom_items)
        f1_avg = sum(item['f1'] for item in bottom_items) / len(bottom_items)

        return {
            'count': bottom_n,
            'total_count': n,
            'ai_avg': round(ai_avg, 4) if ai_avg is not None else None,
            'recall_avg': round(recall_avg, 4),
            'precision_avg': round(precision_avg, 4),
            'f1_avg': round(f1_avg, 4),
            'unit_ids': [item['unit_id'] for item in bottom_items],
            'sort_key': sort_key,  # 记录实际使用的排序方式
        }

    return {
        'overall': get_bottom_20_stats(all_items, sort_by),
        'single_choice': get_bottom_20_stats(sc_items, sort_by),
        'essay': get_bottom_20_stats(essay_items, sort_by),
        'metrics_field_used': metrics_field,
        'sort_by': sort_by,  # 【2026-01 新增】记录排序方式
        'ai_score_available_count': ai_score_count,  # 【2026-01 新增】有 AI 分数的题目数
    }


def update_summary_file(summary_path: Path) -> Tuple[bool, str]:
    """
    更新单个 summary.json 文件

    Returns:
        (success, message)
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        return False, f"读取失败: {e}"

    # 获取 dim_mode
    config = summary.get('config', {})
    dim_mode = config.get('dim_mode', '')

    # 获取 results
    results = summary.get('results', [])
    if not results:
        return False, "无 results 数据"

    # 计算后20%指标
    bottom_20_metrics = calculate_bottom_20_metrics(results, dim_mode)

    # 更新 summary
    summary['bottom_20_metrics'] = bottom_20_metrics

    # 写回文件
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return True, f"已更新 (overall: {bottom_20_metrics['overall']['count']}/{bottom_20_metrics['overall']['total_count']})"
    except Exception as e:
        return False, f"写入失败: {e}"


def find_all_summary_files(outputs_dir: Path) -> List[Path]:
    """
    递归查找所有 summary.json 文件
    """
    summary_files = []

    for item in sorted(outputs_dir.iterdir()):
        if not item.is_dir():
            continue

        # 跳过非实验目录
        if item.name.startswith('.') or item.name.startswith('_'):
            continue

        # 直接在当前目录查找 summary.json
        summary_path = item / 'summary.json'
        if summary_path.exists():
            summary_files.append(summary_path)

        # 检查是否是需要递归的目录（ROUND_, ABLATION 等）
        if item.name.startswith('ROUND_') or item.name == 'ABLATION':
            for subitem in sorted(item.iterdir()):
                if not subitem.is_dir():
                    continue
                sub_summary = subitem / 'summary.json'
                if sub_summary.exists():
                    summary_files.append(sub_summary)

    return summary_files


def main():
    parser = argparse.ArgumentParser(description='更新所有实验的 summary.json，添加后20%低质量题目指标')
    parser.add_argument('--outputs-dir', type=str, default=None,
                        help='outputs 目录路径（默认: 项目根目录下的 outputs）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示将要更新的文件，不实际写入')
    args = parser.parse_args()

    # 确定 outputs 目录
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        outputs_dir = project_root / 'outputs'

    if not outputs_dir.exists():
        print(f'[错误] outputs 目录不存在: {outputs_dir}')
        return

    print(f'[update_bottom20_metrics] 扫描目录: {outputs_dir}')

    # 查找所有 summary.json 文件
    summary_files = find_all_summary_files(outputs_dir)
    print(f'[update_bottom20_metrics] 找到 {len(summary_files)} 个 summary.json 文件')

    if args.dry_run:
        print('\n[DRY RUN] 将要更新以下文件:')
        for path in summary_files:
            print(f'  - {path.parent.name}/summary.json')
        return

    # 更新每个文件
    success_count = 0
    fail_count = 0

    for summary_path in summary_files:
        folder_name = summary_path.parent.name
        success, msg = update_summary_file(summary_path)

        if success:
            success_count += 1
            print(f'  [OK] {folder_name}: {msg}')
        else:
            fail_count += 1
            print(f'  [FAIL] {folder_name}: {msg}')

    print(f'\n[update_bottom20_metrics] 完成: 成功 {success_count}, 失败 {fail_count}')


if __name__ == '__main__':
    main()
