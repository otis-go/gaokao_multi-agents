#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Update experiment summary.json files with bottom-20% low-quality item metrics.

Adds the bottom_20_metrics field:
- overall: recall, precision, and ai_avg for the bottom 20% of all items.
- single_choice: recall, precision, and ai_avg for the bottom 20% single-choice items.
- essay: recall, precision, and ai_avg for the bottom 20% essay items.

Usage:
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
    Calculate metrics for the bottom 20% low-quality items.

    Args:
        results: The results list from summary.json.
        dim_mode: Dimension mode (gk/gk_only/cs/cs_only/gk+cs).

    Returns:
        A dictionary containing overall, single_choice, and essay metrics.

    Updated 2026-01:
    - Fall back to F1 sorting when AI evaluation is disabled.
    - Include sort_by to record the applied sorting key.
    """
    # Choose the metrics field according to dim_mode.
    if dim_mode in ('cs', 'cs_only'):
        metrics_field = 'cs_metrics'
    elif dim_mode in ('gk', 'gk_only'):
        metrics_field = 'gk_metrics'
    else:
        metrics_field = 'ped_metrics'

    # Split single-choice and essay items.
    all_items = []
    sc_items = []  # single-choice
    essay_items = []

    # Count valid AI scores to decide whether to fall back to F1 sorting.
    ai_score_count = 0

    for r in results:
        if r.get('stage2_status') != 'success':
            continue

        ai_score = r.get('ai_overall_score')
        metrics = r.get(metrics_field) or r.get('ped_metrics')

        # Keep all successfully evaluated items even when ai_score is None.
        if metrics is None:
            continue

        if ai_score is not None:
            ai_score_count += 1

        item = {
            'unit_id': r.get('unit_id'),
            'question_type': r.get('question_type'),
            'ai_score': ai_score,  # May be None.
            'precision': metrics.get('precision', 0) if metrics else 0,
            'recall': metrics.get('recall', 0) if metrics else 0,
            'f1': metrics.get('f1', 0) if metrics else 0,
        }

        all_items.append(item)

        if r.get('question_type') == 'single-choice':
            sc_items.append(item)
        elif r.get('question_type') == 'essay':
            essay_items.append(item)

    # Use AI score sorting only when more than half of the items have AI scores.
    use_ai_sort = ai_score_count > len(all_items) * 0.5
    sort_by = 'ai_score' if use_ai_sort else 'f1'

    def get_bottom_20_stats(items: List[Dict], sort_key: str) -> Dict[str, Optional[float]]:
        """Get statistics for the bottom 20% items."""
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
        # Bottom 20%, at least one item.
        bottom_n = max(1, math.ceil(n * 0.2))

        # Sort according to the selected key.
        if sort_key == 'ai_score':
            # Drop items with missing AI scores and sort ascending by AI score.
            valid_items = [item for item in items if item['ai_score'] is not None]
            if not valid_items:
                # Fall back to F1 when no valid AI scores are available.
                sort_key = 'f1'
                sorted_items = sorted(items, key=lambda x: x['f1'])
            else:
                sorted_items = sorted(valid_items, key=lambda x: x['ai_score'])
        else:
            # Sort ascending by F1; lower F1 means lower quality.
            sorted_items = sorted(items, key=lambda x: x['f1'])

        bottom_items = sorted_items[:bottom_n]

        # Compute aggregate values.
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
            'sort_key': sort_key,
        }

    return {
        'overall': get_bottom_20_stats(all_items, sort_by),
        'single_choice': get_bottom_20_stats(sc_items, sort_by),
        'essay': get_bottom_20_stats(essay_items, sort_by),
        'metrics_field_used': metrics_field,
        'sort_by': sort_by,
        'ai_score_available_count': ai_score_count,
    }


def update_summary_file(summary_path: Path) -> Tuple[bool, str]:
    """
    Update one summary.json file.

    Returns:
        (success, message)
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        return False, f"Read failed: {e}"

    # Get dim_mode.
    config = summary.get('config', {})
    dim_mode = config.get('dim_mode', '')

    # Get results.
    results = summary.get('results', [])
    if not results:
        return False, "No results data"

    # Calculate bottom-20% metrics.
    bottom_20_metrics = calculate_bottom_20_metrics(results, dim_mode)

    # Update summary.
    summary['bottom_20_metrics'] = bottom_20_metrics

    # Write back to file.
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return True, f"Updated (overall: {bottom_20_metrics['overall']['count']}/{bottom_20_metrics['overall']['total_count']})"
    except Exception as e:
        return False, f"Write failed: {e}"


def find_all_summary_files(outputs_dir: Path) -> List[Path]:
    """
    Recursively find all summary.json files.
    """
    summary_files = []

    for item in sorted(outputs_dir.iterdir()):
        if not item.is_dir():
            continue

        # Skip non-experiment directories.
        if item.name.startswith('.') or item.name.startswith('_'):
            continue

        # Check the current directory first.
        summary_path = item / 'summary.json'
        if summary_path.exists():
            summary_files.append(summary_path)

        # Recurse into known grouping directories.
        if item.name.startswith('ROUND_') or item.name == 'ABLATION':
            for subitem in sorted(item.iterdir()):
                if not subitem.is_dir():
                    continue
                sub_summary = subitem / 'summary.json'
                if sub_summary.exists():
                    summary_files.append(sub_summary)

    return summary_files


def main():
    parser = argparse.ArgumentParser(description='Update summary.json files with bottom-20% low-quality item metrics')
    parser.add_argument('--outputs-dir', type=str, default=None,
                        help='Path to the outputs directory (default: <project_root>/outputs)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show files that would be updated without writing changes')
    args = parser.parse_args()

    # Resolve outputs directory.
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        outputs_dir = project_root / 'outputs'

    if not outputs_dir.exists():
        print(f'[ERROR] outputs directory does not exist: {outputs_dir}')
        return

    print(f'[update_bottom20_metrics] Scanning directory: {outputs_dir}')

    # Find all summary.json files.
    summary_files = find_all_summary_files(outputs_dir)
    print(f'[update_bottom20_metrics] Found {len(summary_files)} summary.json files')

    if args.dry_run:
        print('\n[DRY RUN] The following files would be updated:')
        for path in summary_files:
            print(f'  - {path.parent.name}/summary.json')
        return

    # Update each file.
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

    print(f'\n[update_bottom20_metrics] Complete: success={success_count}, failed={fail_count}')


if __name__ == '__main__':
    main()
