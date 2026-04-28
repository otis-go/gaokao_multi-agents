#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_multi_model_meta_eval_inputs.py

Self-check script to verify multi_model_meta_eval prompts no longer
contain nonexistent fields like multi_model_accuracy_info or multi_model_vote_info.

This script:
1. Scans persisted prompt_payload or llm_logs from a smoke run
2. Checks that multi_model_meta_eval prompts use only available fields
3. Reports any mismatches

Usage:
    # Check after a smoke run
    python tools/check_multi_model_meta_eval_inputs.py --dir outputs/smoke_tests

    # Check specific evaluation state files
    python tools/check_multi_model_meta_eval_inputs.py --dir outputs/batch_xxx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# Fields that should NOT appear in the new single-model version
DEPRECATED_FIELDS = {
    "multi_model_accuracy_info",
    "multi_model_vote_info",
    "Acc_real",
    "Acc_gen",
    "ΔAcc",
}

# Pattern to detect deprecated fields in prompt text
DEPRECATED_PATTERN = re.compile(
    r"(multi_model_accuracy_info|multi_model_vote_info|Acc_real|Acc_gen|ΔAcc)",
    re.IGNORECASE
)


def check_ai_eval_prompt_json() -> List[str]:
    """Check ai_eval_prompt.json for deprecated fields in multi_model_meta_eval."""
    issues = []

    ai_eval_path = PROJECT_ROOT / "data" / "ai_eval_prompt.json"
    if not ai_eval_path.exists():
        issues.append(f"ai_eval_prompt.json not found: {ai_eval_path}")
        return issues

    with open(ai_eval_path, "r", encoding="utf-8") as f:
        dims = json.load(f)

    for dim in dims:
        if dim.get("id") != "multi_model_meta_eval":
            continue

        # Check definition
        definition = dim.get("definition", "")
        matches = DEPRECATED_PATTERN.findall(definition)
        if matches:
            issues.append(f"Definition contains deprecated fields: {matches}")

        # Check prompt_template
        prompt = dim.get("prompt_template", "")
        matches = DEPRECATED_PATTERN.findall(prompt)
        if matches:
            issues.append(f"Prompt template contains deprecated fields: {matches}")

    return issues


def check_llm_logs(log_dir: Path) -> List[str]:
    """Check LLM logs for deprecated fields in prompts."""
    issues = []

    if not log_dir.exists():
        return issues

    # Find all JSON files in llm_logs directories
    for log_file in log_dir.rglob("llm_logs/**/*.json"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if this is an AI evaluation log
            agent_name = data.get("agent_name", "") or data.get("agent", "")
            if "AICentricEval" not in agent_name:
                continue

            # Check prompt content
            prompt = data.get("prompt", "") or data.get("input", "")
            if not prompt:
                continue

            matches = DEPRECATED_PATTERN.findall(prompt)
            if matches:
                issues.append(
                    f"{log_file.relative_to(log_dir)}: Prompt contains deprecated fields: {matches}"
                )

        except Exception as e:
            issues.append(f"{log_file}: Failed to parse: {e}")

    return issues


def check_evaluation_states(state_dir: Path) -> List[str]:
    """Check evaluation_state.json files for consistency."""
    issues = []

    if not state_dir.exists():
        return issues

    # Find all evaluation_state.json files
    for state_file in state_dir.rglob("**/evaluation_state*.json"):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check ai_eval_result if present
            ai_result = data.get("ai_eval_result") or data.get("ai_centric_eval_result")
            if not ai_result:
                continue

            # Check audit section for dimension_filter
            audit = ai_result.get("audit", {})
            dim_filter = audit.get("dimension_filter", {})

            if dim_filter:
                applied = dim_filter.get("applied_dimensions", [])
                skipped = dim_filter.get("skipped_dimensions", [])
                renorm = dim_filter.get("renormalized_weights", {})

                # Verify consistency
                if set(applied) != set(renorm.keys()):
                    issues.append(
                        f"{state_file.relative_to(state_dir)}: "
                        f"applied_dimensions ({len(applied)}) != renormalized_weights keys ({len(renorm)})"
                    )

                # Check weight sum
                weight_sum = sum(renorm.values())
                if renorm and abs(weight_sum - 1.0) > 0.01:
                    issues.append(
                        f"{state_file.relative_to(state_dir)}: "
                        f"renormalized_weights sum = {weight_sum:.3f}, expected 1.0"
                    )

        except Exception as e:
            issues.append(f"{state_file}: Failed to parse: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Check multi_model_meta_eval prompt inputs for consistency"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="outputs",
        help="Directory to scan for logs/evaluation states",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    scan_dir = Path(args.dir)
    if not scan_dir.is_absolute():
        scan_dir = PROJECT_ROOT / scan_dir

    print("=" * 60)
    print("multi_model_meta_eval Input Consistency Check")
    print("=" * 60)

    all_issues = []

    # Check ai_eval_prompt.json definition
    print("\n[1] Checking ai_eval_prompt.json...")
    issues = check_ai_eval_prompt_json()
    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
        all_issues.extend(issues)
    else:
        print("  OK: No deprecated fields in definition/template")

    # Check LLM logs
    print(f"\n[2] Checking LLM logs in {scan_dir}...")
    if scan_dir.exists():
        issues = check_llm_logs(scan_dir)
        if issues:
            print(f"  Found {len(issues)} issue(s):")
            for issue in issues[:10]:
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
            all_issues.extend(issues)
        else:
            print("  OK: No deprecated fields in LLM prompts")
    else:
        print(f"  Skip: Directory not found")

    # Check evaluation states
    print(f"\n[3] Checking evaluation states in {scan_dir}...")
    if scan_dir.exists():
        issues = check_evaluation_states(scan_dir)
        if issues:
            print(f"  Found {len(issues)} issue(s):")
            for issue in issues[:10]:
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
            all_issues.extend(issues)
        else:
            print("  OK: Evaluation states are consistent")
    else:
        print(f"  Skip: Directory not found")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if not all_issues:
        print("[PASS] All checks passed - multi_model_meta_eval is consistent!")
        print("       - No deprecated multi-model fields in prompts")
        print("       - Definition and implementation aligned")
        sys.exit(0)
    else:
        print(f"[FAIL] Found {len(all_issues)} issue(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
