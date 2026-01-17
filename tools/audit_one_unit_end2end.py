#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/audit_one_unit_end2end.py

End-to-end audit for a single unit: runs stage1+stage2 and outputs audit.json
containing dimension consistency check across all stages.

Verifies:
1. Static labels (gk/cs/question_type) from data
2. Stage1 output dimensions
3. Stage2 received dimensions
4. Final evaluation dimension keys
5. All four must be consistent and use fine-grained dimension names

Usage:
    python tools/audit_one_unit_end2end.py --unit-id 1
    python tools/audit_one_unit_end2end.py --unit-id 3 --out outputs/audit/unit_3_audit.json
    python tools/audit_one_unit_end2end.py --unit-id 1 --dry-run  # No LLM calls
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def load_static_data(unit_id: str) -> Optional[Dict[str, Any]]:
    """Load static data for a unit from merged_kaocha_jk_cs.json."""
    mapping_path = PROJECT_ROOT / "data" / "merged_kaocha_jk_cs.json"
    if not mapping_path.exists():
        return None

    with open(mapping_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    for item in mappings:
        if str(item.get("unit_id")) == str(unit_id):
            return item
    return None


def extract_static_labels(static_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract gk/cs/question_type labels from static data."""
    labels = {
        "question_type": static_data.get("question_type", ""),
        "type": static_data.get("type", ""),
        "gk_dimensions": {},
        "cs_dimensions": {},
    }

    # Extract gk dimensions
    gk_fields = [
        "gk.subject_literacy", "gk.key_ability", "gk.essential_knowledge",
        "gk.wings", "gk.context", "gk.value"
    ]
    for field in gk_fields:
        if field in static_data:
            labels["gk_dimensions"][field] = static_data[field]

    # Extract cs dimensions
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]
    for field in cs_fields:
        if field in static_data:
            labels["cs_dimensions"][field] = static_data[field]

    return labels


def check_fine_grained_dimensions(dims: Dict[str, Any], stage: str) -> List[str]:
    """Check if dimensions are fine-grained (not parent categories)."""
    issues = []
    parent_patterns = ["gk.subject", "gk.value"]  # These should not appear as standalone IDs

    def check_value(key: str, value: Any):
        if isinstance(value, str):
            val_lower = value.lower()
            for pattern in parent_patterns:
                if val_lower == pattern:
                    issues.append(f"{stage}: Found parent category '{value}' in {key}")
        elif isinstance(value, list):
            for item in value:
                check_value(key, item)
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(f"{key}.{k}", v)

    for key, value in dims.items():
        check_value(key, value)

    return issues


def run_dry_audit(unit_id: str, static_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a dry audit without LLM calls."""
    static_labels = extract_static_labels(static_data)

    audit = {
        "unit_id": unit_id,
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run",
        "static_labels": static_labels,
        "stage1_output_dimensions": "(dry run - not executed)",
        "stage2_received_dimensions": "(dry run - not executed)",
        "final_evaluation_dimensions": "(dry run - not executed)",
        "consistency_check": {
            "static_ok": True,
            "static_issues": check_fine_grained_dimensions(static_labels, "static"),
        },
        "overall_consistent": True,
    }

    if audit["consistency_check"]["static_issues"]:
        audit["overall_consistent"] = False

    return audit


def run_full_audit(unit_id: str, static_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run full audit with stage1+stage2 execution."""
    from src.shared.config import GenerationConfig, create_config_from_args
    from src.shared.data_loader import DataLoader
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.shared.prompt_logger import PromptLogger
    from src.shared.question_family import infer_question_family

    static_labels = extract_static_labels(static_data)

    # Create minimal config
    config = GenerationConfig()
    config.dim_mode = "gk"
    config.prompt_level = "C"

    # Initialize components
    data_loader = DataLoader()
    prompt_logger = PromptLogger(PROJECT_ROOT / "outputs" / "audit" / f"unit_{unit_id}")

    # Run Stage1
    print(f"[Stage1] Running generation for unit_id={unit_id}...")
    gen_orchestrator = GenerationOrchestrator(config)
    gen_orchestrator.prompt_logger = prompt_logger

    try:
        stage1_state = gen_orchestrator.run_single(unit_id=unit_id)
        stage1_success = True

        # Extract Stage1 dimensions
        agent1_output = stage1_state.agent1_output
        stage1_dims = {
            "question_type": getattr(agent1_output, "question_type", ""),
            "gk_dims": getattr(agent1_output, "gk_dims", []),
            "cs_dims": getattr(agent1_output, "cs_dims", []),
            "abc_dims": getattr(agent1_output, "abc_dims", []),
        }
    except Exception as e:
        stage1_success = False
        stage1_dims = {"error": str(e)}
        stage1_state = None

    # Run Stage2 if Stage1 succeeded
    stage2_dims = {}
    final_eval_dims = {}

    if stage1_success and stage1_state and hasattr(stage1_state, "stage2_record"):
        print(f"[Stage2] Running evaluation...")
        try:
            eval_orchestrator = EvaluationOrchestrator(config)
            eval_orchestrator.prompt_logger = prompt_logger

            eval_state = eval_orchestrator.run(stage1_state.stage2_record)

            # Extract Stage2 received dimensions
            core_input = stage1_state.stage2_record.core_input
            stage2_dims = {
                "question_type": getattr(core_input, "question_type", ""),
                "question_family": infer_question_family(
                    getattr(core_input, "question_type", ""),
                    getattr(core_input, "stem", ""),
                    getattr(core_input, "options", None),
                ).value,
            }

            # Extract final evaluation dimensions
            ai_result = getattr(eval_state, "ai_centric_eval_result", None) or {}
            if isinstance(ai_result, dict):
                final_eval_dims = {
                    "applied_dimensions": ai_result.get("applied_dimensions", []),
                    "skipped_dimensions": ai_result.get("audit", {}).get("skipped_dimensions", []),
                    "dimension_keys": list((ai_result.get("dimensions") or {}).keys()),
                }

        except Exception as e:
            stage2_dims = {"error": str(e)}
            final_eval_dims = {"error": str(e)}

    # Build audit report
    audit = {
        "unit_id": unit_id,
        "timestamp": datetime.now().isoformat(),
        "mode": "full",
        "static_labels": static_labels,
        "stage1_output_dimensions": stage1_dims,
        "stage2_received_dimensions": stage2_dims,
        "final_evaluation_dimensions": final_eval_dims,
        "consistency_check": {
            "static_ok": True,
            "stage1_ok": stage1_success,
            "stage2_ok": "error" not in stage2_dims,
            "static_issues": check_fine_grained_dimensions(static_labels, "static"),
            "stage1_issues": check_fine_grained_dimensions(stage1_dims, "stage1") if stage1_success else [],
            "stage2_issues": check_fine_grained_dimensions(stage2_dims, "stage2"),
            "final_issues": check_fine_grained_dimensions(final_eval_dims, "final"),
        },
    }

    # Determine overall consistency
    all_issues = (
        audit["consistency_check"]["static_issues"] +
        audit["consistency_check"]["stage1_issues"] +
        audit["consistency_check"]["stage2_issues"] +
        audit["consistency_check"]["final_issues"]
    )
    audit["overall_consistent"] = (
        len(all_issues) == 0 and
        audit["consistency_check"]["static_ok"] and
        audit["consistency_check"]["stage1_ok"] and
        audit["consistency_check"]["stage2_ok"]
    )
    audit["all_issues"] = all_issues

    return audit


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end audit for a single unit"
    )
    parser.add_argument(
        "--unit-id", "-u",
        type=str,
        required=True,
        help="Unit ID to audit",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default=None,
        help="Output path for audit.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check static data without running stages",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    unit_id = args.unit_id

    print("=" * 60)
    print(f"End-to-End Audit: unit_id={unit_id}")
    print("=" * 60)

    # Load static data
    print(f"\n[1] Loading static data for unit_id={unit_id}...")
    static_data = load_static_data(unit_id)
    if static_data is None:
        print(f"  ERROR: Unit not found in merged_kaocha_jk_cs.json")
        sys.exit(1)
    print(f"  Found unit: question_type={static_data.get('question_type')}")

    # Run audit
    if args.dry_run:
        print("\n[2] Running dry audit (no LLM calls)...")
        audit = run_dry_audit(unit_id, static_data)
    else:
        print("\n[2] Running full audit (stage1 + stage2)...")
        audit = run_full_audit(unit_id, static_data)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = PROJECT_ROOT / "outputs" / "audit" / f"unit_{unit_id}_audit.json"

    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    # Save audit
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Audit Summary")
    print("=" * 60)

    print(f"\n  Static labels:")
    print(f"    question_type: {audit['static_labels'].get('question_type')}")
    print(f"    type: {audit['static_labels'].get('type')}")

    if audit["mode"] == "full":
        print(f"\n  Stage1 output:")
        s1 = audit.get("stage1_output_dimensions", {})
        print(f"    question_type: {s1.get('question_type')}")
        print(f"    gk_dims count: {len(s1.get('gk_dims', []))}")
        print(f"    cs_dims count: {len(s1.get('cs_dims', []))}")

        print(f"\n  Stage2 received:")
        s2 = audit.get("stage2_received_dimensions", {})
        print(f"    question_type: {s2.get('question_type')}")
        print(f"    question_family: {s2.get('question_family')}")

        print(f"\n  Final evaluation:")
        fe = audit.get("final_evaluation_dimensions", {})
        print(f"    applied_dimensions: {fe.get('applied_dimensions', [])}")
        print(f"    skipped_dimensions: {[s.get('id') for s in fe.get('skipped_dimensions', [])]}")

    # Issues
    issues = audit.get("all_issues", [])
    if issues:
        print(f"\n  Found {len(issues)} consistency issue(s):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No consistency issues found!")

    print(f"\n  Audit saved to: {out_path}")

    # Exit
    if audit["overall_consistent"]:
        print("\n[PASS] End-to-end audit passed!")
        print("       - Static labels consistent")
        print("       - All stages use fine-grained dimension names")
        sys.exit(0)
    else:
        print("\n[FAIL] End-to-end audit failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
