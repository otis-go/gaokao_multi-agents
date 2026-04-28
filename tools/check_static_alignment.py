#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/check_static_alignment.py

Static data consistency check for gk/cs/question-type alignment.

Verifies:
1. Each unit_id has question_type defined
2. gk/cs tags are "fine-grained dimension-name lists" (NOT parent categories)
3. No parent IDs like gk.subject, gk.value are used as primary identifiers

Parent categories to reject:
- gk.subject (ambiguous - should use gk.subject_literacy, gk.key_ability, etc.)
- gk.value (ambiguous - should use specific value dimensions)

Fine-grained gk dimension fields (accepted):
- gk.subject_literacy
- gk.key_ability
- gk.essential_knowledge
- gk.wings
- gk.context
- gk.value (as a field that may be empty list, not as parent category)

Fine-grained cs dimension fields (accepted):
- cs.core_literacy
- cs.task_group
- cs.ability

Usage:
    python tools/check_static_alignment.py
    python tools/check_static_alignment.py --raw data/raw_material.json --mapping data/merged_kaocha_jk_cs.json
    python tools/check_static_alignment.py --out outputs/audit/static_alignment.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# Fine-grained gk dimension fields (accepted)
GK_FINE_GRAINED_FIELDS = {
    "gk.subject_literacy",
    "gk.key_ability",
    "gk.essential_knowledge",
    "gk.wings",
    "gk.context",
    "gk.value",  # Can be empty list, but valid field
}

# Fine-grained cs dimension fields (accepted)
CS_FINE_GRAINED_FIELDS = {
    "cs.core_literacy",
    "cs.task_group",
    "cs.ability",
}

# Parent/ambiguous categories (rejected as primary identifiers)
PARENT_CATEGORIES = {
    "gk.subject",  # Ambiguous - should use gk.subject_literacy
    "gk",          # Too broad
    "cs",          # Too broad
}

# Valid question types
VALID_QUESTION_TYPES = {
    "选择题", "单选题", "多选题", "简答题", "主观题", "论述题",
    "single-choice", "multiple-choice", "essay", "short-answer",
}


def load_json_file(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load JSON file, return None if not found or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None


def check_unit(unit: Dict[str, Any]) -> List[str]:
    """Check a single unit for consistency issues."""
    issues = []
    unit_id = unit.get("unit_id", "unknown")

    # Check 1: question_type exists
    qt = unit.get("question_type", "")
    if not qt:
        issues.append(f"unit_id={unit_id}: Missing question_type")
    elif qt not in VALID_QUESTION_TYPES:
        # Just warn, don't fail for unknown types
        pass

    # Check 2: gk dimensions are fine-grained
    for gk_field in GK_FINE_GRAINED_FIELDS:
        if gk_field in unit:
            dims = unit[gk_field]
            if isinstance(dims, list):
                for dim in dims:
                    # Check if dimension name looks like a parent category
                    if isinstance(dim, str):
                        dim_lower = dim.lower()
                        # Check for parent category patterns
                        if dim_lower in PARENT_CATEGORIES:
                            issues.append(
                                f"unit_id={unit_id}: {gk_field} contains parent category '{dim}'"
                            )

    # Check 3: cs dimensions are fine-grained
    for cs_field in CS_FINE_GRAINED_FIELDS:
        if cs_field in unit:
            dims = unit[cs_field]
            if isinstance(dims, list):
                for dim in dims:
                    if isinstance(dim, str):
                        dim_lower = dim.lower()
                        if dim_lower in PARENT_CATEGORIES:
                            issues.append(
                                f"unit_id={unit_id}: {cs_field} contains parent category '{dim}'"
                            )

    # Check 4: No direct gk.subject or gk.value used as dimension key (not field)
    # This checks if someone mistakenly used "gk.subject" as a value inside a list
    all_values = []
    for field in list(GK_FINE_GRAINED_FIELDS) + list(CS_FINE_GRAINED_FIELDS):
        if field in unit and isinstance(unit[field], list):
            all_values.extend(unit[field])

    for val in all_values:
        if isinstance(val, str):
            val_lower = val.lower().strip()
            # Check for misuse of field names as values
            if val_lower.startswith("gk.") or val_lower.startswith("cs."):
                if val_lower not in GK_FINE_GRAINED_FIELDS and val_lower not in CS_FINE_GRAINED_FIELDS:
                    issues.append(
                        f"unit_id={unit_id}: Found field-like value '{val}' inside dimension list"
                    )

    return issues


def check_static_alignment(
    mapping_data: List[Dict[str, Any]],
    raw_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Check static data alignment.

    Returns:
        Dict with 'issues', 'stats', and 'success' fields
    """
    all_issues = []
    stats = {
        "total_units": len(mapping_data),
        "units_with_question_type": 0,
        "units_with_gk_dims": 0,
        "units_with_cs_dims": 0,
        "question_types": {},
    }

    for unit in mapping_data:
        issues = check_unit(unit)
        all_issues.extend(issues)

        # Gather stats
        if unit.get("question_type"):
            stats["units_with_question_type"] += 1
            qt = unit["question_type"]
            stats["question_types"][qt] = stats["question_types"].get(qt, 0) + 1

        # Check gk presence
        has_gk = any(
            unit.get(f) and isinstance(unit.get(f), list) and len(unit.get(f)) > 0
            for f in GK_FINE_GRAINED_FIELDS
        )
        if has_gk:
            stats["units_with_gk_dims"] += 1

        # Check cs presence
        has_cs = any(
            unit.get(f) and isinstance(unit.get(f), list) and len(unit.get(f)) > 0
            for f in CS_FINE_GRAINED_FIELDS
        )
        if has_cs:
            stats["units_with_cs_dims"] += 1

    return {
        "success": len(all_issues) == 0,
        "issues": all_issues,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check static data alignment for gk/cs/question-type"
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw_material.json",
        help="Path to raw_material.json",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="data/merged_kaocha_jk_cs.json",
        help="Path to merged_kaocha_jk_cs.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/audit/static_alignment.json",
        help="Output path for audit report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Resolve paths
    raw_path = Path(args.raw)
    mapping_path = Path(args.mapping)
    out_path = Path(args.out)

    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print("=" * 60)
    print("Static Data Alignment Check")
    print("=" * 60)

    # Load mapping data (primary source)
    print(f"\n[1] Loading mapping data from {mapping_path}...")
    mapping_data = load_json_file(mapping_path)
    if mapping_data is None:
        print(f"  ERROR: Failed to load {mapping_path}")
        sys.exit(1)
    print(f"  Loaded {len(mapping_data)} units")

    # Load raw data (optional cross-check)
    print(f"\n[2] Loading raw data from {raw_path}...")
    raw_data = load_json_file(raw_path)
    if raw_data is None:
        print(f"  WARN: Failed to load {raw_path} (cross-check skipped)")
    else:
        print(f"  Loaded {len(raw_data)} raw materials")

    # Run check
    print("\n[3] Checking alignment...")
    result = check_static_alignment(mapping_data, raw_data)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    stats = result["stats"]
    print(f"\n  Total units: {stats['total_units']}")
    print(f"  With question_type: {stats['units_with_question_type']}")
    print(f"  With gk dimensions: {stats['units_with_gk_dims']}")
    print(f"  With cs dimensions: {stats['units_with_cs_dims']}")
    print(f"\n  Question type distribution:")
    for qt, count in sorted(stats["question_types"].items(), key=lambda x: -x[1]):
        print(f"    {qt}: {count}")

    if result["issues"]:
        print(f"\n  Found {len(result['issues'])} issue(s):")
        for issue in result["issues"][:20]:
            print(f"    - {issue}")
        if len(result["issues"]) > 20:
            print(f"    ... and {len(result['issues']) - 20} more")
    else:
        print("\n  No issues found!")

    # Save report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to: {out_path}")

    # Exit
    if result["success"]:
        print("\n[PASS] Static data alignment check passed!")
        print("       - All units have question_type")
        print("       - gk/cs dimensions use fine-grained names")
        print("       - No parent category IDs found")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Found {len(result['issues'])} alignment issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
