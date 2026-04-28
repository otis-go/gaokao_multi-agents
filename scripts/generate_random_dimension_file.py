#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a random-dimension version of merged_kaocha_jk_cs.json.

Used for ablation experiments that test the value of controlled dimensions.

Randomization strategy, updated 2026-01:
- Preserve counts by fine-grained dimension category.
- Do not avoid high-frequency dimensions; all dimensions can be sampled.
- Sample from each corresponding fine-grained category pool.

Usage:
    python scripts/generate_random_dimension_file.py [--seed 42] [--show-comparison]
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_short_name(full_name: str) -> str:
    """
    Extract a short label from a full dimension name.

    Example: "category-label" -> "label".
    """
    if "-" in full_name:
        return full_name.split("-", 1)[-1]
    return full_name


def load_all_dimensions(data_dir: Path, exclude_empty_levelc: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load all dimensions from ABC_prompt.json and group them by category.

    Args:
        data_dir: Data directory path.
        exclude_empty_levelc: Whether to exclude dimensions with empty levelC data.

    Returns:
        gk_dims_by_cat: {category: [short_name, ...]}.
        cs_dims_by_cat: {category: [short_name, ...]}.
    """
    abc_path = data_dir / "ABC_prompt.json"
    with open(abc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gk_dims_by_cat: Dict[str, List[str]] = {}
    cs_dims_by_cat: Dict[str, List[str]] = {}
    excluded_dims: List[str] = []

    for item in data:
        item_id = item.get("id", "")
        full_name = item.get("dimension_name", "")
        short_name = extract_short_name(full_name)

        # Optionally skip dimensions without levelC example content.
        if exclude_empty_levelc:
            levelc = item.get("levelC", {})
            addon = levelc.get("addon", "") if levelc else ""
            if not addon or addon.strip() == "":
                excluded_dims.append(f"{item_id}: {short_name}")
                continue

        if item_id.startswith("gk."):
            gk_dims_by_cat.setdefault(item_id, []).append(short_name)
        elif item_id.startswith("cs."):
            cs_dims_by_cat.setdefault(item_id, []).append(short_name)

    if excluded_dims:
        print(f"[generate_random_dimension_file] Excluded {len(excluded_dims)} dimensions with empty levelC examples:")
        for dim in excluded_dims:
            print(f"  - {dim}")

    return gk_dims_by_cat, cs_dims_by_cat


def load_merged_file(data_dir: Path) -> List[Dict[str, Any]]:
    """Load the original merged_kaocha_jk_cs.json file."""
    merged_path = data_dir / "merged_kaocha_jk_cs.json"
    with open(merged_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_dims(item: Dict, fields: List[str]) -> int:
    """Count the total number of dimensions in the specified fields."""
    total = 0
    for field in fields:
        dims = item.get(field, [])
        if isinstance(dims, str):
            dims = [dims] if dims else []
        total += len(dims)
    return total


def generate_random_dimensions(
    merged_data: List[Dict[str, Any]],
    gk_dims_by_cat: Dict[str, List[str]],
    cs_dims_by_cat: Dict[str, List[str]],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate random dimensions for each item.

    Updated strategy, 2026-01:
    1. Preserve counts by fine-grained dimension category.
    2. Allow high-frequency dimensions to be sampled.
    3. Sample without replacement within each item when possible.
    """
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    result = []
    rng = random.Random(seed)

    # Track warnings for undersized sampling pools.
    pool_size_warnings = []

    def sample_without_replacement(pool: List[str], count: int, field: str, unit_id: str) -> List[str]:
        """
        Sample without replacement when possible.

        If the pool is too small, take the full pool and backfill with replacement.
        """
        if not pool:
            return []

        if len(pool) >= count:
            # Pool is large enough for sampling without replacement.
            return rng.sample(pool, count)
        else:
            # Extreme fallback: take all unique values, then backfill with replacement.
            pool_size_warnings.append(f"unit_id={unit_id}, field={field}: pool_size={len(pool)} < requested={count}")
            selected = list(pool)
            remaining = count - len(pool)
            selected.extend([rng.choice(pool) for _ in range(remaining)])
            return selected

    for idx, item in enumerate(merged_data):
        new_item = dict(item)
        unit_id = str(item.get("unit_id", idx))

        # Process GK dimensions with per-category count preservation.
        for field in gk_fields:
            orig_dims = item.get(field, [])
            if isinstance(orig_dims, str):
                orig_dims = [orig_dims] if orig_dims else []

            dim_count = len(orig_dims)
            if dim_count > 0:
                pool = gk_dims_by_cat.get(field, [])
                new_item[field] = sample_without_replacement(pool, dim_count, field, unit_id)
            else:
                new_item[field] = []

        # Process CS dimensions with per-category count preservation.
        for field in cs_fields:
            orig_dims = item.get(field, [])
            if isinstance(orig_dims, str):
                orig_dims = [orig_dims] if orig_dims else []

            dim_count = len(orig_dims)
            if dim_count > 0:
                pool = cs_dims_by_cat.get(field, [])
                new_item[field] = sample_without_replacement(pool, dim_count, field, unit_id)
            else:
                new_item[field] = []

        result.append(new_item)

    # Emit undersized-pool warnings.
    if pool_size_warnings:
        print(f"\n[WARNING] {len(pool_size_warnings)} sampling pools were undersized; some dimensions may repeat:")
        for warn in pool_size_warnings[:5]:
            print(f"  - {warn}")
        if len(pool_size_warnings) > 5:
            print(f"  ... {len(pool_size_warnings)} total")

    return result


def print_comparison(original: List[Dict], randomized: List[Dict], sample_size: int = 3):
    """Print a comparison between original and randomized dimensions."""
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    print("\n=== Random Dimension Comparison Sample ===\n")

    for i in range(min(sample_size, len(original))):
        orig = original[i]
        rand = randomized[i]

        # Count totals.
        orig_gk_count = count_dims(orig, gk_fields)
        rand_gk_count = count_dims(rand, gk_fields)
        orig_cs_count = count_dims(orig, cs_fields)
        rand_cs_count = count_dims(rand, cs_fields)

        print(f"--- Unit {orig.get('unit_id', i+1)} ---")
        print(f"  GK dimension total: {orig_gk_count} -> {rand_gk_count}")
        print(f"  CS dimension total: {orig_cs_count} -> {rand_cs_count}")

        # GK details.
        print("  GK dimension details:")
        for field in gk_fields:
            orig_dims = orig.get(field, [])
            rand_dims = rand.get(field, [])
            if orig_dims or rand_dims:
                print(f"    {field}: {len(orig_dims)} items {orig_dims} -> {len(rand_dims)} items {rand_dims}")

        # CS details.
        print("  CS dimension details:")
        for field in cs_fields:
            orig_dims = orig.get(field, [])
            rand_dims = rand.get(field, [])
            if orig_dims or rand_dims:
                print(f"    {field}: {len(orig_dims)} items {orig_dims} -> {len(rand_dims)} items {rand_dims}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Generate a random-dimension merged file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--show-comparison", action="store_true", help="Show comparison samples")
    parser.add_argument("--exclude-empty-levelc", action="store_true",
                        help="Exclude dimensions with empty levelC examples (default: include all dimensions)")
    args = parser.parse_args()

    # Resolve data directory.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    print(f"[generate_random_dimension_file] Data directory: {data_dir}")
    print(f"[generate_random_dimension_file] Random seed: {args.seed}")
    print(f"[generate_random_dimension_file] Exclude empty levelC dimensions: {args.exclude_empty_levelc}")

    # Load all dimensions grouped by category.
    print("[generate_random_dimension_file] Loading ABC_prompt.json ...")
    gk_dims_by_cat, cs_dims_by_cat = load_all_dimensions(
        data_dir,
        exclude_empty_levelc=args.exclude_empty_levelc
    )

    # Report pool sizes.
    total_gk = sum(len(v) for v in gk_dims_by_cat.values())
    total_cs = sum(len(v) for v in cs_dims_by_cat.values())
    print(f"[generate_random_dimension_file] GK dimension pool: {total_gk} items across {len(gk_dims_by_cat)} categories")
    print(f"[generate_random_dimension_file] CS dimension pool: {total_cs} items across {len(cs_dims_by_cat)} categories")

    print("\n[generate_random_dimension_file] GK dimension distribution by category:")
    for cat, names in sorted(gk_dims_by_cat.items()):
        print(f"  {cat}: {len(names)} items - {names[:3]}{'...' if len(names) > 3 else ''}")

    print("\n[generate_random_dimension_file] CS dimension distribution by category:")
    for cat, names in sorted(cs_dims_by_cat.items()):
        print(f"  {cat}: {len(names)} items - {names[:3]}{'...' if len(names) > 3 else ''}")

    # Load merged file.
    print("\n[generate_random_dimension_file] Loading merged_kaocha_jk_cs.json ...")
    merged_data = load_merged_file(data_dir)
    print(f"[generate_random_dimension_file] Loaded {len(merged_data)} items")

    # Generate randomized dimensions.
    print("\n[generate_random_dimension_file] Generating random dimensions with per-category count preservation...")
    randomized_data = generate_random_dimensions(
        merged_data, gk_dims_by_cat, cs_dims_by_cat, seed=args.seed
    )

    # Show comparison when requested.
    if args.show_comparison:
        print_comparison(merged_data, randomized_data, sample_size=5)

    # Save result.
    output_path = data_dir / "merged_mix_dimension_jk_cs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(randomized_data, f, ensure_ascii=False, indent=2)

    print(f"\n[generate_random_dimension_file] Saved to: {output_path}")

    # Count dimension changes by category.
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    print("\n[generate_random_dimension_file] Dimension count comparison by category:")

    # GK category totals.
    print("  GK dimensions:")
    for field in gk_fields:
        orig_total = sum(len(item.get(field, [])) for item in merged_data)
        rand_total = sum(len(item.get(field, [])) for item in randomized_data)
        status = "OK" if orig_total == rand_total else "MISMATCH"
        print(f"    {field}: original {orig_total} -> randomized {rand_total} {status}")

    # CS category totals.
    print("  CS dimensions:")
    for field in cs_fields:
        orig_total = sum(len(item.get(field, [])) for item in merged_data)
        rand_total = sum(len(item.get(field, [])) for item in randomized_data)
        status = "OK" if orig_total == rand_total else "MISMATCH"
        print(f"    {field}: original {orig_total} -> randomized {rand_total} {status}")

    # Overall overlap statistics.
    total_gk_orig = 0
    total_gk_rand = 0
    total_cs_orig = 0
    total_cs_rand = 0
    gk_match = 0
    cs_match = 0

    for orig, rand in zip(merged_data, randomized_data):
        # GK
        orig_gk_set = set()
        rand_gk_set = set()
        for field in gk_fields:
            orig_gk_set.update(orig.get(field, []))
            rand_gk_set.update(rand.get(field, []))
        total_gk_orig += len(orig_gk_set)
        total_gk_rand += len(rand_gk_set)
        gk_match += len(orig_gk_set & rand_gk_set)

        # CS
        orig_cs_set = set()
        rand_cs_set = set()
        for field in cs_fields:
            orig_cs_set.update(orig.get(field, []))
            rand_cs_set.update(rand.get(field, []))
        total_cs_orig += len(orig_cs_set)
        total_cs_rand += len(rand_cs_set)
        cs_match += len(orig_cs_set & rand_cs_set)

    print("\n[generate_random_dimension_file] Accidental overlap statistics:")
    if total_gk_orig > 0:
        print(f"  GK: original {total_gk_orig} -> randomized {total_gk_rand} (overlap: {gk_match}, rate: {gk_match/total_gk_orig*100:.1f}%)")
    if total_cs_orig > 0:
        print(f"  CS: original {total_cs_orig} -> randomized {total_cs_rand} (overlap: {cs_match}, rate: {cs_match/total_cs_orig*100:.1f}%)")

    # Verify that a single item does not contain duplicated dimensions.
    print("\n[generate_random_dimension_file] Duplicate dimension check:")
    duplicate_issues = []
    for item in randomized_data:
        unit_id = item.get("unit_id", "?")
        for field in gk_fields + cs_fields:
            dims = item.get(field, [])
            if len(dims) != len(set(dims)):
                duplicates = [d for d in set(dims) if dims.count(d) > 1]
                duplicate_issues.append(f"  unit_id={unit_id}, {field}: duplicates={duplicates}")

    if duplicate_issues:
        print(f"  FAIL: found {len(duplicate_issues)} duplicate issue(s):")
        for issue in duplicate_issues[:10]:
            print(issue)
        if len(duplicate_issues) > 10:
            print(f"  ... {len(duplicate_issues)} total")
    else:
        print("  OK: no duplicate dimensions found")


if __name__ == "__main__":
    main()
