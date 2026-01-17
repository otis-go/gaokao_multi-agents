#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Hard negative control dimension replacement file

Core logic (difference from regular negative control):
1. Global 181-question permutation, no grouping constraints
2. No feasible pool constraints, sample from global dimension pool
3. GK and CS independently replaced (can come from different questions)
4. Maintain dimension count conservation (same quantity replacement)
5. Force difference: if replacement is identical to original, randomly replace 1 dimension
6. Output single dimension overlap rate report

Usage:
    python scripts/generate_hardmix_dimension.py [--seed 42]
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass


# GK and CS field definitions
GK_FIELDS = ["gk.value", "gk.subject_literacy", "gk.key_ability",
             "gk.essential_knowledge", "gk.wings", "gk.context"]
CS_FIELDS = ["cs.core_literacy", "cs.task_group", "cs.ability"]


def load_dim_field_mapping(data_dir: Path) -> Dict[str, str]:
    """
    Load dimension name to field mapping from ABC_prompt.json
    """
    abc_path = data_dir / "ABC_prompt.json"
    with open(abc_path, "r", encoding="utf-8") as f:
        abc_data = json.load(f)

    dim_to_field = {}

    for item in abc_data:
        item_id = item.get("id", "")
        full_name = item.get("dimension_name", "")

        # Extract short name
        if "-" in full_name:
            short_name = full_name.split("-", 1)[-1]
        else:
            short_name = full_name

        # Determine field
        if item_id.startswith("gk.") or item_id.startswith("cs."):
            dim_to_field[short_name] = item_id
            dim_to_field[full_name] = item_id

    return dim_to_field


@dataclass
class QuestionDimInfo:
    """Question dimension information"""
    unit_id: str
    gk_dims: List[str]      # Merged list of all GK dimensions
    cs_dims: List[str]      # Merged list of all CS dimensions
    gk_total: int           # Total GK dimensions
    cs_total: int           # Total CS dimensions
    gk_by_field: Dict[str, List[str]]  # GK dimensions grouped by field
    cs_by_field: Dict[str, List[str]]  # CS dimensions grouped by field


def extract_dim_info(item: Dict) -> QuestionDimInfo:
    """Extract dimension information from question data"""
    unit_id = str(item.get("unit_id", ""))

    gk_by_field = {}
    cs_by_field = {}
    gk_dims = []
    cs_dims = []

    for field in GK_FIELDS:
        dims = item.get(field, [])
        if isinstance(dims, str):
            dims = [dims] if dims else []
        gk_by_field[field] = dims
        gk_dims.extend(dims)

    for field in CS_FIELDS:
        dims = item.get(field, [])
        if isinstance(dims, str):
            dims = [dims] if dims else []
        cs_by_field[field] = dims
        cs_dims.extend(dims)

    return QuestionDimInfo(
        unit_id=unit_id,
        gk_dims=gk_dims,
        cs_dims=cs_dims,
        gk_total=len(gk_dims),
        cs_total=len(cs_dims),
        gk_by_field=gk_by_field,
        cs_by_field=cs_by_field,
    )


def build_global_pools(all_infos: Dict[str, QuestionDimInfo]) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, int]]:
    """
    Build global dimension pool

    Returns:
        (gk_pool, cs_pool, gk_freq, cs_freq)
    """
    gk_counter = Counter()
    cs_counter = Counter()

    for info in all_infos.values():
        gk_counter.update(info.gk_dims)
        cs_counter.update(info.cs_dims)

    gk_pool = list(gk_counter.keys())
    cs_pool = list(cs_counter.keys())
    gk_freq = dict(gk_counter)
    cs_freq = dict(cs_counter)

    return gk_pool, cs_pool, gk_freq, cs_freq


def find_global_replacement(
    target_info: QuestionDimInfo,
    all_infos: Dict[str, QuestionDimInfo],
    dim_type: str,
    rng: random.Random
) -> Tuple[QuestionDimInfo, bool]:
    """
    Find another question with same (or closest) total dimension count in global scope

    Args:
        target_info: Target question information
        all_infos: All question information
        dim_type: "gk" or "cs"
        rng: Random number generator

    Returns:
        (source_info, exact_match): Source question information, whether exact match
    """
    target_count = target_info.gk_total if dim_type == "gk" else target_info.cs_total
    target_dims = set(target_info.gk_dims if dim_type == "gk" else target_info.cs_dims)

    # Exclude self
    candidates = [q for q in all_infos.values() if q.unit_id != target_info.unit_id]

    if not candidates:
        return target_info, False

    def get_dims_set(q: QuestionDimInfo) -> set:
        return set(q.gk_dims if dim_type == "gk" else q.cs_dims)

    def get_count(q: QuestionDimInfo) -> int:
        return q.gk_total if dim_type == "gk" else q.cs_total

    def is_different(q: QuestionDimInfo) -> bool:
        return get_dims_set(q) != target_dims

    # Prioritize exact dimension count matches
    exact_matches = [q for q in candidates if get_count(q) == target_count]

    if exact_matches:
        # Prioritize candidates with different dimension sets
        different_matches = [q for q in exact_matches if is_different(q)]
        if different_matches:
            return rng.choice(different_matches), True
        return rng.choice(exact_matches), True

    # Otherwise find nearest
    candidates_with_diff = [(q, abs(get_count(q) - target_count)) for q in candidates]
    candidates_with_diff.sort(key=lambda x: x[1])
    min_diff = candidates_with_diff[0][1]
    nearest = [q for q, diff in candidates_with_diff if diff == min_diff]

    different_nearest = [q for q in nearest if is_different(q)]
    if different_nearest:
        return rng.choice(different_nearest), False

    return rng.choice(nearest), False


def adjust_dimension_count(
    source_dims: List[str],
    target_count: int,
    pool: List[str],
    freq: Dict[str, int],
    rng: random.Random
) -> Tuple[List[str], bool]:
    """
    Adjust dimension count to match target count (sample from global pool)
    """
    current_count = len(source_dims)

    if current_count == target_count:
        return list(source_dims), False

    adjusted_dims = list(source_dims)

    if current_count > target_count:
        # Need to remove: remove highest frequency dimensions
        to_remove = current_count - target_count
        dims_by_freq = sorted(adjusted_dims, key=lambda d: freq.get(d, 0), reverse=True)
        for dim in dims_by_freq[:to_remove]:
            adjusted_dims.remove(dim)

    elif current_count < target_count:
        # Need to add: randomly add from global pool
        to_add = target_count - current_count
        available = [d for d in pool if d not in adjusted_dims]

        if len(available) >= to_add:
            added = rng.sample(available, to_add)
        else:
            added = list(available)
            remaining = to_add - len(added)
            if pool:
                added.extend(rng.choices(pool, k=remaining))

        adjusted_dims.extend(added)

    return adjusted_dims, True


def replace_dims_field_by_field(
    source_by_field: Dict[str, List[str]],
    target_total: int,
    pool: List[str],
    freq: Dict[str, int],
    fields: List[str],
    rng: random.Random,
    dim_to_field: Dict[str, str] = None
) -> Dict[str, List[str]]:
    """
    Field-level dimension replacement
    """
    result = {f: list(source_by_field.get(f, [])) for f in fields}
    current_total = sum(len(result[f]) for f in fields)

    if current_total == target_total:
        return result

    if current_total > target_total:
        to_remove = current_total - target_total
        all_dims_with_field = []
        for f in fields:
            for dim in result[f]:
                all_dims_with_field.append((dim, f, freq.get(dim, 0)))
        all_dims_with_field.sort(key=lambda x: x[2], reverse=True)
        for i in range(to_remove):
            dim, field, _ = all_dims_with_field[i]
            result[field].remove(dim)

    elif current_total < target_total:
        to_add = target_total - current_total
        existing = set()
        for f in fields:
            existing.update(result[f])
        available = [d for d in pool if d not in existing]

        if len(available) >= to_add:
            added = rng.sample(available, to_add)
        else:
            added = list(available)
            remaining = to_add - len(added)
            if pool:
                added.extend(rng.choices(pool, k=remaining))

        for dim in added:
            correct_field = dim_to_field.get(dim) if dim_to_field else None
            if correct_field and correct_field in result:
                result[correct_field].append(dim)
            else:
                non_empty_fields = [f for f in fields if result[f]]
                if not non_empty_fields:
                    non_empty_fields = [fields[0]]
                result[rng.choice(non_empty_fields)].append(dim)

    return result


def force_difference(
    new_by_field: Dict[str, List[str]],
    original_dims: List[str],
    pool: List[str],
    fields: List[str],
    dim_to_field: Dict[str, str],
    rng: random.Random
) -> Tuple[Dict[str, List[str]], bool]:
    """
    Force difference: if replacement is identical to original, randomly replace 1 dimension
    """
    current_dims = []
    for f in fields:
        current_dims.extend(new_by_field.get(f, []))

    original_set = set(original_dims)
    current_set = set(current_dims)

    if current_set != original_set:
        return new_by_field, False

    # Find dimensions that can be replaced in
    available_new = [d for d in pool if d not in original_set]
    if not available_new:
        return new_by_field, False

    new_dim = rng.choice(available_new)

    if not current_dims:
        return new_by_field, False

    old_dim = rng.choice(current_dims)

    old_field = None
    for f in fields:
        if old_dim in new_by_field.get(f, []):
            old_field = f
            break

    if not old_field:
        return new_by_field, False

    result = {f: list(new_by_field.get(f, [])) for f in fields}
    result[old_field].remove(old_dim)

    correct_field = dim_to_field.get(new_dim) if dim_to_field else None
    if correct_field and correct_field in result:
        result[correct_field].append(new_dim)
    else:
        result[old_field].append(new_dim)

    return result, True


def calculate_overlap_rate(
    original_data: List[Dict],
    hardmix_data: List[Dict]
) -> Tuple[float, float, int]:
    """
    Calculate single dimension overlap rate

    Returns:
        (gk_overlap_rate, cs_overlap_rate, force_diff_count)
    """
    gk_total_original = 0
    gk_overlap_count = 0
    cs_total_original = 0
    cs_overlap_count = 0

    for orig, hardmix in zip(original_data, hardmix_data):
        # GK dimensions
        orig_gk = set()
        new_gk = set()
        for f in GK_FIELDS:
            orig_gk.update(orig.get(f, []))
            new_gk.update(hardmix.get(f, []))

        gk_total_original += len(orig_gk)
        gk_overlap_count += len(orig_gk & new_gk)

        # CS dimensions
        orig_cs = set()
        new_cs = set()
        for f in CS_FIELDS:
            orig_cs.update(orig.get(f, []))
            new_cs.update(hardmix.get(f, []))

        cs_total_original += len(orig_cs)
        cs_overlap_count += len(orig_cs & new_cs)

    gk_rate = (gk_overlap_count / gk_total_original * 100) if gk_total_original > 0 else 0
    cs_rate = (cs_overlap_count / cs_total_original * 100) if cs_total_original > 0 else 0

    # Count questions processed with forced difference
    force_diff_count = sum(
        1 for item in hardmix_data
        if item.get("_permutation_info", {}).get("gk_forced_diff") or
           item.get("_permutation_info", {}).get("cs_forced_diff")
    )

    return gk_rate, cs_rate, force_diff_count


def generate_hardmix_dimensions(
    merged_data: List[Dict[str, Any]],
    seed: int = 42,
    dim_to_field: Dict[str, str] = None
) -> List[Dict[str, Any]]:
    """
    Main function: Generate hard negative control dimension replacement (global permutation)
    """
    rng = random.Random(seed)

    # Extract dimension information from all questions
    all_infos = {str(item.get("unit_id")): extract_dim_info(item) for item in merged_data}

    # Build global dimension pool
    gk_pool, cs_pool, gk_freq, cs_freq = build_global_pools(all_infos)

    print(f"[Global Dimension Pool] GK dimension types: {len(gk_pool)}, CS dimension types: {len(cs_pool)}")

    result = []
    stats = {
        "exact_gk_match": 0,
        "exact_cs_match": 0,
        "gk_adjusted": 0,
        "cs_adjusted": 0,
        "gk_forced_diff": 0,
        "cs_forced_diff": 0,
    }

    for item in merged_data:
        new_item = dict(item)
        unit_id = str(item.get("unit_id", ""))
        target_info = all_infos.get(unit_id)

        if not target_info:
            result.append(new_item)
            continue

        # === GK dimension replacement (global scope) ===
        gk_source, gk_exact = find_global_replacement(
            target_info, all_infos, "gk", rng
        )

        gk_adjusted = False
        gk_forced_diff = False

        gk_by_field = replace_dims_field_by_field(
            source_by_field=gk_source.gk_by_field,
            target_total=target_info.gk_total,
            pool=gk_pool,
            freq=gk_freq,
            fields=GK_FIELDS,
            rng=rng,
            dim_to_field=dim_to_field
        )

        source_gk_total = sum(len(gk_source.gk_by_field.get(f, [])) for f in GK_FIELDS)
        gk_adjusted = (source_gk_total != target_info.gk_total)

        # 强制差异
        gk_by_field, gk_forced_diff = force_difference(
            gk_by_field, target_info.gk_dims,
            gk_pool, GK_FIELDS,
            dim_to_field, rng
        )

        for field, dims in gk_by_field.items():
            new_item[field] = dims

        if gk_exact:
            stats["exact_gk_match"] += 1
        if gk_adjusted:
            stats["gk_adjusted"] += 1
        if gk_forced_diff:
            stats["gk_forced_diff"] += 1

        # === CS dimension replacement (global scope, independently select source question) ===
        cs_source, cs_exact = find_global_replacement(
            target_info, all_infos, "cs", rng
        )

        cs_adjusted = False
        cs_forced_diff = False

        cs_by_field = replace_dims_field_by_field(
            source_by_field=cs_source.cs_by_field,
            target_total=target_info.cs_total,
            pool=cs_pool,
            freq=cs_freq,
            fields=CS_FIELDS,
            rng=rng,
            dim_to_field=dim_to_field
        )

        source_cs_total = sum(len(cs_source.cs_by_field.get(f, [])) for f in CS_FIELDS)
        cs_adjusted = (source_cs_total != target_info.cs_total)

        # Force difference
        cs_by_field, cs_forced_diff = force_difference(
            cs_by_field, target_info.cs_dims,
            cs_pool, CS_FIELDS,
            dim_to_field, rng
        )

        for field, dims in cs_by_field.items():
            new_item[field] = dims

        if cs_exact:
            stats["exact_cs_match"] += 1
        if cs_adjusted:
            stats["cs_adjusted"] += 1
        if cs_forced_diff:
            stats["cs_forced_diff"] += 1

        # Add replacement information tracking
        new_item["_permutation_info"] = {
            "gk_source_unit_id": gk_source.unit_id,
            "cs_source_unit_id": cs_source.unit_id,
            "original_gk_total": target_info.gk_total,
            "original_cs_total": target_info.cs_total,
            "method": "global_replacement_hardmix",
            "gk_exact_match": gk_exact,
            "cs_exact_match": cs_exact,
            "gk_adjusted": gk_adjusted,
            "cs_adjusted": cs_adjusted,
            "gk_forced_diff": gk_forced_diff,
            "cs_forced_diff": cs_forced_diff,
        }

        result.append(new_item)

    print(f"\n[Statistics] GK exact match: {stats['exact_gk_match']}/{len(merged_data)}")
    print(f"[Statistics] CS exact match: {stats['exact_cs_match']}/{len(merged_data)}")
    print(f"[Statistics] GK adjusted: {stats['gk_adjusted']}")
    print(f"[Statistics] CS adjusted: {stats['cs_adjusted']}")
    print(f"[Statistics] GK forced difference: {stats['gk_forced_diff']}")
    print(f"[Statistics] CS forced difference: {stats['cs_forced_diff']}")

    return result


def verify_result(original: List[Dict], randomized: List[Dict]):
    """Verify results"""
    print("\n=== Verification ===")

    gk_count_match = 0
    cs_count_match = 0
    duplicate_issues = []

    for orig, rand in zip(original, randomized):
        unit_id = orig.get("unit_id")

        orig_gk = sum(len(orig.get(f, [])) for f in GK_FIELDS)
        orig_cs = sum(len(orig.get(f, [])) for f in CS_FIELDS)

        rand_gk = sum(len(rand.get(f, [])) for f in GK_FIELDS)
        rand_cs = sum(len(rand.get(f, [])) for f in CS_FIELDS)

        if orig_gk == rand_gk:
            gk_count_match += 1
        if orig_cs == rand_cs:
            cs_count_match += 1

        for field in GK_FIELDS + CS_FIELDS:
            dims = rand.get(field, [])
            if len(dims) != len(set(dims)):
                duplicate_issues.append(f"unit_id={unit_id}, {field}")

    print(f"GK dimension total conservation: {gk_count_match}/{len(original)}")
    print(f"CS dimension total conservation: {cs_count_match}/{len(original)}")

    if duplicate_issues:
        print(f"Found {len(duplicate_issues)} within-field duplicate dimensions")
        for issue in duplicate_issues[:5]:
            print(f"  - {issue}")
    else:
        print("No within-field duplicate dimensions [OK]")


def main():
    parser = argparse.ArgumentParser(description="Generate hard negative control dimension replacement file (global permutation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    print("=" * 60)
    print("Hard Negative Control Dimension Generator (Global Permutation Version)")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Random seed: {args.seed}")

    # Load dimension to field mapping
    dim_to_field = load_dim_field_mapping(data_dir)
    print(f"Loaded dimension field mappings: {len(dim_to_field)} entries")

    # Load original data
    merged_path = data_dir / "merged_kaocha_jk_cs.json"
    with open(merged_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    print(f"Total {len(merged_data)} questions")

    # Generate hard negative control dimensions
    print("\nGenerating global permutation dimensions...")
    hardmix_data = generate_hardmix_dimensions(
        merged_data, seed=args.seed, dim_to_field=dim_to_field
    )

    # Verify
    verify_result(merged_data, hardmix_data)

    # Calculate overlap rate
    gk_rate, cs_rate, force_diff_count = calculate_overlap_rate(merged_data, hardmix_data)

    print("\n" + "=" * 60)
    print("=== Hard Negative Control Dimension Generation Report ===")
    print("=" * 60)
    print(f"Total questions: {len(merged_data)}")
    print(f"GK single dimension overlap rate: {gk_rate:.2f}% (proportion of dimensions same as original after replacement)")
    print(f"CS single dimension overlap rate: {cs_rate:.2f}%")
    print(f"Questions processed with forced difference: {force_diff_count}")

    # Save results
    output_path = data_dir / "merged_hardmix_dimension_jk_cs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hardmix_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
