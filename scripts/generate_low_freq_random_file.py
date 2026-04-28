# -*- coding: utf-8 -*-
"""
generate_low_freq_random_file.py
Generate a random low-frequency-dimension variant of the merged data file.

Refactored version, 2026-01-06:
- Read dimension definitions from gk_cs_eval.json (code -> name mapping).
- Read low-frequency dimension lists from dimension_frequency_analysis.json.
- Extract each dimension's field and label from the real exam dataset.
- Use only dimension values that actually exist in the dataset.

Usage:
    python scripts/generate_low_freq_random_file.py
    python scripts/generate_low_freq_random_file.py --count 3 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
import re

# Add the project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dimension_definitions(data_dir: Path) -> Dict[str, Dict]:
    """
    Load dimension definitions from gk_cs_eval.json.

    Returns a {code: {id, name, field, label}} mapping.
    """
    eval_path = data_dir / "gk_cs_eval.json"
    if not eval_path.exists():
        print(f"[ERROR] Dimension definition file not found: {eval_path}")
        sys.exit(1)

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_config = json.load(f)

    result = {}

    # Process GK dimensions.
    for dim in eval_config.get('gk_dimensions', []):
        code = dim.get('code', '')
        dim_id = dim.get('id', '')
        name = dim.get('name', '')

        # Extract field and label from name.
        # Example: "category-label" -> field=<category field>, label=<label>
        field, label = parse_dimension_name(dim_id, name)

        result[code] = {
            'id': dim_id,
            'name': name,
            'field': field,
            'label': label,
            'type': 'gk'
        }

    # Process CS dimensions.
    for dim in eval_config.get('cs_dimensions', []):
        code = dim.get('code', '')
        dim_id = dim.get('id', '')
        name = dim.get('name', '')

        field, label = parse_dimension_name(dim_id, name)

        result[code] = {
            'id': dim_id,
            'name': name,
            'field': field,
            'label': label,
            'type': 'cs'
        }

    return result


def parse_dimension_name(dim_id: str, name: str) -> Tuple[str, str]:
    """
    Parse field and label from dimension ID and name.

    dim_id examples: gk.value_patriotism, gk.wings_basic, cs.ability_xxx
    name example: "category-label"
    """
    # Known field prefixes.
    KNOWN_FIELDS = [
        'gk.value',
        'gk.subject_literacy',
        'gk.key_ability',
        'gk.wings',
        'gk.context',
        'cs.core_literacy',
        'cs.task_group',
        'cs.ability',
    ]

    # Extract field from dim_id by matching known prefixes.
    field = None
    for known_field in KNOWN_FIELDS:
        if dim_id.startswith(known_field):
            field = known_field
            break

    if field is None:
        # Fallback: use the part before the first underscore.
        field = dim_id.split('_')[0] if '_' in dim_id else dim_id

    # Extract label from name using the final dash-separated segment.
    if '-' in name:
        label = name.split('-')[-1]
    else:
        label = name

    return field, label


def load_low_freq_dims(data_dir: Path) -> Dict[str, List[str]]:
    """
    Load low-frequency dimension lists from dimension_frequency_analysis.json.

    Returns {'gk': [GK01, GK02, ...], 'cs': [CS01, ...]}.
    """
    freq_path = data_dir / "dimension_frequency_analysis.json"
    if not freq_path.exists():
        print(f"[ERROR] Dimension frequency analysis file not found: {freq_path}")
        print("[HINT] Run first: python scripts/analyze_dimension_frequency.py")
        sys.exit(1)

    with open(freq_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    result = {'gk': [], 'cs': []}

    # Extract low-frequency GK dimensions and filter UNMAPPED entries.
    if "gk_analysis" in analysis and "low_freq_dims" in analysis["gk_analysis"]:
        result["gk"] = [
            d for d in analysis["gk_analysis"]["low_freq_dims"]
            if not d.startswith("UNMAPPED:") and d.startswith("GK")
        ]

    # Extract low-frequency CS dimensions and filter UNMAPPED entries.
    if "cs_analysis" in analysis and "low_freq_dims" in analysis["cs_analysis"]:
        result["cs"] = [
            d for d in analysis["cs_analysis"]["low_freq_dims"]
            if not d.startswith("UNMAPPED:") and d.startswith("CS")
        ]

    return result


def verify_dimensions_in_data(
    dim_defs: Dict[str, Dict],
    low_freq_dims: Dict[str, List[str]],
    merged_data: List[Dict]
) -> Dict[str, List[str]]:
    """
    Verify that low-frequency dimensions actually exist in the real exam data.

    Returns validated low-frequency dimensions grouped by type.
    """
    # Collect values that actually appear in each field.
    field_values = {}
    for item in merged_data:
        for field in ['gk.value', 'gk.subject_literacy', 'gk.key_ability', 'gk.wings', 'gk.context',
                      'cs.core_literacy', 'cs.task_group', 'cs.ability']:
            if field not in field_values:
                field_values[field] = set()
            for v in item.get(field, []):
                field_values[field].add(v)

    # Validate each low-frequency dimension.
    valid_dims = {'gk': [], 'cs': []}

    for dim_type in ['gk', 'cs']:
        for code in low_freq_dims.get(dim_type, []):
            if code not in dim_defs:
                print(f"[WARNING] Dimension {code} is not defined in gk_cs_eval.json; skipping")
                continue

            dim_info = dim_defs[code]
            field = dim_info['field']
            label = dim_info['label']

            # Check whether this label appears in the corresponding source-data field.
            if field in field_values and label in field_values[field]:
                valid_dims[dim_type].append(code)
            else:
                print(f"[WARNING] Dimension {code} ({label}) does not appear in source field {field}; skipping")

    print(f"[INFO] Validated low-frequency GK dimensions: {len(valid_dims['gk'])}")
    print(f"[INFO] Validated low-frequency CS dimensions: {len(valid_dims['cs'])}")

    return valid_dims


def generate_random_dims_for_item(
    dim_defs: Dict[str, Dict],
    valid_low_freq: Dict[str, List[str]],
    count: int,
    seed: int
) -> Dict[str, List[str]]:
    """
    Generate random low-frequency dimensions for one item.

    Returns {field: [label1, label2, ...]}.
    """
    random.seed(seed)

    result = {
        "gk.value": [],
        "gk.subject_literacy": [],
        "gk.key_ability": [],
        "gk.wings": [],
        "gk.context": [],
        "cs.core_literacy": [],
        "cs.task_group": [],
        "cs.ability": [],
    }

    # Randomly select low-frequency GK dimensions.
    gk_pool = valid_low_freq.get('gk', [])
    if gk_pool:
        selected_gk = random.sample(gk_pool, min(count, len(gk_pool)))
        for code in selected_gk:
            if code in dim_defs:
                field = dim_defs[code]['field']
                label = dim_defs[code]['label']
                if field in result:
                    result[field].append(label)

    # Randomly select low-frequency CS dimensions.
    cs_pool = valid_low_freq.get('cs', [])
    if cs_pool:
        selected_cs = random.sample(cs_pool, min(count, len(cs_pool)))
        for code in selected_cs:
            if code in dim_defs:
                field = dim_defs[code]['field']
                label = dim_defs[code]['label']
                if field in result:
                    result[field].append(label)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random low-frequency-dimension merged file validated against source data"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of low-frequency dimensions assigned to each item (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path (default: data/merged_kaocha_jk_cs.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/merged_low_freq_k{count}_jk_cs.json)"
    )
    args = parser.parse_args()

    # Resolve paths.
    data_dir = PROJECT_ROOT / "data"
    input_path = Path(args.input) if args.input else data_dir / "merged_kaocha_jk_cs.json"

    # Default output filename includes the k value.
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"merged_low_freq_k{args.count}_jk_cs.json"

    print("=" * 60)
    print("  Generate Random Low-Frequency Dimension File (source-data validated)")
    print("=" * 60)
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Low-frequency dimensions per item: k={args.count}")
    print(f"  Random seed: {args.seed}")
    print("=" * 60)
    print()

    # 1. Load dimension definitions.
    print("[Step 1] Loading dimension definitions...")
    dim_defs = load_dimension_definitions(data_dir)
    print(f"  Loaded {len(dim_defs)} dimension definitions")

    # 2. Load low-frequency dimension lists.
    print("[Step 2] Loading low-frequency dimension lists...")
    low_freq_dims = load_low_freq_dims(data_dir)
    print(f"  Low-frequency GK dimensions: {len(low_freq_dims['gk'])} - {low_freq_dims['gk']}")
    print(f"  Low-frequency CS dimensions: {len(low_freq_dims['cs'])} - {low_freq_dims['cs'][:5]}...")

    # 3. Load the original merged file.
    print("[Step 3] Loading source data...")
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"  Loaded {len(merged_data)} items")

    # 4. Verify that low-frequency dimensions appear in source data.
    print("[Step 4] Validating low-frequency dimensions...")
    valid_low_freq = verify_dimensions_in_data(dim_defs, low_freq_dims, merged_data)

    if not valid_low_freq['gk'] and not valid_low_freq['cs']:
        print("[ERROR] No valid low-frequency dimensions found; check dimension definitions and source data")
        sys.exit(1)

    # 5. Generate random low-frequency dimensions for each item.
    print("[Step 5] Generating random dimensions...")
    result = []
    for i, item in enumerate(merged_data):
        new_item = dict(item)

        # Use unit_id + seed to keep each item's sampling reproducible.
        unit_id = item.get("unit_id", i)
        unit_seed = args.seed + hash(str(unit_id)) % 10000

        # Generate random dimensions.
        random_dims = generate_random_dims_for_item(
            dim_defs, valid_low_freq, args.count, unit_seed
        )

        # Update dimension fields.
        for field, values in random_dims.items():
            new_item[field] = values

        result.append(new_item)

    # 6. Save result.
    print("[Step 6] Saving result...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"[SUCCESS] Generated: {output_path}")
    print(f"  Assigned {args.count} low-frequency GK dimensions and {args.count} low-frequency CS dimensions per item")
    print(f"  Random seed: {args.seed}")
    print("=" * 60)

    # Print sample verification.
    print()
    print("[Sample Verification] Dimension assignment for the first item:")
    sample = result[0]
    for field in ['gk.value', 'gk.subject_literacy', 'gk.wings', 'cs.ability']:
        values = sample.get(field, [])
        if values:
            print(f"  {field}: {values}")


if __name__ == "__main__":
    main()
