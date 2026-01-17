# -*- coding: utf-8 -*-
"""
generate_low_freq_random_file.py
生成随机低频维度版本的 merged 文件

【重构版本 2026-01-06】
- 从 gk_cs_eval.json 读取维度定义（code -> name 映射）
- 从 dimension_frequency_analysis.json 读取低频维度列表
- 从真题数据提取每个维度对应的 field 和 label
- 确保只使用真题中实际存在的维度值

使用方法：
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

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dimension_definitions(data_dir: Path) -> Dict[str, Dict]:
    """
    从 gk_cs_eval.json 加载维度定义
    返回: {code: {id, name, field, label}} 映射
    """
    eval_path = data_dir / "gk_cs_eval.json"
    if not eval_path.exists():
        print(f"[ERROR] 未找到维度定义文件: {eval_path}")
        sys.exit(1)

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_config = json.load(f)

    result = {}

    # 处理 GK 维度
    for dim in eval_config.get('gk_dimensions', []):
        code = dim.get('code', '')
        dim_id = dim.get('id', '')
        name = dim.get('name', '')

        # 从 name 提取 field 和 label
        # 例如: "核心价值-爱国主义情怀" -> field=gk.value, label=爱国主义情怀
        # 例如: "四翼要求-基础性" -> field=gk.wings, label=基础性
        field, label = parse_dimension_name(dim_id, name)

        result[code] = {
            'id': dim_id,
            'name': name,
            'field': field,
            'label': label,
            'type': 'gk'
        }

    # 处理 CS 维度
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
    从维度ID和名称解析 field 和 label

    dim_id 格式: gk.value_patriotism, gk.wings_basic, cs.ability_xxx
    name 格式: "核心价值-爱国主义情怀", "四翼要求-基础性"
    """
    # 已知的 field 前缀列表
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

    # 从 dim_id 提取 field（匹配已知前缀）
    field = None
    for known_field in KNOWN_FIELDS:
        if dim_id.startswith(known_field):
            field = known_field
            break

    if field is None:
        # 回退方案：取第一个下划线之前的部分
        field = dim_id.split('_')[0] if '_' in dim_id else dim_id

    # 从 name 提取 label（取最后一个短横线后的部分）
    if '-' in name:
        label = name.split('-')[-1]
    else:
        label = name

    return field, label


def load_low_freq_dims(data_dir: Path) -> Dict[str, List[str]]:
    """
    从 dimension_frequency_analysis.json 加载低频维度列表
    返回: {'gk': [GK01, GK02, ...], 'cs': [CS01, ...]}
    """
    freq_path = data_dir / "dimension_frequency_analysis.json"
    if not freq_path.exists():
        print(f"[ERROR] 未找到维度频次分析文件: {freq_path}")
        print("[提示] 请先运行: python scripts/analyze_dimension_frequency.py")
        sys.exit(1)

    with open(freq_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    result = {'gk': [], 'cs': []}

    # 提取 GK 低频维度（过滤 UNMAPPED）
    if "gk_analysis" in analysis and "low_freq_dims" in analysis["gk_analysis"]:
        result["gk"] = [
            d for d in analysis["gk_analysis"]["low_freq_dims"]
            if not d.startswith("UNMAPPED:") and d.startswith("GK")
        ]

    # 提取 CS 低频维度（过滤 UNMAPPED）
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
    验证低频维度在真题数据中确实存在
    返回验证后的低频维度列表（按 field 分组）
    """
    # 统计真题数据中各字段的实际值
    field_values = {}
    for item in merged_data:
        for field in ['gk.value', 'gk.subject_literacy', 'gk.key_ability', 'gk.wings', 'gk.context',
                      'cs.core_literacy', 'cs.task_group', 'cs.ability']:
            if field not in field_values:
                field_values[field] = set()
            for v in item.get(field, []):
                field_values[field].add(v)

    # 验证每个低频维度
    valid_dims = {'gk': [], 'cs': []}

    for dim_type in ['gk', 'cs']:
        for code in low_freq_dims.get(dim_type, []):
            if code not in dim_defs:
                print(f"[WARNING] 维度 {code} 未在 gk_cs_eval.json 中定义，跳过")
                continue

            dim_info = dim_defs[code]
            field = dim_info['field']
            label = dim_info['label']

            # 检查该标签是否在真题数据的对应字段中存在
            if field in field_values and label in field_values[field]:
                valid_dims[dim_type].append(code)
            else:
                print(f"[WARNING] 维度 {code} ({label}) 在真题 {field} 字段中不存在，跳过")

    print(f"[INFO] 验证后的 GK 低频维度: {len(valid_dims['gk'])} 个")
    print(f"[INFO] 验证后的 CS 低频维度: {len(valid_dims['cs'])} 个")

    return valid_dims


def generate_random_dims_for_item(
    dim_defs: Dict[str, Dict],
    valid_low_freq: Dict[str, List[str]],
    count: int,
    seed: int
) -> Dict[str, List[str]]:
    """
    为单个题目生成随机低频维度
    返回: {field: [label1, label2, ...]} 格式
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

    # 随机选择 GK 低频维度
    gk_pool = valid_low_freq.get('gk', [])
    if gk_pool:
        selected_gk = random.sample(gk_pool, min(count, len(gk_pool)))
        for code in selected_gk:
            if code in dim_defs:
                field = dim_defs[code]['field']
                label = dim_defs[code]['label']
                if field in result:
                    result[field].append(label)

    # 随机选择 CS 低频维度
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
        description="生成随机低频维度版本的 merged 文件（基于真题数据验证）"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="每题分配的低频维度数量（默认: 3）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入文件路径（默认: data/merged_kaocha_jk_cs.json）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认: data/merged_low_freq_k{count}_jk_cs.json）"
    )
    args = parser.parse_args()

    # 确定路径
    data_dir = PROJECT_ROOT / "data"
    input_path = Path(args.input) if args.input else data_dir / "merged_kaocha_jk_cs.json"

    # 默认输出文件名包含 k 值
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"merged_low_freq_k{args.count}_jk_cs.json"

    print("=" * 60)
    print("  生成低频维度随机文件（基于真题验证）")
    print("=" * 60)
    print(f"  输入文件: {input_path}")
    print(f"  输出文件: {output_path}")
    print(f"  每题低频维度数量 k={args.count}")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)
    print()

    # 1. 加载维度定义
    print("[Step 1] 加载维度定义...")
    dim_defs = load_dimension_definitions(data_dir)
    print(f"  已加载 {len(dim_defs)} 个维度定义")

    # 2. 加载低频维度列表
    print("[Step 2] 加载低频维度列表...")
    low_freq_dims = load_low_freq_dims(data_dir)
    print(f"  GK 低频维度: {len(low_freq_dims['gk'])} 个 - {low_freq_dims['gk']}")
    print(f"  CS 低频维度: {len(low_freq_dims['cs'])} 个 - {low_freq_dims['cs'][:5]}...")

    # 3. 加载原始 merged 文件
    print("[Step 3] 加载原始数据...")
    if not input_path.exists():
        print(f"[ERROR] 未找到输入文件: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"  已加载 {len(merged_data)} 个题目")

    # 4. 验证低频维度在真题中存在
    print("[Step 4] 验证低频维度...")
    valid_low_freq = verify_dimensions_in_data(dim_defs, low_freq_dims, merged_data)

    if not valid_low_freq['gk'] and not valid_low_freq['cs']:
        print("[ERROR] 没有有效的低频维度！请检查维度定义和真题数据")
        sys.exit(1)

    # 5. 为每题生成随机低频维度
    print("[Step 5] 生成随机维度...")
    result = []
    for i, item in enumerate(merged_data):
        new_item = dict(item)

        # 使用 unit_id + seed 作为该题的随机种子，确保可复现
        unit_id = item.get("unit_id", i)
        unit_seed = args.seed + hash(str(unit_id)) % 10000

        # 生成随机维度
        random_dims = generate_random_dims_for_item(
            dim_defs, valid_low_freq, args.count, unit_seed
        )

        # 更新维度字段
        for field, values in random_dims.items():
            new_item[field] = values

        result.append(new_item)

    # 6. 保存结果
    print("[Step 6] 保存结果...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"[SUCCESS] 已生成: {output_path}")
    print(f"  每题分配 {args.count} 个 GK 低频维度 + {args.count} 个 CS 低频维度")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)

    # 打印样本验证
    print()
    print("[样本验证] 第一个题目的维度分配:")
    sample = result[0]
    for field in ['gk.value', 'gk.subject_literacy', 'gk.wings', 'cs.ability']:
        values = sample.get(field, [])
        if values:
            print(f"  {field}: {values}")


if __name__ == "__main__":
    main()
