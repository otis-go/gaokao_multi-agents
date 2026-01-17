#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成随机维度版本的 merged_kaocha_jk_cs.json 文件

用于消融实验：验证受控维度机制的有效性。

【随机策略 - 2026-01 更新】
- 按细分维度类别保持个数一致（如原题 gk.value 有2个，随机版也保持2个）
- 不回避高频维度，所有维度都可被随机选中
- 从同一细分类别的维度池中随机选取

使用方法：
    python scripts/generate_random_dimension_file.py [--seed 42] [--show-comparison]
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_short_name(full_name: str) -> str:
    """
    从完整维度名称中提取简短名称。

    例如：
    - "学科素养-信息获取" -> "信息获取"
    - "核心价值-爱国主义情怀" -> "爱国主义情怀"
    """
    if "-" in full_name:
        return full_name.split("-", 1)[-1]
    return full_name


def load_all_dimensions(data_dir: Path, exclude_empty_levelc: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    从 ABC_prompt.json 加载所有维度，按类别组织。

    Args:
        data_dir: 数据目录路径
        exclude_empty_levelc: 是否排除 levelC 为空的维度，默认 False（不排除）

    Returns:
        gk_dims_by_cat: {category: [short_name, ...]} 例如 {"gk.value": ["爱国主义情怀", ...], ...}
        cs_dims_by_cat: {category: [short_name, ...]} 例如 {"cs.ability": ["比较阅读与辩证判断", ...], ...}
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

        # 检查 levelC 是否为空（可选）
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
        print(f"[generate_random_dimension_file] 已排除 {len(excluded_dims)} 个无真题(levelC为空)的维度:")
        for dim in excluded_dims:
            print(f"  - {dim}")

    return gk_dims_by_cat, cs_dims_by_cat


def load_merged_file(data_dir: Path) -> List[Dict[str, Any]]:
    """加载原始 merged_kaocha_jk_cs.json 文件"""
    merged_path = data_dir / "merged_kaocha_jk_cs.json"
    with open(merged_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_dims(item: Dict, fields: List[str]) -> int:
    """统计一道题目在指定字段中的维度总数"""
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
    为每道题目生成随机维度。

    【2026-01 更新策略】
    1. 按细分维度类别保持个数一致
       - 如原题 gk.value 有2个维度，随机版也从 gk.value 维度池中随机选2个
    2. 不回避高频维度，所有维度都可被随机选中
    3. 使用无放回抽样，保证同一题目内不出现重复维度
       - 如果池子大小 >= 需求数量，使用 random.sample() 无放回抽样
       - 如果池子大小 < 需求数量，先取全部池子，再有放回补齐（极端情况兜底）
    """
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    result = []
    rng = random.Random(seed)

    # 统计池子不够大的警告
    pool_size_warnings = []

    def sample_without_replacement(pool: List[str], count: int, field: str, unit_id: str) -> List[str]:
        """
        无放回抽样，保证不重复。
        如果池子不够大，先取全部再补齐（并记录警告）。
        """
        if not pool:
            return []

        if len(pool) >= count:
            # 池子足够大，无放回抽样
            return rng.sample(pool, count)
        else:
            # 池子不够大（极端情况）：先取全部，再随机补齐
            pool_size_warnings.append(f"unit_id={unit_id}, field={field}: 池子大小={len(pool)} < 需求={count}")
            selected = list(pool)  # 取全部
            # 补齐剩余的（有放回，但已经取了全部不同的）
            remaining = count - len(pool)
            selected.extend([rng.choice(pool) for _ in range(remaining)])
            return selected

    for idx, item in enumerate(merged_data):
        new_item = dict(item)  # 复制原始数据
        unit_id = str(item.get("unit_id", idx))

        # === 处理 GK 维度（按类别保持个数，无放回抽样） ===
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

        # === 处理 CS 维度（按类别保持个数，无放回抽样） ===
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

    # 输出池子大小不够的警告
    if pool_size_warnings:
        print(f"\n[警告] 有 {len(pool_size_warnings)} 处池子大小不够，部分维度可能重复:")
        for warn in pool_size_warnings[:5]:
            print(f"  - {warn}")
        if len(pool_size_warnings) > 5:
            print(f"  ... 共 {len(pool_size_warnings)} 处")

    return result


def print_comparison(original: List[Dict], randomized: List[Dict], sample_size: int = 3):
    """打印原始和随机维度的对比"""
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    print("\n=== 随机维度对比示例 ===\n")

    for i in range(min(sample_size, len(original))):
        orig = original[i]
        rand = randomized[i]

        # 统计总数
        orig_gk_count = count_dims(orig, gk_fields)
        rand_gk_count = count_dims(rand, gk_fields)
        orig_cs_count = count_dims(orig, cs_fields)
        rand_cs_count = count_dims(rand, cs_fields)

        print(f"--- Unit {orig.get('unit_id', i+1)} ---")
        print(f"  GK 维度总数: {orig_gk_count} -> {rand_gk_count}")
        print(f"  CS 维度总数: {orig_cs_count} -> {rand_cs_count}")

        # GK 维度详情
        print("  GK 维度详情:")
        for field in gk_fields:
            orig_dims = orig.get(field, [])
            rand_dims = rand.get(field, [])
            if orig_dims or rand_dims:
                print(f"    {field}: {len(orig_dims)}个 {orig_dims} -> {len(rand_dims)}个 {rand_dims}")

        # CS 维度详情
        print("  CS 维度详情:")
        for field in cs_fields:
            orig_dims = orig.get(field, [])
            rand_dims = rand.get(field, [])
            if orig_dims or rand_dims:
                print(f"    {field}: {len(orig_dims)}个 {orig_dims} -> {len(rand_dims)}个 {rand_dims}")

        print()


def main():
    parser = argparse.ArgumentParser(description="生成随机维度版本的 merged 文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--show-comparison", action="store_true", help="显示对比示例")
    parser.add_argument("--exclude-empty-levelc", action="store_true",
                        help="排除 levelC 为空的维度（默认: 不排除，包含所有维度含高频维度）")
    args = parser.parse_args()

    # 确定数据目录
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    print(f"[generate_random_dimension_file] 数据目录: {data_dir}")
    print(f"[generate_random_dimension_file] 随机种子: {args.seed}")
    print(f"[generate_random_dimension_file] 排除空levelC维度: {args.exclude_empty_levelc}")

    # 加载所有维度（按类别组织）
    print("[generate_random_dimension_file] 正在加载 ABC_prompt.json ...")
    gk_dims_by_cat, cs_dims_by_cat = load_all_dimensions(
        data_dir,
        exclude_empty_levelc=args.exclude_empty_levelc
    )

    # 统计维度池大小
    total_gk = sum(len(v) for v in gk_dims_by_cat.values())
    total_cs = sum(len(v) for v in cs_dims_by_cat.values())
    print(f"[generate_random_dimension_file] GK 维度池: {total_gk} 个（{len(gk_dims_by_cat)} 个类别）")
    print(f"[generate_random_dimension_file] CS 维度池: {total_cs} 个（{len(cs_dims_by_cat)} 个类别）")

    print("\n[generate_random_dimension_file] GK 维度分布（按类别）:")
    for cat, names in sorted(gk_dims_by_cat.items()):
        print(f"  {cat}: {len(names)} 个 - {names[:3]}{'...' if len(names) > 3 else ''}")

    print("\n[generate_random_dimension_file] CS 维度分布（按类别）:")
    for cat, names in sorted(cs_dims_by_cat.items()):
        print(f"  {cat}: {len(names)} 个 - {names[:3]}{'...' if len(names) > 3 else ''}")

    # 加载 merged 文件
    print("\n[generate_random_dimension_file] 正在加载 merged_kaocha_jk_cs.json ...")
    merged_data = load_merged_file(data_dir)
    print(f"[generate_random_dimension_file] 共 {len(merged_data)} 道题目")

    # 生成随机维度
    print("\n[generate_random_dimension_file] 正在生成随机维度（按类别保持个数，含高频维度）...")
    randomized_data = generate_random_dimensions(
        merged_data, gk_dims_by_cat, cs_dims_by_cat, seed=args.seed
    )

    # 显示对比
    if args.show_comparison:
        print_comparison(merged_data, randomized_data, sample_size=5)

    # 保存结果
    output_path = data_dir / "merged_mix_dimension_jk_cs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(randomized_data, f, ensure_ascii=False, indent=2)

    print(f"\n[generate_random_dimension_file] 已保存到: {output_path}")

    # 统计维度变化（按类别）
    gk_fields = ["gk.value", "gk.subject_literacy", "gk.key_ability",
                 "gk.essential_knowledge", "gk.wings", "gk.context"]
    cs_fields = ["cs.core_literacy", "cs.task_group", "cs.ability"]

    print("\n[generate_random_dimension_file] 各类别维度个数对比:")

    # GK 各类别统计
    print("  GK 维度:")
    for field in gk_fields:
        orig_total = sum(len(item.get(field, [])) for item in merged_data)
        rand_total = sum(len(item.get(field, [])) for item in randomized_data)
        print(f"    {field}: 原始 {orig_total} -> 随机 {rand_total} {'✓' if orig_total == rand_total else '✗'}")

    # CS 各类别统计
    print("  CS 维度:")
    for field in cs_fields:
        orig_total = sum(len(item.get(field, [])) for item in merged_data)
        rand_total = sum(len(item.get(field, [])) for item in randomized_data)
        print(f"    {field}: 原始 {orig_total} -> 随机 {rand_total} {'✓' if orig_total == rand_total else '✗'}")

    # 总体统计
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

    print("\n[generate_random_dimension_file] 偶然匹配统计:")
    if total_gk_orig > 0:
        print(f"  GK: 原始 {total_gk_orig} 个 -> 随机 {total_gk_rand} 个 (偶然匹配: {gk_match}, 匹配率: {gk_match/total_gk_orig*100:.1f}%)")
    if total_cs_orig > 0:
        print(f"  CS: 原始 {total_cs_orig} 个 -> 随机 {total_cs_rand} 个 (偶然匹配: {cs_match}, 匹配率: {cs_match/total_cs_orig*100:.1f}%)")

    # 【2026-01 新增】验证：检查同一题目内是否有重复维度
    print("\n[generate_random_dimension_file] 重复维度检查:")
    duplicate_issues = []
    for item in randomized_data:
        unit_id = item.get("unit_id", "?")
        for field in gk_fields + cs_fields:
            dims = item.get(field, [])
            if len(dims) != len(set(dims)):
                duplicates = [d for d in set(dims) if dims.count(d) > 1]
                duplicate_issues.append(f"  unit_id={unit_id}, {field}: 重复={duplicates}")

    if duplicate_issues:
        print(f"  ✗ 发现 {len(duplicate_issues)} 处重复:")
        for issue in duplicate_issues[:10]:
            print(issue)
        if len(duplicate_issues) > 10:
            print(f"  ... 共 {len(duplicate_issues)} 处")
    else:
        print("  ✓ 无重复维度，验证通过！")


if __name__ == "__main__":
    main()
