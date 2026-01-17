#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/merge_markdown_into_readme.py

将项目中所有 .md 文件合并到 README.md 中，并删除原始文件。

功能：
- --dry-run: 预览将要执行的操作，不做实际修改
- --apply: 执行合并和删除操作
- --check: 检查是否只剩下 README.md，没有其他 .md 文件

合并策略：
1. 排除 outputs/ 目录下的生成报告（它们由工具自动生成）
2. 将 LLM_CALL_POSITIONS.md 作为独立章节添加
3. 将 data/README.md 内容添加到 data/ 目录说明处
4. 所有合并内容的标题级别降一级（# -> ##，## -> ###）
5. 添加来源注释和锚点
6. 幂等设计：重复运行不会重复添加内容

使用方法：
    python tools/merge_markdown_into_readme.py --dry-run
    python tools/merge_markdown_into_readme.py --apply
    python tools/merge_markdown_into_readme.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Windows 控制台 UTF-8 编码支持
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 要排除的目录（这些目录下的 .md 文件不参与合并）
EXCLUDE_DIRS = [
    "outputs",        # 工具生成的报告
    "back_up_agent",  # 备份目录
    ".git",           # Git 目录
    "node_modules",   # Node 模块（如果有）
    "__pycache__",    # Python 缓存
]

# 合并标记，用于幂等检测
MERGE_MARKER_START = "<!-- BEGIN:MERGED_CONTENT -->"
MERGE_MARKER_END = "<!-- END:MERGED_CONTENT -->"

# 各文件的合并配置
MERGE_CONFIG: Dict[str, dict] = {
    "LLM_CALL_POSITIONS.md": {
        "section_id": "llm-call-positions",
        "section_title": "LLM Call Positions",
        "description": "大模型调用位置说明文档",
        "priority": 10,  # 数字越小越靠前
    },
    "data/README.md": {
        "section_id": "data-directory",
        "section_title": "Data Directory",
        "description": "静态数据目录说明",
        "priority": 20,
    },
}


def find_markdown_files() -> List[Path]:
    """
    查找项目中所有 .md 文件（排除 README.md 和指定目录）。

    Returns:
        List[Path]: .md 文件路径列表
    """
    md_files = []

    for md_path in PROJECT_ROOT.rglob("*.md"):
        # 跳过 README.md 本身
        if md_path.name == "README.md" and md_path.parent == PROJECT_ROOT:
            continue

        # 跳过排除目录
        rel_path = md_path.relative_to(PROJECT_ROOT)
        parts = rel_path.parts

        skip = False
        for exclude_dir in EXCLUDE_DIRS:
            if exclude_dir in parts:
                skip = True
                break

        if not skip:
            md_files.append(md_path)

    return md_files


def demote_headings(content: str, levels: int = 1) -> str:
    """
    将 Markdown 标题降级指定层数。

    Args:
        content: Markdown 内容
        levels: 降级层数（默认 1）

    Returns:
        处理后的内容
    """
    lines = content.split("\n")
    result = []

    for line in lines:
        # 匹配 Markdown 标题
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            hashes = match.group(1)
            title = match.group(2)
            new_level = len(hashes) + levels
            # 最多 6 级标题
            if new_level > 6:
                new_level = 6
            result.append("#" * new_level + " " + title)
        else:
            result.append(line)

    return "\n".join(result)


def generate_section_content(file_path: Path, config: dict) -> str:
    """
    为指定文件生成合并后的章节内容。

    Args:
        file_path: 文件路径
        config: 合并配置

    Returns:
        格式化的章节内容
    """
    rel_path = file_path.relative_to(PROJECT_ROOT)
    content = file_path.read_text(encoding="utf-8")

    # 去除文件开头的一级标题（避免重复）
    content = re.sub(r'^#\s+[^\n]+\n+', '', content, count=1)

    # 标题降级
    content = demote_headings(content, levels=1)

    # 构建章节
    section_id = config["section_id"]
    section_title = config["section_title"]
    description = config["description"]

    section = f"""
<!-- BEGIN:SECTION:{section_id} -->
## {section_title}

> *{description}*
>
> *原始文件: `{rel_path}`（已合并）*

{content.strip()}

<!-- END:SECTION:{section_id} -->
"""
    return section


def check_already_merged(readme_content: str) -> bool:
    """
    检查 README.md 中是否已经包含合并内容。

    Args:
        readme_content: README.md 内容

    Returns:
        是否已合并
    """
    return MERGE_MARKER_START in readme_content


def remove_existing_merged_content(readme_content: str) -> str:
    """
    移除已存在的合并内容，以便重新合并。

    Args:
        readme_content: README.md 内容

    Returns:
        清理后的内容
    """
    pattern = re.compile(
        re.escape(MERGE_MARKER_START) + r'.*?' + re.escape(MERGE_MARKER_END),
        re.DOTALL
    )
    return pattern.sub('', readme_content).strip()


def merge_files(dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    执行文件合并。

    Args:
        dry_run: 是否为预览模式

    Returns:
        (success, messages): 是否成功，操作消息列表
    """
    messages = []
    readme_path = PROJECT_ROOT / "README.md"

    # 查找要合并的文件
    md_files = find_markdown_files()

    if not md_files:
        messages.append("[INFO] 没有找到需要合并的 .md 文件")
        return True, messages

    messages.append(f"[INFO] 找到 {len(md_files)} 个需要合并的 .md 文件:")
    for f in md_files:
        rel = f.relative_to(PROJECT_ROOT)
        messages.append(f"  - {rel}")

    # 读取 README.md
    readme_content = readme_path.read_text(encoding="utf-8")

    # 检查是否已合并
    if check_already_merged(readme_content):
        messages.append("[INFO] 检测到已有合并内容，将先移除再重新合并")
        readme_content = remove_existing_merged_content(readme_content)

    # 按优先级排序文件
    def get_priority(f: Path) -> int:
        rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel in MERGE_CONFIG:
            return MERGE_CONFIG[rel]["priority"]
        return 100

    md_files.sort(key=get_priority)

    # 生成合并内容
    merged_sections = []
    for md_file in md_files:
        rel_path = str(md_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

        if rel_path in MERGE_CONFIG:
            config = MERGE_CONFIG[rel_path]
        else:
            # 为未配置的文件生成默认配置
            section_id = rel_path.replace("/", "-").replace(".md", "").lower()
            section_id = re.sub(r'[^a-z0-9-]', '-', section_id)
            config = {
                "section_id": section_id,
                "section_title": md_file.stem.replace("_", " ").title(),
                "description": f"从 {rel_path} 合并的内容",
                "priority": 100,
            }

        section_content = generate_section_content(md_file, config)
        merged_sections.append(section_content)
        messages.append(f"[MERGE] {rel_path} -> README.md (section: {config['section_id']})")

    # 构建最终合并块
    merged_block = f"""

{MERGE_MARKER_START}
---

# Appendix: Merged Documentation

> *以下内容从其他 .md 文件合并而来，原始文件已删除。*

{"".join(merged_sections)}
{MERGE_MARKER_END}
"""

    # 合并到 README.md
    new_readme_content = readme_content.rstrip() + merged_block

    if dry_run:
        messages.append("\n[DRY-RUN] 将执行以下操作:")
        messages.append(f"  - 更新 README.md（添加 {len(merged_sections)} 个章节）")
        for md_file in md_files:
            messages.append(f"  - 删除 {md_file.relative_to(PROJECT_ROOT)}")
        messages.append("\n[DRY-RUN] 实际不会修改任何文件")
    else:
        # 写入新的 README.md
        readme_path.write_text(new_readme_content, encoding="utf-8")
        messages.append(f"[WRITE] 已更新 README.md")

        # 删除原始 .md 文件
        for md_file in md_files:
            md_file.unlink()
            messages.append(f"[DELETE] 已删除 {md_file.relative_to(PROJECT_ROOT)}")

    return True, messages


def check_only_readme() -> Tuple[bool, List[str]]:
    """
    检查项目中是否只剩下 README.md。

    Returns:
        (success, messages): 是否只有 README.md，检查消息
    """
    messages = []

    # 查找所有 .md 文件
    all_md_files = list(PROJECT_ROOT.rglob("*.md"))

    # 区分 README.md 和其他文件
    readme_found = False
    other_md_files = []
    excluded_md_files = []

    for md_path in all_md_files:
        rel_path = md_path.relative_to(PROJECT_ROOT)
        parts = rel_path.parts

        # 检查是否在排除目录
        in_excluded = any(exc in parts for exc in EXCLUDE_DIRS)

        if md_path.name == "README.md" and md_path.parent == PROJECT_ROOT:
            readme_found = True
        elif in_excluded:
            excluded_md_files.append(rel_path)
        else:
            other_md_files.append(rel_path)

    messages.append("[CHECK] Markdown 文件检查:")
    messages.append(f"  - README.md: {'存在' if readme_found else '不存在'}")
    messages.append(f"  - 其他 .md 文件: {len(other_md_files)}")
    messages.append(f"  - 排除目录中的 .md 文件: {len(excluded_md_files)}")

    if other_md_files:
        messages.append("\n[WARN] 发现未合并的 .md 文件:")
        for f in other_md_files:
            messages.append(f"  - {f}")
        return False, messages

    if excluded_md_files:
        messages.append("\n[INFO] 排除目录中的 .md 文件（不参与合并）:")
        for f in excluded_md_files:
            messages.append(f"  - {f}")

    # 检查 README.md 中是否有合并标记
    readme_path = PROJECT_ROOT / "README.md"
    if readme_found:
        readme_content = readme_path.read_text(encoding="utf-8")
        if check_already_merged(readme_content):
            messages.append("\n[OK] README.md 包含合并内容标记")
        else:
            messages.append("\n[WARN] README.md 不包含合并内容标记（可能尚未执行合并）")

    messages.append("\n[OK] 检查通过：项目中只有 README.md（排除目录除外）")
    return True, messages


def fix_broken_references(dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    修复指向已删除 .md 文件的引用。

    Args:
        dry_run: 是否为预览模式

    Returns:
        (success, messages): 是否成功，修复消息
    """
    messages = []

    readme_path = PROJECT_ROOT / "README.md"
    readme_content = readme_path.read_text(encoding="utf-8")

    # 已删除的文件 -> 新的锚点映射
    reference_map = {
        "LLM_CALL_POSITIONS.md": "#llm-call-positions",
        "data/README.md": "#data-directory",
        "MIGRATION_GUIDE.md": "#migration-guide",  # 如果存在的话
    }

    new_content = readme_content
    fixes_made = []

    for old_ref, new_anchor in reference_map.items():
        # 匹配 Markdown 链接
        pattern = rf'\[([^\]]+)\]\({re.escape(old_ref)}\)'
        matches = re.findall(pattern, new_content)

        if matches:
            for match_text in matches:
                fixes_made.append(f"  [{match_text}]({old_ref}) -> [{match_text}]({new_anchor})")
            new_content = re.sub(pattern, rf'[\1]({new_anchor})', new_content)

    if fixes_made:
        messages.append(f"[FIX] 修复 {len(fixes_made)} 个引用:")
        messages.extend(fixes_made)

        if not dry_run:
            readme_path.write_text(new_content, encoding="utf-8")
            messages.append("[WRITE] 已更新 README.md")
    else:
        messages.append("[INFO] 没有需要修复的引用")

    return True, messages


def main():
    parser = argparse.ArgumentParser(
        description="合并项目 .md 文件到 README.md"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：显示将要执行的操作，不实际修改文件"
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="执行模式：合并文件并删除原始 .md 文件"
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="检查模式：验证是否只剩下 README.md"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Markdown 文件合并工具")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print()

    if args.dry_run:
        print("[MODE] 预览模式 (--dry-run)")
        print()

        success, messages = merge_files(dry_run=True)
        for msg in messages:
            print(msg)

        print()
        _, ref_messages = fix_broken_references(dry_run=True)
        for msg in ref_messages:
            print(msg)

    elif args.apply:
        print("[MODE] 执行模式 (--apply)")
        print()

        success, messages = merge_files(dry_run=False)
        for msg in messages:
            print(msg)

        if success:
            print()
            _, ref_messages = fix_broken_references(dry_run=False)
            for msg in ref_messages:
                print(msg)

        print()
        print("=" * 60)
        if success:
            print("[DONE] 合并完成！")
        else:
            print("[ERROR] 合并失败")
            sys.exit(1)

    elif args.check:
        print("[MODE] 检查模式 (--check)")
        print()

        success, messages = check_only_readme()
        for msg in messages:
            print(msg)

        print()
        print("=" * 60)
        if success:
            print("[PASS] 检查通过")
        else:
            print("[FAIL] 检查未通过")
            sys.exit(1)


if __name__ == "__main__":
    main()
