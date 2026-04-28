#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check dimension prompt blocks in LLM call logs."""

import argparse

import json
import re
import sys
from pathlib import Path


def _latest_llm_log(project_root: Path) -> Path | None:
    candidates = sorted(
        project_root.glob("outputs/**/llm_logs/llm_calls_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_log_path(path_arg: str | None) -> Path:
    project_root = Path(__file__).resolve().parents[1]

    if not path_arg:
        latest = _latest_llm_log(project_root)
        if latest is None:
            raise FileNotFoundError("No outputs/**/llm_logs/llm_calls_*.jsonl file found")
        return latest

    path = Path(path_arg)
    if path.is_dir():
        logs = sorted(path.glob("llm_logs/llm_calls_*.jsonl"))
        if not logs:
            raise FileNotFoundError(f"No llm_logs/llm_calls_*.jsonl file found in experiment directory: {path}")
        return logs[0]

    if not path.exists():
        raise FileNotFoundError(f"Log file does not exist: {path}")

    return path


def _read_jsonl_line(log_path: Path, line_number: int) -> dict:
    if line_number < 1:
        raise ValueError("--line must be greater than or equal to 1")

    with log_path.open("r", encoding="utf-8") as f:
        for current, line in enumerate(f, start=1):
            if current == line_number:
                return json.loads(line)

    raise ValueError(f"Log has only {current if 'current' in locals() else 0} lines; cannot read line {line_number}")


def main():
    parser = argparse.ArgumentParser(
        description="Check dimension prompt blocks in LLM call logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/check_dims.py
  python scripts/check_dims.py outputs/EXP_ABLATION-NODIM_gkcs_C_20260427_171544
  python scripts/check_dims.py outputs/EXP_xxx/llm_logs/llm_calls_EXP_xxx.jsonl --line 3
""",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Experiment directory or llm_calls_*.jsonl file; defaults to the latest log under outputs",
    )
    parser.add_argument(
        "--line",
        type=int,
        default=1,
        help="JSONL line number to read (default: 1)",
    )
    args = parser.parse_args()

    try:
        log_path = _resolve_log_path(args.path)
        data = _read_jsonl_line(log_path, args.line)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Log file: {log_path}")
    print(f"[INFO] JSONL line: {args.line}")

    messages = data.get("messages", [])
    if not messages:
        print("[WARN] Current log line has no messages field")
        return

    content = messages[0].get("content", "")

    # Match the Chinese dimension-block format used in LLM call prompts.
    pattern = r"【维度(\d+)｜(gk\.subject_literacy)｜([^｜]+)｜命中标签：([^｜]+)｜档位：([^】]+)】"
    matches = re.findall(pattern, content)

    print(f"Found {len(matches)} gk.subject_literacy dimension blocks:")
    for m in matches:
        print(f"  Dimension {m[0]} | {m[1]} | {m[2]} | matched_label: {m[3]} | band: {m[4]}")

    print("\n" + "=" * 60)

    pattern2 = r"【维度(\d+)｜([^｜]+)｜([^｜]+)｜命中标签：([^｜]+)｜"
    all_matches = re.findall(pattern2, content)
    print(f"Found {len(all_matches)} dimension blocks in total:")
    for m in all_matches:
        print(f"  Dimension {m[0]} | {m[1]} | {m[2]} | matched_label: {m[3]}")

    for i in range(11, 16):
        if f"【维度{i}｜" in content:
            idx = content.find(f"【维度{i}｜")
            snippet = content[idx:idx + 150]
            print(f"\nDimension {i} exists: {snippet}...")

if __name__ == "__main__":
    main()
