#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查维度匹配是否正确"""

import json
import re
import sys

def main():
    log_path = "outputs/EXP_SINGLE_U103_gk+cs_C_20251209_015510/llm_logs/llm_calls_EXP_SINGLE_U103_gk+cs_C_20251209_015510.jsonl"

    with open(log_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)
        messages = data.get('messages', [])
        if messages:
            content = messages[0].get('content', '')

            # 找所有 gk.subject_literacy 相关的维度块
            # 格式: 【维度X｜gk.subject_literacy｜学科素养-XXX｜命中标签：XXX｜档位：C】
            pattern = r'【维度(\d+)｜(gk\.subject_literacy)｜([^｜]+)｜命中标签：([^｜]+)｜档位：([^】]+)】'
            matches = re.findall(pattern, content)

            print(f"找到 {len(matches)} 个 gk.subject_literacy 维度块:")
            for m in matches:
                print(f"  维度{m[0]} | {m[1]} | {m[2]} | 命中标签: {m[3]} | 档位: {m[4]}")

            print("\n" + "="*60)

            # 再找所有维度块的概览
            pattern2 = r'【维度(\d+)｜([^｜]+)｜([^｜]+)｜命中标签：([^｜]+)｜'
            all_matches = re.findall(pattern2, content)
            print(f"共找到 {len(all_matches)} 个维度块:")
            for m in all_matches:
                print(f"  维度{m[0]} | {m[1]} | {m[2]} | 命中标签: {m[3]}")

            # 检查是否有维度 11, 12, 13 等
            for i in range(11, 16):
                if f"【维度{i}｜" in content:
                    idx = content.find(f"【维度{i}｜")
                    snippet = content[idx:idx+150]
                    print(f"\n维度{i}存在: {snippet}...")

if __name__ == "__main__":
    main()
