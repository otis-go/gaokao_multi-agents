# -*- coding: utf-8 -*-
"""
【2026-01 新增 / 2026-01 New】Good/Bad Case 展示模块
Good/Bad Case Showcase Module

用于生成完整的案例展示，包括：
Generates complete case presentations, including:
- Stage1 各 Agent 的 prompt/response / Stage1 Agent prompts/responses
- Stage2 评估的 prompt/response 和评估理由 / Stage2 evaluation prompts/responses and reasons
- 通用提示词模板（ABC档次）/ Common prompt templates (ABC levels)
- 迭代过程的完整记录 / Complete iteration records
"""

from .case_collector import CaseCollector, FullCaseRecord
from .case_exporter import CaseExporter
from .prompt_highlighter import PromptHighlighter

__all__ = [
    "CaseCollector",
    "FullCaseRecord",
    "CaseExporter",
    "PromptHighlighter",
]
