# -*- coding: utf-8 -*-
"""
Good/bad case showcase module.

This package generates complete case presentations, including Stage1 agent
prompts/responses, Stage2 evaluation prompts/responses, common prompt templates,
and iteration records.
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
