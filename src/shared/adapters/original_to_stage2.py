# src/shared/adapters/original_to_stage2.py
"""
Original Question to Stage2 Adapter

This module provides the adapter for packaging original questions (from raw_material.json)
directly into Stage2 input contract, bypassing Stage1 entirely.

Design Principles:
1. Pure data transformation - NO LLM calls
2. NO imports from src/generation/** or src/evaluation/**
3. Only imports from src/shared/** (schemas, config, utils)
4. Provides structured validation and skip reasons
5. Preserves source/question_id for traceability

Usage:
    from src.shared.adapters.original_to_stage2 import build_stage2_record_from_original

    raw_item = {"unit_id": 1, "stem": "...", "material": "...", "answer": "A", "analysis": "..."}
    result = build_stage2_record_from_original(raw_item)
    if result.success:
        stage2_record = result.record
    else:
        skip_reasons = result.skip_reasons
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.shared.schemas import (
    Stage2Record,
    Stage2CoreInput,
    Stage1Meta,
)


@dataclass
class OriginalToStage2Result:
    """
    Result of attempting to build a Stage2Record from original question.

    Attributes:
        success: Whether the build was successful
        record: The built Stage2Record (None if failed)
        skip_reasons: List of reasons why the build failed/was skipped
        warnings: Non-fatal issues encountered during build
        inferred_question_type: The inferred question type (for debugging)
    """
    success: bool
    record: Optional[Stage2Record] = None
    skip_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inferred_question_type: str = "unknown"


# =============================================================================
# Question Type Inference Rules
# =============================================================================

# Patterns for single-choice/multiple-choice (mcq/objective)
MCQ_PATTERNS = [
    r'[AaBb][\.\、\s]',                      # A. B. etc
    r'[ABCD][\.\、\s].*[ABCD][\.\、\s]',      # Multiple options
    r'选择.*正确',
    r'选择.*错误',
    r'下列.*正确.*是',
    r'下列.*错误.*是',
    r'以下.*说法.*正确',
    r'以下.*说法.*错误',
    r'符合.*的是',
    r'不符合.*的是',
    r'正确的一项是',
    r'不正确的一项是',
]

# Patterns for fill-in-the-blank
FILL_BLANK_PATTERNS = [
    r'补写',
    r'每处不超过',
    r'填入.*恰当',
    r'填入.*语句',
    r'空[①②③④⑤⑥]',
    r'第[一二三四五六]处',
    r'_+',                                    # Underscores indicating blanks
    r'（\s*）',                               # Empty parentheses
]

# Patterns for essay/subjective
ESSAY_PATTERNS = [
    r'阐述.*理由',
    r'论述',
    r'分析.*原因',
    r'原因是什么',
    r'谈谈.*看法',
    r'你.*认为',
    r'请.*说明',
    r'请.*解释',
    r'请.*概括',
    r'简要.*分析',
    r'结合.*分析',
    r'概括.*特点',
    r'归纳.*要点',
    r'评价',
    r'写.*作文',
    r'请.*写',
]


def infer_question_type_from_content(
    stem: str,
    answer: Optional[str] = None,
    material: Optional[str] = None,
) -> str:
    """
    Infer question type from stem, answer, and material content.

    Priority:
    1. Check answer format first (single letter = mcq)
    2. Pattern matching on stem
    3. Default to 'unknown'

    Returns:
        Question type: 'single-choice', 'fill-blank', 'essay', or 'unknown'
    """
    # Check answer format first
    if answer:
        answer_clean = answer.strip()
        # Single letter answer -> likely MCQ
        if len(answer_clean) == 1 and answer_clean.upper() in 'ABCDEFGH':
            return 'single-choice'
        # Multiple letters separated by comma/space -> multiple choice
        if re.match(r'^[A-H][\s,，、]+[A-H]', answer_clean.upper()):
            return 'single-choice'

    # Check stem patterns
    text_to_check = (stem or '') + ' ' + (material or '')[:500]  # Limit material check

    # MCQ patterns
    for pattern in MCQ_PATTERNS:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return 'single-choice'

    # Fill-blank patterns
    for pattern in FILL_BLANK_PATTERNS:
        if re.search(pattern, text_to_check):
            return 'fill-blank'

    # Essay patterns
    for pattern in ESSAY_PATTERNS:
        if re.search(pattern, text_to_check):
            return 'essay'

    return 'unknown'


def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get value from dict or object."""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _safe_str(value: Any) -> str:
    """Safely convert to string."""
    if value is None:
        return ""
    return str(value).strip()


def _extract_options_from_stem(stem: str) -> Optional[List[Dict[str, str]]]:
    """
    Try to extract options from stem text if it contains embedded options.

    Pattern: A. xxx B. xxx C. xxx D. xxx
    or: A、xxx B、xxx C、xxx D、xxx
    """
    # Pattern for options embedded in stem
    option_pattern = r'([A-D])[\.\、\s]([^A-D]+?)(?=[A-D][\.\、\s]|$)'
    matches = re.findall(option_pattern, stem)

    if len(matches) >= 2:  # At least 2 options to be valid
        options = []
        for label, content in matches:
            content_clean = content.strip()
            if content_clean:
                options.append({"label": label.upper(), "content": content_clean})
        if options:
            return options

    return None


def build_stage2_record_from_original(
    raw_item: Dict[str, Any],
    *,
    experiment_id: str = "BASELINE_ORIGINAL",
    baseline_tag: str = "original",
) -> OriginalToStage2Result:
    """
    Build a Stage2Record from an original question (raw_material.json item).

    This adapter packages original questions directly for Stage2 evaluation,
    bypassing Stage1 entirely. Used for baseline comparison.

    Args:
        raw_item: Dict from raw_material.json with keys:
            - unit_id (required)
            - stem (required) - question text
            - material (required) - reading material
            - answer (required) - gold answer
            - analysis (optional) - gold explanation
            - source (optional) - source identifier
            - question_id (optional) - question identifier
        experiment_id: Experiment identifier for the baseline run
        baseline_tag: Tag to identify this as baseline (default: "original")

    Returns:
        OriginalToStage2Result with success status, record (if successful), and skip reasons

    Constraints:
        - Does NOT call any LLM
        - Does NOT access Stage2 implementation
        - Only performs field extraction, cleanup, type inference, and default filling
    """
    result = OriginalToStage2Result(success=False)

    # === Extract required fields ===

    unit_id = _safe_get(raw_item, "unit_id")
    if unit_id is None:
        result.skip_reasons.append("unit_id is missing")
        return result

    stem = _safe_str(_safe_get(raw_item, "stem"))
    if not stem:
        result.skip_reasons.append("stem is empty")
        return result

    material = _safe_str(_safe_get(raw_item, "material"))
    if not material:
        result.skip_reasons.append("material is empty")
        return result

    answer = _safe_str(_safe_get(raw_item, "answer"))
    if not answer:
        result.skip_reasons.append("answer is empty")
        return result

    # === Extract optional fields ===

    analysis = _safe_str(_safe_get(raw_item, "analysis"))
    source = _safe_str(_safe_get(raw_item, "source"))
    question_id = _safe_get(raw_item, "question_id")

    # === Infer question type ===

    question_type = infer_question_type_from_content(stem, answer, material)
    result.inferred_question_type = question_type

    if question_type == "unknown":
        result.warnings.append("Could not infer question_type, defaulting to 'unknown'")

    # === Build options for MCQ ===

    options = None
    correct_answer = None

    if question_type == "single-choice":
        # Try to extract options from stem
        options = _extract_options_from_stem(stem)

        # Set correct answer
        if len(answer) == 1 and answer.upper() in 'ABCDEFGH':
            correct_answer = answer.upper()
        else:
            correct_answer = answer

        if not options:
            result.warnings.append("MCQ detected but could not extract options from stem")

    # === Build answer points for essay/fill-blank ===

    answer_points = None
    total_score = None

    if question_type in ("essay", "fill-blank", "unknown"):
        # For non-MCQ, package the reference answer as answer_points
        answer_points = [{"point": answer, "score": 1}]
        total_score = 1  # Placeholder score

    # === Build explanation ===

    # Combine answer and analysis for explanation field
    # This ensures Stage2 evaluators have access to the gold answer
    explanation_parts = []
    if answer:
        explanation_parts.append(f"参考答案：{answer}")
    if analysis:
        explanation_parts.append(f"解析：{analysis}")
    explanation = "\n".join(explanation_parts) if explanation_parts else ""

    # === Build Stage2CoreInput ===

    try:
        core = Stage2CoreInput(
            experiment_id=experiment_id,
            unit_id=str(unit_id),
            material_text=material,
            question_type=question_type,
            stem=stem,
            explanation=explanation,
            # Dimension info not available for original questions
            gk_dims={},
            cs_dims={},
            exam_skill={},
            dimension_ids=[],  # Will be empty - pedagogical eval may skip
            # MCQ fields
            options=options,
            correct_answer=correct_answer,
            # Essay/fill-blank fields
            answer_points=answer_points,
            total_score=total_score,
        )
    except Exception as e:
        result.skip_reasons.append(f"Failed to build Stage2CoreInput: {e}")
        return result

    # === Build Stage1Meta (baseline marker) ===

    # For baseline, Stage1 was not run, so create placeholder meta
    meta = Stage1Meta(
        agent5_overall_score=None,  # No Agent5 score - original question
        agent5_layer_scores={},
        agent5_need_revision=False,
        agent5_is_reject=False,
        agent5_issue_types=[],
        ablation_skip_agent="baseline_original",  # Special marker for baseline
    )

    # === Build final record ===

    record = Stage2Record(core_input=core, stage1_meta=meta)

    # === Add metadata for traceability ===

    # Store original source info in a way that's serializable
    # Note: Stage2Record doesn't have a meta field, but we can add it to core_input
    # through the explanation or as warnings
    if source:
        result.warnings.append(f"source={source}")
    if question_id is not None:
        result.warnings.append(f"question_id={question_id}")

    result.success = True
    result.record = record
    return result


def load_raw_materials(
    raw_material_path: str,
    unit_ids: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load raw materials from JSON file, optionally filtering by unit_ids.

    Args:
        raw_material_path: Path to raw_material.json
        unit_ids: Optional list of unit_ids to filter (if None, load all)

    Returns:
        List of raw material dicts
    """
    import json
    from pathlib import Path

    path = Path(raw_material_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw material file not found: {raw_material_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("raw_material.json must be a list of records")

    if unit_ids is None:
        return data

    # Convert unit_ids to set of strings for comparison
    unit_id_set = set(str(uid) for uid in unit_ids)

    filtered = []
    for item in data:
        item_uid = _safe_get(item, "unit_id")
        if item_uid is not None and str(item_uid) in unit_id_set:
            filtered.append(item)

    return filtered


__all__ = [
    "build_stage2_record_from_original",
    "infer_question_type_from_content",
    "load_raw_materials",
    "OriginalToStage2Result",
]
