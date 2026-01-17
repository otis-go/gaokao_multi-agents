# src/shared/question_family.py
"""
Question Family Classification and Dimension Applicability Matrix

This module provides:
1. Unified question family inference: infer_question_family()
2. AI dimension applicability matrix: get_applicable_ai_dimensions()
3. Weight renormalization for filtered dimensions

Question Families:
- objective: Multiple choice (single/multiple), has options (A/B/C/D)
- subjective_short: Short-answer questions with word limits, bullet points
- subjective_essay: Essay questions with structured arguments, open-ended
- unknown: Cannot determine, uses most conservative dimension set
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class QuestionFamily(str, Enum):
    """Question family enumeration."""
    OBJECTIVE = "objective"
    SUBJECTIVE_SHORT = "subjective_short"
    SUBJECTIVE_ESSAY = "subjective_essay"
    UNKNOWN = "unknown"


# =============================================================================
# Question Family Inference
# =============================================================================

# Patterns indicating objective (multiple-choice) questions
OBJECTIVE_PATTERNS = [
    r"[ABCD][\.\)、]",  # Option markers like A. B) C、
    r"选择[题项]",
    r"单选|多选|选出",
    r"正确的[一是]项",
    r"不正确的[一是]项",
    r"以下.*?正确",
    r"以下.*?错误",
    r"下列.*?正确",
    r"下列.*?错误",
]

# Patterns indicating short-answer questions
SHORT_ANSWER_PATTERNS = [
    r"简[要答述]",
    r"概括.*?要点",
    r"不超过\d+字",
    r"限\d+字",
    r"用.*?句话",
    r"分点[回答列]",
    r"列举.*?要点",
    r"归纳.*?特点",
    r"写出.*?特征",
    r"指出.*?特点",
    r"说明.*?原因",
]

# Patterns indicating essay questions
ESSAY_PATTERNS = [
    r"谈谈你的[看理]解",
    r"谈谈你的[观感]",
    r"结合材料.*?分析",
    r"结合.*?谈谈",
    r"结合全文.*?分析",
    r"阐[述释].*?观点",
    r"论[述证]",
    r"评[价析述]",
    r"你认为",
    r"你怎么看",
    r"发表.*?看法",
    r"分析.*?作用",
    r"分析.*?意图",
    r"探究.*?意义",
    r"赏析",
    r"鉴赏",
    r"评论",
]


def infer_question_family(
    question_type: Optional[str] = None,
    stem: Optional[str] = None,
    options: Optional[List[Any]] = None,
) -> QuestionFamily:
    """
    Infer question family from question_type, stem content, and options.

    Rules (in priority order):
    1. If options exist and have content -> objective
    2. If question_type explicitly indicates choice -> objective
    3. If question_type explicitly indicates essay/subjective -> check stem patterns
    4. Check stem patterns for short-answer vs essay indicators
    5. Fallback to unknown

    Args:
        question_type: Explicit question type string (e.g., "single-choice", "essay")
        stem: Question stem text
        options: List of options (if any)

    Returns:
        QuestionFamily enum value
    """
    # Normalize inputs
    qt = (question_type or "").strip().lower()
    stem_text = (stem or "").strip()

    # Rule 1: Has valid options -> objective
    if options:
        valid_options = [
            o for o in options
            if (isinstance(o, dict) and o.get("content"))
            or (hasattr(o, "content") and getattr(o, "content", None))
        ]
        if len(valid_options) >= 2:
            return QuestionFamily.OBJECTIVE

    # Rule 2: Explicit choice type
    choice_keywords = {
        "single-choice", "single_choice", "multiple-choice", "multiple_choice",
        "mcq", "选择题", "单选", "多选", "选择"
    }
    if any(kw in qt for kw in choice_keywords):
        return QuestionFamily.OBJECTIVE

    # Rule 3: Explicit essay/subjective type -> further classify
    essay_keywords = {"essay", "简答", "主观", "问答", "论述"}
    is_subjective_type = any(kw in qt for kw in essay_keywords)

    # Rule 4: Check stem patterns
    if stem_text:
        # Check for short-answer patterns
        for pattern in SHORT_ANSWER_PATTERNS:
            if re.search(pattern, stem_text):
                return QuestionFamily.SUBJECTIVE_SHORT

        # Check for essay patterns
        for pattern in ESSAY_PATTERNS:
            if re.search(pattern, stem_text):
                return QuestionFamily.SUBJECTIVE_ESSAY

        # Check for objective patterns (in case question_type is missing)
        if not is_subjective_type:
            for pattern in OBJECTIVE_PATTERNS:
                if re.search(pattern, stem_text):
                    return QuestionFamily.OBJECTIVE

    # Rule 5: If explicitly subjective type but no clear pattern
    if is_subjective_type:
        # Default subjective to essay (more conservative for rubric applicability)
        return QuestionFamily.SUBJECTIVE_ESSAY

    return QuestionFamily.UNKNOWN


# =============================================================================
# Dimension Applicability Matrix
# =============================================================================

@dataclass
class DimensionApplicability:
    """Defines which AI dimensions apply to each question family.

    [2025-12 Update] Dimension applicability matrix
    Objective question dimensions (7):
        - Exclusive: option_exclusivity_coverage, guessing_lower_asymptote, distractor_headroom, answer_uniqueness
        - Shared: stem_quality, fairness_regional_gender, item_evidence_sufficiency
    Subjective question dimensions (4):
        - Exclusive: rubric_operational
        - Shared: stem_quality, fairness_regional_gender, item_evidence_sufficiency

    Note: Shared dimensions have different prompt_templates for each type, distinguished by applicable_to field.
    """

    # Dimensions only for objective (choice) questions
    objective_only: Set[str] = field(default_factory=lambda: {
        "answer_uniqueness",           # Answer uniqueness - choice questions only
        "option_exclusivity_coverage", # Option exclusivity and coverage - choice questions only
        "distractor_headroom",         # Distractor headroom - choice questions only
        "guessing_lower_asymptote",    # Guessing lower bound - choice questions only
    })

    # Dimensions only for subjective questions (short-answer and essay)
    subjective_only: Set[str] = field(default_factory=lambda: {
        "rubric_operational",          # Rubric operationability - short-answer questions only
    })

    # Shared dimensions (apply to all question types, but with different prompts)
    # These dimensions are defined in both OBJECTIVE_DIMENSIONS and SUBJECTIVE_DIMENSIONS,
    # but use different prompt_templates, selected by applicable_to field
    shared: Set[str] = field(default_factory=lambda: {
        "stem_quality",                # Stem quality - shared (different prompts for different types)
        "fairness_regional_gender",    # Fairness check - shared (different prompts for different types)
        "item_evidence_sufficiency",   # Item evidence sufficiency - shared (different prompts for different types)
    })

    def get_applicable_dimensions(self, family: QuestionFamily) -> Set[str]:
        """Get dimension IDs applicable to a question family."""
        if family == QuestionFamily.OBJECTIVE:
            return self.objective_only | self.shared
        elif family in (QuestionFamily.SUBJECTIVE_SHORT, QuestionFamily.SUBJECTIVE_ESSAY):
            return self.subjective_only | self.shared
        else:
            # Unknown: use most conservative set (shared only)
            return self.shared

    def is_dimension_applicable(self, dim_id: str, family: QuestionFamily) -> bool:
        """Check if a specific dimension applies to a question family."""
        applicable = self.get_applicable_dimensions(family)
        return dim_id in applicable


# Global instance
DIMENSION_APPLICABILITY = DimensionApplicability()


def get_applicable_ai_dimensions(
    family: QuestionFamily,
    all_dimensions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter AI dimensions by question family applicability.

    [2025-12 Fix] Now checks both:
    1. Dimension ID is in the applicable set for this question type (DIMENSION_APPLICABILITY)
    2. Dimension config's applicable_to field matches this question type

    This correctly handles dimensions with same name but different versions for different question types.

    Args:
        family: Question family enum
        all_dimensions: Full list of dimension configs (may contain duplicate IDs for different families)

    Returns:
        Tuple of (applicable_dimensions, skipped_dimensions)
    """
    applicable = []
    skipped = []

    applicable_ids = DIMENSION_APPLICABILITY.get_applicable_dimensions(family)

    # Determine applicable_to keyword for current question family
    if family == QuestionFamily.OBJECTIVE:
        family_key = "objective"
    elif family in (QuestionFamily.SUBJECTIVE_SHORT, QuestionFamily.SUBJECTIVE_ESSAY):
        family_key = "subjective"
    else:
        family_key = None  # unknown - use more lenient matching

    for dim in all_dimensions:
        dim_id = dim.get("id", "")
        dim_applicable_to = dim.get("applicable_to", [])

        # Check 1: Dimension ID in applicable set
        if dim_id not in applicable_ids:
            skipped.append(dim)
            continue

        # Check 2: Dimension's applicable_to field matches current question type
        # If applicable_to is empty or family_key is None, treat as wildcard (backward compatible)
        if family_key and dim_applicable_to:
            if family_key not in dim_applicable_to:
                skipped.append(dim)
                continue

        applicable.append(dim)

    return applicable, skipped


def renormalize_dimension_weights(
    dimensions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Renormalize dimension weights to sum to 1.0.

    Args:
        dimensions: List of dimension configs with "id" and "weight" fields

    Returns:
        Dict mapping dimension_id -> renormalized_weight
    """
    weights = {}
    total = 0.0

    for dim in dimensions:
        dim_id = dim.get("id", "")
        w = float(dim.get("weight", 0.0))
        if dim_id and w > 0:
            weights[dim_id] = w
            total += w

    if total <= 0:
        # Equal weights fallback
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n > 0 else {}

    return {k: v / total for k, v in weights.items()}


@dataclass
class DimensionFilterResult:
    """Result of dimension filtering for audit trail."""
    question_family: str
    applied_dimensions: List[str]
    skipped_dimensions: List[str]
    original_weights: Dict[str, float]
    renormalized_weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_family": self.question_family,
            "applied_dimensions": self.applied_dimensions,
            "skipped_dimensions": self.skipped_dimensions,
            "original_weights": self.original_weights,
            "renormalized_weights": self.renormalized_weights,
        }


def filter_and_renormalize_dimensions(
    question_type: Optional[str],
    stem: Optional[str],
    options: Optional[List[Any]],
    all_dimensions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], DimensionFilterResult]:
    """
    Main entry point: Infer family, filter dimensions, renormalize weights.

    Args:
        question_type: Explicit question type string
        stem: Question stem text
        options: List of options
        all_dimensions: Full dimension config list

    Returns:
        Tuple of (filtered_dimensions, audit_result)
    """
    family = infer_question_family(question_type, stem, options)

    applicable, skipped = get_applicable_ai_dimensions(family, all_dimensions)

    original_weights = {d.get("id", ""): float(d.get("weight", 0)) for d in applicable}
    renormalized = renormalize_dimension_weights(applicable)

    # Update dimension configs with renormalized weights
    filtered = []
    for dim in applicable:
        dim_copy = dict(dim)
        dim_id = dim_copy.get("id", "")
        dim_copy["weight"] = renormalized.get(dim_id, 0.0)
        dim_copy["original_weight"] = original_weights.get(dim_id, 0.0)
        filtered.append(dim_copy)

    result = DimensionFilterResult(
        question_family=family.value,
        applied_dimensions=[d.get("id", "") for d in applicable],
        skipped_dimensions=[d.get("id", "") for d in skipped],
        original_weights=original_weights,
        renormalized_weights=renormalized,
    )

    return filtered, result


__all__ = [
    "QuestionFamily",
    "infer_question_family",
    "DIMENSION_APPLICABILITY",
    "get_applicable_ai_dimensions",
    "renormalize_dimension_weights",
    "filter_and_renormalize_dimensions",
    "DimensionFilterResult",
]
