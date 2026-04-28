# src/shared/adapters/stage1_to_stage2.py
"""
Stage1 to Stage2 Adapter

This module provides the UNIQUE entry point for building Stage2 input from Stage1 output.

Design Principles:
1. Pure data transformation - NO LLM calls
2. NO imports from src/generation/** or src/evaluation/**
3. Only imports from src/shared/** (schemas, config, utils)
4. Provides structured validation and skip reasons
5. Idempotent - same input always produces same output

Usage:
    from src.shared.adapters.stage1_to_stage2 import build_stage2_record

    result = build_stage2_record(stage1_state)
    if result.success:
        stage2_record = result.record
    else:
        skip_reasons = result.skip_reasons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.shared.schemas import (
    GenerationPipelineState,
    Stage2Record,
    Stage2CoreInput,
    Stage1Meta,
)

if TYPE_CHECKING:
    pass


@dataclass
class Stage2RecordBuildResult:
    """
    Result of attempting to build a Stage2Record from Stage1 output.

    Attributes:
        success: Whether the build was successful
        record: The built Stage2Record (None if failed)
        skip_reasons: List of reasons why the build failed/was skipped
        warnings: Non-fatal issues encountered during build
    """
    success: bool
    record: Optional[Stage2Record] = None
    skip_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _safe_getattr(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Safely get nested attributes from an object."""
    current = obj
    for attr in attrs:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(attr, default)
        else:
            current = getattr(current, attr, default)
    return current


def _extract_dimension_ids(agent1_output: Any, agent2_output: Any) -> List[str]:
    """
    Extract dimension IDs for Stage2 evaluation.

    Priority:
    1. Agent1's abc_dims (dimension_name field)
    2. Agent1's abc_dims (id field as fallback)
    3. Agent2's dimension_ids (legacy fallback)
    """
    dim_ids: List[str] = []

    # Try Agent1's abc_dims first (preferred)
    a1_abc = _safe_getattr(agent1_output, "abc_dims") or []
    if a1_abc:
        # First try dimension_name (exact match for ABC lookup)
        for d in a1_abc:
            if isinstance(d, dict):
                dn = (d.get("dimension_name") or "").strip()
            else:
                dn = (_safe_getattr(d, "dimension_name") or "").strip()
            if dn:
                dim_ids.append(dn)

        # Fallback to id if dimension_name is empty
        if not dim_ids:
            for d in a1_abc:
                if isinstance(d, dict):
                    did = (d.get("id") or "").strip()
                else:
                    did = (_safe_getattr(d, "id") or "").strip()
                if did:
                    dim_ids.append(did)

    # Final fallback: Agent2's dimension_ids
    if not dim_ids and agent2_output is not None:
        dim_ids = list(_safe_getattr(agent2_output, "dimension_ids") or [])

    return dim_ids


def _extract_options(question: Any) -> Optional[List[Dict[str, str]]]:
    """Extract options from question object, handling both dict and object forms."""
    opts = _safe_getattr(question, "options") or []
    if not opts:
        return None

    result = []
    for o in opts:
        if isinstance(o, dict):
            label = (o.get("label") or "").strip()
            content = (o.get("content") or "").strip()
        else:
            label = (_safe_getattr(o, "label") or "").strip()
            content = (_safe_getattr(o, "content") or "").strip()
        if label or content:
            result.append({"label": label, "content": content})

    return result if result else None


def _extract_answer_points(question: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract answer points from question object, handling both dict and object forms."""
    aps = _safe_getattr(question, "answer_points") or []
    if not aps:
        return None

    result = []
    for p in aps:
        if isinstance(p, dict):
            result.append(p)
        else:
            result.append({
                "point": _safe_getattr(p, "point"),
                "score": _safe_getattr(p, "score"),
                "evidence_reference": _safe_getattr(p, "evidence_reference"),
            })

    return result if result else None


def validate_stage2_record(record: Stage2Record) -> List[str]:
    """
    Validate a Stage2Record has all required fields.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    core = record.core_input
    if not core:
        errors.append("core_input is None")
        return errors

    # Required fields
    if not core.experiment_id:
        errors.append("core_input.experiment_id is empty")
    if not core.unit_id:
        errors.append("core_input.unit_id is empty")
    if not core.material_text:
        errors.append("core_input.material_text is empty")
    if not core.question_type:
        errors.append("core_input.question_type is empty")
    if not core.stem:
        errors.append("core_input.stem is empty")

    # Type-specific validation
    if core.question_type == "single-choice":
        if not core.options:
            errors.append("single-choice question missing options")
        if not core.correct_answer:
            errors.append("single-choice question missing correct_answer")
    elif core.question_type == "essay":
        if not core.answer_points and core.total_score is None:
            errors.append("essay question missing answer_points and total_score")

    return errors


def build_stage2_record(
    stage1_state: GenerationPipelineState,
    *,
    ablation_skip_agent: str = "none",
) -> Stage2RecordBuildResult:
    """
    Build a Stage2Record from Stage1 pipeline state.

    This is the UNIQUE entry point for constructing Stage2 input from Stage1 output.

    Args:
        stage1_state: The completed Stage1 pipeline state
        ablation_skip_agent: Which agent was skipped in ablation experiment

    Returns:
        Stage2RecordBuildResult with success status, record (if successful), and skip reasons

    Constraints:
        - Does NOT call any LLM
        - Does NOT access Stage2 implementation
        - Only performs field extraction, cleanup, default filling, and type validation
    """
    result = Stage2RecordBuildResult(success=False)

    # === Pre-condition checks ===

    # Check agent1_output exists
    a1 = _safe_getattr(stage1_state, "agent1_output")
    if a1 is None:
        result.skip_reasons.append("agent1_output is None - Stage1 failed at Agent1")
        return result

    # Check agent5_output exists
    v = _safe_getattr(stage1_state, "agent5_output")
    if v is None:
        result.skip_reasons.append("agent5_output is None - Stage1 failed at Agent5")
        return result

    # Check agent5 didn't reject
    is_reject = _safe_getattr(v, "is_reject", default=False)
    if is_reject:
        result.skip_reasons.append("agent5 marked is_reject=True - question failed quality check")
        return result

    # Check agent5 doesn't need revision (unless at max iterations)
    need_revision = _safe_getattr(v, "need_revision", default=False)
    if need_revision:
        result.skip_reasons.append("agent5 marked need_revision=True - question needs more iteration")
        return result

    # Get agent2 output (optional - may be skipped in ablation)
    a2 = _safe_getattr(stage1_state, "agent2_output")

    # === Extract question object ===
    # Priority: agent5.original_question > agent5.question > agent4_output
    q = (
        _safe_getattr(v, "original_question")
        or _safe_getattr(v, "question")
        or _safe_getattr(stage1_state, "agent4_output")
    )

    if q is None:
        result.skip_reasons.append("No question object found in agent5_output or agent4_output")
        return result

    # === Build Stage2CoreInput ===

    # Extract unit_id
    unit_id = _safe_getattr(a1, "unit_id") or _safe_getattr(a1, "material_id")
    if not unit_id:
        result.skip_reasons.append("unit_id not found in agent1_output")
        return result

    # Extract material_text
    material_text = _safe_getattr(a1, "material_text")
    if not material_text:
        result.skip_reasons.append("material_text not found in agent1_output")
        return result

    # Extract question_type
    question_type = _safe_getattr(q, "question_type") or _safe_getattr(a1, "question_type")
    if not question_type:
        result.skip_reasons.append("question_type not found")
        return result

    # Extract stem
    stem = _safe_getattr(q, "stem")
    if not stem:
        result.skip_reasons.append("stem not found in question")
        return result

    # Extract dimension IDs
    dimension_ids = _extract_dimension_ids(a1, a2)
    if not dimension_ids:
        result.warnings.append("dimension_ids is empty - pedagogical eval may skip all dimensions")

    # [2026-01 New] Extract anchor info from agent3_output
    # Only experiments using Stage1 without agent2 ablation will have anchor info
    anchors: Optional[List[Dict[str, Any]]] = None
    anchor_count = 0
    if ablation_skip_agent != "agent2":
        a3 = _safe_getattr(stage1_state, "agent3_output")
        if a3 is not None:
            raw_anchors = _safe_getattr(a3, "anchors") or []
            if raw_anchors:
                anchors = []
                for anchor in raw_anchors:
                    if isinstance(anchor, dict):
                        anchors.append({
                            "snippet": anchor.get("snippet", ""),
                            "reason_for_anchor": anchor.get("reason_for_anchor", ""),
                            "loc": anchor.get("loc", ""),
                            "paragraph_idx": anchor.get("paragraph_idx"),
                        })
                    else:
                        anchors.append({
                            "snippet": _safe_getattr(anchor, "snippet") or "",
                            "reason_for_anchor": _safe_getattr(anchor, "reason_for_anchor") or "",
                            "loc": _safe_getattr(anchor, "loc") or "",
                            "paragraph_idx": _safe_getattr(anchor, "paragraph_idx"),
                        })
                anchor_count = len(anchors)

    # Extract options and answer_points
    options = _extract_options(q)
    answer_points = _extract_answer_points(q)

    # Extract correct_answer (handle both field names)
    correct_answer = _safe_getattr(q, "standard_answer") or _safe_getattr(q, "correct_answer")

    # Build core input
    try:
        core = Stage2CoreInput(
            experiment_id=_safe_getattr(stage1_state, "pipeline_id") or "",
            unit_id=str(unit_id),
            material_text=material_text,
            question_type=question_type,
            stem=stem,
            explanation=_safe_getattr(q, "explanation") or "",
            gk_dims=_safe_getattr(a1, "gk_dims") or {},
            cs_dims=_safe_getattr(a1, "cs_dims") or {},
            exam_skill=_safe_getattr(a1, "exam_skill") or {},
            dimension_ids=dimension_ids,
            options=options,
            correct_answer=correct_answer,
            answer_points=answer_points,
            total_score=_safe_getattr(q, "total_score"),
            # [2026-01 New] Anchor info
            anchors=anchors,
            anchor_count=anchor_count,
        )
    except Exception as e:
        result.skip_reasons.append(f"Failed to build Stage2CoreInput: {e}")
        return result

    # === Build Stage1Meta ===
    quality_score = _safe_getattr(v, "quality_score")
    issues = _safe_getattr(v, "issues") or []

    issue_types = []
    for iss in issues:
        if isinstance(iss, dict):
            it = iss.get("issue_type")
        else:
            it = _safe_getattr(iss, "issue_type")
        if it:
            issue_types.append(it)

    meta = Stage1Meta(
        agent5_overall_score=_safe_getattr(quality_score, "overall_score"),
        agent5_layer_scores=_safe_getattr(quality_score, "layer_scores") or {},
        agent5_need_revision=need_revision,
        agent5_is_reject=is_reject,
        agent5_issue_types=issue_types,
        ablation_skip_agent=ablation_skip_agent,
    )

    # === Build final record ===
    record = Stage2Record(core_input=core, stage1_meta=meta)

    # === Validate ===
    validation_errors = validate_stage2_record(record)
    if validation_errors:
        result.warnings.extend(validation_errors)
        # These are warnings, not fatal - we still return the record

    result.success = True
    result.record = record
    return result


__all__ = [
    "build_stage2_record",
    "validate_stage2_record",
    "Stage2RecordBuildResult",
]
