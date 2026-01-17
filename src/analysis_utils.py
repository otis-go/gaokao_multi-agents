# from pathlib import Path
# from src.analysis.analysis_utils import iter_analysis_rows
# import pandas as pd
#
# pipeline_dir = Path("outputs/demo_exp_001/pipeline_states")
# eval_dir = Path("outputs/demo_exp_001/evaluation_states")
#
# rows = list(iter_analysis_rows(pipeline_dir, eval_dir))
# df = pd.DataFrame([r.__dict__ for r in rows])
#
# # Then df can be used for groupby / plotting / exporting to csv


# src/analysis/analysis_utils.py
# Analysis Utilities Module

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from src.shared.schemas import EvaluationPipelineState


@dataclass
class AnalysisRow:
    """
    A single row for downstream DataFrame/statistics (one row per question)
    """

    experiment_id: str
    pipeline_id: str

    unit_id: Optional[str]
    question_type: Optional[str]

    # Dimension info (from Stage 1)
    dimension_ids: List[str]

    # Stage 1 Agent5 (lightweight verifier) results (if any)
    agent5_overall_score: Optional[float]
    agent5_need_revision: Optional[bool]

    # Stage 2 AI evaluation results (if any)
    ai_score: Optional[float]
    ai_decision: Optional[str]

    # Stage 2 pedagogical evaluation results (if any)
    ped_overall_score: Optional[float]
    ped_decision: Optional[str]
    ped_dim_count: Optional[int]

    # Orchestrator final decision
    final_decision: Optional[str]


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_stage1_fields(state_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fields we care about from pipeline_state_xxx.json
    """
    pipeline_id = state_json.get("pipeline_id", "")
    experiment_id = state_json.get("experiment_id", "")

    # Question info (assuming Stage1 state has unified_question / stage2_record.core_input)
    # First try stage2_record.core_input, then fallback
    core_input = _safe_get(state_json, "stage2_record", "core_input", default={}) or {}
    unit_id = core_input.get("unit_id") or _safe_get(state_json, "agent1_output", "unit_id")
    question_type = core_input.get("question_type") or _safe_get(
        state_json, "agent1_output", "question_type"
    )
    dimension_ids = core_input.get("dimension_ids") or []

    # Agent5 results (if agent5_output exists in state)
    agent5 = state_json.get("agent5_output") or {}
    agent5_overall_score = agent5.get("overall_score")
    agent5_need_revision = agent5.get("need_revision")

    return dict(
        experiment_id=experiment_id,
        pipeline_id=pipeline_id,
        unit_id=unit_id,
        question_type=question_type,
        dimension_ids=dimension_ids,
        agent5_overall_score=agent5_overall_score,
        agent5_need_revision=agent5_need_revision,
        )


def _extract_stage2_fields(eval_state_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AI / Ped summary fields from evaluation_state_xxx.json
    """
    state = EvaluationPipelineState(**eval_state_json)  # Use existing wrapper

    # AI evaluation
    ai = getattr(state, "ai_eval_result", None)
    ai_score = None
    ai_decision = None
    if isinstance(ai, dict):
        ai_score = ai.get("overall_score") or ai.get("score")
        ai_decision = ai.get("decision")

    # Pedagogical evaluation
    ped = getattr(state, "pedagogical_eval_result", None)
    ped_overall_score = None
    ped_decision = None
    ped_dim_count = None
    if ped is not None:
        if hasattr(ped, "overall_score"):
            ped_overall_score = getattr(ped, "overall_score")
        if hasattr(ped, "decision"):
            ped_decision = getattr(ped, "decision")
        dim_results = getattr(ped, "dimension_results", None)
        if isinstance(dim_results, (list, tuple)):
            ped_dim_count = len(dim_results)

    return dict(
        ai_score=ai_score,
        ai_decision=ai_decision,
        ped_overall_score=ped_overall_score,
        ped_decision=ped_decision,
        ped_dim_count=ped_dim_count,
        final_decision=getattr(state, "final_decision", None),
    )


def iter_analysis_rows(
    pipeline_state_dir: Path,
    evaluation_state_dir: Path,
) -> Iterable[AnalysisRow]:
    """
    Iterate over two directories and combine pipeline_state_xxx + evaluation_state_xxx into AnalysisRow.

    Convention:
    - The * part of pipeline_state_*_final.json and evaluation_state_*_final.json should match.
    """
    # First index evaluation_state: key = middle segment id
    eval_index: Dict[str, Dict[str, Any]] = {}
    for eval_path in evaluation_state_dir.glob("evaluation_state_*_final.json"):
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = eval_path.name.replace("evaluation_state_", "").replace("_final.json", "")
        eval_index[key] = data

    # Then iterate over pipeline_state
    for pipe_path in pipeline_state_dir.glob("pipeline_state_*_final.json"):
        key = pipe_path.name.replace("pipeline_state_", "").replace("_final.json", "")
        with pipe_path.open("r", encoding="utf-8") as f:
            pipe_json = json.load(f)

        stage1_fields = _extract_stage1_fields(pipe_json)
        stage2_json = eval_index.get(key)
        stage2_fields: Dict[str, Any] = (
            _extract_stage2_fields(stage2_json) if stage2_json is not None else {}
        )

        row_kwargs = {**stage1_fields, **stage2_fields}
        yield AnalysisRow(**row_kwargs)
