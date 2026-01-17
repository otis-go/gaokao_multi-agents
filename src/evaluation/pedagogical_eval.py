# -*- coding: utf-8 -*-
# pedagogical_eval.py
# Stage2 Pedagogical Dimension Evaluator - Supports gk (Gaokao dimensions) and cs (Curriculum Standard dimensions) modes
"""
Stage2 Pedagogical Dimension Evaluation Module

- Supports gk (Gaokao dimensions, 17) and cs (Curriculum Standard dimensions, 21) modes, 38 total dimensions
- LLM judges each dimension: hit: true/false + reason
- Calculates Precision/Recall/F1 scores
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging

from src.shared.schemas import (
    PedagogicalHitBasedResult,
    PedagogicalRoundAggregation,
)
from src.shared.config import (
    STAGE2_EVAL_MODELS,
    STAGE2_LLM_PARAMS,
)
from src.shared.llm_interface import LLMClient
from src.shared.prompt_logger import PromptLogger
from src.shared.api_config import is_no_temperature_model
from src.shared.llm_json import extract_json_candidate, repair_common_json

logger = logging.getLogger(__name__)


# ============================================================================
# Dimension Definition Data Class
# ============================================================================

@dataclass
class DimensionDefinition:
    """
    Dimension Definition
    """
    code: str           # GK01-GK17(17) or CS01-CS21(21)
    id: str             # Dimension ID e.g. gk.value_patriotism
    name: str           # Chinese name
    hit_true: str       # Hit condition
    hit_false: str      # Non-hit condition


# ============================================================================
# Load Dimension Definitions from gk_cs_eval.json
# ============================================================================

import os as _os

def _load_gk_cs_eval_config() -> dict:
    """Load gk_cs_eval.json configuration file"""
    # Find configuration file path
    current_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root = _os.path.dirname(_os.path.dirname(current_dir))
    config_path = _os.path.join(project_root, "data", "gk_cs_eval.json")

    if not _os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_dimensions_from_config(config: dict, dim_type: str) -> list:
    """Build DimensionDefinition list from configuration"""
    key = f"{dim_type}_dimensions"
    if key not in config:
        raise KeyError(f"Missing {key} in configuration")

    dimensions = []
    for item in config[key]:
        dimensions.append(DimensionDefinition(
            code=item["code"],
            id=item["id"],
            name=item["name"],
            hit_true=item["hit_true"],
            hit_false=item["hit_false"],
        ))
    return dimensions

# Load configuration and build dimension lists
_GK_CS_EVAL_CONFIG = _load_gk_cs_eval_config()
STAGE2_GK_DIMENSIONS: list = _build_dimensions_from_config(_GK_CS_EVAL_CONFIG, "gk")
STAGE2_CS_DIMENSIONS: list = _build_dimensions_from_config(_GK_CS_EVAL_CONFIG, "cs")
_SYSTEM_PROMPT_TEMPLATE: str = _GK_CS_EVAL_CONFIG.get("system_prompt", "")

# ============================================================================
# High-Frequency Dimension Definitions (for evaluation statistics excluding high-freq dims)
# ============================================================================
# [2026-01-09 Refactoring] Unified import from src.shared.dimension_config
# No longer define loading logic repeatedly in this file

from src.shared.dimension_config import (
    GK_HIGH_FREQ_DIMS,
    CS_HIGH_FREQ_DIMS,
    GK_LOW_FREQ_DIMS,
    CS_LOW_FREQ_DIMS,
    get_high_freq_dims_by_mode,
    get_low_freq_dims_by_mode,
)

# Log loaded high-frequency dimensions (confirm unified source)
logger.info(f"[High-frequency dimensions] GK: {GK_HIGH_FREQ_DIMS} (source: dimension_config)")
logger.info(f"[High-frequency dimensions] CS: {CS_HIGH_FREQ_DIMS} (source: dimension_config)")







# ============================================================================
# Helper Functions: Get dimension lists and mappings by dim_mode
# ============================================================================

def get_dimensions_by_mode(dim_mode: str, low_freq_only: bool = False) -> List["DimensionDefinition"]:
    """
    Return dimension list based on dim_mode

    Args:
        dim_mode: Dimension mode ("gk", "gk_only", "cs", "cs_only")
        low_freq_only: Whether to return only low-frequency dimensions (for low-freq experiments)

    Returns:
        List of dimension definitions
    """
    if dim_mode in ("gk", "gk_only"):
        all_dims = STAGE2_GK_DIMENSIONS
    elif dim_mode in ("cs", "cs_only"):
        all_dims = STAGE2_CS_DIMENSIONS
    else:
        # Default to gk dimensions
        all_dims = STAGE2_GK_DIMENSIONS

    if low_freq_only:
        # Return only low-frequency dimensions
        low_freq_codes = get_low_freq_dims_by_mode(dim_mode)
        if low_freq_codes:
            return [d for d in all_dims if d.code in low_freq_codes]
        # If no low-frequency dimension info found, return all (compatible with old data)
        logger.warning(f"[get_dimensions_by_mode] Low-frequency dimension info not found for {dim_mode}, returning all dimensions")

    return all_dims


def get_dimension_codes_by_mode(dim_mode: str, low_freq_only: bool = False) -> List[str]:
    """
    Return dimension code list based on dim_mode

    Args:
        dim_mode: Dimension mode
        low_freq_only: Whether to return only low-frequency dimension codes
    """
    dims = get_dimensions_by_mode(dim_mode, low_freq_only=low_freq_only)
    return [d.code for d in dims]


def get_dimension_by_code_map(dim_mode: str) -> Dict[str, "DimensionDefinition"]:
    """Return mapping from dimension code to definition based on dim_mode"""
    dims = get_dimensions_by_mode(dim_mode)
    return {d.code: d for d in dims}


def get_dimension_by_id_map(dim_mode: str) -> Dict[str, "DimensionDefinition"]:
    """Return mapping from dimension ID to definition based on dim_mode"""
    dims = get_dimensions_by_mode(dim_mode)
    return {d.id: d for d in dims}


def get_dimension_by_name_map(dim_mode: str) -> Dict[str, "DimensionDefinition"]:
    """Return mapping from dimension name to definition based on dim_mode"""
    dims = get_dimensions_by_mode(dim_mode)
    return {d.name: d for d in dims}


# ============================================================================
# Complete Stage2 Pedagogical Evaluation Prompt Templates (dynamic gk/cs mode support)
# ============================================================================

def _get_dim_info(dim_mode: str, low_freq_only: bool = False) -> tuple:
    """
    Return dimension mode related information

    Args:
        dim_mode: Dimension mode
        low_freq_only: Whether to count only low-frequency dimensions

    Returns:
        (dimension type description, dimension range description, dimension count)
    """
    # Dynamically get actual dimension count
    if low_freq_only:
        dims = get_dimensions_by_mode(dim_mode, low_freq_only=True)
        dim_count = len(dims)
        low_freq_codes = get_low_freq_dims_by_mode(dim_mode)
        if dim_mode in ("gk", "gk_only"):
            dim_range = f"低频GK维度({dim_count}个: {', '.join(sorted(low_freq_codes)[:5])}...)"
            return ("高考维度-低频(gk-low)", dim_range, dim_count)
        elif dim_mode in ("cs", "cs_only"):
            dim_range = f"低频CS维度({dim_count}个)"
            return ("课标维度-低频(cs-low)", dim_range, dim_count)
    else:
        gk_count = len(STAGE2_GK_DIMENSIONS)
        cs_count = len(STAGE2_CS_DIMENSIONS)
        if dim_mode in ("gk", "gk_only"):
            return ("高考维度(gk)", f"GK(17个)", gk_count)
        elif dim_mode in ("cs", "cs_only"):
            return ("课标维度(cs)", f"CS(21个)", cs_count)
        else:
            return ("高考维度(gk)", f"GK(17个)", gk_count)


def build_system_prompt(dim_mode: str) -> str:
    """Build system prompt based on dim_mode (loaded from gk_cs_eval.json)"""
    return _SYSTEM_PROMPT_TEMPLATE

def build_dimension_rules(dim_mode: str, low_freq_only: bool = False) -> str:
    """
    Build dimension judgment rules text based on dim_mode

    Args:
        dim_mode: Dimension mode
        low_freq_only: Whether to build rules only for low-frequency dimensions (for low-freq experiments)
    """
    dim_label, dim_range, dim_count = _get_dim_info(dim_mode, low_freq_only=low_freq_only)
    dimensions = get_dimensions_by_mode(dim_mode, low_freq_only=low_freq_only)

    lines = [f"【{dim_count} 个{dim_label}二分类判定门槛（binary rules）】"]
    for dim in dimensions:
        lines.append(f"\n{dim.code} {dim.name}（{dim.id}）")
        lines.append(f"- hit=true：{dim.hit_true}")
        lines.append(f"- hit=false：{dim.hit_false}")
    return "\n".join(lines)


def build_output_format(dim_mode: str, low_freq_only: bool = False) -> str:
    """
    Build output format text based on dim_mode

    Args:
        dim_mode: Dimension mode
        low_freq_only: Whether to build output format only for low-frequency dimensions (for low-freq experiments)
    """
    dim_label, dim_range, dim_count = _get_dim_info(dim_mode, low_freq_only=low_freq_only)
    dimensions = get_dimensions_by_mode(dim_mode, low_freq_only=low_freq_only)

    lines = [f'【输出格式（必须严格 JSON，且必须包含全部 {dim_count} 个维度）】']
    lines.append('{')
    lines.append('  "question_id": "<question_id>",')
    lines.append('  "dimensions": {')

    for i, dim in enumerate(dimensions):
        comma = "," if i < len(dimensions) - 1 else ""
        if i == 0:
            lines.append(f'    "{dim.code}": {{"hit": true/false, "reason": "…(1–2句，指向stem/analysis证据)…"}}{comma}')
        else:
            lines.append(f'    "{dim.code}": {{"hit": true/false, "reason": "…"}}{comma}')

    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)


# ============================================================================
# Helper Functions
# ============================================================================

def _response_to_text(resp: Any) -> str:
    """Unify various return formats from LLMClient.generate to str."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("content", "text", "message", "output"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(resp, ensure_ascii=False)
    return str(resp)


def _safe_json_loads(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """json.loads that never throws exceptions."""
    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Core Functions
# ============================================================================

def build_hit_based_prompt(question: Any, dim_mode: str = "gk", low_freq_only: bool = False) -> str:
    """
    Build complete hit-based evaluation prompt.

    Args:
        question: Question object
        dim_mode: Dimension mode, "gk" or "cs"
        low_freq_only: Whether to evaluate only low-frequency dimensions (for low-freq experiments)

    Uses built-in dimension definitions, selects GK or CS dimensions based on dim_mode.
    """
    dim_label, dim_range, dim_count = _get_dim_info(dim_mode, low_freq_only=low_freq_only)

    stem = str(getattr(question, "stem", "") or "").strip()
    material = str(getattr(question, "material_text", "") or "").strip()
    explanation = str(getattr(question, "explanation", "") or "").strip()
    question_type = str(getattr(question, "question_type", "") or "").strip()

    # [2025-12 Fix] Answer handling: distinguish multiple choice vs essay questions
    # Multiple choice uses correct_answer, essay questions use answer_points
    correct_answer = (
        getattr(question, "correct_answer", None)
        or getattr(question, "standard_answer", None)
        or ""
    )
    correct_answer = str(correct_answer).strip()

    # [2025-12 New] Answer points handling for essay questions
    answer_text = correct_answer  # Default to correct_answer
    answer_points = getattr(question, "answer_points", None) or []
    total_score = getattr(question, "total_score", None)

    if question_type == "essay" and answer_points:
        # Essay question: build answer text from answer_points
        lines = []
        for i, pt in enumerate(answer_points, 1):
            if isinstance(pt, dict):
                point_text = str(pt.get("point", "")).strip()
                point_score = pt.get("score", "")
                evidence_ref = pt.get("evidence_reference", "")
            else:
                point_text = str(getattr(pt, "point", "") or "").strip()
                point_score = getattr(pt, "score", "")
                evidence_ref = getattr(pt, "evidence_reference", "")

            line = f"  {i}. {point_text}"
            if point_score:
                line += f"（{point_score}分）"
            if evidence_ref:
                if isinstance(evidence_ref, list):
                    evidence_ref = ", ".join(str(e) for e in evidence_ref)
                line += f" [依据: {evidence_ref}]"
            lines.append(line)

        answer_text = "\n".join(lines)
        if total_score:
            answer_text += f"\n  （总分：{total_score}分）"

    # Option handling (append to stem)
    options_lines: List[str] = []
    opts = getattr(question, "options", None) or []
    for idx, opt in enumerate(opts, 1):
        if isinstance(opt, dict):
            label = str(opt.get("label", "") or "").strip()
            content = str(opt.get("content", "") or "").strip()
        else:
            label = str(getattr(opt, "label", "") or "").strip()
            content = str(getattr(opt, "content", "") or "").strip()
        if not label and content:
            label = str(idx)
        if label or content:
            options_lines.append(f"{label}. {content}".rstrip())

    # Append options to stem
    stem_with_options = stem
    if options_lines:
        stem_with_options = stem + "\n" + "\n".join(options_lines)

    # Get question_id
    question_id = str(getattr(question, "question_id", "") or "").strip()
    if not question_id:
        question_id = stem[:30] + "..." if len(stem) > 30 else stem

    # Dynamically build prompt
    system_prompt = build_system_prompt(dim_mode)
    dimension_rules = build_dimension_rules(dim_mode, low_freq_only=low_freq_only)
    output_format = build_output_format(dim_mode, low_freq_only=low_freq_only)

    # [2025-12 Fix] Adjust answer label based on question type
    answer_label = "answer（标准答案/答案要点）" if question_type == "essay" else "answer（标准答案）"

    # Build complete prompt
    return f"""{system_prompt}

{dimension_rules}

{output_format}

---
【待评估题目】
- question_id: {question_id}
- question_type: {question_type or "未知"}
- material（阅读材料）: {material or "(未提供)"}
- stem（题干/设问/选项）: {stem_with_options or "(未提供)"}
- {answer_label}: {answer_text or "(未提供)"}
- analysis（解析/评分要点）: {explanation or "(未提供)"}

请严格按照上述格式输出 JSON，确保包含全部 {dim_count} 个维度的评估结果。""".strip()


def parse_hit_based_response(
    response: Any,
    model_name: str,
    dim_mode: str = "gk",
    low_freq_only: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Parse hit-based response, return dimension evaluation results.

    Args:
        response: LLM response
        model_name: Model name
        dim_mode: Dimension mode, "gk" or "cs"
        low_freq_only: Whether to parse only low-frequency dimensions (for low-freq experiments)

    Returns:
    - dimensions_result: {GK01/CS01: {"hit": bool, "reason": str}, ...}
    - audit: Parse audit information
    """
    text = _response_to_text(response)
    audit: Dict[str, Any] = {"model": model_name, "parse_status": "ok"}

    # Get dimension code list for current mode
    all_dim_codes = get_dimension_codes_by_mode(dim_mode, low_freq_only=low_freq_only)

    # Default: all not hit
    dimensions_result: Dict[str, Dict[str, Any]] = {
        code: {"hit": False, "reason": "Parse failed, default not hit"}
        for code in all_dim_codes
    }

    # Use unified JSON extraction function
    cand = extract_json_candidate(text)
    if not cand:
        audit["parse_status"] = "fallback"
        audit["parse_error"] = "no_json_object_found"
        return dimensions_result, audit

    # Use unified JSON repair function
    cand = repair_common_json(cand)
    parsed, err = _safe_json_loads(cand)

    if parsed is None:
        audit["parse_status"] = "fallback"
        audit["parse_error"] = f"json_loads_failed: {err}"
        return dimensions_result, audit

    if not isinstance(parsed, dict):
        audit["parse_status"] = "fallback"
        audit["parse_error"] = f"top_level_not_dict: {type(parsed).__name__}"
        return dimensions_result, audit

    dims_data = parsed.get("dimensions")
    if dims_data is None and all(isinstance(k, str) for k in parsed.keys()):
        # May have directly returned {GK01: {...}, GK02: {...}}
        dims_data = parsed

    if not isinstance(dims_data, dict):
        audit["parse_status"] = "fallback"
        audit["parse_error"] = f"dimensions_not_dict: {type(dims_data).__name__}"
        return dimensions_result, audit

    # Parse each dimension (in GK01-GK17 or CS01-CS21 format)
    for code in all_dim_codes:
        raw_item = dims_data.get(code)

        if not isinstance(raw_item, dict):
            continue

        hit_val = raw_item.get("hit")
        if isinstance(hit_val, bool):
            hit = hit_val
        elif isinstance(hit_val, str):
            hit = hit_val.lower() in ("true", "1", "yes", "是")
        else:
            hit = False

        reason = str(raw_item.get("reason", "") or "").strip() or "No reason provided"

        dimensions_result[code] = {"hit": hit, "reason": reason}

    return dimensions_result, audit


def calculate_prf(
    gold_dims: List[str],
    predicted_dims: List[str],
    exclude_dims: set = None,
) -> Tuple[int, int, int, float, float, float]:
    """
    Calculate Precision/Recall/F1.

    Args:
        gold_dims: Gold standard dimension list (GK01-GK17/CS01-CS21 format)
        predicted_dims: Predicted hit dimension list
        exclude_dims: Set of high-frequency dimensions to exclude (optional)

    Returns:
        (tp, fp, fn, precision, recall, f1)
    """
    gold_set = set(gold_dims)
    pred_set = set(predicted_dims)

    # Exclude high-frequency dimensions
    if exclude_dims:
        gold_set = gold_set - exclude_dims
        pred_set = pred_set - exclude_dims

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def compute_model_bias_analysis(
    results: List['PedagogicalHitBasedResult'],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    [2026-01 New] Compute per-model evaluation bias analysis

    Analyze P/R/F1 performance for each evaluation model, determine bias tendency:
    - Precision-biased (conservative): Precision significantly higher than Recall
    - Recall-biased (lenient): Recall significantly higher than Precision
    - Balanced: Small difference between the two

    Args:
        results: List of evaluation results for each question

    Returns:
        {"model_name": {"avg_precision": 0.78, "avg_recall": 0.65, "avg_f1": 0.71, "bias": "Precision-biased (conservative)"}, ...}
    """
    if not results:
        return None

    # Collect prediction results for each model
    model_preds: Dict[str, List[Tuple[List[str], List[str]]]] = {}  # model -> [(gold, pred), ...]

    for r in results:
        if not hasattr(r, 'model_results') or not r.model_results:
            continue

        gold_dims = r.gold_dimensions

        for model_name, dims_result in r.model_results.items():
            if model_name not in model_preds:
                model_preds[model_name] = []

            # Get this model's prediction for this question
            predicted = [code for code, item in dims_result.items() if item.get("hit")]
            model_preds[model_name].append((gold_dims, predicted))

    if not model_preds:
        return None

    # Calculate metrics for each model
    analysis = {}
    for model_name, pred_pairs in model_preds.items():
        total_tp, total_fp, total_fn = 0, 0, 0

        for gold, pred in pred_pairs:
            tp, fp, fn, _, _, _ = calculate_prf(gold, pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Micro average
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Determine bias tendency
        diff = precision - recall
        if diff > 0.1:
            bias = "Precision-biased (conservative)"
        elif diff < -0.1:
            bias = "Recall-biased (lenient)"
        else:
            bias = "Balanced"

        analysis[model_name] = {
            "avg_precision": round(precision, 3),
            "avg_recall": round(recall, 3),
            "avg_f1": round(f1, 3),
            "bias": bias,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "questions_evaluated": len(pred_pairs),
        }

    return analysis if analysis else None


# ============================================================================
# Main Evaluation Class
# ============================================================================

class PedagogicalHitBasedEval:
    """
    Pedagogical Evaluator - Hit/not-hit based P/R/F1 evaluation

    Supports gk (Gaokao dimensions GK01-GK17) and cs (Curriculum Standard CS01-CS21) modes.

    [2026-01 Added] Low-frequency dimension experiment support
    - low_freq_only=True: Only use low-frequency dimensions for evaluation (for low-freq dimension experiments)
    - In low-freq mode, evaluation prompts only contain low-freq dims, not high-freq dims
    """

    def __init__(
        self,
        llm_clients: Optional[List[LLMClient]] = None,
        prompt_logger: Optional[PromptLogger] = None,
        dim_mode: str = "gk",
        low_freq_only: bool = False,  # [2026-01 New] Low-frequency dimension experiment mode
        **kwargs,  # Ignore other parameters (like data_loader)
    ):
        self.prompt_logger = prompt_logger
        self.dim_mode = dim_mode
        self.low_freq_only = low_freq_only  # [2026-01 New] Save low-freq mode flag

        # Get dimension list based on low-freq mode
        self.dimensions = get_dimensions_by_mode(dim_mode, low_freq_only=low_freq_only)
        self.dimension_codes = get_dimension_codes_by_mode(dim_mode, low_freq_only=low_freq_only)

        # Print initialization info
        if low_freq_only:
            print(f"[PedagogicalHitBasedEval] Initialized in low-frequency mode: dim_mode={dim_mode}, dimension_count={len(self.dimension_codes)}")
            print(f"  Low-freq dimensions: {self.dimension_codes}")
        else:
            print(f"[PedagogicalHitBasedEval] Initialized in full dimension mode: dim_mode={dim_mode}, dimension_count={len(self.dimension_codes)}")

        # LLM clients
        self.llm_clients: List[LLMClient] = llm_clients or self._build_default_clients()

    def _build_default_clients(self) -> List[LLMClient]:
        clients: List[LLMClient] = []
        for model_name in STAGE2_EVAL_MODELS:
            clients.append(LLMClient(api_type="openai", model_name=model_name, verbose=False))
        return clients

    def evaluate_single(
        self,
        question: Any,
        gold_dimensions: List[str],
    ) -> Optional[PedagogicalHitBasedResult]:
        """
        Perform hit-based evaluation on a single question.

        Args:
            question: Question object
            gold_dimensions: Gold standard dimension list (GK01-GK17/CS01-CS21 format or dimension ID format)

        Returns:
            PedagogicalHitBasedResult, or None if question has no dimensions of corresponding type (skip)
        """
        question_id = (getattr(question, "stem", "") or "")[:30] + "..."

        # Convert gold_dimensions to dimension code format
        gold_codes = self._normalize_dimensions(gold_dimensions)

        # [2026-01 Fix] In low-freq mode, ensure gold_dimensions only contains low-freq dimensions
        # Prevent high-freq dimensions from accidentally mixing in and causing abnormal Recall
        if self.low_freq_only:
            low_freq_codes = get_low_freq_dims_by_mode(self.dim_mode)
            original_count = len(gold_codes)
            gold_codes = [c for c in gold_codes if c in low_freq_codes]
            filtered_count = original_count - len(gold_codes)
            if filtered_count > 0:
                logger.debug(f"[PedagogicalHitBasedEval] Low-freq mode filtered {filtered_count} high-freq gold dimensions")

        # [2025-12-31 New] If question has no gold dimensions of corresponding type, skip evaluation
        if not gold_codes:
            logger.info(f"[PedagogicalHitBasedEval] Skip question (no {self.dim_mode} dimensions): {question_id}")
            print(f"[PedagogicalHitBasedEval] Skip question (no {self.dim_mode} dimensions): {question_id}")
            return None

        # [2026-01 Modified] Use low_freq_only parameter to build prompt
        prompt = build_hit_based_prompt(question, self.dim_mode, low_freq_only=self.low_freq_only)

        model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        audits: List[Dict[str, Any]] = []

        for client in self.llm_clients:
            model_name = client.model_name
            try:
                messages = [{"role": "user", "content": prompt}]
                gen_kwargs = {
                    "max_tokens": STAGE2_LLM_PARAMS.get("max_tokens", 4096),
                    "metadata": {"agent": "PedagogicalHitBasedEval"},
                }
                if not is_no_temperature_model(model_name or ""):
                    gen_kwargs["temperature"] = STAGE2_LLM_PARAMS.get("temperature", 0.0)

                resp = client.generate(messages, **gen_kwargs)

                if self.prompt_logger:
                    self.prompt_logger.save_agent_log(
                        agent_name="PedagogicalHitBasedEval",
                        stage="pedagogical_hit_based",
                        prompt=prompt,
                        response=resp,
                        metadata={"question_id": question_id, "model_name": model_name, "dim_mode": self.dim_mode, "low_freq_only": self.low_freq_only},
                        model=model_name,
                    )

                # [2026-01 Modified] Use low_freq_only parameter to parse response
                dims_result, audit = parse_hit_based_response(resp, model_name, self.dim_mode, low_freq_only=self.low_freq_only)
                model_results[model_name] = dims_result
                audits.append(audit)

            except Exception as e:
                audits.append({"model": model_name, "parse_status": "error", "parse_error": str(e)})
                logger.warning(f"[PedagogicalHitBasedEval] Model {model_name} failed: {e}")

        if not model_results:
            raise RuntimeError(f"All models failed. audits={audits[:3]}")

        # Aggregate multi-model results: majority voting
        aggregated: Dict[str, Dict[str, Any]] = {}
        for code in self.dimension_codes:
            hit_votes = 0
            total_votes = 0
            reasons: List[str] = []

            for model_name, dims_result in model_results.items():
                item = dims_result.get(code, {})
                if item.get("hit"):
                    hit_votes += 1
                total_votes += 1
                if item.get("reason"):
                    reasons.append(item["reason"])

            # Majority voting determines hit
            final_hit = hit_votes > total_votes / 2
            final_reason = reasons[0] if reasons else "No reason provided"

            aggregated[code] = {"hit": final_hit, "reason": final_reason}

        # Extract predicted hit dimensions
        predicted_dims = [code for code, item in aggregated.items() if item.get("hit")]

        # Calculate P/R/F1
        tp, fp, fn, precision, recall, f1 = calculate_prf(gold_codes, predicted_dims)

        # [2026-01 New] Calculate Off-target (false positive rate)
        off_target = 1.0 - precision  # FP / (TP + FP)

        # Calculate missing and extra dimensions
        gold_set = set(gold_codes)
        pred_set = set(predicted_dims)
        missing_dims = sorted(list(gold_set - pred_set))  # FN: should hit but didn't
        extra_dims = sorted(list(pred_set - gold_set))    # FP: shouldn't hit but did

        # Print missing dimension info
        if missing_dims:
            print(f"[PedagogicalHitBasedEval] Missing dimensions ({len(missing_dims)}): {missing_dims}")
        print(f"[PedagogicalHitBasedEval] Completed: TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Off-target={off_target:.3f}")

        return PedagogicalHitBasedResult(
            question_id=question_id,
            gold_dimensions=gold_codes,
            predicted_dimensions=predicted_dims,
            missing_dimensions=missing_dims,
            extra_dimensions=extra_dims,
            dimension_results=aggregated,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            off_target=off_target,
            model_results=model_results,
            audit={"model_audits": audits, "models_count": len(self.llm_clients), "dim_mode": self.dim_mode},
        )

    def _normalize_dimensions(self, dimensions: List[str]) -> List[str]:
        """
        Normalize dimension list to dimension code format.

        Supported input formats:
        - GK01, GK02, ... or CS01, CS02, ... (use directly)
        - gk.xxx, cs.xxx dimension ID format (convert to corresponding code)
        - Chinese names like "核心价值-爱国主义情怀" (convert via name mapping)

        Note:
        - Question type dimensions (like "错误选项的设置类型-有无") do not belong to pedagogical dimensions, will be silently filtered
        """
        # Get dimension mappings for current mode
        dim_by_code = get_dimension_by_code_map(self.dim_mode)
        dim_by_id = get_dimension_by_id_map(self.dim_mode)
        dim_by_name = get_dimension_by_name_map(self.dim_mode)

        # [2025-12 Fix] Question type dimension keyword list - these don't belong to pedagogical evaluation dimensions, silently ignore
        # Question type dimension naming format usually contains these keywords
        QUESTION_TYPE_KEYWORDS = (
            "错误选项", "正确选项", "选项设置", "题型梳理",
            "主观题", "选择题", "简答题", "干扰项",
            "信息筛选", "论点与论据", "同义替换", "合理推断", "合理概括", "偷换概念",
        )

        result = []
        for dim in dimensions:
            dim = dim.strip()
            if not dim:
                continue

            # Check if it's a question type dimension - skip silently without warning
            is_question_type_dim = any(kw in dim for kw in QUESTION_TYPE_KEYWORDS)
            if is_question_type_dim:
                # Question type dimensions don't belong to pedagogical evaluation scope, silently ignore
                continue

            if dim in dim_by_code:
                # Already in dimension code format
                result.append(dim)
            elif dim in dim_by_id:
                # Is dimension ID format, convert to code
                result.append(dim_by_id[dim].code)
            elif dim in dim_by_name:
                # Is standard Chinese name
                result.append(dim_by_name[dim].code)
            else:
                # Unknown format, log warning but don't add
                logger.warning(f"[PedagogicalHitBasedEval] Unknown dimension format: {dim} (dim_mode={self.dim_mode})")
        return result

    @staticmethod
    def aggregate_round(
        results: List[PedagogicalHitBasedResult],
        round_id: str = "",
        dim_mode: str = "gk",
        success_threshold: float = 0.8,  # [2026-01 Refactoring] Threshold changed from 0.6 to 0.8
    ) -> PedagogicalRoundAggregation:
        """
        Aggregate results from one evaluation round, calculate micro/macro averages.

        Args:
            results: List of evaluation results for each question
            round_id: Round identifier
            dim_mode: Dimension mode (gk/cs), used to determine high-frequency dimensions
            success_threshold: Success@k threshold (default 0.8)

        Returns:
            PedagogicalRoundAggregation
        """
        if not results:
            return PedagogicalRoundAggregation(round_id=round_id)

        # Calculate global TP/FP/FN (for micro average)
        total_tp = sum(r.tp for r in results)
        total_fp = sum(r.fp for r in results)
        total_fn = sum(r.fn for r in results)

        # Micro average
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # [2026-01 New] Off-target and Success@k
        off_target = 1.0 - micro_precision  # Overall false positive rate
        success_count = sum(1 for r in results if r.recall >= success_threshold)
        success_at_k = success_count / len(results) if results else 0.0

        # [2026-01-07 Fix] Macro average: dimension perspective (calculate P/R/F1 for each dimension, then average)
        dim_stats_macro: Dict[str, Dict[str, int]] = {}
        for r in results:
            gold = set(r.gold_dimensions) if r.gold_dimensions else set()
            pred = set(r.predicted_dimensions) if r.predicted_dimensions else set()
            all_dims = gold | pred
            for d in all_dims:
                if d not in dim_stats_macro:
                    dim_stats_macro[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats_macro[d]["tp"] += 1
                elif d in pred and d not in gold:
                    dim_stats_macro[d]["fp"] += 1
                elif d in gold and d not in pred:
                    dim_stats_macro[d]["fn"] += 1

        dim_p_list, dim_r_list, dim_f1_list = [], [], []
        for d, st in dim_stats_macro.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p_list.append(p_d)
            dim_r_list.append(r_d)
            dim_f1_list.append(f1_d)

        macro_precision = sum(dim_p_list) / len(dim_p_list) if dim_p_list else 0.0
        macro_recall = sum(dim_r_list) / len(dim_r_list) if dim_r_list else 0.0
        macro_f1 = sum(dim_f1_list) / len(dim_f1_list) if dim_f1_list else 0.0

        # Dimension statistics
        dimension_stats: Dict[str, Dict[str, int]] = {}
        for r in results:
            for code, item in r.dimension_results.items():
                if code not in dimension_stats:
                    dimension_stats[code] = {"hit_count": 0, "total": 0}
                dimension_stats[code]["total"] += 1
                if item.get("hit"):
                    dimension_stats[code]["hit_count"] += 1

        print(f"[PedagogicalHitBasedEval] Round aggregation: {len(results)} questions")
        print(f"  Micro: P={micro_precision:.3f}, R={micro_recall:.3f}, F1={micro_f1:.3f}")
        print(f"  Off-target={off_target:.3f}, Success@{success_threshold}={success_at_k:.3f} ({success_count}/{len(results)})")
        print(f"  Macro: P={macro_precision:.3f}, R={macro_recall:.3f}, F1={macro_f1:.3f}")

        # ========== Count skipped questions with no dimensions ==========
        # [2025-12-31 New] Count questions with empty original gold dimensions
        skipped_no_dims = sum(1 for r in results if not r.gold_dimensions)
        if skipped_no_dims > 0:
            print(f"  Note: {skipped_no_dims} questions have no {dim_mode.upper()} dimensions (skipped)")

        # ========== [2026-01 Refactoring] metrics_rare_only: low-frequency dimensions only metrics ==========
        # If question has empty gold after removing high-freq dimensions, skip that question (don't count in statistics)
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        metrics_rare_only = None

        if high_freq_dims:
            # Recalculate TP/FP/FN after excluding high-frequency dimensions
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0  # Number of questions skipped for having only high-freq dimensions
            excl_valid_results = []  # Valid results list (with dimensions after removing high-freq) (with P/R/F1)

            for r in results:
                # Check if there are remaining gold dimensions after removing high-freq dimensions
                gold_after_excl = set(r.gold_dimensions) - high_freq_dims
                if not gold_after_excl:
                    # This question only has high-freq dimensions, skip from statistics
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, p_ex, r_ex, f1_ex = calculate_prf(
                    r.gold_dimensions, r.predicted_dimensions, exclude_dims=high_freq_dims
                )
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                # Save for Macro and Success@k calculation
                # [2026-01-07 Fix] Also save gold/pred after excluding high-freq for dimension perspective macro calculation
                gold_excl = set(r.gold_dimensions) - high_freq_dims if r.gold_dimensions else set()
                pred_excl = set(r.predicted_dimensions) - high_freq_dims if r.predicted_dimensions else set()
                excl_valid_results.append({
                    'precision': p_ex, 'recall': r_ex, 'f1': f1_ex,
                    'gold': gold_excl, 'pred': pred_excl
                })

            # Calculate Micro average after excluding high-freq
            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # [2026-01-07 Fix] Calculate Macro average after excluding high-freq: dimension perspective
            excl_dim_stats: Dict[str, Dict[str, int]] = {}
            for er in excl_valid_results:
                gold = er['gold']
                pred = er['pred']
                all_dims = gold | pred
                for d in all_dims:
                    if d not in excl_dim_stats:
                        excl_dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                    if d in gold and d in pred:
                        excl_dim_stats[d]["tp"] += 1
                    elif d in pred and d not in gold:
                        excl_dim_stats[d]["fp"] += 1
                    elif d in gold and d not in pred:
                        excl_dim_stats[d]["fn"] += 1

            excl_dim_p, excl_dim_r, excl_dim_f1 = [], [], []
            for d, st in excl_dim_stats.items():
                tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
                p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
                r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
                f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
                excl_dim_p.append(p_d)
                excl_dim_r.append(r_d)
                excl_dim_f1.append(f1_d)

            excl_macro_p = sum(excl_dim_p) / len(excl_dim_p) if excl_dim_p else 0.0
            excl_macro_r = sum(excl_dim_r) / len(excl_dim_r) if excl_dim_r else 0.0
            excl_macro_f1 = sum(excl_dim_f1) / len(excl_dim_f1) if excl_dim_f1 else 0.0

            # Calculate Off-target and Success@k after excluding high-freq
            excl_off_target = 1.0 - excl_micro_p
            excl_success_count = sum(1 for r in excl_valid_results if r['recall'] >= success_threshold)
            excl_success_at_k = excl_success_count / len(excl_valid_results) if excl_valid_results else 0.0

            metrics_rare_only = {
                "excluded_dims": sorted(list(high_freq_dims)),
                "note": "Only count low-frequency dimensions (after excluding high-frequency dimensions)",
                "micro_precision": excl_micro_p,
                "micro_recall": excl_micro_r,
                "micro_f1": excl_micro_f1,
                "macro_precision": excl_macro_p,
                "macro_recall": excl_macro_r,
                "macro_f1": excl_macro_f1,
                "total_tp": excl_total_tp,
                "total_fp": excl_total_fp,
                "total_fn": excl_total_fn,
                "valid_questions": len(excl_valid_results),
                "skipped_only_high_freq": skipped_only_high_freq,
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            print(f"  [metrics_rare_only] Excluding high-freq dimensions {sorted(high_freq_dims)} (skipped: {skipped_only_high_freq}, valid: {len(excl_valid_results)}):")
            print(f"    Micro: P={excl_micro_p:.3f}, R={excl_micro_r:.3f}, F1={excl_micro_f1:.3f}")
            print(f"    Macro: P={excl_macro_p:.3f}, R={excl_macro_r:.3f}, F1={excl_macro_f1:.3f}")
            print(f"    Off-target={excl_off_target:.3f}, Success@{success_threshold}={excl_success_at_k:.3f} ({excl_success_count}/{len(excl_valid_results)})")

        # ========== [2026-01 New] Per-model evaluation bias analysis ==========
        model_bias_analysis = compute_model_bias_analysis(results)
        if model_bias_analysis:
            print("  [Per-model bias analysis]")
            for model_name, bias_data in model_bias_analysis.items():
                print(f"    {model_name}: P={bias_data['avg_precision']:.3f}, R={bias_data['avg_recall']:.3f}, F1={bias_data['avg_f1']:.3f} -> {bias_data['bias']}")

        return PedagogicalRoundAggregation(
            round_id=round_id,
            question_results=results,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            dimension_stats=dimension_stats,
            skipped_no_dims=skipped_no_dims,  # [2025-12-31] Number of questions skipped for having no original dimensions
            metrics_rare_only=metrics_rare_only,  # [2026-01] Renamed: exclude_high_freq -> metrics_rare_only
            off_target=off_target,
            success_at_k=success_at_k,
            success_threshold=success_threshold,
            model_bias_analysis=model_bias_analysis,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "PedagogicalHitBasedEval",
    "build_hit_based_prompt",
    "parse_hit_based_response",
    "calculate_prf",
    # Dimension definitions
    "STAGE2_GK_DIMENSIONS",
    "STAGE2_CS_DIMENSIONS",
    "DimensionDefinition",
    # High-frequency dimension definitions
    "GK_HIGH_FREQ_DIMS",
    "CS_HIGH_FREQ_DIMS",
    "get_high_freq_dims_by_mode",
    # Dimension helper functions
    "get_dimensions_by_mode",
    "get_dimension_codes_by_mode",
    "get_dimension_by_code_map",
    "get_dimension_by_id_map",
    "get_dimension_by_name_map",
    # Prompt building functions
    "build_system_prompt",
    "build_dimension_rules",
    "build_output_format",
]
