# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os as _os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# =============================================================================
# [2025-12 Refactor] AI Dimension Evaluation Module
#
# Core Design:
# 1) Config-driven: anchor validation rules loaded from data/ai_eval.json
# 2) Parse-first: Parse "hard anchors" in reason (K / N / |C| / low-level templates)
# 3) Pure validation: Non-compliant outputs are marked INVALID
# 4) No repair calls: INVALID model-dimension pairs are excluded during aggregation
#
# Config File Structure (data/ai_eval.json):
# - system_prompts: System prompts
# - dimension_prompts: Dimension evaluation rules
# - output_constraints: Output format constraints
# - anchor_validation_rules: Anchor validation rules
# =============================================================================


class AIEvalConfigError(Exception):
    """AI evaluation config loading error."""
    pass


def _load_ai_eval_config() -> dict:
    """
    Load AI evaluation config from data/ai_eval.json (fail-fast mode).

    Returns:
        dict: Complete config dict containing anchor_validation_rules etc.

    Raises:
        AIEvalConfigError: Raised when config file doesn't exist or parse fails.
    """
    current_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root = _os.path.dirname(_os.path.dirname(current_dir))
    config_path = _os.path.join(project_root, "data", "ai_eval.json")

    # [2025-12 Refactor] Fail-fast mode: raise error on config load failure, no silent fallback to {}
    if not _os.path.exists(config_path):
        raise AIEvalConfigError(
            f"[FATAL] AI evaluation config file not found: {config_path}\n"
            f"Please ensure data/ai_eval.json exists and is properly formatted."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise AIEvalConfigError(
            f"[FATAL] AI evaluation config JSON parse failed: {config_path}\n"
            f"Error details: {e}\n"
            f"Please check JSON syntax."
        ) from e
    except Exception as e:
        raise AIEvalConfigError(
            f"[FATAL] AI evaluation config read failed: {config_path}\n"
            f"Error details: {e}"
        ) from e

    # Validate required fields exist
    required_keys = ["score_mappings", "dimension_prompts", "anchor_validation_rules"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise AIEvalConfigError(
            f"[FATAL] AI evaluation config missing required fields: {missing_keys}\n"
            f"Config file: {config_path}"
        )

    return config


# Load AI evaluation config
_AI_EVAL_CONFIG = _load_ai_eval_config()


# -----------------------------------------------------------------------------
# 0) Helper Functions: Text Normalization & Negation Protection
# -----------------------------------------------------------------------------
_NEG_PREFIX = r"(?:未|不|无|没有|并未|不存在|未见|未发现)"
_QUOTE_SIGNS = ["“", "”", "‘", "’", "\"", "'"]


def _norm_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", "", s)


def _has_any(text: str, phrases: List[str]) -> bool:
    t = _norm_text(text)
    return any(p.replace(" ", "") in t for p in phrases)


def _count_any(text: str, phrases: List[str]) -> int:
    t = _norm_text(text)
    return sum(1 for p in phrases if p.replace(" ", "") in t)


def _contains_quote_signal(text: str) -> bool:
    return any(q in text for q in _QUOTE_SIGNS) or ("引用" in text) or ("触发片段" in text) or ("原文摘录" in text)


def _risk_term_is_negated(text: str, term: str, window: int = 4) -> bool:
    """
    If reason contains negation prefix + term, consider term in negated context (no escalation).
    window controls max gap between negation prefix and term (by character count).
    """
    if not text or not term:
        return False
    pattern = rf"{_NEG_PREFIX}.{{0,{window}}}{re.escape(term)}"
    return re.search(pattern, text) is not None


# -----------------------------------------------------------------------------
# [2025-12 Refactor] Anchor Parsing Rules
#
# Priority loads from anchor_validation_rules in data/ai_eval.json
# Falls back to code defaults if config doesn't exist
#
# Design Principles:
# - Only parse hard anchors (K / N / |C| / low-level templates)
# - Pure validation mode: no override/fallback, non-compliant returns INVALID
# -----------------------------------------------------------------------------
def _build_reason_parse_rules() -> Dict[str, Dict[str, Any]]:
    """
    Build anchor parsing rules from ai_eval.json.

    Priority uses JSON config, falls back to defaults if not present.
    """
    # Default rules (fallback)
    default_rules: Dict[str, Dict[str, Any]] = {
        "answer_uniqueness": {
            "regex": r"\|C\|\s*=?\s*([0-9]+)",
            "map": {
                1: "唯一",
                0: "次唯一/开放",
                "ge2": "不唯一",
            }
        },
        "distractor_headroom": {
            "regex": r"(?:最终计入|N\s*=?\s*)([0-9]+)\s*条?",
            "map": {3: "3", 2: "2", 1: "1", 0: "0"}
        },
        "guessing_lower_asymptote": {
            "regex": r"K\s*=?\s*([0-9]+)\+?",
            "map": {
                0: "正常四选一",
                1: "略易蒙对",
                2: "明显易蒙对",
                3: "接近送分题",
                "ge3": "接近送分题",
            }
        },
        "fairness_regional_gender": {
            "low_templates": ["未发现可引用", "无可引用触发", "故判低", "低（100）", "判低（100）"],
            "risk_terms_regional": ["区域门槛", "地方知识", "方言", "地方政策", "地方习俗", "区域特有经验"],
            "risk_terms_stereo": ["刻板印象", "歧视", "歧视性表述", "污名化", "侮辱", "贬损", "负面绑定", "地域黑"],
            "severity_terms_high": ["严重", "明显", "显著不公", "特定地区考生明显优势", "必须依赖", "关键依赖"],
            "citation_signal_terms": ["触发片段", "原文摘录", "可引用触发点", "引用", "例如", "如"] + _QUOTE_SIGNS
        },
        "rubric_operational": {
            "point_terms": ["要点", "采分点", "评分要点", "①", "②", "③", "1）", "2）", "3）", "一是", "二是", "三是"],
            "boundary_terms": ["边界", "不给分", "扣分", "不得分", "不计分", "不得", "不应给分"]
        }
    }

    # Try loading from config and merge
    json_rules = _AI_EVAL_CONFIG.get("anchor_validation_rules", {})

    if json_rules:
        # Merge JSON config (JSON takes priority)
        for dim_id, rule in json_rules.items():
            if dim_id not in default_rules:
                default_rules[dim_id] = {}

            # Convert JSON format to code format
            if "regex" in rule:
                default_rules[dim_id]["regex"] = rule["regex"]

            if "level_mapping" in rule:
                # Convert level_mapping format
                lm = rule["level_mapping"]
                new_map = {}
                for k, v in lm.items():
                    try:
                        new_map[int(k)] = v
                    except ValueError:
                        new_map[k] = v  # Preserve non-numeric keys like "ge2"
                default_rules[dim_id]["map"] = new_map

            # Merge fairness special fields
            for key in ["low_templates", "risk_terms_regional", "risk_terms_stereo"]:
                if key in rule:
                    default_rules[dim_id][key] = rule[key]

    return default_rules


# Build anchor parsing rules (loaded from JSON config)
REASON_PARSE_RULES: Dict[str, Dict[str, Any]] = _build_reason_parse_rules()


def _parse_level_from_reason(dim_id: str, reason: str) -> Optional[str]:
    """
    Returns: If level can be stably inferred from hard anchors (string), return that level; otherwise None.
    """
    if not reason:
        return None

    r = reason  # Don't normalize, regex needs spaces etc.
    nr = _norm_text(reason)

    # 1) answer_uniqueness: |C|
    if dim_id == "answer_uniqueness":
        m = re.search(REASON_PARSE_RULES[dim_id]["regex"], nr)
        if not m:
            return None
        c = int(m.group(1))
        if c == 1:
            return "唯一"  # Unique
        if c == 0:
            return "次唯一/开放"  # Near-unique/Open
        return REASON_PARSE_RULES[dim_id]["map"]["ge2"]

    # 2) distractor_headroom: N
    if dim_id == "distractor_headroom":
        # Compatible with Chinese natural-language count markers and "N=2".
        m = re.search(r"(?:最终计入\s*|N\s*=?\s*)([0-9]+)", nr)
        if not m:
            return None
        n = int(m.group(1))
        return str(n) if n in (0, 1, 2, 3) else ("3" if n >= 3 else None)

    # 3) guessing_lower_asymptote: K
    if dim_id == "guessing_lower_asymptote":
        m = re.search(r"K\s*=?\s*([0-9]+)\+?", nr)
        if not m:
            return None
        k = int(m.group(1))
        # Note: distinguish direct presence from negated-context mentions.
        # [2025-12 Fix] Increased window to 50 for matching long sentences
        if (("题干直给" in reason and not _risk_term_is_negated(reason, "题干直给", window=50)) or
                ("模式线索" in reason and not _risk_term_is_negated(reason, "模式线索", window=50))):
            return "接近送分题"  # Near-giveaway
        if k >= 3:
            return "接近送分题"  # Near-giveaway
        if k == 2:
            return "明显易蒙对"  # Obviously easy to guess
        if k == 1:
            return "略易蒙对"  # Slightly easy to guess
        # k==0: Distinguish 100 vs 85 based on keywords
        if _has_any(reason, ["控制良好", "难以蒙对", "高度对称", "结构均衡", "形式对称", "结构对称"]) and not _has_any(
                reason, ["轻微可疑", "略有可排除"]):
            return "控制良好（难以蒙对）"  # Well controlled (hard to guess)
        return "正常四选一"  # Normal 4-choice

    # 4) fairness_regional_gender: Low templates take priority; medium/high need citation + risk terms (non-negated)
    if dim_id == "fairness_regional_gender":
        rules = REASON_PARSE_RULES[dim_id]
        # Low templates
        if _has_any(reason, rules["low_templates"]):
            return "低"  # Low

        # Risk detection (with negation protection)
        risk_terms = rules["risk_terms_regional"] + rules["risk_terms_stereo"]
        found_risk = False
        for t in risk_terms:
            if t in reason and not _risk_term_is_negated(reason, t, window=12):
                found_risk = True
                break

        has_cite = _has_any(reason, rules["citation_signal_terms"]) or _contains_quote_signal(reason)

        if found_risk and has_cite:
            # High vs Medium: check severity terms
            if _has_any(reason, rules["severity_terms_high"]):
                return "高"  # High
            return "中"  # Medium

        # If only general mention of risk without citation: don't force parse, return None
        # [2025-12 Refactor] Don't use keyword-based level inference, let upper layer decide
        return None

    # 5) rubric_operational: Structure validation for determining if "100/85" holds
    if dim_id == "rubric_operational":
        rules = REASON_PARSE_RULES[dim_id]
        pts = _count_any(reason, rules["point_terms"])
        bds = _count_any(reason, rules["boundary_terms"])
        # Only return "high tier" signal when structure clearly satisfied
        if pts >= 2 and bds >= 1:
            return "STRUCT_OK"
        return "STRUCT_WEAK"

    return None


# -----------------------------------------------------------------------------
# [2025-12 Refactor] Schema/Anchor Validation Entry
# Pure validation mode: only returns VALID/INVALID, no override/fallback
# Invalid model-dimension pairs are excluded by the aggregator; no repair call is made.
# -----------------------------------------------------------------------------
def validate_and_fix_level(
        dim_id: str,
        level: str,
        reason: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    [2025-12 Refactor] Pure validation mode

    Returns: (is_valid, errors)
    - is_valid: True=passed validation, False=non-compliant
    - errors: Validation details dict containing dim_id, orig_level, parsed_level, invalid_reason etc.

    Validation rules:
    1. Parse hard anchors (K / N / |C| / low-level templates) from reason
    2. If parsed level is clear and doesn't match output -> INVALID (is_valid=False)
    3. Other cases VALID (is_valid=True)

    Note:
    - This function does NOT return "corrected level", only validation result
    - No override/fallback; invalid outputs remain invalid for aggregation-time filtering
    """
    errors: Dict[str, Any] = {"dim_id": dim_id, "orig_level": level}

    # 1) Parse-first: Parse hard anchors from reason
    parsed = _parse_level_from_reason(dim_id, reason)
    errors["parsed_level"] = parsed

    # rubric_operational special case: parsed returns STRUCT_OK/STRUCT_WEAK
    if dim_id == "rubric_operational":
        if parsed == "STRUCT_WEAK" and level == "100":
            # 100 requires points>=2 and boundaries>=1, otherwise INVALID
            errors["invalid_reason"] = "rubric_operational: level=100 but structure weak (points<2 or boundaries<1)"
            return False, errors
        # STRUCT_OK or other tiers -> VALID
        return True, errors

    # fairness: If parse gets clear level (low/medium/high)
    if parsed in ("低", "中", "高"):
        if parsed != level:
            errors["invalid_reason"] = f"fairness: parsed_level={parsed} != output_level={level}"
            return False, errors
        return True, errors

    # 2) Other dimensions: If parse gets clear level, do consistency check
    if parsed and parsed not in ("STRUCT_OK", "STRUCT_WEAK"):
        if parsed != level:
            errors["invalid_reason"] = f"anchor_mismatch: parsed_level={parsed} != output_level={level}"
            return False, errors
        return True, errors

    # 3) Cannot parse clear level from reason -> VALID (no assumptions)
    return True, errors


# -----------------------------------------------------------------------------
# [2025-12 Added] D1) Per-Dimension Schema/Anchor Validator
# Integrates all validation logic and returns (valid, errors) for audit/filtering
# -----------------------------------------------------------------------------
def schema_validate_dim(
        dim_id: str,
        level: Optional[str],
        score: Optional[float],
        reason: Optional[str],
        dim_cfg: Dict[str, Any],
        guessing_c: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    [D1] Per-Dimension Schema/Anchor Validator

    Integrates all validation logic and returns specific errors for audit/filtering.

    Args:
        dim_id: Dimension ID
        level: Model output level
        score: Model output score (optional)
        reason: Model output reason
        dim_cfg: Dimension config containing allowed_levels, score_mapping etc.
        guessing_c: Guessing probability (for specific dimension validation)

    Returns:
        (valid, errors)
        - valid: True=passed validation, False=non-compliant and should be filtered
        - errors: Specific error list for audit output
    """
    errors: List[str] = []

    # ========== 1) JSON field completeness check ==========
    if level is None or level == "":
        errors.append("Missing required field 'level'")
    if reason is None or reason == "":
        errors.append("Missing required field 'reason'")

    if errors:
        return False, errors

    # ========== 2) Level validity check ==========
    sm = dim_cfg.get("score_mapping", {}) or {}
    if sm.get("type") == "levels":
        levels_cfg = sm.get("levels", []) or []
        allowed_levels = [str(x.get("label", "")).strip() for x in levels_cfg if "label" in x]
        allowed_levels = [x for x in allowed_levels if x]
    else:
        allowed_levels = ["低", "中", "高"]

    if level not in allowed_levels:
        errors.append(f"level='{level}' not in allowed list {allowed_levels}")

    # ========== 3) Anchor validation (for dimensions requiring anchors) ==========
    reason_text = reason or ""

    # 3.0) stem_quality must contain the required action/object markers.
    if dim_id == "stem_quality":
        pattern = r"任务动作词=.+;\s*范围/对象=.+"
        if not re.search(pattern, reason_text):
            errors.append("stem_quality requires reason to contain '任务动作词=<verb>; 范围/对象=<noun phrase>' (both required)")

    # 3.0b) option_exclusivity_coverage must contain the required comparison-axis markers.
    if dim_id == "option_exclusivity_coverage":
        pattern = r"竞争轴=.+;\s*[A-D]与?[A-D]=.+"
        if not re.search(pattern, reason_text):
            errors.append("option_exclusivity_coverage requires reason to contain '竞争轴=<axis>; <option pair>=<relation>' (e.g., '竞争轴=信息理解; B与C=近同义')")

    # 3.1) answer_uniqueness: Must contain |C|=n
    if dim_id == "answer_uniqueness":
        m = re.search(r"\|C\|\s*=\s*([0-9]+)", _norm_text(reason_text))
        if not m:
            errors.append("answer_uniqueness requires reason to contain |C|=<0-4> (candidate correct set size)")
        else:
            c_val = int(m.group(1))
            # Check anchor-level contradiction
            if c_val == 1 and level != "唯一":
                errors.append(f"|C|=1 should correspond to level='唯一', but output level='{level}'")
            elif c_val == 0 and level not in ("次唯一/开放",):
                errors.append(f"|C|=0 should correspond to level='次唯一/开放', but output level='{level}'")
            elif c_val >= 2 and level not in ("不唯一",):
                errors.append(f"|C|={c_val}>=2 should correspond to level='不唯一', but output level='{level}'")

    # 3.2) distractor_headroom: Must contain N=<0-3> (N=effective misreading path count)
    if dim_id == "distractor_headroom":
        m = re.search(r"N\s*=\s*([0-3])", _norm_text(reason_text))
        if not m:
            errors.append("distractor_headroom requires reason to contain N=<0-3> (N=effective misreading path count)")
        else:
            n_val = int(m.group(1))
            # Check N vs level consistency (N=3 means >=3, level="3")
            level_n_map = {"0": 0, "1": 1, "2": 2, "3": 3}
            expected_n = level_n_map.get(level)
            if expected_n is not None and n_val != expected_n:
                errors.append(f"level='{level}' expects N={expected_n}, but reason reports N={n_val}")

    # 3.3) guessing_lower_asymptote: Must contain K=<0-3> (K=options excludable by form cues)
    if dim_id == "guessing_lower_asymptote":
        m = re.search(r"K\s*=\s*([0-3])", _norm_text(reason_text))
        if not m:
            errors.append("guessing_lower_asymptote requires reason to contain K=<0-3> (K=options excludable by form cues)")
        else:
            k_val = int(m.group(1))
            # Check K vs level consistency (allowing adjacent tiers)
            k_level_mapping = {
                0: ["控制良好（难以蒙对）", "正常四选一"],
                1: ["正常四选一", "略易蒙对"],
                2: ["略易蒙对", "明显易蒙对"],
                3: ["明显易蒙对", "接近送分题"],
            }
            valid_levels = k_level_mapping.get(k_val, [])
            if level not in valid_levels:
                errors.append(f"K={k_val} should correspond to {valid_levels}, but output level='{level}'")

    # 3.4) fairness_regional_gender: Low tier requires low template phrases
    if dim_id == "fairness_regional_gender":
        low_templates = ["未发现可引用", "无可引用触发", "故判低", "低（100）", "判低（100）"]
        parsed = _parse_level_from_reason(dim_id, reason_text)
        if parsed == "低" and level != "低":
            errors.append(f"reason contains low template phrase (e.g., '未发现可引用触发点'), but level='{level}' is not '低'")
        elif parsed in ("中", "高") and level != parsed:
            errors.append(f"reason parses to level='{parsed}', but output level='{level}', contradiction exists")

    # 3.5) item_evidence_sufficiency: Different anchors required by level
    if dim_id == "item_evidence_sufficiency":
        # level 0/1 requires a concrete gap marker.
        # level 2/3 requires a concrete closed-evidence marker.
        low_levels = ["0", "1"]
        high_levels = ["2", "3"]
        has_gap = bool(re.search(r"缺口点=.+", reason_text))
        has_evidence = bool(re.search(r"闭环证据=.+", reason_text))

        if level in low_levels and not has_gap:
            errors.append(f"item_evidence_sufficiency level='{level}' (partial self-sufficient or lower) requires reason to contain '缺口点=<specific gap>'")
        elif level in high_levels and not has_evidence:
            errors.append(f"item_evidence_sufficiency level='{level}' (basic self-sufficient or higher) requires reason to contain '闭环证据=<evidence point>'")

    # 3.6) rubric_operational must contain point-count and boundary-count markers.
    if dim_id == "rubric_operational":
        pattern = r"要点数=(\d+);\s*边界数=(\d+)"
        m = re.search(pattern, reason_text)
        if not m:
            errors.append("rubric_operational requires reason to contain '要点数=<m>; 边界数=<n>' (both required)")
        else:
            m_val = int(m.group(1))
            n_val = int(m.group(2))
            # Check points/boundaries vs level consistency
            level_requirements = {
                "100": {"m_min": 3, "n_min": 2},
                "85": {"m_min": 2, "n_min": 1},
                "70": {"m_min": 1, "n_min": 0},
                "40": {"m_min": 1, "n_min": 0},
                "0": {"m_min": 0, "n_min": 0},
            }
            req = level_requirements.get(level)
            if req:
                if m_val < req["m_min"]:
                    errors.append(f"level='{level}' requires points>={req['m_min']}, but reports points={m_val}")
                if n_val < req["n_min"]:
                    errors.append(f"level='{level}' requires boundaries>={req['n_min']}, but reports boundaries={n_val}")

    # ========== 4) Forbidden phrases hit check ==========
    dim_rules = REASON_VALIDITY_RULES.get("by_dimension", {}).get(dim_id, {})
    forbidden = dim_rules.get("forbidden_phrases", [])
    if forbidden and _rv_has_any(reason_text, forbidden):
        matched = [p for p in forbidden if p in reason_text]
        errors.append(f"reason contains forbidden phrases: {matched[:3]}...")  # Only show first 3

    # ========== 5) Semantic fit overreach check (guessing_lower_asymptote) ==========
    if dim_id == "guessing_lower_asymptote":
        sem_phrases = dim_rules.get("semantic_fit_phrases", [])
        form_phrases = dim_rules.get("form_evidence_phrases", [])
        option_regex = dim_rules.get("option_pointer_regex", r"")

        has_sem = _rv_has_any(reason_text, sem_phrases)
        has_form = _rv_has_any(reason_text, form_phrases)
        has_option_pointer = bool(re.search(option_regex, reason_text)) if option_regex else False

        if has_sem and has_option_pointer and (not has_form):
            matched_sem = [p for p in sem_phrases if p in reason_text]
            errors.append(f"Semantic fit overreach: reason uses intuition/fit words {matched_sem[:2]}, but lacks form evidence (like length/syntax/symmetry etc.)")

    # Return result
    is_valid = len(errors) == 0
    return is_valid, errors


# =============================================================================
# Module Configuration
# =============================================================================

from src.shared.schemas import (
    GeneratedQuestion,
    VerifiedQuestionSet,
    AICentricEvalResult,
    Stage2CoreInput,
    Stage2Record,
)
from src.shared.llm_interface import LLMClient
from src.shared.config import STAGE2_EVAL_MODELS, MODEL_WEIGHT, STAGE2_LLM_PARAMS
from src.shared.api_config import is_no_temperature_model
from src.shared.question_family import (
    QuestionFamily,
    infer_question_family,
    filter_and_renormalize_dimensions,
    DimensionFilterResult,
)

logger = logging.getLogger(__name__)

# =============================================================================
# =============================================================================
# [2025-12 Refactor] AI Evaluation Dimension Definitions - Loaded from data/ai_eval.json
# Split into objective question dimensions and subjective question dimensions
# Config includes: score_mappings (weights and score mapping) + dimension_prompts (prompts)
# =============================================================================

# [REMOVED HARDCODE] Original OBJECTIVE_DIMENSIONS and SUBJECTIVE_DIMENSIONS
# Now dynamically loaded from ai_eval.json via _build_dimensions_from_config()


def _build_dimensions_from_config(q_type: str) -> List[Dict[str, Any]]:
    """
    Build dimension configs for a specific question type from ai_eval.json.

    Args:
        q_type: "objective" or "subjective".

    Returns:
        Dimension config list.
    """
    dims: List[Dict[str, Any]] = []
    score_mappings = _AI_EVAL_CONFIG.get("score_mappings", {}).get(q_type, {})
    dimension_prompts = _AI_EVAL_CONFIG.get("dimension_prompts", {}).get(q_type, {})

    for dim_id, score_cfg in score_mappings.items():
        prompt_cfg = dimension_prompts.get(dim_id, {})
        dim_config = {
            "id": dim_id,
            "name": dim_id,
            "weight": score_cfg.get("weight", 1/7 if q_type == "objective" else 0.25),
            "applicable_to": [q_type],
            "score_mapping": {
                "type": score_cfg.get("type", "levels"),
                "levels": score_cfg.get("levels", [])
            },
            "prompt_eval": prompt_cfg.get("prompt_eval", ""),
            "allowed_levels": prompt_cfg.get("allowed_levels", []),
            "level_default": prompt_cfg.get("level_default", "")
        }
        dims.append(dim_config)
    return dims


def get_ai_dimensions_by_question_type(question_type: str) -> List[Dict[str, Any]]:
    """
    Return corresponding AI evaluation dimension config based on question type (loaded from ai_eval.json).

    Args:
        question_type: Question type
            - "single-choice", "objective", etc. returns objective question dimensions
            - "essay", "subjective", etc. returns subjective question dimensions

    Returns:
        Dimension config list for the corresponding type
    """
    qt_lower = str(question_type or "").lower().strip()

    # Subjective question type keywords
    subjective_keywords = ["essay", "subjective", "简答", "主观", "short_answer", "open"]

    # Objective question type keywords
    objective_keywords = ["single-choice", "objective", "选择", "客观", "multiple", "choice"]

    # Determine question type
    is_subjective = any(kw in qt_lower for kw in subjective_keywords)
    is_objective = any(kw in qt_lower for kw in objective_keywords)

    if is_subjective:
        return _build_dimensions_from_config("subjective")
    elif is_objective:
        return _build_dimensions_from_config("objective")
    else:
        # Default to objective dimensions (backward compatibility)
        return _build_dimensions_from_config("objective")


# =============================================================================
# JSON Parsing Tools: Extract + Repair (never crash)
# =============================================================================


def _extract_json_candidate(text: str) -> Optional[str]:
    """Extract outermost JSON {...} from model response, supporting code blocks/prefix text."""
    if not text:
        return None
    t = text.strip()

    # Prefer code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        return m.group(1).strip()

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(t)):
        ch = t[i]

        if escape_next:
            escape_next = False
            continue

        if in_string and ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]

    # Return greedy result when unclosed
    return t[start:] if depth > 0 else None


def _repair_common_json(raw: str) -> str:
    """Repair common JSON errors: trailing comma, missing quotes, single quotes, newlines etc."""
    if not raw:
        return "{}"
    s = raw

    # Single quotes -> double quotes (simple version)
    s = s.replace("'", '"')

    # Control chars -> spaces (except newline)
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', s)

    # Trailing comma (only before }])
    s = re.sub(r',(\s*)([}\]])', r'\1\2', s)

    return s
# =============================================================================
# Level normalization helpers.
# =============================================================================

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．／", "0123456789./")

# =============================================================================
# Reason validity rules.
#
# Design:
# - Pure validation mode: forbidden phrases or anchor conflicts become INVALID.
# - Invalid dimensions are excluded from aggregation; no extra repair calls are made.
# =============================================================================

REASON_VALIDITY_RULES = {
    "by_dimension": {
        # 1) option_exclusivity_coverage: source-attribution overreach, dimension-specific.
        "option_exclusivity_coverage": {
            "forbidden_phrases": [
                "材料一", "材料二", "材料三",
                "来源不一致", "来源不同", "出处不同", "原文出处",
                "不在材料", "出自材料", "来自材料一", "来自材料二",
                "对应材料一", "对应材料二", "材料来源",
                "哪段材料", "哪一段材料", "材料第几段",
            ],
        },

        # 2) guessing_lower_asymptote: semantic-fit / intuition overreach checks.
        "guessing_lower_asymptote": {
            # Semantic-fit terms that are unsafe when tied to an option without formal evidence.
            "semantic_fit_phrases": [
                "最贴合", "更贴合",
                "一眼看出", "直觉",
                "语义明显", "显然选",
                "最像正确", "明显正确", "更合理",
            ],
            # Option pointer regex used to detect references to specific options.
            "option_pointer_regex": r"(?:选项)?\s*(?:[A-Da-d]|[甲乙丙丁])\b|(?:只有|选)\s*(?:[A-Da-d]|[甲乙丙丁])\b",
            # Formal evidence phrases that can justify the reasoning.
            "form_evidence_phrases": [
                "长度", "句式", "语体", "语体断裂", "风格突变",
                "对称", "形式对称",
                "异常突出", "唯一异常",
                "跑题", "任务不匹配", "明显无关",
                "常识矛盾", "自相矛盾", "自我否定",
                "模式线索", "题干直给",
            ],
        },

        # 3) distractor_headroom: N-value consistency check.
        "distractor_headroom": {
            # Minimum N required for each score.
            "score_min_n": {
                0: 0,
                50: 1,
                75: 2,
                100: 3,
            },
        },
    },
}


def _rv_has_any(s: str, kws: List[str]) -> bool:
    if not s:
        return False
    return any(k in s for k in kws)


def _rv_extract_k_guessing(reason: str) -> Optional[int]:
    # Support "K=2", "K=3+", and equivalent Chinese rubric wording.
    m = re.search(r"(?:K\s*=\s*)(\d)\+?", reason or "")
    if m:
        return int(m.group(1))
    # Fallback for common Chinese numeral wording in model reasons.
    r = reason or ""
    if "三个" in r and "强可排除" in r:
        return 3
    if "两个" in r and "强可排除" in r:
        return 2
    if "一个" in r and "强可排除" in r:
        return 1
    return None


def _rv_extract_n_distractor(reason: str) -> Optional[int]:
    # Support "N=2" and equivalent Chinese rubric wording.
    r = reason or ""
    m = re.search(r"(?:N\s*=\s*)(\d)", r)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:最终计入)\s*(\d)\s*条", r)
    if m:
        return int(m.group(1))
    return None


def enforce_reason_validity(
        dim_id: str,
        canonical_level: str,
        reason: str,
        score_mapping: Dict[str, Any],
        guessing_c: Optional[float] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Pure reason-validity checker.

    Detects overreach or level-inconsistent signals in the reason field.
    Non-compliant results become INVALID and are excluded from aggregation.
    No automatic fallback, overwrite, or extra repair call is performed.

    Returns:
        (result_level, reason, meta).
    """
    meta: Dict[str, Any] = {"valid": True, "flags": [], "dim_id": dim_id}
    lvl = canonical_level
    r = reason or ""

    # Fetch rules for the current dimension.
    dim_rules = REASON_VALIDITY_RULES.get("by_dimension", {}).get(dim_id, {})
    if not dim_rules:
        return lvl, r, meta

    # Estimate current score for consistency checks.
    try:
        cur_score = map_level_to_score(dim_id, lvl, score_mapping, guessing_c)
    except Exception:
        cur_score = None

    # ========== A) option_exclusivity_coverage: source-attribution overreach ==========
    if dim_id == "option_exclusivity_coverage":
        forbidden = dim_rules.get("forbidden_phrases", [])
        if _rv_has_any(r, forbidden):
            meta["valid"] = False
            meta["flags"].append("overreach: source attribution")
            meta["invalid_reason"] = "option_exclusivity_coverage: source-attribution overreach detected"
            return "INVALID", r, meta

    # ========== B) guessing_lower_asymptote: semantic-fit overreach + K consistency ==========
    if dim_id == "guessing_lower_asymptote":
        sem_phrases = dim_rules.get("semantic_fit_phrases", [])
        form_phrases = dim_rules.get("form_evidence_phrases", [])
        option_regex = dim_rules.get("option_pointer_regex", r"")

        has_sem = _rv_has_any(r, sem_phrases)
        has_form = _rv_has_any(r, form_phrases)
        has_option_pointer = bool(re.search(option_regex, r)) if option_regex else False

        # B1) Semantic-fit + option pointer + no formal evidence -> INVALID.
        if has_sem and has_option_pointer and (not has_form):
            meta["valid"] = False
            meta["flags"].append("overreach: semantic fit without formal evidence")
            meta["invalid_reason"] = "guessing_lower_asymptote: semantic-fit or intuition-based reason lacks formal evidence"
            return "INVALID", r, meta

        # B2) K parsing and consistency checks.
        k = _rv_extract_k_guessing(r)

        if k is None:
            # Low tiers must report K unless they provide direct-stem or pattern evidence.
            if lvl in {"明显易蒙对", "接近送分题"} and ("题干直给" not in r and "模式线索" not in r):
                meta["valid"] = False
                meta["flags"].append("inconsistent: low tier without K")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level={lvl} lacks K and direct-stem/pattern evidence"
                return "INVALID", r, meta
        else:
            # K-value and level consistency.
            if lvl == "明显易蒙对" and k < 2:
                meta["valid"] = False
                meta["flags"].append("inconsistent: level expects K>=2")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level='明显易蒙对' but K={k}<2"
                return "INVALID", r, meta

            if lvl == "接近送分题" and (k < 3) and ("题干直给" not in r):
                meta["valid"] = False
                meta["flags"].append("inconsistent: level expects K>=3 or direct-stem evidence")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level='接近送分题' but K={k}<3 and no direct-stem evidence"
                return "INVALID", r, meta

    # ========== C) distractor_headroom: N parsing and score consistency ==========
    if dim_id == "distractor_headroom":
        score_min_n = dim_rules.get("score_min_n", {0: 0, 50: 1, 75: 2, 100: 3})

        # Parse N.
        n = _rv_extract_n_distractor(r)

        if n is None:
            # Low tiers must report N.
            if lvl in {"0", "1"}:
                meta["valid"] = False
                meta["flags"].append("inconsistent: low tier without N")
                meta["invalid_reason"] = f"distractor_headroom: level={lvl} does not report N"
                return "INVALID", r, meta
        else:
            # Check score <-> N consistency.
            if cur_score is not None:
                min_n_required = score_min_n.get(int(cur_score), 0)

                # score=0 with N>=1 is contradictory.
                if cur_score == 0 and n >= 1:
                    meta["valid"] = False
                    meta["flags"].append("inconsistent: score=0 but N>=1")
                    meta["invalid_reason"] = f"distractor_headroom: score=0 but reason reports N={n}"
                    return "INVALID", r, meta

                # Positive score requires the configured minimum N.
                if cur_score > 0 and n < min_n_required:
                    meta["valid"] = False
                    meta["flags"].append(f"inconsistent: score={cur_score} but N={n}<{min_n_required}")
                    meta["invalid_reason"] = f"distractor_headroom: score={cur_score} but N={n} is below required minimum {min_n_required}"
                    return "INVALID", r, meta

    return lvl, r, meta


def _clean_level_text(x: Any) -> str:
    s = str(x).strip()
    s = s.translate(_FULLWIDTH_DIGITS)
    s = re.sub(r"\s+", " ", s)

    return s


def canonicalize_level(dim_id: str, raw_level: Any, allowed_levels: List[str], default_level: str) -> str:
    """
    Normalize a model-produced level into one of allowed_levels.

    - Exact matches are returned directly.
    - Numeric labels such as "85.0" are normalized and matched to the closest allowed numeric label.
    - Otherwise, default_level is returned.
    """
    if raw_level is None:
        return default_level

    allowed = [str(a) for a in allowed_levels]
    allowed_set = set(allowed)
    s = _clean_level_text(raw_level)

    if s in allowed_set:
        return s

    # Numeric labels: choose the nearest allowed numeric label.
    if re.fullmatch(r"-?\d+(\.\d+)?", s) and all(re.fullmatch(r"\d+", a) for a in allowed):
        try:
            v = float(s)
            best = min(allowed, key=lambda a: abs(float(a) - v))
            return str(best)
        except Exception:
            return default_level

    # Normalize full-width and half-width slashes.
    s2 = s.replace("／", "/")
    if s2 in allowed_set:
        return s2

    return default_level


# =============================================================================
# Prompt rendering for the dimension list.
# =============================================================================

AI_BATCHED_OUTPUT_CONSTRAINTS = """
【输出要求（必须严格遵守）】
1) 只输出一个 JSON 对象；不要输出任何解释性文字；不要输出 markdown 代码块。
2) JSON 顶层必须且只能包含一个字段："dimensions"
3) "dimensions" 必须包含上述维度列表中的全部维度 id（一个不少，一个不多），key 必须与维度 id 完全一致。
4) 每个维度对象必须且只能包含两个字段：
   - "level": 字符串，必须 EXACT MATCH 允许列表之一（保持原样，包括数字、斜杠、空格等）
   - "reason": 中文 1-2 句，简短说明
5) 禁止输出 score；禁止输出数值区间（如"70-85"）；禁止输出自造标签。

6) 若不确定，请直接输出该维度的 level_default。
""".strip()


def render_dim_specs(dim_specs: List[Dict[str, Any]], guessing_c: Optional[float]) -> str:
    """
    【2025-12 重构】渲染维度评估规则列表。

    【重要修复】使用 prompt_eval 字段作为完整评估规则。
    每个维度的 prompt_eval 包含了该维度的完整评估提示词，包括：
    - 评估关注点
    - 评分口径
    - 锚点要求（K/N/|C|等）
    - 输出格式要求
    """
    dim_count = len(dim_specs)
    lines: List[str] = []

    # Overall instructions.
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  【评估任务】请对题目进行 {dim_count} 个维度的独立评估                   ║")
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("【整体评估要求】")
    lines.append("1. 请按顺序逐个评估每个维度，每个维度独立打分，不要让一个维度的判断影响另一个维度")
    lines.append("2. 每个维度都有独立的评分口径和锚点要求，请严格遵循")
    lines.append("3. level 必须从 allowed_levels 中原样选一个；禁止改写、禁止区间、禁止额外解释")
    lines.append("4. 如不确定，一律选择 level_default")
    if guessing_c is not None:
        lines.append(
            f"5. 提示：guessing_lower_asymptote 已提供 c={guessing_c}（仅用于你判断低/中/高，最终 score 由系统计算）"
        )
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    for i, d in enumerate(dim_specs, 1):
        dim_id = d["id"]
        name = d.get("name", "")
        allowed = d.get("allowed_levels") or []
        default = d.get("level_default") or (allowed[len(allowed) // 2] if allowed else "")

        # Prefer prompt_eval because it contains the complete evaluation prompt.
        prompt_eval = (d.get("prompt_eval") or "").strip()
        if not prompt_eval:
            # Compatibility fallback when prompt_eval is absent.
            prompt_eval = (d.get("eval_rules") or "").strip()
            if not prompt_eval:
                definition = (d.get("definition") or "").strip().replace("\n", " ")
                if len(definition) > 200:
                    definition = definition[:200] + "..."
                prompt_eval = definition

        allowed_render = " | ".join([json.dumps(str(x), ensure_ascii=False) for x in allowed])

        # Clear dimension separator.
        lines.append("┌─────────────────────────────────────────────────────────────────┐")
        lines.append(f"│  【维度 {i}/{dim_count}】{name}（{dim_id}）")
        lines.append("└─────────────────────────────────────────────────────────────────┘")
        lines.append(f"allowed_levels = {allowed_render}")
        lines.append(f"level_default = {json.dumps(str(default), ensure_ascii=False)}")

        if dim_id == "guessing_lower_asymptote":
            lines.append("特殊说明：低/中/高判断阈值（只为可读性，最终 score 由系统公式算）：")
            lines.append("  c <= 0.08 -> 低；0.08 < c <= 0.20 -> 中；c > 0.20 -> 高")

        # Output complete evaluation rules from prompt_eval.
        lines.append("")
        lines.append("【本维度评估规则（请严格遵循）】")
        lines.append(prompt_eval)
        lines.append("")
        lines.append("─" * 70)
        lines.append("")

    return "\n".join(lines).strip()


def build_ai_batched_prompt(question: Any, dims_cfg: List[Dict], guessing_c: Optional[float]) -> str:
    """
    兼容多种题目对象：
    - Stage2CoreInput（推荐）
    - GeneratedQuestion
    - 其他含 stem/material_text/options/... 属性的对象
    """
    # AI evaluation only uses material, stem, and options for objective items.
    # Do not feed answers, explanations, or scoring points so evaluation remains blind.
    stem = getattr(question, "stem", "") or ""
    material = getattr(question, "material_text", "") or ""
    question_type = getattr(question, "question_type", "") or ""

    options_text = ""
    opts = getattr(question, "options", None) or []
    for opt in opts:
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).strip()
            content = str(opt.get("content", "")).strip()
        else:
            label = str(getattr(opt, "label", "") or "").strip()
            content = str(getattr(opt, "content", "") or "").strip()
        if label or content:
            options_text += f"  {label}. {content}\n"

    total_score = getattr(question, "total_score", None)
    total_score_text = f"{total_score}分" if total_score else "(未提供)"

    # Dimension specs come strictly from config; formula dimensions provide ordered levels.
    # Ensure prompt_eval is included for the full evaluation prompt.
    dim_specs: List[Dict[str, Any]] = []
    for d in dims_cfg:
        sm = d.get("score_mapping", {})
        sm_type = sm.get("type", "levels")

        if sm_type == "levels":
            levels = sm.get("levels", []) or []
            allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
            allowed = [x for x in allowed if x]
            default = allowed[len(allowed) // 2] if allowed else "N/A"
        else:
            allowed = ["低", "中", "高"]
            default = "中"

        dim_specs.append(
            {
                "id": d["id"],
                "name": d.get("name", ""),
                "definition": d.get("definition", ""),
                "prompt_eval": d.get("prompt_eval", ""),  # Full evaluation prompt.
                "eval_rules": d.get("eval_rules", ""),  # Backward-compatible legacy config.
                "allowed_levels": allowed,
                "level_default": default,
            }
        )

    dim_specs_text = render_dim_specs(dim_specs, guessing_c)

    example_dims = {d["id"]: {"level": d["level_default"], "reason": "..."} for d in dim_specs}
    json_example = json.dumps({"dimensions": example_dims}, ensure_ascii=False, indent=2)

    # Render different fields by question type: objective items show options, essays show total score.
    # Do not feed answers or explanations so AI evaluation remains blind.
    if question_type == "essay":
        # Essay template.
        # AI evaluation uses only the source material and question, not answer/analysis.
        prompt = f"""你是一位资深的中文阅读理解题目质量评估专家。系统已经以固定字段格式提供了一道简答题的完整信息，你不需要再去猜测或补全题目内容，只需基于这些字段在给定的维度上做出评价。

【题目信息（由系统按固定字段提供）】
- 题型：简答题（essay）

- 材料（material_text）：这是论述类或说明类阅读材料的全文或节选，用于支撑题干设问。
  当前材料内容如下：
{material if material else "(未提供)"}

- 题干（stem）：这是要求学生作答的核心问题或指令。
  当前题干为：
{stem if stem else "(未提供)"}

- 总分（total_score）：本题的总分值。
  当前总分为：{total_score_text}

{dim_specs_text}

{AI_BATCHED_OUTPUT_CONSTRAINTS}

【输出示例（仅示例结构，禁止照抄理由）】
{json_example}

现在输出你的评估结果（只输出 JSON）：
"""
    else:
        # Objective-question template.
        # AI evaluation uses only the source material and question, not answer/analysis.
        prompt = f"""你是一位资深的中文阅读理解题目质量评估专家。系统已经以固定字段格式提供了一道题目的完整信息，你不需要再去猜测或补全题目内容，只需基于这些字段在给定的维度上做出评价。

【题目信息（由系统按固定字段提供）】
- 材料（material_text）：这是论述类或说明类阅读材料的全文或节选，用于支撑题干设问。
  当前材料内容如下：
{material if material else "(未提供)"}

- 题干（stem）：这是要求学生作答的核心问题或指令。
  当前题干为：
{stem if stem else "(未提供)"}

- 选项列表（options_if_any）：若为选择题，这里给出所有选项，每个选项含 label（A/B/C/D 等）和 content（选项内容）；若为非选择题，则本字段为空。
  当前选项为：
{options_text if options_text else "(无选项；可能为非选择题)"}

{dim_specs_text}

{AI_BATCHED_OUTPUT_CONSTRAINTS}

【输出示例（仅示例结构，禁止照抄理由）】
{json_example}

现在输出你的评估结果（只输出 JSON）：
"""
    return prompt


# =============================================================================
# Score calculation: level mapping plus formula-based dimensions.
# =============================================================================


def _score_by_levels(level: str, levels: List[Dict[str, Any]]) -> float:
    if not levels:
        return 50.0
    label_to_score = {str(x.get("label")): float(x.get("score", 0)) for x in levels if "label" in x}
    if level in label_to_score:
        return float(label_to_score[level])
    # Default to the middle tier.
    scores_sorted = [float(x.get("score", 0)) for x in levels]
    mid = scores_sorted[len(scores_sorted) // 2]
    return float(mid)


def _score_guessing_lower_asymptote(c: Optional[float]) -> float:
    """
    你的要求：score = 100 * (1 - c/0.35), clamp [0,100]
    """
    if c is None:
        return 50.0
    try:
        s = 100.0 * (1.0 - float(c) / 0.35)
        return max(0.0, min(100.0, float(s)))
    except Exception:
        return 50.0


def map_level_to_score(dim_id: str, canonical_level: str, score_mapping: Dict[str, Any],
                       guessing_c: Optional[float]) -> float:
    mtype = (score_mapping or {}).get("type", "levels")

    if mtype == "levels":
        levels = score_mapping.get("levels", []) or []
        return _score_by_levels(canonical_level, levels)

    # Formula support is intentionally limited; do not use eval.
    if mtype == "formula":
        if dim_id == "guessing_lower_asymptote":
            return _score_guessing_lower_asymptote(guessing_c)
        return 50.0

    return 50.0


# =============================================================================
# Batched response parser: never raises on malformed model output.
# =============================================================================


def parse_batched_ai_response(
        response: str,
        dims_cfg: List[Dict[str, Any]],
        guessing_c: Optional[float],
        model_name: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    audit: Dict[str, Any] = {
        "model": model_name,
        "parse_status": "ok",
        "parse_error": None,
        "raw_response_len": len(response) if response else 0,
        "level_validations": [],  # Records secondary level validation results.
        "reason_validations": [],
    }

    # Default output placeholders. INVALID entries are not scored.
    dimensions_final: Dict[str, Dict[str, Any]] = {}
    dim_cfg_map = {d["id"]: d for d in dims_cfg}

    for d in dims_cfg:
        sm = d.get("score_mapping", {}) or {}
        if sm.get("type") == "levels":
            levels = sm.get("levels", []) or []
            allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
            allowed = [x for x in allowed if x]
            default_level = allowed[len(allowed) // 2] if allowed else "N/A"
            default_score = _score_by_levels(default_level, levels)
        else:
            default_level = "中" if d["id"] == "guessing_lower_asymptote" else "中"
            default_score = map_level_to_score(d["id"], default_level, sm, guessing_c)

        # Initialize as INVALID instead of silently falling back to default_level.
        dimensions_final[d["id"]] = {
            "level": "INVALID",
            "reason": "initialized pending parse",
            "score": None,
            "invalid": True,
            "invalid_reason": "dimension not parsed yet",
        }

    if not response or not response.strip():
        audit["parse_status"] = "empty_response"
        audit["parse_error"] = "empty response"
        return dimensions_final, audit

    extracted = _extract_json_candidate(response)
    if not extracted:
        audit["parse_status"] = "salvaged"
        audit["parse_error"] = "no json candidate"
        return dimensions_final, audit

    parsed: Any = None
    try:
        parsed = json.loads(extracted)
    except Exception:
        try:
            parsed = json.loads(_repair_common_json(extracted))
            audit["parse_status"] = "repaired_ok"
        except Exception:
            audit["parse_status"] = "salvaged"
            audit["parse_error"] = "json.loads failed after repair"
            return dimensions_final, audit

    # parsed may be a list or another type; never call .get unless it is a dict.
    if not isinstance(parsed, dict):
        audit["parse_status"] = "salvaged"
        audit["parse_error"] = f"top-level json is not dict: {type(parsed)}"
        return dimensions_final, audit

    dims_data = parsed.get("dimensions", {})
    if not isinstance(dims_data, dict):
        audit["parse_status"] = "salvaged"
        audit["parse_error"] = "dimensions missing or not dict"
        return dimensions_final, audit

    # Parse each dimension and run pure validation. Invalid entries are skipped later.
    invalid_dimensions: List[Dict[str, Any]] = []

    for dim_id, cfg in dim_cfg_map.items():
        # Missing dimensions are marked INVALID.
        if dim_id not in dims_data:
            logger.warning(f"[AICentricEval] {model_name} dimension {dim_id}: INVALID - missing in response")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": None,
                "reason": "",
                "invalid_reason": "missing dimension in response",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": "",
                "score": None,
                "invalid": True,
                "invalid_reason": "missing dimension in response",
                "validation_errors": ["missing dimension in response; output every requested dimension"],
            }
            continue

        item = dims_data.get(dim_id)
        if not isinstance(item, dict):
            logger.warning(f"[AICentricEval] {model_name} dimension {dim_id}: INVALID - dimension payload is not a dict")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": None,
                "reason": "",
                "invalid_reason": "dimension payload is not a dict",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": "",
                "score": None,
                "invalid": True,
                "invalid_reason": "dimension payload is not a dict",
                "validation_errors": ["dimension payload must be a dict containing level and reason"],
            }
            continue

        raw_level = item.get("level", None)
        reason = item.get("reason", "")

        sm = cfg.get("score_mapping", {}) or {}
        if sm.get("type") == "levels":
            levels = sm.get("levels", []) or []
            allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
            allowed = [x for x in allowed if x]
            default_level = allowed[len(allowed) // 2] if allowed else "N/A"
        else:
            allowed = ["低", "中", "高"]
            default_level = "中"

        # Detect invalid levels without silently falling back to defaults.
        # canonicalize_level is used only for formatting normalization.
        canonical_level = canonicalize_level(dim_id, raw_level, allowed, default_level)

        # If a non-empty raw level normalizes to default_level but is not allowed, mark it invalid.
        raw_level_str = str(raw_level).strip() if raw_level is not None else ""
        if raw_level_str and canonical_level == default_level and raw_level_str not in allowed:
            logger.warning(f"[AICentricEval] {model_name} dimension {dim_id}: INVALID - level='{raw_level_str}' not in allowed list {allowed}")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": raw_level_str,
                "reason": reason,
                "invalid_reason": f"level='{raw_level_str}' not in allowed list",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": str(reason)[:400] if reason else "",
                "score": None,
                "invalid": True,
                "invalid_reason": f"level='{raw_level_str}' not in allowed list {allowed}",
                "original_level": raw_level_str,
                "validation_errors": [f"level='{raw_level_str}' not in allowed list {allowed}; use an allowed level value"],
            }
            continue

        # Pure validation mode: return VALID/INVALID only, with no overwrite or fallback.
        # Step 1: schema/anchor validation.
        anchor_valid, anchor_errors = validate_and_fix_level(
            dim_id=dim_id,
            level=canonical_level,
            reason=reason,
        )
        audit["level_validations"].append({
            "dim_id": dim_id,
            "stage": "anchor",
            "canonical_level": canonical_level,
            "is_valid": anchor_valid,
            "errors": anchor_errors,
        })

        # Step 2: reason-validity validation.
        validity_valid = True
        validity_errors: Dict[str, Any] = {}
        if anchor_valid:
            validity_level, _, validity_errors = enforce_reason_validity(
                dim_id=dim_id,
                canonical_level=canonical_level,
                reason=reason,
                score_mapping=sm,
                guessing_c=guessing_c,
            )
            validity_valid = (validity_level != "INVALID")
            audit["reason_validations"].append({
                "dim_id": dim_id,
                "stage": "validity",
                "is_valid": validity_valid,
                "errors": validity_errors,
            })

        # Any validation failure makes the dimension INVALID.
        is_invalid = (not anchor_valid) or (not validity_valid)

        if is_invalid:
            invalid_reason = anchor_errors.get("invalid_reason") or validity_errors.get("invalid_reason", "validation failed")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": canonical_level,
                "reason": reason,
                "invalid_reason": invalid_reason,
            })
            logger.warning(
                f"[AICentricEval] {model_name} dimension {dim_id}: INVALID - {invalid_reason}"
            )

            # Invalid dimensions are not scored and will be excluded from aggregation.
            result_entry: Dict[str, Any] = {
                "level": "INVALID",
                "reason": str(reason)[:400] if reason else "",
                "score": None,
                "invalid": True,
                "invalid_reason": invalid_reason,
                "original_level": canonical_level,
            }
        else:
            # Passed validation: score the original canonical level without overwrite.
            score = map_level_to_score(dim_id, canonical_level, sm, guessing_c)
            result_entry = {
                "level": canonical_level,
                "reason": str(reason)[:400] if reason else "",
                "score": float(score),
            }

        dimensions_final[dim_id] = result_entry

    # Record invalid dimensions in the audit payload.
    audit["invalid_dimensions"] = invalid_dimensions

    return dimensions_final, audit


# =============================================================================
# AICentricEval main class (batched-only).
# =============================================================================


class AICentricEval:
    """
    AI-centric evaluator (batched-only).

    Selects dimensions by question type:
    - objective questions: objective AI dimensions.
    - subjective questions: subjective AI dimensions.

    Each model is called once per question with all applicable dimensions.

    Supported inputs:
    - Stage2Record
    - Stage2CoreInput
    - VerifiedQuestionSet / GeneratedQuestion for compatibility.
    """

    def __init__(
            self,
            config: Optional[Any] = None,
            llm_client: Optional[LLMClient] = None,
            prompt_logger: Optional[Any] = None,
            eval_prompt_file: Optional[Path] = None,  # Deprecated compatibility parameter.
            eval_models: Optional[List[str]] = None,
            model_weights: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.prompt_logger = prompt_logger

        # Dimension config is loaded from data/ai_eval.json.
        # eval_prompt_file is deprecated; keep it for caller compatibility.

        # Model group and weights can be overridden by callers; weights are normalized internally.
        self.eval_models = (
                eval_models or getattr(config, "models", None)
                and [m.model_name for m in config.models]
                or STAGE2_EVAL_MODELS
        )

        if model_weights is not None:
            self.model_weights: Any = model_weights
        elif config is not None and getattr(config, "models", None):
            self.model_weights = {m.model_name: float(m.weight) for m in config.models}
        else:
            self.model_weights = MODEL_WEIGHT

        # The orchestrator can inject real clients.
        self.llm_clients: Optional[List[LLMClient]] = None

        # dims_cfg contains all objective/subjective dimensions and is filtered per question.
        self.dims_cfg: List[Dict[str, Any]] = self._load_dims_config()
        self.dim_weights: Dict[str, float] = {d["id"]: float(d.get("weight", 1.0)) for d in self.dims_cfg}

        # Decision threshold.
        self.pass_threshold: float = float(getattr(config, "pass_threshold", 70.0)) if config is not None else 70.0

    def _load_dims_config(self) -> List[Dict[str, Any]]:
        """
        Load dimension config from data/ai_eval.json.

        Config structure:
        - score_mappings: score mapping and weights for objective/subjective dimensions.
        - dimension_prompts: prompt_eval, allowed_levels, level_default, etc.

        Returns all dimensions, including IDs shared by different question types.
        Runtime filtering uses question type plus applicable_to.

        Returns:
            All dimension configs.
        """
        all_dims: List[Dict[str, Any]] = []

        # Read from the loaded module-level config.
        score_mappings = _AI_EVAL_CONFIG.get("score_mappings", {})
        dimension_prompts = _AI_EVAL_CONFIG.get("dimension_prompts", {})

        # Objective dimensions.
        obj_scores = score_mappings.get("objective", {})
        obj_prompts = dimension_prompts.get("objective", {})
        for dim_id, score_cfg in obj_scores.items():
            prompt_cfg = obj_prompts.get(dim_id, {})
            dim_config = {
                "id": dim_id,
                "name": dim_id,
                "weight": score_cfg.get("weight", 1/7),
                "applicable_to": ["objective"],
                "score_mapping": {
                    "type": score_cfg.get("type", "levels"),
                    "levels": score_cfg.get("levels", [])
                },
                "prompt_eval": prompt_cfg.get("prompt_eval", ""),
                "allowed_levels": prompt_cfg.get("allowed_levels", []),
                "level_default": prompt_cfg.get("level_default", "")
            }
            all_dims.append(dim_config)

        # Subjective dimensions.
        subj_scores = score_mappings.get("subjective", {})
        subj_prompts = dimension_prompts.get("subjective", {})
        for dim_id, score_cfg in subj_scores.items():
            prompt_cfg = subj_prompts.get(dim_id, {})
            dim_config = {
                "id": dim_id,
                "name": dim_id,
                "weight": score_cfg.get("weight", 0.25),
                "applicable_to": ["subjective"],
                "score_mapping": {
                    "type": score_cfg.get("type", "levels"),
                    "levels": score_cfg.get("levels", [])
                },
                "prompt_eval": prompt_cfg.get("prompt_eval", ""),
                "allowed_levels": prompt_cfg.get("allowed_levels", []),
                "level_default": prompt_cfg.get("level_default", "")
            }
            all_dims.append(dim_config)

        obj_count = len(obj_scores)
        subj_count = len(subj_scores)
        logger.info(
            f"[AICentricEval] Loaded dimension config from ai_eval.json: "
            f"objective={obj_count}, subjective={subj_count}, total={len(all_dims)}")
        return all_dims

    def _normalize_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """
        Normalize self.model_weights to a dict regardless of input shape.
        """
        n = max(len(model_names), 1)
        w = self.model_weights

        if isinstance(w, dict):
            base = {m: float(w.get(m, 1.0 / n)) for m in model_names}

        elif isinstance(w, list):
            # Align list values by model_names order.
            base = {m: float(w[i]) if i < len(w) else 1.0 / n for i, m in enumerate(model_names)}
        else:
            base = {m: 1.0 / n for m in model_names}

        s = sum(max(v, 0.0) for v in base.values())
        if s <= 0:
            return {m: 1.0 / n for m in model_names}
        return {m: max(v, 0.0) / s for m, v in base.items()}

    # ---------- public entry point ----------
    def run(
            self,
            question_like: Union[Stage2Record, Stage2CoreInput, VerifiedQuestionSet, GeneratedQuestion, Any],
            guessing_c: Optional[float] = None,
            hit_dimensions: Optional[List[str]] = None,  # Orchestrator compatibility; ignored here.
    ) -> Dict[str, Any]:
        """
        Unified entry point for supported question-like objects.

        - Stage2Record -> uses core_input.
        - Stage2CoreInput -> used directly.
        - VerifiedQuestionSet -> uses original_question.
        - GeneratedQuestion -> used directly.
        - Other objects -> attempts .core_input or .original_question.
        """
        if isinstance(question_like, Stage2Record):
            core: Any = question_like.core_input
        elif isinstance(question_like, Stage2CoreInput):
            core = question_like

        elif isinstance(question_like, VerifiedQuestionSet):
            core = question_like.original_question
        elif isinstance(question_like, GeneratedQuestion):
            core = question_like
        else:
            # Light backward compatibility for older wrappers.
            core = getattr(question_like, "core_input", None) or getattr(question_like, "original_question", None)
            if core is None:
                raise ValueError(f"AICentricEval.run: unsupported input {type(question_like)}")

        return self._run_batched(core, guessing_c=guessing_c)

    # ---------- batched main path ----------
    def _run_batched(self, question: Any, guessing_c: Optional[float]) -> Dict[str, Any]:
        if not self.dims_cfg:
            return {
                "dimensions": {},
                "overall_score": 0.0,
                "total_score": 0.0,
                "decision": "error",
                "model_results": {},
                "audit": {
                    "call_count": 0,

                    "models_called": 0,
                    "models_success": 0,
                    "details": ["no dimensions configured"],
                },
            }

        # -------------------------
        # Use question_family classification and the shared dimension applicability matrix.
        # Shared dimensions can have different prompt templates by applicable_to.
        # -------------------------
        qt = str(getattr(question, "question_type", "") or "").strip()
        stem = str(getattr(question, "stem", "") or "").strip()
        opts = getattr(question, "options", None)

        # Use the unified dimension filtering with question family inference
        dims_cfg_eff, dim_filter_result = filter_and_renormalize_dimensions(
            question_type=qt,
            stem=stem,
            options=opts,
            all_dimensions=self.dims_cfg,
        )

        # Build effective dimension IDs and weights from filtered result
        eff_ids: set[str] = set(dim_filter_result.applied_dimensions)
        dim_weights_eff: Dict[str, float] = dict(dim_filter_result.renormalized_weights)

        # Build skipped list for audit
        skipped: List[Dict[str, str]] = [
            {"id": dim_id, "reason": f"not applicable for question_family={dim_filter_result.question_family}"}
            for dim_id in dim_filter_result.skipped_dimensions
        ]

        logger.info(
            f"[AICentricEval] Question family: {dim_filter_result.question_family}, "
            f"Applied: {dim_filter_result.applied_dimensions}, "
            f"Skipped: {dim_filter_result.skipped_dimensions}"
        )

        # -------------------------
        # Prefer clients injected by the orchestrator.
        # -------------------------
        clients: List[LLMClient]
        if self.llm_clients:
            clients = list(self.llm_clients)
        else:
            api_type = getattr(self.llm_client, "api_type", "openai") if self.llm_client else "openai"
            clients = [LLMClient(api_type=api_type, model_name=m, verbose=False) for m in self.eval_models]

        model_names = [c.model_name for c in clients]
        weights = self._normalize_model_weights(model_names)

        # Prompting and parsing use the filtered effective dimension config.
        prompt = build_ai_batched_prompt(question, dims_cfg_eff, guessing_c)

        model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        audits: List[Dict[str, Any]] = []
        this_run_calls = 0

        # -------------------------
        # Initial model evaluation
        # -------------------------
        for client in clients:
            name = client.model_name
            try:
                gen_kwargs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": STAGE2_LLM_PARAMS.get("max_tokens", 4096),
                    "metadata": {"agent": "AICentricEval", "mode": "batched"},
                }
                if not is_no_temperature_model(name or ""):
                    gen_kwargs["temperature"] = 0.0

                resp = client.generate(**gen_kwargs)
                this_run_calls += 1

                dims_final, audit = parse_batched_ai_response(
                    response=resp,
                    dims_cfg=dims_cfg_eff,
                    guessing_c=guessing_c,
                    model_name=name,
                )
                model_results[name] = dims_final
                audits.append(audit)
            except Exception as e:
                audits.append({"model": name, "parse_status": "error", "parse_error": str(e)})

        # INVALID dimensions are excluded from aggregation without additional LLM repair calls.
        final_invalid_count = 0
        for model_name, dims in model_results.items():
            for dim_id, dim_result in dims.items():
                if dim_result.get("invalid") or dim_result.get("level") == "INVALID":
                    final_invalid_count += 1

        if final_invalid_count > 0:
            logger.warning(
                f"[AICentricEval] {final_invalid_count} invalid model-dimension pairs will be excluded from aggregation"
            )

        # Aggregate only applicable dimensions; invalid dimension outputs are skipped.
        final_dimensions = self._aggregate(
            model_results,
            weights,
            dims_cfg=dims_cfg_eff,
            dim_weights=dim_weights_eff,
        )

        # Add skipped dimensions back as N/A with zero weight so totals are unchanged.
        for item in skipped:
            dim_id = item.get("id", "")
            if dim_id and dim_id not in final_dimensions:
                final_dimensions[dim_id] = {
                    "level": "N/A",
                    "reason": item.get("reason", ""),
                    "score": None,
                    "weight": 0.0,
                }

        overall = self._weighted_total(final_dimensions)
        decision = "pass"

        return {
            "dimensions": final_dimensions,
            "overall_score": overall,
            "total_score": overall,  # alias
            "decision": decision,
            "model_results": model_results,
            "audit": {
                "call_count": this_run_calls,
                "models_called": len(clients),
                "models_success": len(model_results),
                "details": audits,
                "skipped_dimensions": skipped,
                "dimension_filter": dim_filter_result.to_dict(),
                "final_invalid_count": final_invalid_count,
            },
            "applied_dimensions": dim_filter_result.applied_dimensions,
            "renormalized_weights": dim_filter_result.renormalized_weights,
        }

    def _aggregate(
            self,
            model_results: Dict[str, Dict[str, Dict[str, Any]]],
            model_weights: Dict[str, float],
            *,
            dims_cfg: Optional[List[Dict[str, Any]]] = None,
            dim_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate multi-model evaluation results.

        Returns:
            - aggregated: {dim_id: {"level", "reason", "score", "weight"}}
        """
        if not model_results:
            return {}

        dims_cfg = self.dims_cfg if dims_cfg is None else dims_cfg
        dim_weights = self.dim_weights if dim_weights is None else dim_weights

        aggregated: Dict[str, Dict[str, Any]] = {}
        dim_ids = [d["id"] for d in (dims_cfg or []) if isinstance(d, dict) and d.get("id")]

        for dim_id in dim_ids:
            scored: List[Tuple[float, float]] = []
            levels: List[str] = []
            reasons: List[str] = []

            for model_name, dims in model_results.items():
                if dim_id not in dims:
                    continue

                w = float(model_weights.get(model_name, 0.0))
                s = (dims.get(dim_id) or {}).get("score", None)
                level_val = str((dims.get(dim_id) or {}).get("level", ""))

                try:
                    if s is None:
                        continue
                    score_float = float(s)
                    scored.append((score_float, w))
                except Exception:
                    continue

                levels.append(level_val)
                r = str((dims.get(dim_id) or {}).get("reason", "")).strip()
                if r:
                    reasons.append(r)

            if not scored:
                continue

            total_w = sum(w for _, w in scored)
            if total_w > 0:
                score = sum(s * w for s, w in scored) / total_w
            else:

                score = sum(s for s, _ in scored) / float(len(scored))

            level = Counter(levels).most_common(1)[0][0] if levels else ""
            reason = "; ".join(reasons)[:500] if reasons else ""
            weight = float((dim_weights or {}).get(dim_id, 1.0))

            aggregated[dim_id] = {
                "level": level,
                "reason": reason,
                "score": round(float(score), 2),
                "weight": weight,
            }

        return aggregated

    def _weighted_total(self, dimensions: Dict[str, Dict[str, Any]]) -> float:
        if not dimensions:
            return 0.0

        total_w = 0.0
        total = 0.0

        for _, d in dimensions.items():
            w = float(d.get("weight", 1.0) or 0.0)
            s = d.get("score", None)

            if s is None or w <= 0:
                continue

            try:
                s = float(s)
            except Exception:
                continue

            total += s * w
            total_w += w

        return round(total / total_w, 2) if total_w > 0 else 0.0


# =============================================================================
# Convenience demo wrapper.
# =============================================================================


def evaluate_question_ai_centric(
        question: GeneratedQuestion,
        guessing_c: Optional[float] = None,
        eval_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    evaluator = AICentricEval(eval_models=eval_models)
    return evaluator.run(question, guessing_c=guessing_c)


__all__ = ["AICentricEval", "evaluate_question_ai_centric"]
