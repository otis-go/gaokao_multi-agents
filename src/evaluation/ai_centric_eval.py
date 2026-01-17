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
# 3) Pure validation: Non-compliant returns INVALID, triggers re-evaluation (max 3 rounds)
# 4) No auto-fallback/override: Only override dimension result after successful re-evaluation
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
        # Compatible with format like "最终计入N条"/"N=2"
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
        # Note: Distinguish between "present" vs negated context "未见/无...present"
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
# Upper layer triggers re-evaluation mechanism (max 3 rounds)
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
    - No override/fallback, upper layer decides whether to trigger re-evaluation
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
# Integrates all validation logic, returns (valid, errors) for re-evaluation
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

    Integrates all validation logic, returns specific errors for re-evaluation prompts.

    Args:
        dim_id: Dimension ID
        level: Model output level
        score: Model output score (optional)
        reason: Model output reason
        dim_cfg: Dimension config containing allowed_levels, score_mapping etc.
        guessing_c: Guessing probability (for specific dimension validation)

    Returns:
        (valid, errors)
        - valid: True=passed validation, False=non-compliant needs re-evaluation
        - errors: Specific error list for re-evaluation prompts
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

    # 3.0) stem_quality: Must contain "任务动作词=...; 范围/对象=..."
    if dim_id == "stem_quality":
        pattern = r"任务动作词=.+;\s*范围/对象=.+"
        if not re.search(pattern, reason_text):
            errors.append("stem_quality requires reason to contain '任务动作词=<verb>; 范围/对象=<noun phrase>' (both required)")

    # 3.0b) option_exclusivity_coverage: Must contain "竞争轴=...; 选项对=关系"
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
        # level=0,1 (partial self-sufficient or lower): Must contain 缺口点=...
        # level=2,3 (basic self-sufficient or higher): Must contain 闭环证据=...
        low_levels = ["0", "1"]
        high_levels = ["2", "3"]
        has_gap = bool(re.search(r"缺口点=.+", reason_text))
        has_evidence = bool(re.search(r"闭环证据=.+", reason_text))

        if level in low_levels and not has_gap:
            errors.append(f"item_evidence_sufficiency level='{level}' (partial self-sufficient or lower) requires reason to contain '缺口点=<specific gap>'")
        elif level in high_levels and not has_evidence:
            errors.append(f"item_evidence_sufficiency level='{level}' (basic self-sufficient or higher) requires reason to contain '闭环证据=<evidence point>'")

    # 3.6) rubric_operational: Must contain "要点数=m; 边界数=n"
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
    从 ai_eval.json 构建指定题型的维度配置列表。

    Args:
        q_type: "objective" 或 "subjective"

    Returns:
        维度配置列表
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
# level 规范化（尽量减少映射失败）
# =============================================================================

_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．／", "0123456789./")

# =============================================================================
# 【2025-12 重构】越权/不一致理由校验规则配置
#
# 设计原则：
# - 纯校验模式：命中 forbidden_phrases 或锚点矛盾 → INVALID（不回退、不改分）
# - 由上层触发重评机制（最多3轮）
# =============================================================================

REASON_VALIDITY_RULES = {
    "by_dimension": {
        # 1) option_exclusivity_coverage：材料归属/来源考据越权（仅此维度生效）
        "option_exclusivity_coverage": {
            "forbidden_phrases": [
                "材料一", "材料二", "材料三",
                "来源不一致", "来源不同", "出处不同", "原文出处",
                "不在材料", "出自材料", "来自材料一", "来自材料二",
                "对应材料一", "对应材料二", "材料来源",
                "哪段材料", "哪一段材料", "材料第几段",
            ],
        },

        # 2) guessing_lower_asymptote：语义贴合/直觉式越权检测
        "guessing_lower_asymptote": {
            # 禁止的语义贴合词（需同时指向具体选项且缺少形式证据）
            "semantic_fit_phrases": [
                "最贴合", "更贴合",
                "一眼看出", "直觉",
                "语义明显", "显然选",
                "最像正确", "明显正确", "更合理",
            ],
            # 选项指向正则（用于判断是否指向具体选项）
            "option_pointer_regex": r"(?:选项)?\s*(?:[A-Da-d]|[甲乙丙丁])\b|(?:只有|选)\s*(?:[A-Da-d]|[甲乙丙丁])\b",
            # 形式证据（有这些则不判越权）
            "form_evidence_phrases": [
                "长度", "句式", "语体", "语体断裂", "风格突变",
                "对称", "形式对称",
                "异常突出", "唯一异常",
                "跑题", "任务不匹配", "明显无关",
                "常识矛盾", "自相矛盾", "自我否定",
                "模式线索", "题干直给",
            ],
        },

        # 3) distractor_headroom：N值一致性检查
        "distractor_headroom": {
            # score 与 N 的最低一致性门槛
            "score_min_n": {
                0: 0,   # score=0 允许 N=0（若 N>=1 则矛盾 → INVALID）
                50: 1,  # score=50 至少 N>=1
                75: 2,  # score=75 至少 N>=2
                100: 3, # score=100 至少 N>=3
            },
        },
    },
}


def _rv_has_any(s: str, kws: List[str]) -> bool:
    if not s:
        return False
    return any(k in s for k in kws)


def _rv_extract_k_guessing(reason: str) -> Optional[int]:
    # 支持 “K=2 / K=3+ / 强可排除项数量K=2”
    m = re.search(r"(?:K\s*=\s*)(\d)\+?", reason or "")
    if m:
        return int(m.group(1))
    # 兜底：若写了“三个强可排除项/两个强可排除项”
    r = reason or ""
    if "三个" in r and "强可排除" in r:
        return 3
    if "两个" in r and "强可排除" in r:
        return 2
    if "一个" in r and "强可排除" in r:
        return 1
    return None


def _rv_extract_n_distractor(reason: str) -> Optional[int]:
    # 支持 “N=2 / 最终计入2条”
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
    【2025-12 重构】越权/不一致理由校验器（纯校验模式）

    检测 reason 中的越权表述或与 level 不一致的信号。
    不合规时标记 INVALID，由上层触发重评。
    不再做任何自动回退/覆盖。

    Returns:
        (result_level, reason, meta)
        - result_level: 原 level 或 "INVALID"
        - reason: 原 reason（不再追加 [校验回退] 注记）
        - meta: 校验详情，包含 valid, flags, invalid_reason 等
    """
    meta: Dict[str, Any] = {"valid": True, "flags": [], "dim_id": dim_id}
    lvl = canonical_level
    r = reason or ""

    # 获取当前维度的规则配置
    dim_rules = REASON_VALIDITY_RULES.get("by_dimension", {}).get(dim_id, {})
    if not dim_rules:
        return lvl, r, meta

    # 预估当前 score（便于做不一致检查）
    try:
        cur_score = map_level_to_score(dim_id, lvl, score_mapping, guessing_c)
    except Exception:
        cur_score = None

    # ========== A) option_exclusivity_coverage：材料归属/来源考据越权 ==========
    if dim_id == "option_exclusivity_coverage":
        forbidden = dim_rules.get("forbidden_phrases", [])
        if _rv_has_any(r, forbidden):
            meta["valid"] = False
            meta["flags"].append("越权：材料来源考据")
            meta["invalid_reason"] = "option_exclusivity_coverage: 检测到材料归属/来源考据越权表述"
            return "INVALID", r, meta

    # ========== B) guessing_lower_asymptote：语义贴合越权 + K值一致性检查 ==========
    if dim_id == "guessing_lower_asymptote":
        sem_phrases = dim_rules.get("semantic_fit_phrases", [])
        form_phrases = dim_rules.get("form_evidence_phrases", [])
        option_regex = dim_rules.get("option_pointer_regex", r"")

        has_sem = _rv_has_any(r, sem_phrases)
        has_form = _rv_has_any(r, form_phrases)
        has_option_pointer = bool(re.search(option_regex, r)) if option_regex else False

        # B1) 语义贴合 + 选项指向 + 无形式证据 → 越权 INVALID
        if has_sem and has_option_pointer and (not has_form):
            meta["valid"] = False
            meta["flags"].append("越权：语义贴合无形式证据")
            meta["invalid_reason"] = "guessing_lower_asymptote: 语义贴合/直觉式理由但缺少形式证据"
            return "INVALID", r, meta

        # B2) K值解析与一致性检查
        k = _rv_extract_k_guessing(r)

        if k is None:
            # 低档未报告K且无特殊证据 → INVALID
            if lvl in {"明显易蒙对", "接近送分题"} and ("题干直给" not in r and "模式线索" not in r):
                meta["valid"] = False
                meta["flags"].append("不一致：低档未报告K值")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level={lvl}但未报告K值且无'题干直给/模式线索'证据"
                return "INVALID", r, meta
        else:
            # K值与level不一致检查
            if lvl == "明显易蒙对" and k < 2:
                meta["valid"] = False
                meta["flags"].append("不一致：明显易蒙对但K<2")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level='明显易蒙对'但K={k}<2"
                return "INVALID", r, meta

            if lvl == "接近送分题" and (k < 3) and ("题干直给" not in r):
                meta["valid"] = False
                meta["flags"].append("不一致：接近送分题但K<3")
                meta["invalid_reason"] = f"guessing_lower_asymptote: level='接近送分题'但K={k}<3且无'题干直给'证据"
                return "INVALID", r, meta

    # ========== C) distractor_headroom：N值解析与score一致性检查 ==========
    if dim_id == "distractor_headroom":
        score_min_n = dim_rules.get("score_min_n", {0: 0, 50: 1, 75: 2, 100: 3})

        # 解析N值
        n = _rv_extract_n_distractor(r)

        if n is None:
            # 无法解析N且为低档 → INVALID
            if lvl in {"0", "1"}:
                meta["valid"] = False
                meta["flags"].append("不一致：低档未报告N值")
                meta["invalid_reason"] = f"distractor_headroom: level={lvl}但未报告'最终计入N条'"
                return "INVALID", r, meta
        else:
            # 检查 score ↔ N 一致性
            if cur_score is not None:
                min_n_required = score_min_n.get(int(cur_score), 0)

                # score=0 但 N>=1 → 矛盾 INVALID
                if cur_score == 0 and n >= 1:
                    meta["valid"] = False
                    meta["flags"].append("不一致：score=0但N>=1")
                    meta["invalid_reason"] = f"distractor_headroom: level=0(score=0)但理由报告N={n}条可构造路径"
                    return "INVALID", r, meta

                # score>0 但 N < 最低要求 → 矛盾 INVALID
                if cur_score > 0 and n < min_n_required:
                    meta["valid"] = False
                    meta["flags"].append(f"不一致：score={cur_score}但N={n}<{min_n_required}")
                    meta["invalid_reason"] = f"distractor_headroom: score={cur_score}但N={n}不满足最低要求{min_n_required}"
                    return "INVALID", r, meta

    return lvl, r, meta


def _clean_level_text(x: Any) -> str:
    s = str(x).strip()
    s = s.translate(_FULLWIDTH_DIGITS)
    s = re.sub(r"\s+", " ", s)

    return s


def canonicalize_level(dim_id: str, raw_level: Any, allowed_levels: List[str], default_level: str) -> str:
    """
    将模型输出 level 归一化到 allowed_levels 中的一个：
    - EXACT MATCH 直接返回
    - 数值："85.0" -> "85" 并选最近 allowed 数字
    - 兜底：default_level
    """
    if raw_level is None:
        return default_level

    allowed = [str(a) for a in allowed_levels]
    allowed_set = set(allowed)
    s = _clean_level_text(raw_level)

    if s in allowed_set:
        return s

    # 数字型标签：选最近
    if re.fullmatch(r"-?\d+(\.\d+)?", s) and all(re.fullmatch(r"\d+", a) for a in allowed):
        try:
            v = float(s)
            best = min(allowed, key=lambda a: abs(float(a) - v))
            return str(best)
        except Exception:
            return default_level

    # 斜杠全角/半角
    s2 = s.replace("／", "/")
    if s2 in allowed_set:
        return s2

    return default_level


# =============================================================================
# prompt：维度列表渲染（强约束）
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

    # 总体说明
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

        # 【2025-12 修复】优先使用 prompt_eval 字段（包含完整评估提示词）
        prompt_eval = (d.get("prompt_eval") or "").strip()
        if not prompt_eval:
            # 兼容：若无 prompt_eval，尝试使用 eval_rules 或 definition
            prompt_eval = (d.get("eval_rules") or "").strip()
            if not prompt_eval:
                definition = (d.get("definition") or "").strip().replace("\n", " ")
                if len(definition) > 200:
                    definition = definition[:200] + "..."
                prompt_eval = definition

        allowed_render = " | ".join([json.dumps(str(x), ensure_ascii=False) for x in allowed])

        # 清晰的维度分隔
        lines.append("┌─────────────────────────────────────────────────────────────────┐")
        lines.append(f"│  【维度 {i}/{dim_count}】{name}（{dim_id}）")
        lines.append("└─────────────────────────────────────────────────────────────────┘")
        lines.append(f"allowed_levels = {allowed_render}")
        lines.append(f"level_default = {json.dumps(str(default), ensure_ascii=False)}")

        if dim_id == "guessing_lower_asymptote":
            lines.append("特殊说明：低/中/高判断阈值（只为可读性，最终 score 由系统公式算）：")
            lines.append("  c <= 0.08 -> 低；0.08 < c <= 0.20 -> 中；c > 0.20 -> 高")

        # 输出完整的评估规则（来自 prompt_eval）
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
    # 【2025-12 重构】AI 评估只使用：材料、题干、选项（选择题）
    # 不喂入答案、解析、答案要点等信息，确保盲评
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

    # 维度规格（严格来自配置；formula 维度提供低/中/高）
    # 【2025-12 修复】确保包含 prompt_eval 字段，用于完整的评估提示词
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
                "prompt_eval": d.get("prompt_eval", ""),  # 【关键修复】包含完整评估提示词
                "eval_rules": d.get("eval_rules", ""),  # 兼容旧配置
                "allowed_levels": allowed,
                "level_default": default,
            }
        )

    dim_specs_text = render_dim_specs(dim_specs, guessing_c)

    example_dims = {d["id"]: {"level": d["level_default"], "reason": "..."} for d in dim_specs}
    json_example = json.dumps({"dimensions": example_dims}, ensure_ascii=False, indent=2)

    # 【2025-12 重构】根据题型区分显示：选择题显示选项，简答题显示总分
    # 不喂入答案、解析等信息，确保 AI 盲评
    if question_type == "essay":
        # 简答题模板
        # 【2025-12 修改】AI 评估只使用原材料和题目，不再使用 answer/analysis
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
        # 选择题模板（默认）
        # 【2025-12 修改】AI 评估只使用原材料和题目，不再使用 answer/analysis
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
# score 计算：levels 映射 + formula（guessing_lower_asymptote）
# =============================================================================


def _score_by_levels(level: str, levels: List[Dict[str, Any]]) -> float:
    if not levels:
        return 50.0
    label_to_score = {str(x.get("label")): float(x.get("score", 0)) for x in levels if "label" in x}
    if level in label_to_score:
        return float(label_to_score[level])
    # 中位档默认
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

    # formula：目前只支持 guessing_lower_asymptote（避免 eval）
    if mtype == "formula":
        if dim_id == "guessing_lower_asymptote":
            return _score_guessing_lower_asymptote(guessing_c)
        return 50.0

    return 50.0


# =============================================================================
# batched 响应解析：永不崩溃
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
        "level_validations": [],  # 【2025-12 新增】记录 level 二次校验结果
        "reason_validations": [],
    }

    # 默认输出（中位档 or formula 默认）
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

        # 【2025-12 修复】初始化为 INVALID，不使用 default_level 作为兜底
        # 解析失败的维度标记为 INVALID，触发重评而不是静默使用默认值
        dimensions_final[d["id"]] = {
            "level": "INVALID",
            "reason": "初始化（待解析）",
            "score": None,
            "invalid": True,
            "invalid_reason": "维度尚未解析",
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

    # 关键修复：parsed 可能不是 dict（例如 list），不能直接 .get
    if not isinstance(parsed, dict):
        audit["parse_status"] = "salvaged"
        audit["parse_error"] = f"top-level json is not dict: {type(parsed)}"
        return dimensions_final, audit

    dims_data = parsed.get("dimensions", {})
    if not isinstance(dims_data, dict):
        audit["parse_status"] = "salvaged"
        audit["parse_error"] = "dimensions missing or not dict"
        return dimensions_final, audit

    # 【2025-12 重构】逐维解析 + 纯校验（不做回退，不合规标记 INVALID）
    invalid_dimensions: List[Dict[str, Any]] = []  # 记录 INVALID 维度，供上层重评

    for dim_id, cfg in dim_cfg_map.items():
        # 【F1】维度缺失检查：缺失的维度标记为 INVALID
        if dim_id not in dims_data:
            logger.warning(f"[AICentricEval] {model_name} 维度 {dim_id}: INVALID - 响应中缺失该维度")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": None,
                "reason": "",
                "invalid_reason": "响应中缺失该维度",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": "",
                "score": None,
                "invalid": True,
                "invalid_reason": "响应中缺失该维度",
                "validation_errors": ["响应中缺失该维度，请确保输出所有要求的维度"],
            }
            continue

        item = dims_data.get(dim_id)
        if not isinstance(item, dict):
            logger.warning(f"[AICentricEval] {model_name} 维度 {dim_id}: INVALID - 维度数据格式错误（非字典）")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": None,
                "reason": "",
                "invalid_reason": "维度数据格式错误（非字典）",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": "",
                "score": None,
                "invalid": True,
                "invalid_reason": "维度数据格式错误（非字典）",
                "validation_errors": ["维度数据格式错误，请确保每个维度是包含 level 和 reason 字段的字典"],
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

        # 【G1】检测无效 level：不静默转换，直接标记 INVALID
        # canonicalize_level 只用于格式归一化（大小写、空格），不用于静默 fallback
        canonical_level = canonicalize_level(dim_id, raw_level, allowed, default_level)

        # 检查：如果 raw_level 存在但 canonical_level 变成了 default_level（说明被静默转换了）
        # 则需要判断原始 level 是否真的无效
        raw_level_str = str(raw_level).strip() if raw_level is not None else ""
        if raw_level_str and canonical_level == default_level and raw_level_str not in allowed:
            # 原始 level 既不为空，也不在 allowed 中，但被静默转换为 default_level
            # 这是无效 level，应标记为 INVALID
            logger.warning(f"[AICentricEval] {model_name} 维度 {dim_id}: INVALID - level='{raw_level_str}' 不在允许列表 {allowed} 中")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": raw_level_str,
                "reason": reason,
                "invalid_reason": f"level='{raw_level_str}' 不在允许列表中",
            })
            dimensions_final[dim_id] = {
                "level": "INVALID",
                "reason": str(reason)[:400] if reason else "",
                "score": None,
                "invalid": True,
                "invalid_reason": f"level='{raw_level_str}' 不在允许列表 {allowed} 中",
                "original_level": raw_level_str,
                "validation_errors": [f"level='{raw_level_str}' 不在允许列表 {allowed} 中，请使用允许的 level 值"],
            }
            continue

        # 【2025-12 重构】纯校验模式：只返回 VALID/INVALID，不做覆盖/回退
        # Step 1: Schema/Anchor 校验
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

        # Step 2: Reason-validity 校验（越权/不一致检测）
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

        # 判断是否 INVALID（任一校验失败即为 INVALID）
        is_invalid = (not anchor_valid) or (not validity_valid)

        if is_invalid:
            # 记录 INVALID 维度，供上层重评
            invalid_reason = anchor_errors.get("invalid_reason") or validity_errors.get("invalid_reason", "校验不通过")
            invalid_dimensions.append({
                "dim_id": dim_id,
                "original_level": canonical_level,
                "reason": reason,
                "invalid_reason": invalid_reason,
            })
            logger.warning(
                f"[AICentricEval] {model_name} 维度 {dim_id}: INVALID - {invalid_reason}"
            )

            # INVALID 维度：暂时保留默认值，等待重评覆盖
            # 不计算 score（保持默认的解析失败值）
            result_entry: Dict[str, Any] = {
                "level": "INVALID",
                "reason": str(reason)[:400] if reason else "",
                "score": None,  # INVALID 不计分
                "invalid": True,
                "invalid_reason": invalid_reason,
                "original_level": canonical_level,
            }
        else:
            # 通过校验：使用原始 canonical_level 计算 score（不做任何覆盖）
            score = map_level_to_score(dim_id, canonical_level, sm, guessing_c)
            result_entry = {
                "level": canonical_level,
                "reason": str(reason)[:400] if reason else "",
                "score": float(score),
            }

        dimensions_final[dim_id] = result_entry

    # 【2025-12 新增】在 audit 中记录 INVALID 维度列表
    audit["invalid_dimensions"] = invalid_dimensions

    return dimensions_final, audit


# =============================================================================
# AICentricEval 主类（batched-only）
# =============================================================================


class AICentricEval:
    """
    【2025-12 重构】AI角度评估器（batched-only）

    根据题型选择不同的维度进行评估：
    - 选择题(objective): 8个维度
    - 简答题(subjective): 5个维度

    每模型一次调用，所有适用维度一起评估。

    现在支持多种输入：
    - Stage2Record（推荐：从 Stage 1 state.stage2_record 直接传入）
    - Stage2CoreInput
    - VerifiedQuestionSet / GeneratedQuestion（旧版兼容）
    """

    def __init__(
            self,
            config: Optional[Any] = None,
            llm_client: Optional[LLMClient] = None,
            prompt_logger: Optional[Any] = None,
            eval_prompt_file: Optional[Path] = None,  # 【2025-12 废弃】保留参数但不再使用
            eval_models: Optional[List[str]] = None,
            model_weights: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.prompt_logger = prompt_logger

        # 【2025-12 重构】维度配置从 data/ai_eval.json 加载
        # eval_prompt_file 参数已废弃，统一使用 ai_eval.json

        # 模型组 & 权重（外部可覆盖；内部会 normalize）
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

        # orchestrator 会注入真实 client 列表（推荐）
        self.llm_clients: Optional[List[LLMClient]] = None

        # 【2025-12 重构】使用嵌入的维度配置
        # dims_cfg 包含所有维度（客观题+主观题），实际评估时根据题型筛选
        self.dims_cfg: List[Dict[str, Any]] = self._load_dims_config()
        self.dim_weights: Dict[str, float] = {d["id"]: float(d.get("weight", 1.0)) for d in self.dims_cfg}

        # 阈值（用于 decision）
        self.pass_threshold: float = float(getattr(config, "pass_threshold", 70.0)) if config is not None else 70.0

    def _load_dims_config(self) -> List[Dict[str, Any]]:
        """
        【2025-12 重构】从 data/ai_eval.json 加载维度配置。

        配置文件结构：
        - score_mappings: 包含 objective/subjective 两类维度的 score_mapping 和 weight
        - dimension_prompts: 包含 prompt_eval、allowed_levels、level_default 等

        返回所有维度配置，包括同名但不同题型版本的维度（如 stem_quality 客观题版和主观题版）。
        实际评估时根据题型 + applicable_to 字段筛选使用。

        Returns:
            List[Dict[str, Any]]: 所有维度配置列表（可能包含同 ID 不同题型版本）
        """
        all_dims: List[Dict[str, Any]] = []

        # 从已加载的配置中获取
        score_mappings = _AI_EVAL_CONFIG.get("score_mappings", {})
        dimension_prompts = _AI_EVAL_CONFIG.get("dimension_prompts", {})

        # 处理客观题维度
        obj_scores = score_mappings.get("objective", {})
        obj_prompts = dimension_prompts.get("objective", {})
        for dim_id, score_cfg in obj_scores.items():
            prompt_cfg = obj_prompts.get(dim_id, {})
            dim_config = {
                "id": dim_id,
                "name": dim_id,  # 可以从其他地方获取中文名称
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

        # 处理主观题维度
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
            f"[AICentricEval] 从 ai_eval.json 加载维度配置: 客观题{obj_count}维, 主观题{subj_count}维, 合计{len(all_dims)}维")
        return all_dims

    def _normalize_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """
        关键：无论 self.model_weights 是 dict 还是 list，都 normalize 成 dict，避免 `.get()` 崩溃。
        """
        n = max(len(model_names), 1)
        w = self.model_weights

        if isinstance(w, dict):
            base = {m: float(w.get(m, 1.0 / n)) for m in model_names}

        elif isinstance(w, list):
            # 假设按 model_names 顺序对齐
            base = {m: float(w[i]) if i < len(w) else 1.0 / n for i, m in enumerate(model_names)}
        else:
            base = {m: 1.0 / n for m in model_names}

        s = sum(max(v, 0.0) for v in base.values())
        if s <= 0:
            return {m: 1.0 / n for m in model_names}
        return {m: max(v, 0.0) / s for m, v in base.items()}

    # ---------- 【2025-12 新增】D3) 单维度重评方法（含 repair feedback）----------
    def _build_single_dim_reevaluate_prompt(
            self,
            question: Any,
            dim_cfg: Dict[str, Any],
            guessing_c: Optional[float],
            previous_level: str,
            previous_reason: str,
            validation_errors: List[str],
    ) -> str:
        """
        【D3】构建针对单个维度的重评 prompt（含 repair feedback）。

        包含：
        1. 原题目信息（材料、题干、选项 - 原样喂入保证上下文完整）
        2. 维度定义与分档 rubric
        3. 上一轮该维度的输出（level/reason）
        4. validator 的 errors 列表，明确要求"按 errors 修正并重写输出 JSON"
        """
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

        dim_id = dim_cfg["id"]
        prompt_eval = dim_cfg.get("prompt_eval", "") or ""

        sm = dim_cfg.get("score_mapping", {}) or {}
        if sm.get("type") == "levels":
            levels = sm.get("levels", []) or []
            allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
            allowed = [x for x in allowed if x]
            default = allowed[len(allowed) // 2] if allowed else "N/A"
        else:
            allowed = ["低", "中", "高"]
            default = "中"

        allowed_render = " | ".join([json.dumps(str(x), ensure_ascii=False) for x in allowed])

        # 格式化 errors 列表
        errors_text = ""
        for i, err in enumerate(validation_errors, 1):
            errors_text += f"  {i}. {err}\n"

        # 构建重评 prompt（包含 repair feedback）
        prompt = f"""你是一位资深的中文阅读理解题目质量评估专家。

╔══════════════════════════════════════════════════════════════════╗
║  【重评任务】你上次的评估结果未通过校验，请按要求修正后重新输出   ║
╚══════════════════════════════════════════════════════════════════╝

【原题目信息（完整上下文）】
- 材料（material_text）：
{material if material else "(未提供)"}

- 题干（stem）：
{stem if stem else "(未提供)"}

- 选项（options）：
{options_text if options_text else "(无选项)"}

────────────────────────────────────────────────────────────────────
【维度定义与分档规则】
维度ID: {dim_id}
allowed_levels = {allowed_render}
level_default = {json.dumps(str(default), ensure_ascii=False)}

{prompt_eval}

────────────────────────────────────────────────────────────────────
【上一轮你的输出】
- level: {previous_level}
- reason: {previous_reason}

────────────────────────────────────────────────────────────────────
【校验错误列表（必须全部修正）】
{errors_text}
⚠️ 请仔细阅读上述错误，理解问题所在，然后按要求修正并重新输出。

────────────────────────────────────────────────────────────────────
【输出要求】
1) 只输出一个 JSON 对象，格式如下：
   {{"level": "...", "reason": "..."}}

2) level 必须从 allowed_levels 中选一个（EXACT MATCH）
3) reason 必须包含支持判断的关键锚点证据：
   - answer_uniqueness 必须报告 |C|=n
   - distractor_headroom 必须报告 N=n 或"最终计入N条"
   - guessing_lower_asymptote 必须报告 K=n（或提供"题干直给/模式线索"证据）
4) 禁止输出 score；禁止自造标签

现在请按上述要求修正并重新输出 JSON：
"""
        return prompt

    def _reevaluate_single_dimension(
            self,
            client: LLMClient,
            question: Any,
            dim_cfg: Dict[str, Any],
            guessing_c: Optional[float],
            previous_level: str,
            previous_reason: str,
            validation_errors: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        对单个模型的单个维度进行重评。

        Args:
            client: LLM 客户端
            question: 题目对象
            dim_cfg: 维度配置
            guessing_c: 蒙对概率
            previous_level: 上一轮输出的 level
            previous_reason: 上一轮输出的 reason
            validation_errors: schema_validate_dim() 返回的具体错误列表

        Returns:
            (dim_result, audit)
            - dim_result: {"level", "reason", "score"} 或 {"level": "INVALID", ...}
            - audit: 重评审计信息
        """
        dim_id = dim_cfg["id"]
        model_name = client.model_name

        audit: Dict[str, Any] = {
            "dim_id": dim_id,
            "model": model_name,
            "reevaluate": True,
            "input_errors": validation_errors,  # 记录输入的错误列表
        }

        prompt = self._build_single_dim_reevaluate_prompt(
            question=question,
            dim_cfg=dim_cfg,
            guessing_c=guessing_c,
            previous_level=previous_level,
            previous_reason=previous_reason,
            validation_errors=validation_errors,
        )

        try:
            gen_kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": STAGE2_LLM_PARAMS.get("max_tokens", 2048),
                "metadata": {"agent": "AICentricEval", "mode": "reevaluate", "dim_id": dim_id},
            }
            if not is_no_temperature_model(model_name or ""):
                gen_kwargs["temperature"] = 0.0

            resp = client.generate(**gen_kwargs)
            audit["response_len"] = len(resp) if resp else 0

            # 解析单维度响应
            extracted = _extract_json_candidate(resp)
            if not extracted:
                audit["parse_status"] = "no_json"
                return {"level": "INVALID", "reason": "重评解析失败：无 JSON", "score": None, "invalid": True}, audit

            try:
                parsed = json.loads(extracted)
            except Exception:
                try:
                    parsed = json.loads(_repair_common_json(extracted))
                except Exception:
                    audit["parse_status"] = "json_error"
                    return {"level": "INVALID", "reason": "重评解析失败：JSON 错误", "score": None, "invalid": True}, audit

            if not isinstance(parsed, dict):
                audit["parse_status"] = "not_dict"
                return {"level": "INVALID", "reason": "重评解析失败：非字典", "score": None, "invalid": True}, audit

            raw_level = parsed.get("level", None)
            reason = parsed.get("reason", "")

            # 规范化 level
            sm = dim_cfg.get("score_mapping", {}) or {}
            if sm.get("type") == "levels":
                levels = sm.get("levels", []) or []
                allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
                allowed = [x for x in allowed if x]
                default_level = allowed[len(allowed) // 2] if allowed else "N/A"
            else:
                allowed = ["低", "中", "高"]
                default_level = "中"

            canonical_level = canonicalize_level(dim_id, raw_level, allowed, default_level)

            # 【G1】检测无效 level：不静默转换，直接标记 INVALID
            raw_level_str = str(raw_level).strip() if raw_level is not None else ""
            if raw_level_str and canonical_level == default_level and raw_level_str not in allowed:
                # 原始 level 被静默转换为 default_level，标记为 INVALID
                audit["parse_status"] = "invalid_level"
                return {
                    "level": "INVALID",
                    "reason": reason,
                    "score": None,
                    "invalid": True,
                    "original_level": raw_level_str,
                    "validation_errors": [f"重评后 level='{raw_level_str}' 仍不在允许列表 {allowed} 中"],
                }, audit

            # 再次校验：使用 schema_validate_dim 统一校验
            score_val = parsed.get("score")
            is_valid, new_validation_errors = schema_validate_dim(
                dim_id=dim_id,
                level=canonical_level,
                score=score_val,
                reason=reason,
                dim_cfg=dim_cfg,
                guessing_c=guessing_c,
            )
            audit["validation_errors"] = new_validation_errors

            # 判断是否仍为 INVALID
            if not is_valid:
                audit["parse_status"] = "still_invalid"
                return {
                    "level": "INVALID",
                    "reason": reason,
                    "score": None,
                    "invalid": True,
                    "original_level": canonical_level,  # 保留原始 level 供下轮重评
                    "validation_errors": new_validation_errors,  # 保留错误列表供下轮重评
                }, audit

            # 通过校验：使用原始 canonical_level（不做覆盖）
            score = map_level_to_score(dim_id, canonical_level, sm, guessing_c)
            audit["parse_status"] = "ok"

            return {
                "level": canonical_level,
                "reason": str(reason)[:400] if reason else "",
                "score": float(score),
            }, audit

        except Exception as e:
            audit["parse_status"] = "error"
            audit["error"] = str(e)
            return {"level": "INVALID", "reason": f"重评异常: {e}", "score": None, "invalid": True}, audit

    # -------------------------------------------------------------------------
    # 【2025-12 新增】Batched 重评：将同一模型的多个 INVALID 维度打包成一次调用
    # -------------------------------------------------------------------------

    def _build_batched_reevaluate_prompt(
            self,
            question: Any,
            invalid_dims_info: List[Dict[str, Any]],
            guessing_c: Optional[float],
    ) -> str:
        """
        构建 batched 重评 prompt。

        Args:
            question: 题目对象
            invalid_dims_info: 待重评维度信息列表，每项包含：
                - dim_cfg: 维度配置
                - previous_level: 上一轮 level
                - previous_reason: 上一轮 reason
                - validation_errors: 校验错误列表
            guessing_c: 蒙对概率

        Returns:
            batched 重评 prompt
        """
        # 提取题目信息
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

        # 构建维度列表（含错误反馈）
        dims_text_parts = []
        dim_ids_for_output = []

        for info in invalid_dims_info:
            dim_cfg = info["dim_cfg"]
            dim_id = dim_cfg["id"]
            previous_level = info.get("previous_level", "")
            previous_reason = info.get("previous_reason", "")
            validation_errors = info.get("validation_errors", [])

            dim_ids_for_output.append(dim_id)

            # 获取 allowed_levels
            sm = dim_cfg.get("score_mapping", {}) or {}
            if sm.get("type") == "levels":
                levels = sm.get("levels", []) or []
                allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
                allowed = [x for x in allowed if x]
                default = allowed[len(allowed) // 2] if allowed else "N/A"
            else:
                allowed = ["低", "中", "高"]
                default = "中"

            allowed_render = " | ".join([json.dumps(str(x), ensure_ascii=False) for x in allowed])
            prompt_eval = dim_cfg.get("prompt_eval", "") or ""

            # 格式化 errors 列表
            errors_text = ""
            for i, err in enumerate(validation_errors, 1):
                errors_text += f"    {i}. {err}\n"

            dim_text = f"""
┌─────────────────────────────────────────────────────────────────┐
│ 维度ID: {dim_id}
│ allowed_levels = {allowed_render}
│ level_default = {json.dumps(str(default), ensure_ascii=False)}
└─────────────────────────────────────────────────────────────────┘
{prompt_eval}

【上一轮你的输出】
- level: {previous_level}
- reason: {previous_reason}

【校验错误（必须修正）】
{errors_text if errors_text else "    (无具体错误信息)"}
"""
            dims_text_parts.append(dim_text)

        dims_combined = "\n".join(dims_text_parts)

        # 构建输出示例
        example_dims = {}
        for info in invalid_dims_info:
            dim_cfg = info["dim_cfg"]
            dim_id = dim_cfg["id"]
            sm = dim_cfg.get("score_mapping", {}) or {}
            if sm.get("type") == "levels":
                levels = sm.get("levels", []) or []
                allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
                default = allowed[len(allowed) // 2] if allowed else "N/A"
            else:
                default = "中"
            example_dims[dim_id] = {"level": default, "reason": "..."}

        json_example = json.dumps({"dimensions": example_dims}, ensure_ascii=False, indent=2)

        # 构建完整 prompt
        prompt = f"""你是一位资深的中文阅读理解题目质量评估专家。

╔══════════════════════════════════════════════════════════════════╗
║  【重评任务】你上次的部分评估结果未通过校验，请按要求修正后重新输出   ║
╚══════════════════════════════════════════════════════════════════╝

【原题目信息（完整上下文）】
- 材料（material_text）：
{material if material else "(未提供)"}

- 题干（stem）：
{stem if stem else "(未提供)"}

- 选项（options）：
{options_text if options_text else "(无选项)"}

════════════════════════════════════════════════════════════════════
【需要重评的维度列表】（共 {len(invalid_dims_info)} 个维度）
════════════════════════════════════════════════════════════════════
{dims_combined}

════════════════════════════════════════════════════════════════════
【输出要求】
════════════════════════════════════════════════════════════════════
1) 只输出一个 JSON 对象，格式如下：
   {{"dimensions": {{"dim_id1": {{"level": "...", "reason": "..."}}, ...}}}}

2) "dimensions" 必须包含上述全部 {len(invalid_dims_info)} 个维度 id：
   {dim_ids_for_output}

3) 每个维度的 level 必须从该维度的 allowed_levels 中选一个（EXACT MATCH）

4) reason 必须包含支持判断的关键锚点证据：
   - answer_uniqueness 必须报告 |C|=n
   - distractor_headroom 必须报告 N=n 或"最终计入N条"
   - guessing_lower_asymptote 必须报告 K=n（或提供"题干直给/模式线索"证据）

5) 禁止输出 score；禁止自造标签

⚠️ 请仔细阅读每个维度的【校验错误】，理解问题所在，然后按要求修正。

【输出示例（仅示例结构，禁止照抄理由）】
{json_example}

现在请按上述要求修正并重新输出 JSON：
"""
        return prompt

    def _reevaluate_batched_dimensions(
            self,
            client: LLMClient,
            question: Any,
            invalid_dims_info: List[Dict[str, Any]],
            guessing_c: Optional[float],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        对单个模型的多个 INVALID 维度进行 batched 重评。

        Args:
            client: LLM 客户端
            question: 题目对象
            invalid_dims_info: 待重评维度信息列表
            guessing_c: 蒙对概率

        Returns:
            (results, audit)
            - results: {dim_id: dim_result} 字典
            - audit: 重评审计信息
        """
        model_name = client.model_name
        dim_ids = [info["dim_cfg"]["id"] for info in invalid_dims_info]

        audit: Dict[str, Any] = {
            "model": model_name,
            "reevaluate": True,
            "batched": True,
            "dim_ids": dim_ids,
            "dim_results": {},
        }

        # 构建 dim_cfg_map 便于后续解析
        dim_cfg_map = {info["dim_cfg"]["id"]: info["dim_cfg"] for info in invalid_dims_info}

        prompt = self._build_batched_reevaluate_prompt(
            question=question,
            invalid_dims_info=invalid_dims_info,
            guessing_c=guessing_c,
        )

        results: Dict[str, Dict[str, Any]] = {}

        try:
            gen_kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": STAGE2_LLM_PARAMS.get("max_tokens", 4096),
                "metadata": {"agent": "AICentricEval", "mode": "batched_reevaluate", "dim_ids": dim_ids},
            }
            if not is_no_temperature_model(model_name or ""):
                gen_kwargs["temperature"] = 0.0

            resp = client.generate(**gen_kwargs)
            audit["response_len"] = len(resp) if resp else 0

            # 解析响应
            extracted = _extract_json_candidate(resp)
            if not extracted:
                audit["parse_status"] = "no_json"
                # 所有维度都标记为 INVALID
                for dim_id in dim_ids:
                    results[dim_id] = {
                        "level": "INVALID",
                        "reason": "重评解析失败：无 JSON",
                        "score": None,
                        "invalid": True,
                        "validation_errors": ["重评响应中未找到有效 JSON"],
                    }
                return results, audit

            try:
                parsed = json.loads(extracted)
            except Exception:
                try:
                    parsed = json.loads(_repair_common_json(extracted))
                except Exception:
                    audit["parse_status"] = "json_error"
                    for dim_id in dim_ids:
                        results[dim_id] = {
                            "level": "INVALID",
                            "reason": "重评解析失败：JSON 错误",
                            "score": None,
                            "invalid": True,
                            "validation_errors": ["重评响应 JSON 解析失败"],
                        }
                    return results, audit

            # 提取 dimensions
            if isinstance(parsed, dict) and "dimensions" in parsed:
                dims_data = parsed["dimensions"]
            elif isinstance(parsed, dict):
                dims_data = parsed
            else:
                audit["parse_status"] = "invalid_structure"
                for dim_id in dim_ids:
                    results[dim_id] = {
                        "level": "INVALID",
                        "reason": "重评解析失败：响应结构无效",
                        "score": None,
                        "invalid": True,
                        "validation_errors": ["重评响应结构无效"],
                    }
                return results, audit

            # 逐维度处理
            for dim_id in dim_ids:
                dim_cfg = dim_cfg_map[dim_id]
                dim_data = dims_data.get(dim_id, {})

                if not isinstance(dim_data, dict):
                    results[dim_id] = {
                        "level": "INVALID",
                        "reason": f"重评解析失败：维度 {dim_id} 数据无效",
                        "score": None,
                        "invalid": True,
                        "validation_errors": [f"维度 {dim_id} 的输出不是有效字典"],
                    }
                    audit["dim_results"][dim_id] = "invalid_data"
                    continue

                raw_level = dim_data.get("level")
                reason = dim_data.get("reason", "")

                # 规范化 level
                sm = dim_cfg.get("score_mapping", {}) or {}
                if sm.get("type") == "levels":
                    levels = sm.get("levels", []) or []
                    allowed = [str(x.get("label", "")).strip() for x in levels if "label" in x]
                    allowed = [x for x in allowed if x]
                    default_level = allowed[len(allowed) // 2] if allowed else "N/A"
                else:
                    allowed = ["低", "中", "高"]
                    default_level = "中"

                canonical_level = canonicalize_level(dim_id, raw_level, allowed, default_level)

                # 检测无效 level
                raw_level_str = str(raw_level).strip() if raw_level is not None else ""
                if raw_level_str and canonical_level == default_level and raw_level_str not in allowed:
                    results[dim_id] = {
                        "level": "INVALID",
                        "reason": reason,
                        "score": None,
                        "invalid": True,
                        "original_level": raw_level_str,
                        "validation_errors": [f"重评后 level='{raw_level_str}' 仍不在允许列表 {allowed} 中"],
                    }
                    audit["dim_results"][dim_id] = "invalid_level"
                    continue

                # 再次校验
                score_val = dim_data.get("score")
                is_valid, new_validation_errors = schema_validate_dim(
                    dim_id=dim_id,
                    level=canonical_level,
                    score=score_val,
                    reason=reason,
                    dim_cfg=dim_cfg,
                    guessing_c=guessing_c,
                )

                if not is_valid:
                    results[dim_id] = {
                        "level": "INVALID",
                        "reason": reason,
                        "score": None,
                        "invalid": True,
                        "original_level": canonical_level,
                        "validation_errors": new_validation_errors,
                    }
                    audit["dim_results"][dim_id] = "still_invalid"
                    continue

                # 通过校验
                score = map_level_to_score(dim_id, canonical_level, sm, guessing_c)
                results[dim_id] = {
                    "level": canonical_level,
                    "reason": str(reason)[:400] if reason else "",
                    "score": float(score),
                }
                audit["dim_results"][dim_id] = "ok"

            audit["parse_status"] = "ok"
            return results, audit

        except Exception as e:
            audit["parse_status"] = "error"
            audit["error"] = str(e)
            for dim_id in dim_ids:
                results[dim_id] = {
                    "level": "INVALID",
                    "reason": f"重评异常: {e}",
                    "score": None,
                    "invalid": True,
                    "validation_errors": [f"重评调用异常: {e}"],
                }
            return results, audit

    # ---------- 唯一对外入口 ----------
    def run(
            self,
            question_like: Union[Stage2Record, Stage2CoreInput, VerifiedQuestionSet, GeneratedQuestion, Any],
            guessing_c: Optional[float] = None,
            hit_dimensions: Optional[List[str]] = None,  # 为兼容 orchestrator；AI 侧忽略
    ) -> Dict[str, Any]:
        """
        question_like 统一入口：
        - Stage2Record -> 使用其 core_input
        - Stage2CoreInput -> 直接使用
        - VerifiedQuestionSet -> 使用 original_question
        - GeneratedQuestion -> 直接使用
        - 其他：尝试读取 .core_input / .original_question
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
            # 尝试做一点向后兼容
            core = getattr(question_like, "core_input", None) or getattr(question_like, "original_question", None)
            if core is None:
                raise ValueError(f"AICentricEval.run: unsupported input {type(question_like)}")

        return self._run_batched(core, guessing_c=guessing_c)

    # ---------- batched 主干 ----------
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
        # ✅ 【2025-12 更新】题型/选项适配：使用统一的 question_family 分类和维度适配矩阵
        # - objective（选择题）: 7维（权重均分 1/7）
        #   - 专属: answer_uniqueness, option_exclusivity_coverage, distractor_headroom, guessing_lower_asymptote
        #   - 共享: stem_quality, fairness_regional_gender, item_evidence_sufficiency
        # - subjective（简答题）: 4维（权重均分 1/4）
        #   - 专属: rubric_operational
        #   - 共享: stem_quality, fairness_regional_gender, item_evidence_sufficiency
        # 注意：共享维度在两种题型中有不同的 prompt_template，通过 applicable_to 字段区分
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
        # 选择 clients（orchestrator 注入优先）
        # -------------------------
        clients: List[LLMClient]
        if self.llm_clients:
            clients = list(self.llm_clients)
        else:
            api_type = getattr(self.llm_client, "api_type", "openai") if self.llm_client else "openai"
            clients = [LLMClient(api_type=api_type, model_name=m, verbose=False) for m in self.eval_models]

        model_names = [c.model_name for c in clients]
        weights = self._normalize_model_weights(model_names)

        # ✅ 【2025-12 更新】关键：prompt/parse 使用过滤后的维度配置 dims_cfg_eff
        # 选择题使用8维，简答题使用5维（具体见上方注释）
        prompt = build_ai_batched_prompt(question, dims_cfg_eff, guessing_c)

        model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        audits: List[Dict[str, Any]] = []
        reevaluate_audits: List[Dict[str, Any]] = []  # 【2025-12 新增】重评审计
        this_run_calls = 0

        # 构建 client 字典，便于重评时按 model_name 查找
        client_map: Dict[str, LLMClient] = {c.model_name: c for c in clients}

        # 构建 dim_cfg 字典，便于重评时按 dim_id 查找
        dim_cfg_map: Dict[str, Dict[str, Any]] = {d["id"]: d for d in dims_cfg_eff}

        # -------------------------
        # 【第一轮】初次评估
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

        # -------------------------
        # 【2025-12 新增】重评循环：最多 3 轮
        # 只针对"出现问题的模型 + 出现问题的维度"进行重评
        # -------------------------
        MAX_REEVALUATE_ROUNDS = 3

        for reevaluate_round in range(1, MAX_REEVALUATE_ROUNDS + 1):
            # 收集所有仍为 INVALID 的 (model_name, dim_id) 对
            invalid_pairs: List[Tuple[str, str, Dict[str, Any]]] = []

            for model_name, dims in model_results.items():
                for dim_id, dim_result in dims.items():
                    if dim_result.get("invalid") or dim_result.get("level") == "INVALID":
                        invalid_pairs.append((model_name, dim_id, dim_result))

            if not invalid_pairs:
                # 没有 INVALID 维度，退出重评循环
                logger.info(f"[AICentricEval] 重评检查 round={reevaluate_round}/{MAX_REEVALUATE_ROUNDS}：无 INVALID 维度，跳过重评")
                break

            # 【G2】按模型分组显示 INVALID 维度
            model_dim_groups: Dict[str, List[str]] = {}
            for model_name, dim_id, _ in invalid_pairs:
                if model_name not in model_dim_groups:
                    model_dim_groups[model_name] = []
                model_dim_groups[model_name].append(dim_id)

            for model_name, dim_ids in model_dim_groups.items():
                logger.info(
                    f"[AICentricEval] INVALID -> retry model={model_name} dims={dim_ids} round={reevaluate_round}/{MAX_REEVALUATE_ROUNDS}"
                )

            # 【2025-12 重构】按模型分组进行 batched 重评（每个模型只调用一次）
            for model_name, dim_ids in model_dim_groups.items():
                client = client_map.get(model_name)
                if not client:
                    logger.warning(f"[AICentricEval] 重评跳过：找不到 client for model={model_name}")
                    continue

                # 收集该模型的所有 INVALID 维度信息
                invalid_dims_info: List[Dict[str, Any]] = []
                for dim_id in dim_ids:
                    dim_cfg = dim_cfg_map.get(dim_id)
                    if not dim_cfg:
                        logger.warning(f"[AICentricEval] 重评跳过维度：找不到 dim_cfg for dim_id={dim_id}")
                        continue

                    old_result = model_results[model_name].get(dim_id, {})
                    previous_level = old_result.get("original_level") or old_result.get("level", "")
                    previous_reason = old_result.get("reason", "")

                    # 优先使用已有的 validation_errors
                    validation_errors = old_result.get("validation_errors")
                    if not validation_errors:
                        previous_score = old_result.get("score")
                        _, validation_errors = schema_validate_dim(
                            dim_id=dim_id,
                            level=previous_level if previous_level != "INVALID" else None,
                            score=previous_score,
                            reason=previous_reason,
                            dim_cfg=dim_cfg,
                            guessing_c=guessing_c,
                        )

                    if not validation_errors:
                        validation_errors = ["校验不通过，请严格按照评分规则重新评估"]

                    invalid_dims_info.append({
                        "dim_cfg": dim_cfg,
                        "previous_level": previous_level,
                        "previous_reason": previous_reason,
                        "validation_errors": validation_errors,
                    })

                if not invalid_dims_info:
                    continue

                # 调用 batched 重评（一次调用处理该模型的所有 INVALID 维度）
                new_results, reeval_audit = self._reevaluate_batched_dimensions(
                    client=client,
                    question=question,
                    invalid_dims_info=invalid_dims_info,
                    guessing_c=guessing_c,
                )
                this_run_calls += 1

                # 记录重评审计
                reeval_audit["round"] = reevaluate_round
                reevaluate_audits.append(reeval_audit)

                # 用新结果覆盖旧结果
                for dim_id, new_result in new_results.items():
                    model_results[model_name][dim_id] = new_result

                    if new_result.get("invalid") or new_result.get("level") == "INVALID":
                        logger.warning(
                            f"[AICentricEval] 第 {reevaluate_round} 轮重评 {model_name}/{dim_id}：仍为 INVALID"
                        )
                    else:
                        logger.info(
                            f"[AICentricEval] 第 {reevaluate_round} 轮重评 {model_name}/{dim_id}：成功，level={new_result.get('level')}"
                        )

        # 【2025-12 新增】统计最终仍为 INVALID 的维度
        final_invalid_count = 0
        for model_name, dims in model_results.items():
            for dim_id, dim_result in dims.items():
                if dim_result.get("invalid") or dim_result.get("level") == "INVALID":
                    final_invalid_count += 1

        if final_invalid_count > 0:
            logger.warning(
                f"[AICentricEval] 重评完成后仍有 {final_invalid_count} 个 INVALID (模型+维度) 对，将不参与聚合"
            )

        # ✅ 只对"适用维度(dims_cfg_eff)"聚合+计分（包含高方差检测）
        # 【2025-12 更新】聚合时跳过 INVALID 维度
        final_dimensions, high_variance_dims = self._aggregate(
            model_results,
            weights,
            dims_cfg=dims_cfg_eff,
            dim_weights=dim_weights_eff,
        )

        # 【2025-12 新增】打印高方差警告汇总
        if high_variance_dims:
            print(f"[AICentricEval] [WARN] 发现 {len(high_variance_dims)} 个高方差维度，需人工复核")

        # ✅ 把跳过的维度补回输出（N/A，weight=0，不影响总分）
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
        # 【2025-12 简化】不再判断阈值，只输出分数
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
                # ✅ 新增：维度适配审计信息（用于 evaluation_state.json 审计）
                "dimension_filter": dim_filter_result.to_dict(),
                # 【2025-12 新增】高方差维度列表
                "high_variance_dims": high_variance_dims,
                # 【2025-12 新增】重评审计信息
                "reevaluate_audits": reevaluate_audits,
                "final_invalid_count": final_invalid_count,
            },
            # ✅ 顶层审计字段（便于直接读取）
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
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        聚合多模型评估结果。

        【2025-12 新增】返回高方差维度列表用于日志记录。

        Returns:
            - aggregated: {dim_id: {"level", "reason", "score", "weight"}}
            - high_variance_dims: 高方差维度列表
        """
        if not model_results:
            return {}, []

        dims_cfg = self.dims_cfg if dims_cfg is None else dims_cfg
        dim_weights = self.dim_weights if dim_weights is None else dim_weights

        aggregated: Dict[str, Dict[str, Any]] = {}
        high_variance_dims: List[Dict[str, Any]] = []  # 高方差维度记录
        dim_ids = [d["id"] for d in (dims_cfg or []) if isinstance(d, dict) and d.get("id")]

        # 方差检测阈值：分数差距超过此值认为方差过大
        # 【2025-12 调整】从40提高到50，避免过于敏感
        # 50分差距意味着真正显著的模型分歧，需要人工复核
        SCORE_DIFF_THRESHOLD = 50.0

        for dim_id in dim_ids:
            scored: List[Tuple[float, float]] = []
            levels: List[str] = []
            reasons: List[str] = []

            # 【2025-12 新增】收集各模型的评分用于方差检测
            model_scores: Dict[str, float] = {}
            model_levels: Dict[str, str] = {}

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
                    # 记录各模型评分
                    model_scores[model_name] = score_float
                    model_levels[model_name] = level_val
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

            # 【2025-12 新增】检测方差过大
            if len(model_scores) >= 2:
                scores_list = list(model_scores.values())
                max_score = max(scores_list)
                min_score = min(scores_list)
                score_diff = max_score - min_score

                if score_diff >= SCORE_DIFF_THRESHOLD:
                    # 找出最高分和最低分的模型
                    max_model = max(model_scores.items(), key=lambda x: x[1])[0]
                    min_model = min(model_scores.items(), key=lambda x: x[1])[0]

                    high_variance_dims.append({
                        "dimension_id": dim_id,
                        "dimension_name": dim_id,  # AI评估中 dim_id 就是名称
                        "score_diff": score_diff,
                        "max_score": max_score,
                        "max_model": max_model,
                        "max_level": model_levels.get(max_model, ""),
                        "min_score": min_score,
                        "min_model": min_model,
                        "min_level": model_levels.get(min_model, ""),
                        "all_model_scores": model_scores,
                        "all_model_levels": model_levels,
                        "final_score": score,
                        "eval_type": "ai_centric",
                    })

                    # 打印警告日志
                    print(
                        f"[AICentricEval] [WARN] 高方差警告 - 维度'{dim_id}': "
                        f"{max_model}={max_score:.0f}分({model_levels.get(max_model)}) vs "
                        f"{min_model}={min_score:.0f}分({model_levels.get(min_model)}), "
                        f"差距={score_diff:.0f}分"
                    )

        return aggregated, high_variance_dims

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
# 便捷函数（保留一种 demo 调用方式）
# =============================================================================


def evaluate_question_ai_centric(
        question: GeneratedQuestion,
        guessing_c: Optional[float] = None,
        eval_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    evaluator = AICentricEval(eval_models=eval_models)
    return evaluator.run(question, guessing_c=guessing_c)


__all__ = ["AICentricEval", "evaluate_question_ai_centric"]
