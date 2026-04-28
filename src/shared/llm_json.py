# src/shared/llm_json.py
# LLM JSON parsing toolkit

"""
[Module Description]
Provides robust LLM JSON output parsing functionality, supporting:
1. JSON extraction from LLM responses (supports markdown code blocks, explanatory text before/after, etc.)
2. Lightweight JSON repair (trailing commas, common format issues)
3. Field salvage (when JSON parsing fails, attempts to extract key fields from text)

[Usage Scenarios]
- Stage2 AI-centric evaluation: parse score/level/reason
- Stage2 Pedagogical evaluation: parse hit_level/score/reason
- Agent5 Layer2 extreme gating: parse answer/confidence/evidence

[Design Principles]
- Recover useful information from non-standard output as much as possible
- Don't crash on failure, return reasonable default values
- Log parse status (ok/repaired_ok/salvaged/failed) for audit purposes

[2025-12 New Addition]
Used to solve the problem of numerous json.loads parsing failure warnings during Stage2 evaluation.
"""

import re
import json
from typing import Dict, Any, Optional, Tuple


# ============================================================================
# 1. JSON extraction: Allow extra text before/after JSON & code blocks
# ============================================================================

def extract_json_candidate(text: str) -> Optional[str]:
    """
    Extract JSON candidate string from LLM response text.

    [Supported Formats]
    1. ```json\n{...}\n``` code block (extracted with priority)
    2. ``` {...} ``` plain code block
    3. {...} object with explanatory text before/after

    [Implementation Notes]
    - Prioritize matching markdown code blocks
    - Otherwise find first { and track brace balance

    Args:
        text: LLM response text

    Returns:
        Extracted JSON string candidate, or None (if extraction fails)
    """
    if not text:
        return None

    t = text.strip()

    # 1) Prioritize extracting ```json ... ``` or ``` ... ``` code blocks
    # Allow explanatory text before code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        return m.group(1).strip()

    # 2) Otherwise, find first { and use brace balance tracking
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

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1].strip()

    # If no complete brace pair found, return from { to last }
    # (may be incomplete, but can attempt repair)
    end = t.rfind("}")
    if end > start:
        return t[start:end + 1].strip()

    return None


# ============================================================================
# 2. Lightweight JSON repair
# ============================================================================

def repair_common_json(s: str) -> str:
    """
    Lightweight JSON repair: fix common JSON format issues.

    [Supported Repairs]
    1. Remove trailing commas: { "a": 1, } -> { "a": 1 }
    2. Remove trailing commas in arrays: [1, 2, ] -> [1, 2]
    3. [2025-12 New] Fix unterminated strings (Unterminated string)
    4. [2025-12 New] Complete missing } and ]
    5. [2025-12 New] Remove illegal control characters

    Args:
        s: JSON string to repair

    Returns:
        Repaired JSON string
    """
    if not s:
        return s

    # 1. Remove illegal control characters (preserve newlines, tabs, carriage returns)
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

    # 2. Remove trailing commas (before } or ])
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 3. Attempt to fix unterminated strings
    s = _repair_unterminated_strings(s)

    # 4. Complete missing brackets
    s = _repair_missing_brackets(s)

    return s


def _repair_unterminated_strings(s: str) -> str:
    """
    [2025-12 New] Fix unterminated strings.

    When LLM output is truncated, strings may be cut in the middle, causing "Unterminated string" errors.
    This function attempts to fix by adding closing quotes.

    [Strategy]
    1. Detect if unterminated strings exist (odd number of quotes)
    2. If last character is not a quote and truncated in middle of string, attempt to close

    Args:
        s: JSON string

    Returns:
        Repaired string
    """
    if not s:
        return s

    # Count non-escaped double quotes
    quote_count = 0
    i = 0
    while i < len(s):
        if s[i] == '"':
            # Check if escaped
            num_backslashes = 0
            j = i - 1
            while j >= 0 and s[j] == '\\':
                num_backslashes += 1
                j -= 1
            # Even number of backslashes means quote is not escaped
            if num_backslashes % 2 == 0:
                quote_count += 1
        i += 1

    # If quote count is odd, there's an unterminated string
    if quote_count % 2 == 1:
        # Find last valid JSON structure point
        # Strategy: Add closing quote at appropriate position

        # First, try to find position of last unterminated string
        # Look backward from end for first position that can be closed
        s_stripped = s.rstrip()

        # If end is newline or content after whitespace, string may be truncated
        # Try adding " at end
        if not s_stripped.endswith('"'):
            # Check if inside string (check if there's opening quote after last :)
            last_colon = s_stripped.rfind(':')
            if last_colon != -1:
                after_colon = s_stripped[last_colon+1:].strip()
                if after_colon.startswith('"') and not after_colon.endswith('"'):
                    # Truncated inside string, add closing quote
                    s = s_stripped + '"'
                    print("[llm_json] [Repair] Added closing quote to fix truncated string")

    return s


def _repair_missing_brackets(s: str) -> str:
    """
    [2025-12 New] Complete missing brackets.

    When LLM output is truncated, closing } or ] may be missing.

    Args:
        s: JSON string

    Returns:
        Repaired string
    """
    if not s:
        return s

    # Count brackets (need to consider brackets inside strings don't count)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Complete missing brackets
    repairs = []
    if open_brackets > 0:
        repairs.append(']' * open_brackets)
    if open_braces > 0:
        repairs.append('}' * open_braces)

    if repairs:
        # Need to close in correct order: ] first, } second
        s = s.rstrip()
        suffix = ''.join(repairs)
        s = s + suffix
        print(f"[llm_json] [Repair] Completed missing brackets: {suffix}")

    return s


# ============================================================================
# 3. Field salvage: Extract key fields from text when JSON parsing fails
# ============================================================================

def salvage_ai_fields(text: str) -> Dict[str, Any]:
    """
    Salvage AI-centric evaluation required fields from non-JSON text.

    [Salvaged Fields]
    - score: 0-100 score
    - level: Evaluation level
    - reason: Evaluation reason (if found)

    [Matching Patterns]
    - "score": 80 / score: 80 / score: 80 / score: 80 (Chinese variants)
    - "level": "xxx" / level: xxx / level: xxx (Chinese variants)

    Args:
        text: LLM response text (may be non-JSON format)

    Returns:
        Dictionary containing score, level, reason, parse_recovered=True
    """
    result = {
        "score": 50.0,  # Default middle value
        "level": "Parse Salvaged",
        "reason": "",
        "parse_recovered": True,
    }

    if not text:
        return result

    # Salvage score
    # Support multiple formats:
    # - "score": 80 or "score":80
    # - score: 80 or score=80
    # - Chinese equivalents for score
    score_patterns = [
        r'(?:"score"|score|分数|得分|评分)\s*[:=：]\s*(\d+(?:\.\d+)?)',
        r'(\d{2,3})\s*分',  # "80 points" or "100 points" (Chinese)
    ]

    for pattern in score_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                score = float(m.group(1))
                # Limit to 0-100 range
                score = max(0.0, min(100.0, score))
                result["score"] = score
                break
            except (ValueError, TypeError):
                pass

    # Salvage level
    # Support multiple formats:
    # - "level": "xxx" or "level":"xxx"
    # - level: xxx or level: xxx (Chinese variants)
    level_patterns = [
        r'(?:"level"|level|等级|评级)\s*[:=：]\s*"([^"]+)"',
        r'(?:"level"|level|等级|评级)\s*[:=：]\s*([^\s,}"\[\]]+)',
    ]

    for pattern in level_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result["level"] = m.group(1).strip()
            break

    # Salvage reason
    # Support multiple formats:
    # - "reason": "..." or Chinese equivalents
    reason_patterns = [
        r'(?:"reason"|"reasoning"|reason|理由|说明)\s*[:=：]\s*"([^"]+)"',
    ]

    for pattern in reason_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result["reason"] = m.group(1).strip()
            break

    # If no reason found, use first 100 characters of response
    if not result["reason"]:
        result["reason"] = f"[Salvaged from non-standard output] {text[:100]}..."

        return result


def salvage_pedagogical_fields(text: str) -> Dict[str, Any]:
    """
    Salvage Pedagogical evaluation required fields from non-JSON text.

    [Salvaged Fields]
    - hit_level: Hit level (High Match/Basic Match/Partial Match/Clear Deviation)
    - score: Corresponding score (mapped from hit_level or directly extracted)
    - reason: Evaluation reason

    [Hit Level Mapping]
    - High Match: 100
    - Basic Match: 80
    - Partial Match: 60
    - Clear Deviation: 30

    Args:
        text: LLM response text (may be non-JSON format)

    Returns:
        Dictionary containing hit_level, score, reason, parse_recovered=True
    """
    HIT_LEVEL_SCORES = {
        "高度命中": 100.0,
        "基本命中": 80.0,
        "部分相关": 60.0,
        "明显偏离": 30.0,
    }

    result = {
        "hit_level": "部分相关",  # Default value (Partial Match)
        "score": 60.0,
        "reason": "",
        "parse_recovered": True,
    }

    if not text:
        return result

    # Salvage hit_level
    # Directly match four hit levels in text
    for level, score in HIT_LEVEL_SCORES.items():
        if level in text:
            result["hit_level"] = level
            result["score"] = score
            break

    # If no hit level matched, try extracting from JSON format
    m = re.search(r'(?:"hit_level"|hit_level|命中等级)\s*[:=：]\s*"?([^"}\s,]+)"?', text, re.IGNORECASE)
    if m:
        level_text = m.group(1).strip()
        if level_text in HIT_LEVEL_SCORES:
            result["hit_level"] = level_text
            result["score"] = HIT_LEVEL_SCORES[level_text]

    # If score can be found directly, use directly found score
    score_patterns = [
        r'(?:"score"|score|分数|得分)\s*[:=：]\s*(\d+(?:\.\d+)?)',
    ]

    for pattern in score_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                score = float(m.group(1))
                score = max(0.0, min(100.0, score))
                result["score"] = score
                break
            except (ValueError, TypeError):
                pass

    # Salvage reason
    reason_patterns = [
        r'(?:"reason"|"reasoning"|reason|理由|说明)\s*[:=：]\s*"([^"]+)"',
    ]

    for pattern in reason_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result["reason"] = m.group(1).strip()
            break

    # If no reason found, use first 100 characters of response
    if not result["reason"]:
        result["reason"] = f"[Salvaged from non-standard output] {text[:100]}..."

    return result


# ============================================================================
# 4. Unified parsing entry point: Three-step parsing method
# ============================================================================

def parse_llm_json_response(
    response: str,
    salvage_func: callable = None,
    model_name: str = "unknown",
    dimension_id: str = "unknown",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Three-step method to parse LLM JSON response.

    [Parsing Flow]
    1. Extract JSON candidate (extract_json_candidate)
    2. Lightweight repair + json.loads (repair_common_json)
    3. If failed, call salvage_func to salvage fields

    [Return Values]
    - data: Parsed data dictionary
    - audit: Audit information, including:
      - raw_head: First 200 characters of raw response
      - json_candidate_head: First 200 characters of extracted JSON candidate
      - parse_status: "ok" | "repaired_ok" | "salvaged" | "failed"
      - parse_error: Parse error message (if any)

    Args:
        response: LLM response text
        salvage_func: Field salvage function (optional, e.g. salvage_ai_fields)
        model_name: Model name (for logging)
        dimension_id: Dimension ID (for logging)

    Returns:
        (data, audit) tuple
    """
    audit = {
        "raw_head": (response[:200] if response else "")[:200],
        "json_candidate_head": "",
        "parse_status": "failed",
        "parse_error": None,
    }

    if not response:
        audit["parse_error"] = "Empty response"
        if salvage_func:
            data = salvage_func("")
            audit["parse_status"] = "salvaged"
            return data, audit
        return {}, audit

    # Step 1: Extract JSON candidate
    candidate = extract_json_candidate(response)
    audit["json_candidate_head"] = (candidate[:200] if candidate else "")[:200]

    if not candidate:
        # Cannot extract JSON, attempt salvage
        print(f"[WARN][Stage2][{model_name}][{dimension_id}] json extract failed | head={response[:200]!r}")
        if salvage_func:
            data = salvage_func(response)
            audit["parse_status"] = "salvaged"
            return data, audit
        audit["parse_error"] = "No JSON candidate found"
        return {}, audit

    # Step 2: Attempt direct parsing
    try:
        data = json.loads(candidate)
        audit["parse_status"] = "ok"
        return data, audit
    except json.JSONDecodeError as e1:
        # Step 2.5: Attempt parsing after repair
        repaired = repair_common_json(candidate)
        try:
            data = json.loads(repaired)
            audit["parse_status"] = "repaired_ok"
            return data, audit
        except json.JSONDecodeError as e2:
            # Step 3: Salvage
            audit["parse_error"] = str(e2)
            print(f"[WARN][Stage2][{model_name}][{dimension_id}] json parse failed: {e2} | head={response[:200]!r}")

            if salvage_func:
                data = salvage_func(response)
                audit["parse_status"] = "salvaged"
                return data, audit

            return {}, audit


# ============================================================================
# 5. Convenience functions: Specifically for AI-centric and Pedagogical evaluation
# ============================================================================

def parse_ai_centric_response(
    response: str,
    dimension_id: str,
    dimension_name: str,
    model_name: str = "unknown",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function for parsing AI-centric evaluation response.

    [Expected Fields]
    - score: 0-100 score
    - level: Evaluation level
    - reason: Evaluation reason

    Args:
        response: LLM response text
        dimension_id: Dimension ID
        dimension_name: Dimension name
        model_name: Model name

    Returns:
        (data, audit) tuple
    """
    data, audit = parse_llm_json_response(
        response=response,
        salvage_func=salvage_ai_fields,
        model_name=model_name,
        dimension_id=dimension_id,
    )

    # Ensure necessary fields exist and are reasonable
    if "score" not in data:
        data["score"] = 50.0
    else:
        try:
            data["score"] = float(data["score"])
            data["score"] = max(0.0, min(100.0, data["score"]))
        except (ValueError, TypeError):
            data["score"] = 50.0

    if "level" not in data:
        data["level"] = "Unknown"

    if "reason" not in data:
        data["reason"] = "No explanation"

    return data, audit


def parse_pedagogical_response(
    response: str,
    dimension_id: str,
    dimension_name: str,
    model_name: str = "unknown",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function for parsing Pedagogical evaluation response.

    [Expected Fields]
    - hit_level: Hit level (High Match/Basic Match/Partial Match/Clear Deviation)
    - score: Corresponding score
    - reason: Evaluation reason

    Args:
        response: LLM response text
        dimension_id: Dimension ID
        dimension_name: Dimension name
        model_name: Model name

    Returns:
        (data, audit) tuple
    """
    HIT_LEVEL_SCORES = {
        "高度命中": 100.0,
        "基本命中": 80.0,
        "部分相关": 60.0,
        "明显偏离": 30.0,
    }

    data, audit = parse_llm_json_response(
        response=response,
        salvage_func=salvage_pedagogical_fields,
        model_name=model_name,
        dimension_id=dimension_id,
    )

    # Ensure hit_level exists
    if "hit_level" not in data:
        data["hit_level"] = "部分相关"

    # If hit_level not in predefined list, use default value
    if data["hit_level"] not in HIT_LEVEL_SCORES:
        data["hit_level"] = "部分相关"

    # Ensure score exists and is reasonable
    if "score" not in data:
        data["score"] = HIT_LEVEL_SCORES.get(data["hit_level"], 60.0)
    else:
        try:
            data["score"] = float(data["score"])
            data["score"] = max(0.0, min(100.0, data["score"]))
        except (ValueError, TypeError):
            data["score"] = HIT_LEVEL_SCORES.get(data["hit_level"], 60.0)

    # Compatible with reasoning field (some responses may use reasoning instead of reason)
    # Prioritize using reasoning field (if exists)
    if "reasoning" in data:
        data["reason"] = data["reasoning"]
    elif "reason" not in data:
        data["reason"] = "No explanation"

    return data, audit


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "extract_json_candidate",
    "repair_common_json",
    "salvage_ai_fields",
    "salvage_pedagogical_fields",
    "parse_llm_json_response",
    "parse_ai_centric_response",
    "parse_pedagogical_response",
]
