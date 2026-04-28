# -*- coding: utf-8 -*-
"""
dimension_config.py - Unified High-frequency/Low-frequency Dimension Definitions

IMPORTANT: This file is the single source of truth for high-frequency dimension definitions!
All modules needing high-frequency dimensions should import from here, no hardcoding elsewhere.

Data Source: data/dimension_frequency_analysis.json
- Uses Kneedle algorithm (Elbow Method) to determine thresholds
- GK threshold: >15 hits = high frequency
- CS threshold: >25 hits = high frequency

Usage:
    from src.shared.dimension_config import (
        GK_HIGH_FREQ_DIMS,
        CS_HIGH_FREQ_DIMS,
        get_high_freq_dims,
        get_low_freq_dims,
    )
"""

import json
import logging
from pathlib import Path
from typing import Set, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Analysis file path
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
_ANALYSIS_FILE = _PROJECT_ROOT / "data" / "dimension_frequency_analysis.json"

# Cached analysis data
_analysis_cache: Optional[Dict[str, Any]] = None


def _load_analysis_data() -> Dict[str, Any]:
    """
    Load dimension frequency analysis data (with caching).

    Returns:
        Analysis data dictionary
    """
    global _analysis_cache

    if _analysis_cache is not None:
        return _analysis_cache

    if not _ANALYSIS_FILE.exists():
        logger.warning(f"Dimension frequency analysis file not found: {_ANALYSIS_FILE}, using defaults")
        return {}

    try:
        with open(_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            _analysis_cache = json.load(f)
        return _analysis_cache
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to load dimension frequency analysis file: {e}, using defaults")
        return {}


def get_high_freq_dims(domain: str) -> Set[str]:
    """
    Get high-frequency dimension set for specified domain.

    Args:
        domain: "gk" or "cs" (case-insensitive)

    Returns:
        High-frequency dimension code set, e.g. {"GK07", "GK08", "GK10", "GK13", "GK15"}
    """
    analysis = _load_analysis_data()
    key = f"{domain.lower()}_analysis"

    if key in analysis and "high_freq_dims" in analysis[key]:
        # Filter out UNMAPPED dimensions
        high_freq = [d for d in analysis[key]["high_freq_dims"] if not d.startswith("UNMAPPED:")]
        return set(high_freq)

    # Defaults (based on Kneedle algorithm encoding)
    if domain.lower() in ("gk", "gk_only"):
        return {"GK07", "GK08", "GK10", "GK13", "GK15"}
    elif domain.lower() in ("cs", "cs_only"):
        return {"CS03", "CS11", "CS13"}

    return set()


def get_low_freq_dims(domain: str) -> Set[str]:
    """
    Get low-frequency dimension set for specified domain.

    Args:
        domain: "gk" or "cs" (case-insensitive)

    Returns:
        Low-frequency dimension code set
    """
    analysis = _load_analysis_data()
    key = f"{domain.lower()}_analysis"

    if key in analysis and "low_freq_dims" in analysis[key]:
        # Filter out UNMAPPED dimensions
        low_freq = [d for d in analysis[key]["low_freq_dims"] if not d.startswith("UNMAPPED:")]
        return set(low_freq)

    return set()


def get_elbow_threshold(domain: str) -> int:
    """
    Get Kneedle threshold for specified domain.

    Args:
        domain: "gk" or "cs"

    Returns:
        Threshold (hit count), values above this are high-frequency
    """
    analysis = _load_analysis_data()
    key = f"{domain.lower()}_analysis"

    if key in analysis and "elbow_threshold" in analysis[key]:
        return analysis[key]["elbow_threshold"]

    # Defaults
    return 15 if domain.lower() == "gk" else 25


def get_dimension_frequencies(domain: str) -> Dict[str, int]:
    """
    Get frequency counts for all dimensions in specified domain.

    Args:
        domain: "gk" or "cs"

    Returns:
        Dimension to frequency mapping, e.g. {"GK07": 179, "GK15": 156, ...}
    """
    analysis = _load_analysis_data()
    key = f"{domain.lower()}_analysis"

    if key in analysis and "frequencies" in analysis[key]:
        return dict(analysis[key]["frequencies"])

    return {}


# ============================================================
# Pre-loaded Constants (for direct import usage)
# ============================================================

# GK High-frequency dimensions
# GK07=Wings-Foundational(179), GK15=Subject Literacy-Comprehension(156), GK10=Key Ability-Thinking Cognition(127)
# GK13=Subject Literacy-Information Acquisition(95), GK08=Wings-Comprehensive(60)
GK_HIGH_FREQ_DIMS: Set[str] = get_high_freq_dims("gk")

# CS High-frequency dimensions
# CS11=Chinese Ability-Viewpoint Material Relationship(124), CS03=Core Literacy-Thinking Development(97)
# CS13=Chinese Ability-Key Information Extraction(87)
CS_HIGH_FREQ_DIMS: Set[str] = get_high_freq_dims("cs")

# GK Low-frequency dimensions
GK_LOW_FREQ_DIMS: Set[str] = get_low_freq_dims("gk")

# CS Low-frequency dimensions
CS_LOW_FREQ_DIMS: Set[str] = get_low_freq_dims("cs")


# ============================================================
# Compatibility Functions (for legacy code)
# ============================================================

def get_high_freq_dims_by_mode(dim_mode: str) -> Set[str]:
    """
    Get high-frequency dimension set by dim_mode (legacy interface compatibility).

    Args:
        dim_mode: "gk", "gk_only", "cs", "cs_only"

    Returns:
        High-frequency dimension set
    """
    if dim_mode in ("gk", "gk_only"):
        return GK_HIGH_FREQ_DIMS
    elif dim_mode in ("cs", "cs_only"):
        return CS_HIGH_FREQ_DIMS
    return set()


def get_low_freq_dims_by_mode(dim_mode: str) -> Set[str]:
    """
    Get low-frequency dimension set by dim_mode (legacy interface compatibility).

    Args:
        dim_mode: "gk", "gk_only", "cs", "cs_only"

    Returns:
        Low-frequency dimension set
    """
    if dim_mode in ("gk", "gk_only"):
        return GK_LOW_FREQ_DIMS
    elif dim_mode in ("cs", "cs_only"):
        return CS_LOW_FREQ_DIMS
    return set()


# ============================================================
# Logging Output (on module load)
# ============================================================

logger.info(f"[dimension_config] High-freq dimension source: {_ANALYSIS_FILE}")
logger.info(f"[dimension_config] GK high-freq dimensions: {sorted(GK_HIGH_FREQ_DIMS)}")
logger.info(f"[dimension_config] CS high-freq dimensions: {sorted(CS_HIGH_FREQ_DIMS)}")


__all__ = [
    # Constants
    "GK_HIGH_FREQ_DIMS",
    "CS_HIGH_FREQ_DIMS",
    "GK_LOW_FREQ_DIMS",
    "CS_LOW_FREQ_DIMS",
    # Functions
    "get_high_freq_dims",
    "get_low_freq_dims",
    "get_high_freq_dims_by_mode",
    "get_low_freq_dims_by_mode",
    "get_elbow_threshold",
    "get_dimension_frequencies",
]
