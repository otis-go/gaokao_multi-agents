# -*- coding: utf-8 -*-
"""
output_analysis configuration file

Contains high-frequency dimensions, path configuration, metric descriptions, and unified settings

GK/CS independent scope principle:
- GK and CS are two independent dimension systems, no cross-system numerical comparison
- All output files must have domain identifier (_gk or _cs)
"""

from pathlib import Path
from typing import Set, Dict, Any, List
import re
import sys

# Ensure src modules can be imported
_OUTPUT_ANALYSIS_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _OUTPUT_ANALYSIS_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================
# Path Configuration
# ============================================================

# output_analysis module root directory
MODULE_ROOT = Path(__file__).resolve().parent

# Project root directory
PROJECT_ROOT = MODULE_ROOT.parent

# Module internal data directory
DATA_DIR = MODULE_ROOT / "data"
BASELINE_DIR = MODULE_ROOT / "baselines"

# Default paths (all presentation-related files are in output_analysis/)
DEFAULT_END_PLAN_PATH = DATA_DIR / "end_plan.xlsx"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = MODULE_ROOT / "results"

# Baseline experiment directory
NONEDIM_BASELINE_DIR = BASELINE_DIR / "nonedim_deepseek_tem1"

# ============================================================
# Domain Configuration (GK/CS independent systems)
# ============================================================

# List of supported domains
DOMAINS: List[str] = ["GK", "CS"]

# ============================================================
# High-frequency dimension configuration (imported from unified location)
# ============================================================

# Import high-frequency dimension definitions from src/shared/dimension_config.py
# This is the sole source of high-frequency dimensions, all modules should get them from here
from src.shared.dimension_config import (
    GK_HIGH_FREQ_DIMS,
    CS_HIGH_FREQ_DIMS,
    GK_LOW_FREQ_DIMS,
    CS_LOW_FREQ_DIMS,
    get_high_freq_dims as _get_high_freq_dims_impl,
    get_low_freq_dims,
    get_elbow_threshold,
)


def get_high_freq_dims(domain: str) -> Set[str]:
    """
    Get high-frequency dimension set based on domain

    Args:
        domain: "GK" or "CS"

    Returns:
        Set of high-frequency dimensions

    Note:
        This function delegates to src.shared.dimension_config.get_high_freq_dims
        High-frequency dimension definition source: data/dimension_frequency_analysis.json
    """
    return _get_high_freq_dims_impl(domain)


# ============================================================
# Experiment Identification Configuration
# ============================================================

def parse_experiment_id(exp_id: str) -> Dict[str, str]:
    """
    Parse experiment_id to extract domain, prompt_level and other information

    Args:
        exp_id: Experiment ID, e.g. "EXP_FULL_gk_C_20260107_061725"

    Returns:
        {
            "domain": "GK" or "CS",
            "prompt_level": "A"/"B"/"C"/"baseline",
            "is_baseline": True/False,
            "is_permutation": True/False,
            "is_hardmix": True/False,
            "is_low_freq": True/False,
            "low_freq_k": 1/3/5/None
        }
    """
    exp_id_upper = exp_id.upper()

    # Determine domain
    if "_GK_" in exp_id_upper or exp_id_upper.startswith("GK"):
        domain = "GK"
    elif "_CS_" in exp_id_upper or exp_id_upper.startswith("CS"):
        domain = "CS"
    else:
        domain = "UNKNOWN"

    # Determine prompt_level
    if "NODIM" in exp_id_upper or "ABLATION" in exp_id_upper:
        prompt_level = "baseline"
        is_baseline = True
    elif "_A_" in exp_id_upper:
        prompt_level = "A"
        is_baseline = False
    elif "_B_" in exp_id_upper:
        prompt_level = "B"
        is_baseline = False
    elif "_C_" in exp_id_upper:
        prompt_level = "C"
        is_baseline = False
    else:
        prompt_level = "UNKNOWN"
        is_baseline = False

    # Determine permutation (regular negative control)
    is_permutation = "RANDOM" in exp_id_upper or "PERM" in exp_id_upper or "RANDDIM" in exp_id_upper

    # Determine hardmix (hard negative control)
    is_hardmix = "HARDMIX" in exp_id_upper

    # Determine low_freq
    is_low_freq = "LOW_FREQ" in exp_id_upper
    low_freq_k = None
    if is_low_freq:
        if "K1" in exp_id_upper:
            low_freq_k = 1
        elif "K3" in exp_id_upper:
            low_freq_k = 3
        elif "K5" in exp_id_upper:
            low_freq_k = 5

    return {
        "domain": domain,
        "prompt_level": prompt_level,
        "is_baseline": is_baseline,
        "is_permutation": is_permutation,
        "is_hardmix": is_hardmix,
        "is_low_freq": is_low_freq,
        "low_freq_k": low_freq_k
    }


# ============================================================
# Bootstrap and Statistical Testing Configuration
# ============================================================

BOOTSTRAP_N_ITERATIONS = 10000
BOOTSTRAP_CI_LEVEL = 0.95
MIN_SAMPLE_SIZE_FOR_WILCOXON = 5
