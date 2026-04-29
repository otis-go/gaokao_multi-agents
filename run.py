#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run.py
# Unified CLI Entry Point - Data-driven Question Type Auto-detection

# =============================================================================
# Windows UTF-8 Encoding Auto-configuration (no need to manually set PYTHONUTF8=1)
# =============================================================================
import sys
import os

# Set Python UTF-8 mode (Windows compatible)
if sys.platform == "win32":
    # Set environment variable (affects subprocesses)
    os.environ.setdefault("PYTHONUTF8", "1")
    # Reconfigure standard output streams to UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # May fail in some environments, handle silently

"""
Script Description:
This is the refactored unified command-line entry file.

[2025-12 Architecture Refactoring: LEGO-style Decoupling]
- CLI only handles: determine unit_id list -> call two orchestrators -> collect results
- CLI does no DataLoader pre-validation (question type/dimension/material pre-reading)
- Schema is the only data flow contract
- Stage1 and Stage2 orchestrate independently, CLI only chains the two stages
- Statistics only depend on stable fields in evaluation_state.json

Usage:

1. Single question mode (must specify unit_id):
   python run.py --run-mode single --unit-id 10 --dim-mode gk --prompt-level C

2. Full mode (run all 181 questions):
   python run.py --run-mode full --dim-mode gk --prompt-level C

3. Subset sampling mode (40 or 60 questions proportional stratified):
   python run.py --run-mode full --subset-size 40 --subset-strategy proportional_stratified --subset-seed 42

4. Load subset from file:
   python run.py --run-mode full --subset-file outputs/exp1/subset_unit_ids.json

5. Use round-id to organize experiments in the same round (recommended for subset40/60 comparison):
   # Step 1: Run subset40
   python run.py --run-mode full --subset-size 40 --round-id ROUND_20251208_A_deepseek --dim-mode gk --prompt-level C

   # Step 2: Run subset60 (same round-id)
   python run.py --run-mode full --subset-size 60 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

   # Output structure:
   # outputs/ROUND_20251207_A/
   #   subset40_stratified_seed42_gk_C_20251207_120000/
   #     summary.json, subset_unit_ids.json, subset_stats.json, stage2/, llm_logs/
   #   subset60_stratified_seed42_gk_C_20251207_121500/
   #     summary.json, etc.
   #   round_manifest.jsonl

6. View help:
   python run.py --help

Parameter Description:
  --run-mode, -r        Run mode: single / full
  --unit-id, -u         Required for single mode: specify unit_id
  --dim-mode, -d        Dimension mode: gk (Gaokao) / cs (Curriculum Standard)
  --prompt-level, -p    Prompt level: A / B / C
  --generator-model     Generator model name (overrides default)
  --experiment-id, -e   Custom experiment ID (optional)
  --output-dir, -o      Custom output dir (optional; represents output_root when round-id enabled)
  --round-id            Experiment round ID (optional)
  --verbose, -v         Show verbose logs

  Subset Sampling Parameters (only for full mode):
  --subset-size         Subset size (40/60), default full
  --subset-strategy     Sampling strategy: proportional_stratified (recommended) / stratified / random
  --subset-seed         Random seed, default: 42
  --subset-file         Load unit_ids from file (mutually exclusive with --subset-size)

Output:
- Generated content per question (with data-driven question type)
- AI-centric evaluation results (generation-family-decoupled evaluator ensemble)
- Pedagogical evaluation results
- summary.json (full mode)
- subset_unit_ids.json / subset_stats.json (subset mode)
- round_manifest.jsonl (records each run index when round-id enabled)

Core Changes:
1. Removed manual question type: auto-determined from data
2. Unified run modes: single or full
3. Unified LLM control via LLMRouter
4. Stage 2 dual evaluation shares the same decoupled evaluator group
5. [2025-12] Support stratified subset sampling
6. [2025-12] Support round-id for organizing multiple runs
7. [2025-12] CLI does no DataLoader pre-validation
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

# [2025-12 Update] API config centralized to src/shared/api_config.py
# No need to set environment variables, just modify api_config.py
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(usecwd=True), override=False)

# Import config
from src.shared.config import (
    ExperimentConfig,
    create_default_config,
    STAGE1_LLM_PARAMS,
    STAGE2_EVAL_MODELS,
)

# [2026-01 New] Import API config (for recording LLM config in summary.json)
from src.shared import api_config

# Import LLM logger
from src.shared.llm_logger import init_global_logger

# [2025-12 New] Import LLM retry audit
from src.shared.llm_interface import get_retry_audit, clear_retry_audit

# [2026-01 New] Import bottom_20_metrics calculation function
from update_bottom20_metrics import calculate_bottom_20_metrics

# [2025-12 New] Import MD report generator
from src.shared.report_generator import generate_reports_from_summary


# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)
def _print_runtime_files():
    """[2025-12 Refactoring] Print actual file paths used at runtime, confirm correct files are being modified"""
    print(f"[cli] file={__file__}")
    try:
        from src.generation.pipeline import generation_orchestrator
        print(f"[cli] generation_orchestrator file={generation_orchestrator.__file__}")
    except Exception as e:
        print(f"[cli] generation_orchestrator import error: {e}")
    try:
        from src.evaluation import evaluation_orchestrator
        print(f"[cli] evaluation_orchestrator file={evaluation_orchestrator.__file__}")
    except Exception as e:
        print(f"[cli] evaluation_orchestrator import error: {e}")


# ============================================================================
# Utility Functions: Path Sanitization, run-folder Naming, Manifest Writing
# ============================================================================
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=8)

def sanitize_path_component(s: str) -> str:
    """
    Convert string to path-safe form:
    - Remove or replace unsafe characters (spaces, slashes, colons, backslashes, etc.)
    - Keep letters, numbers, underscores, hyphens, dots
    """
    if not s:
        return ""
    # Replace common separators with underscore
    s = re.sub(r'[\s/\\:*?"<>|]+', '_', s)
    # Remove other unsafe characters
    s = re.sub(r'[^\w\-.]', '', s)
    # Strip leading/trailing underscores
    s = s.strip('_')
    return s or "unknown"
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)


def build_run_folder_name(
    run_mode: str,
    unit_id: Optional[str] = None,
    subset_size: Optional[int] = None,
    subset_file: Optional[str] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    dim_mode: str = "gk",
    prompt_level: str = "C",
    generator_model: Optional[str] = None,
    timestamp: str = None,
    stage1_skip: str = "none",
    skip_qtype_prompt: bool = False,
    use_random_dims: bool = False,
    use_hardmix_dims: bool = False,
    low_freq_random: bool = False,
    low_freq_ablation: bool = False,
) -> str:
    """
    Generate run-folder name, ensuring uniqueness and readability.

    Naming rules:
    - single mode: single_U{unit_id}_{dim-mode}_{prompt-level}_{timestamp}
    - full (all questions): full_all_{dim-mode}_{prompt-level}_{timestamp}
    - subset-size: subset{size}_{strategy}_seed{seed}_{dim-mode}_{prompt-level}_{timestamp}
    - subset-file: subsetfile_{file_stem}_{dim-mode}_{prompt-level}_{timestamp}

    [2025-12 Added] Ablation experiment identifier
    - If stage1_skip != "none", append _SKIP{N} (e.g., _SKIP2, _SKIP3, _SKIP5)

    Optionally append generator_model (if different from default)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = []

    if run_mode == "single":
        parts.append("single")
        parts.append(f"U{sanitize_path_component(str(unit_id))}")
    elif run_mode == "baseline":
        # Baseline evaluation mode: may run on single-unit / subset / full real questions; no prompt_level
        parts.append("baseline")
        if unit_id:
            parts.append(f"U{sanitize_path_component(str(unit_id))}")
        elif subset_file:
            parts.append("subsetfile")
            file_stem = Path(subset_file).stem
            parts.append(sanitize_path_component(file_stem))
        elif subset_size:
            parts.append(f"subset{subset_size}")
            parts.append(sanitize_path_component(subset_strategy))
            parts.append(f"seed{subset_seed}")
        else:
            parts.append("all")
        parts.append(sanitize_path_component(dim_mode))
        parts.append(timestamp)
        return "_".join(parts)  # Return early, no prompt_level added
    # [2026-01 Deprecated] Year-based experiment division no longer used
    # elif run_mode == "baseline-recent":
    #     # [2025-12 Added] Recent six years real exam baseline evaluation mode (2020-2025)
    #     parts.append("baseline")
    #     parts.append("recent")  # Distinguish from all
    #     parts.append(sanitize_path_component(dim_mode))
    #     parts.append(timestamp)
    #     return "_".join(parts)  # Return early, no prompt_level added
    elif run_mode == "ablation-nodim":
        # [2025-12 Added] Stage1 ablation experiment mode
        parts.append("ABLATION_NODIM")
        if subset_size:
            parts.append(f"subset{subset_size}")
        else:
            parts.append("all")
        parts.append(timestamp)
        return "_".join(parts)  # Return early, no dim_mode and prompt_level added
    elif subset_file:
        parts.append("subsetfile")
        file_stem = Path(subset_file).stem
        parts.append(sanitize_path_component(file_stem))
    elif subset_size:
        parts.append(f"subset{subset_size}")
        parts.append(sanitize_path_component(subset_strategy))
        parts.append(f"seed{subset_seed}")
    else:
        parts.append("full")
        parts.append("all")

    parts.append(sanitize_path_component(dim_mode))
    parts.append(sanitize_path_component(prompt_level))

    # [2025-12-26 Improved] Ablation experiment identifier - clearer naming
    if stage1_skip and stage1_skip != "none":
        # From "agent2" -> "SKIP_AGENT2", "agent4" -> "SKIP_AGENT4"
        skip_tag = stage1_skip.upper().replace("AGENT", "_AGENT")
        if not skip_tag.startswith("SKIP"):
            skip_tag = f"SKIP{skip_tag}"
        parts.append(skip_tag)

    # [2025-12-28 Added] Question type dimension ablation identifier
    if skip_qtype_prompt:
        parts.append("NOQTYPE")

    # [2026-01 Added] Random dimension ablation identifier
    if use_random_dims:
        parts.append("RANDDIM")

    # [2026-01-14 Added] Hard negative control identifier
    if use_hardmix_dims:
        parts.append("HARDMIX")

    # [2026-01-05 Added] Low-frequency dimension experiment identifier
    if low_freq_random:
        parts.append("LOWFREQ_RAND")
    if low_freq_ablation:
        parts.append("LOWFREQ_ABL")

    # Optional: append generator model name (shortened form)
    if generator_model:
        model_short = sanitize_path_component(generator_model)
        if len(model_short) > 20:
            model_short = model_short[:20]
        # Not appended to avoid long paths; uncomment if needed
        # parts.append(model_short)

    parts.append(timestamp)

    # DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)
    return "_".join(parts)


def append_round_manifest(
    round_root: Path,
    record: Dict[str, Any],
) -> None:
    """
    Append a record to round_root/round_manifest.jsonl.
    Create the file if it doesn't exist.

    record should contain:
    - round_id
    - run_folder
    - run_id (experiment_id)
    - subset_size / subset_file
    - dim_mode
    - prompt_level
    - generator_model
    - summary_path (relative path)
    - timestamp
    """
    manifest_path = round_root / "round_manifest.jsonl"
    round_root.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Round Manifest] Record appended to: {manifest_path}")


# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)
# ============================================================================
# Command-line argument parsing
# ============================================================================

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Experiment CLI Entry Point - Data-driven Question Type Auto-detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question mode (must specify unit_id)
  python run.py --run-mode single --unit-id 1 --dim-mode gk --prompt-level C

  # Full mode (run all questions)
  python run.py --run-mode full --dim-mode gk --prompt-level C

  # Subset sampling mode (40 or 60 questions proportional stratified)
  python run.py --run-mode full --subset-size 40 --subset-strategy proportional_stratified --subset-seed 42

  # Load subset from file
  python run.py --run-mode full --subset-file outputs/exp1/subset_unit_ids.json

  # Use round-id to organize experiments in the same round (recommended for subset40/60 comparison)
  # Step 1: Run subset40
  python run.py --run-mode full --subset-size 40 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

  # Step 2: Run subset60 (same round-id)
  python run.py --run-mode full --subset-size 60 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

  # Output structure example (when round-id enabled):
  # outputs/ROUND_20251207_A/
  # ├── subset40_stratified_seed42_gk_C_20251207_120000/
  # │   ├── summary.json
  # │   └── ...
  # ├── subset60_stratified_seed42_gk_C_20251207_121500/
  # │   └── ...
  # └── round_manifest.jsonl

Note: When --round-id enabled, --output-dir represents output_root (default: outputs),
      actual output directory is output_root/round-id/run-folder/
        """
    )

    # ========== Core parameters ==========
    parser.add_argument(
        "--run-mode", "-r",
        type=str,
        # [2026-01 Deprecated] Removed baseline-recent, year-based experiment division no longer used
        choices=["single", "full", "baseline", "extract", "stage1-only", "stage2-only", "ablation-nodim"],
        required=True,
        help="Run mode: single (single question) / full (all or subset) / baseline (real-question Stage2 direct evaluation; use --baseline-dim-source to choose random vs gold dimensions) / extract (extract questions) / stage1-only (Stage1 only) / stage2-only (Stage2 only) / ablation-nodim (ablation experiment: no dimension prompts, direct generation)"
    )

    parser.add_argument(
        "--unit-id", "-u",
        type=str,
        default=None,
        help="Single mode required: specify the unit_id to run"
    )

    parser.add_argument(
        "--dim-mode", "-d",
        type=str,
        choices=["gk", "cs"],
        default="gk",
        help="Dimension mode: gk (Gaokao dimension) / cs (Curriculum Standard dimension), default: gk"
    )

    parser.add_argument(
        "--prompt-level", "-p",
        type=str,
        choices=["A", "B", "C"],
        default="C",
        help="Prompt level: A / B / C, default: C"
    )

    # ========== Model configuration ==========
    default_model = STAGE1_LLM_PARAMS.get("model", "deepseek-v3.2-exp")
    parser.add_argument(
        "--generator-model",
        type=str,
        default=default_model,
        help=f"Generator model name, default: {default_model}"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Stage1 generation temperature (override default 1.0). Options: 0.5/1.0/1.5/2.0"
    )

    # ========== Optional parameters ==========
    parser.add_argument(
        "--experiment-id", "-e",
        type=str,
        default=None,
        help="Custom experiment ID (optional, auto-generated by default)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Custom output directory (optional; represents output_root when round-id enabled, default: outputs)"
    )

    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Experiment round ID (optional; used to organize subset40/60 etc. under the same parent directory, e.g., ROUND_20251207_A)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose logs"
    )

    # ========== Subset sampling parameters (only for full mode) ==========
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        choices=[40, 60],
        help="Subset size (40 / 60), if not set use full set"
    )

    parser.add_argument(
        "--subset-strategy",
        type=str,
        choices=["proportional_stratified", "stratified", "random"],
        default="proportional_stratified",
        help="Sampling strategy: proportional_stratified (proportional stratified, recommended) / stratified (coverage priority) / random (random), default: proportional_stratified"
    )

    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed, default: 42"
    )

    parser.add_argument(
        "--subset-file",
        type=str,
        default=None,
        help="Load unit_id list from file (mutually exclusive with --subset-size)"
    )

    # ========== Stage1 ablation experiment parameters ==========
    # ========== [2026-01 Added] Checkpoint resume parameters ==========
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="[Checkpoint resume] Specify existing experiment directory to continue from specified position and overwrite subsequent results. "
             "Must be used with --start-from."
    )

    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="[Checkpoint resume] Start from specified unit_id (inclusive). "
             "For example, --start-from 61 means run units 61-181, skip 1-60. "
             "Must be used with --resume-dir."
    )

    parser.add_argument(
        "--stage1-skip",
        type=str,
        choices=["none", "agent2", "agent4"],
        default="none",
        help="Stage1 ablation experiment: skip specified Agent (default: none, no skip). "
             "agent2=skip anchor finding, agent4=skip lightweight verification. "
             "Note: agent3 cannot be skipped (otherwise cannot generate questions)."
    )

    parser.add_argument(
        "--skip-qtype-prompt",
        action="store_true",
        help="Stage1 ablation experiment: skip question type dimension prompt extraction. "
             "When enabled, only use pedagogical dimensions (gk/cs), not question type dimensions (total 25). "
             "Used to verify the impact of question type dimensions on generation quality."
    )

    parser.add_argument(
        "--use-random-dims",
        action="store_true",
        help="Stage1 ablation experiment: use random dimensions for experiment. "
             "When enabled, use merged_mix_dimension_jk_cs.json (dimension count per question unchanged, but dimension names randomly selected). "
             "Used to verify the effectiveness of controlled dimension mechanism. "
             "Note: need to run python scripts/generate_random_dimension_file.py first to generate random dimension file."
    )

    # [2026-01-14 Added] Hard negative control experiment parameters
    parser.add_argument(
        "--use-hardmix-dims",
        action="store_true",
        help="Hard negative control experiment: use globally permuted random dimensions. "
             "Difference from --use-random-dims: no grouping constraint, randomly select source questions from global 181 questions. "
             "Used for stricter negative control experiment. "
             "Note: need to run python scripts/generate_hardmix_dimension.py first to generate file."
    )

    # [2026-01-05 Added] Low-frequency dimension experiment parameters
    parser.add_argument(
        "--low-freq-random",
        action="store_true",
        help="Low-frequency dimension experiment: assign N random low-frequency dimensions to each question. "
             "Used to verify the effectiveness of controlled dimension mechanism on low-frequency dimensions. "
             "Note: need to run python scripts/generate_low_freq_random_file.py first to generate file."
    )

    parser.add_argument(
        "--low-freq-count",
        type=int,
        choices=[1, 3, 5],  # [2026-01 Refactored] Limit k=1/3/5 experiment settings
        default=3,
        help="[--low-freq-random] Number of low-frequency dimensions assigned per question, options: 1/3/5, default: 3"
    )

    parser.add_argument(
        "--low-freq-ablation",
        action="store_true",
        help="Low-frequency ablation experiment: for questions with low-frequency dimensions, no dimension prompts during generation. "
             "Used to compare differences in hit results with/without low-frequency dimension prompts."
    )

    # ========== [2026-01 Added] No-control Baseline parameters ==========
    parser.add_argument(
        "--skip-stage1-dims",
        action="store_true",
        help="No-control Baseline: skip Stage1 dimension selection and constraints, directly use materials to generate questions. "
             "Used to compare with controlled dimension Pipeline."
    )

    # ========== [2025-12 Added] Question extraction parameters (only for extract mode) ==========
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="[extract mode] Experiment directory path to extract questions from"
    )

    parser.add_argument(
        "--extract-format",
        type=str,
        choices=["text", "markdown", "json"],
        default="text",
        help="[extract mode] Output format: text (default) / markdown / json"
    )

    parser.add_argument(
        "--extract-output",
        type=str,
        default=None,
        help="[extract mode] Output file path (print to terminal if not specified)"
    )

    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="[extract mode] Do not display score information"
    )

    # ========== [2025-12 Added] Stage separation mode parameters ==========
    parser.add_argument(
        "--stage1-dir",
        type=str,
        default=None,
        help="[stage2-only mode] Specify completed Stage1 output directory for reading generated questions for evaluation"
    )

    parser.add_argument(
        "--stage2-output-dir",
        type=str,
        default=None,
        help="[stage2-only mode] Specify evaluation result output directory. If not specified, append evaluation results to stage1-dir original directory. "
             "If specified, will copy questions to new directory and evaluate in new directory."
    )

    # ========== [2025-12 Added] Evaluation mode parameters ==========
    parser.add_argument(
        "--eval-mode",
        type=str,
        default=None,
        choices=["ai", "gk", "cs", "gk+cs", "ai+gk", "ai+cs", "ai+gk+cs"],
        help="Evaluation mode (control which evaluations Stage2 runs):\n"
              "  ai: AI evaluation only\n"
              "  gk: GK pedagogical evaluation only (default)\n"
              "  cs: CS pedagogical evaluation only\n"
             "  gk+cs: GK + CS pedagogical evaluation\n"
              "  ai+gk: AI + GK pedagogical evaluation\n"
             "  ai+cs: AI + CS pedagogical evaluation\n"
             "  ai+gk+cs: AI + GK + CS pedagogical evaluation\n"
             "AI evaluation not enabled by default, specify ai or ai+gk/ai+cs/ai+gk+cs if needed"
    )

    parser.add_argument(
        "--incremental-ai",
        action="store_true",
        help="[stage2-only mode] Incremental AI evaluation: skip completed units (success=True and overall_score>0)"
    )

    # [2026-01 Deprecated] Year range filtering parameters no longer used
    # # ========== [2025-12 Added] Year range filtering parameters ==========
    # parser.add_argument(
    #     "--year-start",
    #     type=int,
    #     default=None,
    #     help="[baseline mode] Start year (inclusive), e.g., --year-start 2021 means 2021 and later"
    # )
    #
    # parser.add_argument(
    #     "--year-end",
    #     type=int,
    #     default=None,
    #     help="[baseline mode] End year (inclusive), e.g., --year-end 2020 means 2020 and earlier"
    # )

    # ========== [2025-12 Added] Exam type filtering parameters ==========
    parser.add_argument(
        "--exam-type",
        type=str,
        choices=["all", "national", "local"],
        default="all",
        help="[baseline mode] Exam type filtering: "
             "all (all, default) / "
             "national (national exams only: New Curriculum Standard, National A/B, New Gaokao) / "
             "local (local exams only: Beijing, Tianjin, Shandong, Jiangsu, Zhejiang, etc.)"
    )
    parser.add_argument(
        "--baseline-dim-source",
        type=str,
        choices=["random", "gold"],
        default="random",
        help="[baseline mode] Stage2 input dimension source: "
             "random (default: input uses merged_mix_dimension_jk_cs.json; pedagogical gold still uses original dimensions) / "
             "gold (human oracle: input and gold both use merged_kaocha_jk_cs.json)"
    )

    args = parser.parse_args()

    # ========== Parameter validation ==========
    if args.run_mode == "single" and args.unit_id is None:
        parser.error("Single mode (--run-mode single) must specify --unit-id")

    if args.run_mode == "full" and args.unit_id is not None:
        print(f"[WARN] Full/subset mode ignores --unit-id parameter")
        args.unit_id = None

    if args.run_mode == "baseline" and args.unit_id is not None:
        print(f"[INFO] Baseline mode will only evaluate specified --unit-id (omit this param for all questions)")
        if args.subset_size is not None or args.subset_file is not None:
            print(f"[WARN] Baseline mode prioritizes --unit-id and ignores --subset-size / --subset-file")
            args.subset_size = None
            args.subset_file = None

    # [2026-01 Deprecated] baseline-recent no longer used
    # if args.run_mode == "baseline-recent" and args.unit_id is not None:
    #     print(f"[WARN] Recent five years baseline evaluation mode ignores --unit-id parameter (auto-filter 2021-2025 questions)")
    #     args.unit_id = None

    if args.run_mode == "single":
        if args.subset_size is not None or args.subset_file is not None:
            print(f"[WARN] Single mode ignores --subset-size / --subset-file parameters")
            args.subset_size = None
            args.subset_file = None

    if args.run_mode == "baseline":
        if args.prompt_level != "C":
            print(f"[WARN] Baseline mode ignores --prompt-level parameter (skips Stage1, no prompt needed)")

    # [2026-01 Deprecated] baseline-recent no longer used
    # if args.run_mode == "baseline-recent":
    #     if args.subset_size is not None or args.subset_file is not None:
    #         print(f"[WARN] Recent five years baseline evaluation mode ignores --subset-size / --subset-file parameters")
    #         args.subset_size = None
    #         args.subset_file = None
    #     if args.prompt_level != "C":
    #         print(f"[WARN] Recent five years baseline evaluation mode ignores --prompt-level parameter (skips Stage1, no prompt generation needed)")

    if args.subset_size is not None and args.subset_file is not None:
        parser.error("--subset-size and --subset-file are mutually exclusive, specify only one")

    # ========== extract mode validation ==========
    if args.run_mode == "extract":
        if args.extract_dir is None:
            parser.error("Extract mode (--run-mode extract) must specify --extract-dir")

    # ========== stage2-only mode validation ==========
    if args.run_mode == "stage2-only":
        if args.stage1_dir is None:
            parser.error("stage2-only mode must specify --stage1-dir (Stage1 output directory)")
        stage1_path = Path(args.stage1_dir)
        if not stage1_path.exists():
            parser.error(f"Specified Stage1 directory does not exist: {args.stage1_dir}")
        stage2_dir = stage1_path / "stage2"
        if not stage2_dir.exists():
            parser.error(f"stage2 subdirectory does not exist in Stage1 directory: {stage2_dir}")

        # [2026-01 Fix] Infer dim_mode from folder name to avoid configuration errors from default value
        exp_name = stage1_path.name
        if "_cs_" in exp_name.lower():
            inferred_dim_mode = "cs"
        elif "_gk_" in exp_name.lower():
            inferred_dim_mode = "gk"
        else:
            inferred_dim_mode = args.dim_mode  # fallback to command-line parameter

        # If user didn't explicitly specify (using default "gk"), override with inferred value
        if args.dim_mode == "gk" and inferred_dim_mode != "gk":
            args.dim_mode = inferred_dim_mode
            print(f"[Stage2-Only] Inferred dim_mode from folder name: {inferred_dim_mode}")

    # ========== stage1-only mode validation ==========
    if args.run_mode == "stage1-only":
        if args.stage1_dir is not None:
            print(f"[WARN] stage1-only mode ignores --stage1-dir parameter")
            args.stage1_dir = None

    # ========== ablation-nodim mode validation ==========
    if args.run_mode == "ablation-nodim":
        # Ablation mode ignores dimension and prompt-related parameters
        if args.dim_mode != "gk":
            print(f"[TIP] ablation-nodim mode ignores --dim-mode parameter (ablation experiment does not use dimensions)")
        if args.prompt_level != "C":
            print(f"[TIP] ablation-nodim mode ignores --prompt-level parameter (ablation experiment does not use dimension prompts)")
        if args.unit_id is not None and (args.subset_size is not None or args.subset_file is not None):
            print(f"[WARN] ablation-nodim mode prioritizes --unit-id and ignores --subset-size / --subset-file")
            args.subset_size = None
            args.subset_file = None
        # [2026-01 Modified] AI evaluation not enabled by default
        if args.eval_mode is None:
            args.eval_mode = "gk"
            print(f"[TIP] ablation-nodim mode default eval_mode=gk (GK pedagogical evaluation only, specify --eval-mode ai+gk+cs for AI+GK+CS evaluation)")
        else:
            print(f"[TIP] ablation-nodim mode using specified eval_mode={args.eval_mode}")

    # ========== [2026-01 Added] Checkpoint resume mode validation ==========
    if args.resume_dir is not None or args.start_from is not None:
        # Checkpoint resume parameters must be specified together
        if args.resume_dir is None:
            parser.error("--start-from must be used with --resume-dir")
        if args.start_from is None:
            parser.error("--resume-dir must be used with --start-from")

        # Verify directory exists
        resume_path = Path(args.resume_dir)
        if not resume_path.exists():
            parser.error(f"Specified checkpoint resume directory does not exist: {args.resume_dir}")
        stage2_dir = resume_path / "stage2"
        if not stage2_dir.exists():
            parser.error(f"stage2 subdirectory does not exist in checkpoint resume directory: {stage2_dir}")

        # Verify start_from range
        if args.start_from < 1 or args.start_from > 181:
            parser.error(f"--start-from must be in range 1-181, current value: {args.start_from}")

        # Checkpoint resume mode only supports full mode
        if args.run_mode != "full":
            parser.error(f"Checkpoint resume (--resume-dir) only supports full mode, current mode: {args.run_mode}")

        # Ignore subset-related parameters
        if args.subset_size is not None or args.subset_file is not None:
            print(f"[WARN] Checkpoint resume mode ignores --subset-size / --subset-file parameters")
            args.subset_size = None
            args.subset_file = None

        print(f"[Resume] Directory: {args.resume_dir}")
        print(f"[Resume] Starting from unit {args.start_from} (inclusive)")

    # Inject unified timestamp (avoid calling datetime.now() in multiple places)
    args._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return args

# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)

# ============================================================================
# Experiment configuration creation
# ============================================================================

def create_experiment_config(args) -> ExperimentConfig:
    """
    Create experiment configuration based on command-line arguments.

    [2025-12 Added round-id support]
    - round_id empty: keep original behavior, output to outputs/<experiment_id>
    - round_id non-empty: output to <output_root>/<round_id>/<run_folder>/
      where output_root defaults to "outputs", can be overridden by --output-dir

    config.experiment_id = run_id (unique identifier for this run)
    config.output_dir = actual output directory
    config.round_id = round_id (can be None)
    config.run_folder = run_folder name (can be None)
    """
    timestamp = getattr(args, "_timestamp", None) or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output_root
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path("outputs")

    # Get stage1_skip parameter (compatible with old args without this field)
    stage1_skip = getattr(args, "stage1_skip", "none") or "none"

    # Generate run_folder name
    # Get ablation experiment parameters
    use_random_dims = getattr(args, "use_random_dims", False)
    use_hardmix_dims = getattr(args, "use_hardmix_dims", False)
    low_freq_random = getattr(args, "low_freq_random", False)
    low_freq_ablation = getattr(args, "low_freq_ablation", False)

    run_folder = build_run_folder_name(
        run_mode=args.run_mode,
        unit_id=args.unit_id,
        subset_size=args.subset_size,
        subset_file=args.subset_file,
        subset_strategy=args.subset_strategy,
        subset_seed=args.subset_seed,
        dim_mode=args.dim_mode,
        prompt_level=args.prompt_level,
        generator_model=args.generator_model,
        timestamp=timestamp,
        stage1_skip=stage1_skip,
        skip_qtype_prompt=getattr(args, "skip_qtype_prompt", False),
        use_random_dims=use_random_dims,
        use_hardmix_dims=use_hardmix_dims,
        low_freq_random=low_freq_random,
        low_freq_ablation=low_freq_ablation,
    )

    # Determine experiment_id (run_id)
    # [2026-01 Fix] stage2-only mode special handling: when stage2-output-dir not specified, use original stage1-dir
    if args.run_mode == "stage2-only" and not args.stage2_output_dir:
        # Extract original experiment_id from stage1_dir
        stage1_path = Path(args.stage1_dir)
        experiment_id = stage1_path.name  # Use original directory name as experiment_id
        final_output_dir = stage1_path  # Output to original directory
    elif args.experiment_id:
        experiment_id = args.experiment_id
        # Determine final output directory
        if args.round_id:
            round_id = sanitize_path_component(args.round_id)
            final_output_dir = output_root / round_id / run_folder
        else:
            if args.output_dir and not args.round_id:
                final_output_dir = output_root
            else:
                final_output_dir = output_root / experiment_id
    else:
        # Use run_folder as experiment_id (ensure uniqueness)
        mode_tag = args.run_mode.upper()
        # [2026-01 Refactored] Remove gk+cs mode, infer dim_mode tag from eval_mode
        eval_mode = getattr(args, "eval_mode", None)
        if eval_mode:
            # Infer dimension tag from eval_mode
            if "gk" in eval_mode and "cs" in eval_mode:
                dim_tag = "gkcs"
            elif "gk" in eval_mode:
                dim_tag = "gk"
            elif "cs" in eval_mode:
                dim_tag = "cs"
            else:
                dim_tag = args.dim_mode  # AI evaluation only, keep original dim_mode
        else:
            dim_tag = args.dim_mode

        # [2025-12-26 Added] Add ablation experiment identifier to experiment_id
        ablation_tag = ""
        if stage1_skip and stage1_skip != "none":
            ablation_tag = f"_SKIP_{stage1_skip.upper()}"

        baseline_source_tag = ""
        if args.run_mode == "baseline":
            baseline_dim_source = getattr(args, "baseline_dim_source", "random") or "random"
            baseline_source_tag = "_ORACLE" if baseline_dim_source == "gold" else "_RANDDIM"

        if args.run_mode == "single":
            experiment_id = f"EXP_{mode_tag}_U{args.unit_id}_{dim_tag}_{args.prompt_level}{ablation_tag}_{timestamp}"
        elif args.run_mode == "baseline":
            if args.unit_id:
                experiment_id = f"EXP_{mode_tag}_U{sanitize_path_component(str(args.unit_id))}_{dim_tag}{baseline_source_tag}_{timestamp}"
            elif args.subset_file:
                file_stem = sanitize_path_component(Path(args.subset_file).stem)
                experiment_id = f"EXP_{mode_tag}_SUBSETFILE_{file_stem}_{dim_tag}{baseline_source_tag}_{timestamp}"
            elif args.subset_size:
                experiment_id = f"EXP_{mode_tag}_SUBSET{args.subset_size}_{dim_tag}{baseline_source_tag}_{timestamp}"
            else:
                experiment_id = f"EXP_{mode_tag}_{dim_tag}{baseline_source_tag}_{timestamp}"
        else:
            experiment_id = f"EXP_{mode_tag}_{dim_tag}_{args.prompt_level}{ablation_tag}_{timestamp}"

        # Determine final output directory
        if args.round_id:
            # round-id mode: output_root/round_id/run_folder/
            round_id = sanitize_path_component(args.round_id)
            final_output_dir = output_root / round_id / run_folder
        else:
            # Original mode: output_root/experiment_id (or custom output_dir)
            if args.output_dir and not args.round_id:
                # User directly specified output_dir, use it as final directory
                final_output_dir = output_root
            else:
                final_output_dir = output_root / experiment_id

    # Create configuration
    config = create_default_config(experiment_id)

    # Dimension mode
    dim_mode_map = {
        "gk": "gk_only",
        "cs": "cs_only",
    }
    config.pipeline.agent1.dimension_mode = dim_mode_map.get(args.dim_mode, "gk_only")

    # Prompt level (using prompt_extraction configuration)
    config.pipeline.prompt_extraction.prompt_level = args.prompt_level

    # Model (keep original behavior)
    config.llm.model_name = args.generator_model
    config.llm.verbose = args.verbose

    # [2026-01 Added] Temperature parameter handling
    if args.temperature is not None:
        from src.shared import api_config
        api_config.STAGE1_TEMPERATURE = args.temperature
        print(f"[Config] Stage1 temperature set to: {args.temperature}")
    else:
        print(f"[Config] Stage1 temperature using default: 1.0")

    # Output directory
    # [2026-01 Added] Checkpoint resume mode: directly use existing directory, don't create new directory
    resume_dir = getattr(args, "resume_dir", None)
    if resume_dir:
        config.output_dir = Path(resume_dir)
        # Checkpoint resume directory should already exist, no need to create
    else:
        config.output_dir = final_output_dir
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Attach round-id related information (for subsequent manifest writing and audit)
    config.round_id = args.round_id if args.round_id else None
    config.run_folder = run_folder if args.round_id else None
    config.round_root = (output_root / sanitize_path_component(args.round_id)) if args.round_id else None

    # [2025-12 Added] Stage1 ablation experiment configuration
    config.pipeline.stage1_ablation.skip_agent = stage1_skip

    # [2026-01 Added] Random dimension ablation configuration (negative control experiment)
    if use_random_dims:
        config.pipeline.stage1_ablation.use_random_dims = True
        config.is_negative_control = True  # Mark as negative control experiment for subsequent analysis identification
        print("[CLI] Random dimension ablation enabled (negative control): using merged_mix_dimension_jk_cs.json")

    # [2026-01-14 Added] Hard negative control experiment configuration (global permutation)
    if use_hardmix_dims:
        config.pipeline.stage1_ablation.use_hardmix_dims = True
        config.is_hard_negative_control = True  # Mark as Hard negative control experiment
        print("[CLI] Hard negative control enabled: using merged_hardmix_dimension_jk_cs.json (global permutation)")

    # [2026-01-05 Added] Low-frequency dimension experiment configuration
    if low_freq_random:
        config.pipeline.stage1_ablation.use_low_freq_random = True
        config.pipeline.stage1_ablation.low_freq_random_count = getattr(args, "low_freq_count", 3)
        # [2026-01-09 Fix] Synchronize config top-level attribute for summary.json recording
        config.is_lowfreq = True
        config.lowfreq_k = config.pipeline.stage1_ablation.low_freq_random_count
        print(f"[CLI] Low-frequency random dimensions enabled: assigning {config.pipeline.stage1_ablation.low_freq_random_count} low-freq dims per question")

    if low_freq_ablation:
        config.pipeline.stage1_ablation.low_freq_ablation = True
        print("[CLI] Low-frequency ablation enabled: no dimension prompts for questions with low-freq dims")

    # [2026-01 Added] No-control Baseline configuration
    skip_stage1_dims = getattr(args, "skip_stage1_dims", False)
    if skip_stage1_dims:
        config.pipeline.stage1_ablation.skip_dims = True
        print("[CLI] No-control baseline enabled: skipping Stage1 dimension selection, generating directly")

    # [2025-12-28 Added] Question type dimension ablation configuration
    if getattr(args, "skip_qtype_prompt", False):
        config.pipeline.prompt_extraction.skip_question_type_prompt = True
        print("[CLI] Question-type dimension ablation enabled: skipping question-type dims, using only pedagogical dims")

    # [2025-12 Added] Evaluation mode configuration
    eval_mode = getattr(args, "eval_mode", None)
    if eval_mode:
        # Set pedagogical evaluation dim_mode based on eval_mode
        # [2025-12 Fix] Also update pipeline.agent1.dimension_mode to ensure baseline mode can read correctly
        if "gk" in eval_mode and "cs" in eval_mode:
            config.evaluation.pedagogical.dim_mode = "gk+cs"
            # Keep legacy-compatible single value on pipeline side; explicit eval_mode controls actual Stage2 execution.
            config.pipeline.agent1.dimension_mode = "gk_only"
        elif "gk" in eval_mode:
            config.evaluation.pedagogical.dim_mode = "gk"
            config.pipeline.agent1.dimension_mode = "gk_only"
        elif "cs" in eval_mode:
            config.evaluation.pedagogical.dim_mode = "cs"
            config.pipeline.agent1.dimension_mode = "cs_only"
        else:
            # AI evaluation only, disable pedagogical evaluation
            config.evaluation.pedagogical.enabled = False

        # [2026-01 Modified] If eval_mode contains ai, enable AI evaluation (disabled by default)
        if "ai" in eval_mode:
            config.evaluation.ai_centric.enabled = True

    # Save eval_mode to config for subsequent use
    config.eval_mode = eval_mode
    config.baseline_dim_source = getattr(args, "baseline_dim_source", "random") or "random"
    config.run_mode = args.run_mode
    config.is_baseline = args.run_mode == "baseline"
    config.stage1_generation_model = None if args.run_mode in ("baseline", "stage2-only") else config.llm.model_name

    return config
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)


# ============================================================================
# Helper functions
# ============================================================================

def _get_overall_score(x):
    if not x:
        return None
    if isinstance(x, dict):
        return x.get("overall_score", x.get("total_score"))
    return getattr(x, "overall_score", None)


def _detect_stage1_generation_model(*summaries: Optional[Dict[str, Any]]) -> Optional[str]:
    """Detect the Stage1 generation model from saved experiment metadata."""
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        llm_config = summary.get("llm_config", {}) if isinstance(summary.get("llm_config"), dict) else {}
        config_info = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
        for candidate in (
            config_info.get("generator_model"),
            summary.get("generator_model"),
            llm_config.get("stage1_model"),
        ):
            text = str(candidate or "").strip()
            if text and text.lower() not in {"unknown", "none", "n/a"}:
                return text
    return None


def _get_overall_score_number(x, default=None):
    """Convert overall_score to float to avoid formatting crashes."""
    s = _get_overall_score(x)
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(s)
    except Exception:
        return default

def _extract_eval_scores(evaluation_state):
    """
    Extract Stage2 scores from the unified evaluation state.

    Reads:
    - ai_eval_result.overall_score -> AI overall score.
    - pedagogical_eval_result -> pedagogical P/R/F1 metrics.
    - gk_eval_result -> independent GK P/R/F1 metrics.
    - cs_eval_result -> independent CS P/R/F1 metrics.

    Returns:
    - ai_score: float | None
    - ped_metrics: dict | None with {f1, precision, recall, tp, fp, fn}
    - gk_metrics: dict | None with independent GK metrics
    - cs_metrics: dict | None with independent CS metrics
    """
    def _get_overall(x):
        if not x:
            return None
        if isinstance(x, dict):
            return x.get("overall_score")
        return getattr(x, "overall_score", None)

    def _extract_ped_metrics(ped_result):
        """Extract P/R/F1 metrics from a pedagogical evaluation result."""
        if ped_result is None:
            return None
        # Extract metrics from PedagogicalHitBasedResult or equivalent dict.
        if isinstance(ped_result, dict):
            f1 = ped_result.get("f1")
            precision = ped_result.get("precision")
            recall = ped_result.get("recall")
            tp = ped_result.get("tp")
            fp = ped_result.get("fp")
            fn = ped_result.get("fn")
            # Extract dimension lists for high-frequency exclusion stats.
            gold_dimensions = ped_result.get("gold_dimensions", [])
            predicted_dimensions = ped_result.get("predicted_dimensions", [])
        else:
            f1 = getattr(ped_result, "f1", None)
            precision = getattr(ped_result, "precision", None)
            recall = getattr(ped_result, "recall", None)
            tp = getattr(ped_result, "tp", None)
            fp = getattr(ped_result, "fp", None)
            fn = getattr(ped_result, "fn", None)
            # Extract dimension lists for high-frequency exclusion stats.
            gold_dimensions = getattr(ped_result, "gold_dimensions", [])
            predicted_dimensions = getattr(ped_result, "predicted_dimensions", [])

        # Build a metrics dict if any metric field is present.
        if any(x is not None for x in [f1, precision, recall, tp, fp, fn]):
            return {
                "f1": float(f1) if f1 is not None else 0.0,
                "precision": float(precision) if precision is not None else 0.0,
                "recall": float(recall) if recall is not None else 0.0,
                "tp": int(tp) if tp is not None else 0,
                "fp": int(fp) if fp is not None else 0,
                "fn": int(fn) if fn is not None else 0,
                # Keep dimension lists for downstream grouped metrics.
                "gold_dimensions": list(gold_dimensions) if gold_dimensions else [],
                "predicted_dimensions": list(predicted_dimensions) if predicted_dimensions else [],
            }
        return None

    # AI score.
    ai = _get_overall(getattr(evaluation_state, "ai_eval_result", None))

    try:
        ai = float(ai) if ai is not None else None
    except Exception:
        ai = None

    # Pedagogical evaluation uses hit-based P/R/F1 metrics.
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    ped_metrics = _extract_ped_metrics(ped_result)

    # Independent GK/CS pedagogical evaluation results.
    gk_result = getattr(evaluation_state, "gk_eval_result", None)
    gk_metrics = _extract_ped_metrics(gk_result)

    cs_result = getattr(evaluation_state, "cs_eval_result", None)
    cs_metrics = _extract_ped_metrics(cs_result)

    return ai, ped_metrics, gk_metrics, cs_metrics


def _round_floats(obj: Any, decimals: int = 3) -> Any:
    """
    Recursively round floats inside dictionaries/lists.

    Args:
        obj: Object to process.
        decimals: Number of decimals to keep.

    Returns:
        Processed object.
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(item, decimals) for item in obj]
    else:
        return obj


def _calculate_prf(
    gold_dims: List[str],
    predicted_dims: List[str],
    exclude_dims: Optional[set] = None,
) -> Tuple[int, int, int, float, float, float]:
    """Calculate TP/FP/FN and precision/recall/F1 for dimension-code lists."""
    gold_set = set(gold_dims or [])
    pred_set = set(predicted_dims or [])

    if exclude_dims:
        gold_set -= exclude_dims
        pred_set -= exclude_dims

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def _compute_iteration_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute iteration-related statistics.

    Tracks iteration failures, examples, triggered count, and triggered rate.

    Args:
        results: stats["results"], each optionally containing iteration_count and max_iteration_exceeded.

    Returns:
        Iteration stats dict.
    """
    if not results:
        return {
            "iteration_failed_count": 0,
            "iteration_failed_examples": [],
            "iteration_success_rate": 1.0,
            "iteration_triggered_count": 0,
            "iteration_triggered_rate": 0.0,
            "total_questions": 0,
        }

    total = len(results)
    failed_count = 0
    failed_examples = []
    triggered_count = 0

    for r in results:
        iteration_count = r.get("iteration_count", 0)
        max_exceeded = r.get("max_iteration_exceeded", False)

        # Iteration triggered means iteration_count > 1.
        if iteration_count > 1:
            triggered_count += 1

        # Iteration failed if max_iteration_exceeded is True.
        if max_exceeded:
            failed_count += 1
            if len(failed_examples) < 5:
                failed_examples.append(r.get("unit_id", "unknown"))

    # Iteration success rate among triggered cases.
    if triggered_count > 0:
        iteration_success_rate = round((triggered_count - failed_count) / triggered_count, 3)
    else:
        iteration_success_rate = 1.0

    triggered_rate = round(triggered_count / total, 3) if total > 0 else 0.0

    return {
        "iteration_failed_count": failed_count,
        "iteration_failed_examples": failed_examples,
        "iteration_success_rate": iteration_success_rate,
        "iteration_triggered_count": triggered_count,
        "iteration_triggered_rate": triggered_rate,
        "total_questions": total,
    }


def _extract_eval_details(evaluation_state, model_weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Extract dimension-level evaluation details for summary transparency.

    Return shape:
    {
        "ai_eval": {
            "overall_score": float,
            "dimensions": {dim_id: {"score": float, "weight": float, "level": str}},
            "model_summary": {model_name: {"average_score": float, "weight": float}}
        },
        "pedagogical_eval": {
            "overall_score": float,
            "dimensions": {dim_name: {"score": float, "hit_level": str}},
            "model_summary": {model_name: {"average_score": float, "weight": float}}
        }
    }
    """
    result = {
        "ai_eval": None,
        "pedagogical_eval": None,
    }

    model_weights = model_weights or {}

    # Extract AI evaluation details.
    ai_result = getattr(evaluation_state, "ai_eval_result", None)
    if ai_result:
        ai_details = {"overall_score": None, "dimensions": {}, "model_summary": {}}

        if isinstance(ai_result, dict):
            ai_details["overall_score"] = ai_result.get("overall_score") or ai_result.get("total_score")
            dimensions = ai_result.get("dimensions", {})
            model_results = ai_result.get("model_results", {})
        else:
            ai_details["overall_score"] = getattr(ai_result, "overall_score", None) or getattr(ai_result, "total_score", None)
            dimensions = getattr(ai_result, "dimensions", {}) or {}
            model_results = getattr(ai_result, "model_results", {}) or {}

        # Dimension details with contribution values.
        for dim_id, dim_data in (dimensions if isinstance(dimensions, dict) else {}).items():
            if isinstance(dim_data, dict):
                dim_score = dim_data.get("score")
                dim_weight = dim_data.get("weight", 1.0)
                ai_details["dimensions"][dim_id] = {
                    "score": dim_score,
                    "weight": dim_weight,
                    "contribution": round(dim_score * dim_weight, 2) if dim_score is not None else None,
                    "level": dim_data.get("level"),
                }

        # Model summary with contribution values.
        for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
            if isinstance(model_dims, dict):
                scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                if scores:
                    model_weight = model_weights.get(model_name, 0.0)
                    model_avg = round(sum(scores) / len(scores), 2)
                    ai_details["model_summary"][model_name] = {
                        "average_score": model_avg,
                        "weight": model_weight,
                        "contribution": round(model_avg * model_weight, 2),
                        "dimension_count": len(scores),
                    }

        result["ai_eval"] = ai_details

    # Extract pedagogical evaluation details.
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    if ped_result:
        ped_details = {"overall_score": None, "dimensions": {}, "model_summary": {}}

        if hasattr(ped_result, "overall_score"):
            ped_details["overall_score"] = float(ped_result.overall_score)
            dimension_results = getattr(ped_result, "dimension_results", []) or []
            model_results = getattr(ped_result, "model_results", {}) or {}
        elif isinstance(ped_result, dict):
            ped_details["overall_score"] = ped_result.get("overall_score")
            dimension_results = ped_result.get("dimension_results", []) or []
            model_results = ped_result.get("model_results", {}) or {}
        else:
            dimension_results = []
            model_results = {}

        # Dimension details with equal weights and contribution values.
        dim_count = len(dimension_results) if dimension_results else 1
        dim_weight = round(1.0 / dim_count, 4) if dim_count > 0 else 0

        for dim_res in dimension_results:
            if hasattr(dim_res, "dimension_name"):
                dim_name = dim_res.dimension_name
                dim_score = float(getattr(dim_res, "score", 0))
                ped_details["dimensions"][dim_name] = {
                    "score": dim_score,
                    "weight": dim_weight,
                    "contribution": round(dim_score * dim_weight, 2),
                    "hit_level": getattr(dim_res, "hit_level", ""),
                }
            elif isinstance(dim_res, dict):
                dim_name = dim_res.get("dimension_name", "")
                if dim_name:
                    dim_score = dim_res.get("score", 0)
                    ped_details["dimensions"][dim_name] = {
                        "score": dim_score,
                        "weight": dim_weight,
                        "contribution": round(dim_score * dim_weight, 2),
                        "hit_level": dim_res.get("hit_level", ""),
                    }

        # Model summary with contribution values.
        for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
            if isinstance(model_dims, dict):
                scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                if scores:
                    model_weight = model_weights.get(model_name, 0.0)
                    model_avg = round(sum(scores) / len(scores), 2)
                    ped_details["model_summary"][model_name] = {
                        "average_score": model_avg,
                        "weight": model_weight,
                        "contribution": round(model_avg * model_weight, 2),
                        "dimension_count": len(scores),
                    }

        result["pedagogical_eval"] = ped_details

    return result


def _extract_missing_dimensions(evaluation_state, unit_id: str, question_type: str = "") -> Optional[Dict[str, Any]]:
    """
    Extract missing-dimension information from evaluation state.

    Missing dimensions are gold_dimensions - predicted_dimensions (false negatives).

    Return shape:
    {
        "unit_id": str,
        "question_type": str,
        "gold_dimensions": List[str],
        "predicted_dimensions": List[str],
        "missing_dimensions": List[str],
        "extra_dimensions": List[str],
        "tp": int,
        "fp": int,
        "fn": int,
        "precision": float,
        "recall": float,
        "f1": float,
    }
    Returns None if no pedagogical evaluation result is available.
    """
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    if ped_result is None:
        return None

    # Extract fields from PedagogicalHitBasedResult or equivalent dict.
    if hasattr(ped_result, "gold_dimensions"):
        gold_dims = list(getattr(ped_result, "gold_dimensions", []) or [])
        predicted_dims = list(getattr(ped_result, "predicted_dimensions", []) or [])
        missing_dims = list(getattr(ped_result, "missing_dimensions", []) or [])
        extra_dims = list(getattr(ped_result, "extra_dimensions", []) or [])
        tp = getattr(ped_result, "tp", 0)
        fp = getattr(ped_result, "fp", 0)
        fn = getattr(ped_result, "fn", 0)
        precision = getattr(ped_result, "precision", 0.0)
        recall = getattr(ped_result, "recall", 0.0)
        f1 = getattr(ped_result, "f1", 0.0)
    elif isinstance(ped_result, dict):
        gold_dims = list(ped_result.get("gold_dimensions", []) or [])
        predicted_dims = list(ped_result.get("predicted_dimensions", []) or [])
        missing_dims = list(ped_result.get("missing_dimensions", []) or [])
        extra_dims = list(ped_result.get("extra_dimensions", []) or [])
        tp = ped_result.get("tp", 0)
        fp = ped_result.get("fp", 0)
        fn = ped_result.get("fn", 0)
        precision = ped_result.get("precision", 0.0)
        recall = ped_result.get("recall", 0.0)
        f1 = ped_result.get("f1", 0.0)
    else:
        return None

    # Manually compute missing/extra dimensions if not provided.
    if not missing_dims and gold_dims:
        gold_set = set(gold_dims)
        pred_set = set(predicted_dims)
        missing_dims = sorted(list(gold_set - pred_set))
        extra_dims = sorted(list(pred_set - gold_set))

    return {
        "unit_id": unit_id,
        "question_type": question_type,
        "gold_dimensions": gold_dims,
        "predicted_dimensions": predicted_dims,
        "missing_dimensions": missing_dims,
        "extra_dimensions": extra_dims,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _save_missing_dimensions_report(
    missing_dims_records: List[Dict[str, Any]],
    output_dir: Path,
    experiment_id: str,
) -> Tuple[Path, Path]:
    """
    Save pedagogical dimension evaluation summary reports.

    Outputs:
    1. missing_dimensions_report.json
    2. missing_dimensions_report.txt

    Includes TP/FP/FN details for each question.

    Returns: (json_path, txt_path).
    """
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Keep valid records.
    valid_records = [r for r in missing_dims_records if r]

    # Missing dimension frequency (FN).
    missing_freq: Dict[str, int] = {}
    for r in valid_records:
        for dim in r.get("missing_dimensions", []):
            missing_freq[dim] = missing_freq.get(dim, 0) + 1

    # Extra dimension frequency (FP).
    extra_freq: Dict[str, int] = {}
    for r in valid_records:
        for dim in r.get("extra_dimensions", []):
            extra_freq[dim] = extra_freq.get(dim, 0) + 1

    # Sort by frequency.
    sorted_missing_freq = sorted(missing_freq.items(), key=lambda x: -x[1])
    sorted_extra_freq = sorted(extra_freq.items(), key=lambda x: -x[1])

    # Overall metrics.
    total_tp = sum(r.get("tp", 0) for r in valid_records)
    total_fp = sum(r.get("fp", 0) for r in valid_records)
    total_fn = sum(r.get("fn", 0) for r in valid_records)
    avg_precision = sum(r.get("precision", 0) for r in valid_records) / len(valid_records) if valid_records else 0
    avg_recall = sum(r.get("recall", 0) for r in valid_records) / len(valid_records) if valid_records else 0
    avg_f1 = sum(r.get("f1", 0) for r in valid_records) / len(valid_records) if valid_records else 0

    # Build summary.
    summary = {
        "experiment_id": experiment_id,
        "generated_at": timestamp,
        "total_questions": len(valid_records),
        "overall_metrics": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1": round(avg_f1, 4),
        },
        "missing_dimension_frequency": dict(sorted_missing_freq),
        "extra_dimension_frequency": dict(sorted_extra_freq),
        "detail_records": valid_records,
    }

    # Save JSON.
    json_path = output_dir / "missing_dimensions_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save a human-readable TXT report.
    txt_path = output_dir / "missing_dimensions_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Pedagogical Dimension Evaluation Summary\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Total questions: {len(valid_records)}\n")
        f.write(f"\n")

        # Overall metrics.
        f.write(f"[Overall Metrics]\n")
        f.write(f"-" * 60 + "\n")
        f.write(f"  Total TP (correct hits): {total_tp}\n")
        f.write(f"  Total FP (extra predictions): {total_fp}\n")
        f.write(f"  Total FN (missed dimensions): {total_fn}\n")
        f.write(f"  Average Precision: {avg_precision:.4f}\n")
        f.write(f"  Average Recall:    {avg_recall:.4f}\n")
        f.write(f"  Average F1:        {avg_f1:.4f}\n")
        f.write(f"\n")

        # Missing dimension frequency (FN).
        f.write(f"[Missing Dimensions (FN), descending frequency]\n")
        f.write(f"-" * 60 + "\n")
        if sorted_missing_freq:
            for dim, freq in sorted_missing_freq:
                f.write(f"  {dim}: {freq} times\n")
        else:
            f.write(f"  No missing dimensions.\n")
        f.write(f"\n")

        # Extra prediction frequency (FP).
        f.write(f"[Extra Predictions (FP), descending frequency]\n")
        f.write(f"-" * 60 + "\n")
        if sorted_extra_freq:
            for dim, freq in sorted_extra_freq:
                f.write(f"  {dim}: {freq} times\n")
        else:
            f.write(f"  No extra predictions.\n")
        f.write(f"\n")

        # Per-question details.
        f.write(f"[Per-Question Dimension Hit Details]\n")
        f.write(f"-" * 60 + "\n")
        for r in valid_records:
            unit_id = r.get("unit_id", "unknown")
            qt = r.get("question_type", "unknown")
            missing = r.get("missing_dimensions", [])
            extra = r.get("extra_dimensions", [])
            gold = r.get("gold_dimensions", [])
            predicted = r.get("predicted_dimensions", [])
            tp = r.get("tp", 0)
            fp = r.get("fp", 0)
            fn = r.get("fn", 0)
            precision = r.get("precision", 0.0)
            recall = r.get("recall", 0.0)
            f1 = r.get("f1", 0.0)

            # Calculate hit dimensions (TP).
            gold_set = set(gold)
            pred_set = set(predicted)
            hit_dims = sorted(gold_set & pred_set)

            f.write(f"\nQuestion unit_id={unit_id} (question_type: {qt})\n")
            f.write(f"  Gold dimensions ({len(gold)} items): {gold}\n")
            f.write(f"  Predicted dimensions ({len(predicted)} items): {predicted}\n")
            f.write(f"  ---\n")
            f.write(f"  TP hits ({len(hit_dims)} items): {hit_dims}\n")
            f.write(f"  FN missing ({len(missing)} items): {missing}\n")
            f.write(f"  FP extra ({len(extra)} items): {extra}\n")
            f.write(f"  ---\n")
            f.write(f"  TP={tp}, FP={fp}, FN={fn} | P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")

    print(f"\n[INFO] Missing dimension report saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")

    return json_path, txt_path


def collect_good_bad_cases(
    results: List[Dict[str, Any]],
    generation_states: Dict[str, Any] = None,
    max_cases: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect Good Case and Bad Case examples for showcase output.

    Bad case priority:
    1. Iteration-limit failures.
    2. F1=0 cases.
    3. Lowest-F1 cases.

    Good case priority:
    1. F1=1.0 cases.
    2. Highest-F1 cases.

    Args:
        results: Evaluation result list.
        generation_states: Optional unit_id -> generation_state mapping for iteration info.
        max_cases: Maximum number of cases per category.

    Returns:
        {"good_cases": [...], "bad_cases": [...]}
    """
    generation_states = generation_states or {}

    # Collect bad cases.
    bad_cases = []

    # 1. First collect iteration-limit failures.
    for uid, gen_state in generation_states.items():
        if gen_state.get("max_iteration_exceeded", False):
            bad_cases.append({
                "unit_id": uid,
                "fail_type": "iteration_exceeded",
                "iteration_count": gen_state.get("iteration_count", 0),
                "fail_reason": gen_state.get("iteration_fail_reason", "iteration limit exceeded"),
                "f1": None,
                "stem_preview": gen_state.get("stem_preview", ""),
            })

    # 2. Collect low-F1 cases from evaluation results.
    def _get_f1(r):
        """Extract F1 from a result item."""
        ped = r.get("ped_metrics") or r.get("gk_metrics") or r.get("cs_metrics")
        if ped:
            return ped.get("f1")
        return r.get("f1")

    def _get_metrics(r):
        """Extract evaluation metrics from a result item."""
        ped = r.get("ped_metrics") or r.get("gk_metrics") or r.get("cs_metrics") or {}
        return {
            "f1": ped.get("f1", r.get("f1", 0)),
            "precision": ped.get("precision", r.get("precision", 0)),
            "recall": ped.get("recall", r.get("recall", 0)),
            "gold_dimensions": ped.get("gold_dimensions", r.get("gold_dimensions", [])),
            "predicted_dimensions": ped.get("predicted_dimensions", r.get("predicted_dimensions", [])),
        }

    valid_results = [r for r in results if _get_f1(r) is not None]
    sorted_by_f1 = sorted(valid_results, key=lambda x: _get_f1(x) or 0)

    for r in sorted_by_f1:
        if len(bad_cases) >= max_cases:
            break
        # Skip cases already collected as iteration-limit failures.
        if any(bc["unit_id"] == r.get("unit_id") for bc in bad_cases):
            continue
        m = _get_metrics(r)
        bad_cases.append({
            "unit_id": r.get("unit_id"),
            "fail_type": "low_f1",
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "gold_dimensions": m["gold_dimensions"],
            "predicted_dimensions": m["predicted_dimensions"],
            "stem_preview": "",
            "question_type": r.get("question_type", ""),
        })

    # Collect good cases.
    good_cases = []
    sorted_by_f1_desc = sorted(valid_results, key=lambda x: _get_f1(x) or 0, reverse=True)

    for r in sorted_by_f1_desc:
        if len(good_cases) >= max_cases:
            break
        m = _get_metrics(r)
        good_cases.append({
            "unit_id": r.get("unit_id"),
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "gold_dimensions": m["gold_dimensions"],
            "predicted_dimensions": m["predicted_dimensions"],
            "stem_preview": "",
            "question_type": r.get("question_type", ""),
        })

    return {
        "good_cases": good_cases[:max_cases],
        "bad_cases": bad_cases[:max_cases],
    }


def print_good_bad_cases(cases: Dict[str, List[Dict[str, Any]]]):
    """
    [2026-01 Added] Print Good Cases and Bad Cases.
    """
    print("\n" + "=" * 70)
    print("Good Cases / Bad Cases Examples")
    print("=" * 70)

    # Print Bad Cases
    bad_cases = cases.get("bad_cases", [])
    if bad_cases:
        print(f"\n[Bad Cases] ({len(bad_cases)} items)")
        print("-" * 60)
        for i, bc in enumerate(bad_cases, 1):
            print(f"\n  [{i}] unit_id={bc['unit_id']}")
            if bc.get("fail_type") == "iteration_exceeded":
                print(f"      Type: Iteration limit exceeded")
                print(f"      Iteration count: {bc.get('iteration_count', 'N/A')}")
                print(f"      Reason: {bc.get('fail_reason', 'N/A')}")
            else:
                print(f"      Type: Low F1 score")
                print(f"      F1={bc.get('f1', 0):.4f}, P={bc.get('precision', 0):.4f}, R={bc.get('recall', 0):.4f}")
                print(f"      Question type: {bc.get('question_type', 'N/A')}")
                print(f"      Gold: {bc.get('gold_dimensions', [])}")
                print(f"      Pred: {bc.get('predicted_dimensions', [])}")
            if bc.get("stem_preview"):
                print(f"      Stem preview: {bc['stem_preview']}")
    else:
        print("\n[Bad Cases] None")

    # Print Good Cases
    good_cases = cases.get("good_cases", [])
    if good_cases:
        print(f"\n[Good Cases] ({len(good_cases)} items)")
        print("-" * 60)
        for i, gc in enumerate(good_cases, 1):
            print(f"\n  [{i}] unit_id={gc['unit_id']}")
            print(f"      F1={gc.get('f1', 0):.4f}, P={gc.get('precision', 0):.4f}, R={gc.get('recall', 0):.4f}")
            print(f"      Question type: {gc.get('question_type', 'N/A')}")
            print(f"      Gold: {gc.get('gold_dimensions', [])}")
            print(f"      Pred: {gc.get('predicted_dimensions', [])}")
            if gc.get("stem_preview"):
                print(f"      Stem preview: {gc['stem_preview']}")
    else:
        print("\n[Good Cases] None")

    print("\n" + "=" * 70)


# ============================================================================
# [2025-12 Architecture Refactoring] Unified unit execution pipeline
# ============================================================================

def run_units(
    config: "ExperimentConfig",
    unit_ids: List[str],
    router: "LLMRouter",
    generation_orchestrator: "GenerationOrchestrator",
    evaluation_orchestrator: "EvaluationOrchestrator",
) -> Dict[str, Any]:
    """
    [2025-12 Architecture Refactoring] Unified unit execution pipeline.

    single/subset/full three modes share this function.

    For each uid, execute:
    1. stage1_state = generation_orchestrator.run_single(uid)
    2. record = stage1_state.stage2_record
    3. if not record: append skipped; continue
    4. stage2_state = evaluation_orchestrator.run(record)
    5. append result

    Returns aggregated statistics.
    """
    all_results: List[Dict[str, Any]] = []
    generation_success = 0
    evaluation_success = 0
    stage2_skipped = 0
    skipped_no_dim = 0  # [2025-12-31 Added] Number of questions skipped due to no corresponding dimensions

    ai_scores: List[float] = []
    # [2025-12 Refactored] Pedagogical evaluation uses P/R/F1 metric collection
    ped_metrics_list: List[Dict[str, Any]] = []  # Each question's {f1, precision, recall, tp, fp, fn}
    # [2025-12 Added] Independent GK/CS evaluation metric collection
    gk_metrics_list: List[Dict[str, Any]] = []

    # [2025-12-31 Added] Load year mapping (for year-based statistics)
    from src.shared.data_loader import DataLoader
    data_loader = DataLoader()
    unit_year_mapping = data_loader.load_unit_year_mapping()
    cs_metrics_list: List[Dict[str, Any]] = []

    question_type_counts = {"single-choice": 0, "essay": 0, "other": 0}
    score_buckets = {
        "single-choice": {"ai": [], "ped_f1": []},
        "essay": {"ai": [], "ped_f1": []},
        "other": {"ai": [], "ped_f1": []},
    }

    # [2025-12-28 Added] Pedagogical metrics collection grouped by question type (for calculating P/R/F1 by question type)
    ped_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    gk_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    cs_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }

    agent_errors = {"agent1": 0, "agent2": 0, "agent3": 0, "agent4": 0, "agent5": 0, "eval": 0}

    # [2025-12 Added] Missing dimension collection
    all_missing_dims_records: List[Dict[str, Any]] = []

    # [2026-01 Added] Collect generation_state iteration information (for good/bad case analysis)
    all_generation_states: Dict[str, Dict[str, Any]] = {}

    run_total = len(unit_ids)

    for i, uid in enumerate(unit_ids):
        uid = str(uid)
        print(f"\n>>> [{i+1}/{run_total}] Processing unit_id={uid}...")

        result_item = {
            "unit_id": uid,
            "stage1_status": "pending",
            "stage2_status": "skip",
            "question_type": None,
            "ai_overall_score": None,
            "ped_overall_score": None,
            "skip_reason": None,
            # [2026-01 Added] Iteration statistics fields
            "iteration_count": 0,
            "max_iteration_exceeded": False,
        }

        try:
            # -------- Stage 1 --------
            generation_state = generation_orchestrator.run_single(unit_id=uid)
            stage2_record = getattr(generation_state, "stage2_record", None)

            # [2026-01 Added] Collect iteration information (for good/bad case analysis)
            all_generation_states[uid] = {
                "max_iteration_exceeded": getattr(generation_state, "max_iteration_exceeded", False),
                "iteration_count": getattr(generation_state, "iteration_count", 0),
                "iteration_fail_reason": getattr(generation_state, "iteration_fail_reason", ""),
                "current_agent": getattr(generation_state, "current_agent", ""),
                # Get question stem preview (for display)
                "stem_preview": "",
            }
            # [2026-01 Added] Synchronize iteration information to result_item (for statistics)
            result_item["iteration_count"] = all_generation_states[uid]["iteration_count"]
            result_item["max_iteration_exceeded"] = all_generation_states[uid]["max_iteration_exceeded"]
            # Try to get stem preview from agent3_output
            agent3_out = getattr(generation_state, "agent3_output", None)
            if agent3_out and hasattr(agent3_out, "generated_question"):
                gq = agent3_out.generated_question
                if gq and hasattr(gq, "stem"):
                    stem_text = gq.stem or ""
                    all_generation_states[uid]["stem_preview"] = stem_text[:100] + ("..." if len(stem_text) > 100 else "")

            if stage2_record is None:
                stage2_skipped += 1
                result_item["stage1_status"] = "no_stage2_record"
                result_item["stage2_status"] = "skip"

                # [2025-12-31 Added] Check if skipped due to no dimensions
                current_agent = getattr(generation_state, "current_agent", "")
                if current_agent == "skipped_no_dim":
                    result_item["skip_reason"] = "no_matching_dimension"
                    skipped_no_dim += 1  # [2025-12-31 Added] Increment no-dimension skip count
                    print(f"    [SKIP] No matching dimensions, skipping this question")
                else:
                    result_item["skip_reason"] = "stage1_no_stage2_record"
                    # Count agent errors
                    if not getattr(generation_state, "agent1_success", False):
                        agent_errors["agent1"] += 1
                    elif not getattr(generation_state, "agent2_success", False):
                        agent_errors["agent2"] += 1
                    elif not getattr(generation_state, "agent3_success", False):
                        agent_errors["agent3"] += 1
                    elif not getattr(generation_state, "agent4_success", False):
                        agent_errors["agent4"] += 1
                    elif not getattr(generation_state, "agent5_success", False):
                        agent_errors["agent5"] += 1
                    else:
                        agent_errors["agent5"] += 1
                    print("    [SKIP] Stage1 did not produce stage2_record, skipping Stage2")

                all_results.append(result_item)
                continue

            generation_success += 1
            result_item["stage1_status"] = "success"

            # Extract question type
            qt = getattr(stage2_record.core_input, "question_type", None) or "other"
            if qt not in question_type_counts:
                qt = "other"
            question_type_counts[qt] += 1
            result_item["question_type"] = qt

            # [2026-01 Added] Extract anchor information (only when Stage1 is used and agent2 ablation is not performed)
            anchor_count = getattr(stage2_record.core_input, "anchor_count", 0)
            anchors = getattr(stage2_record.core_input, "anchors", None)
            result_item["anchor_count"] = anchor_count
            if anchors:
                result_item["anchors"] = anchors

            # -------- Stage 2 --------
            try:
                evaluation_state = evaluation_orchestrator.run(stage2_record)
            except Exception as e:
                agent_errors["eval"] += 1
                result_item["stage2_status"] = "error"
                result_item["skip_reason"] = f"stage2_exception: {e}"
                print(f"    [EXCEPTION] Evaluation exception: {e}")
                all_results.append(result_item)
                continue

            if getattr(evaluation_state, "current_stage", None) != "completed":
                agent_errors["eval"] += 1
                result_item["stage2_status"] = "fail"
                print("    [FAIL] Evaluation failed")
                all_results.append(result_item)
                continue

            evaluation_success += 1
            result_item["stage2_status"] = "success"

            # [2025-12 Architecture Refactoring] Read AI scores and pedagogical P/R/F1 metrics
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            result_item["ai_overall_score"] = ai_score
            result_item["ped_metrics"] = ped_metrics  # Contains f1, precision, recall, tp, fp, fn
            result_item["gk_metrics"] = gk_metrics  # [2025-12 Added] Independent GK evaluation metrics
            result_item["cs_metrics"] = cs_metrics  # [2025-12 Added] Independent CS evaluation metrics

            if ai_score is not None:
                ai_scores.append(ai_score)
                score_buckets[qt]["ai"].append(ai_score)
            if ped_metrics is not None:
                # [2025-12-31 Added] Add year and question type information (for grouped statistics)
                ped_metrics["year"] = unit_year_mapping.get(uid)
                ped_metrics["question_type"] = qt
                ped_metrics["unit_id"] = uid
                ped_metrics_list.append(ped_metrics)
                score_buckets[qt]["ped_f1"].append(ped_metrics["f1"])
                ped_metrics_by_qtype[qt].append(ped_metrics)  # [2025-12-28 Added] Collect by question type
            # [2025-12 Added] Collect independent GK/CS evaluation metrics
            if gk_metrics is not None:
                # [2025-12-31 Added] Add year and question type information
                gk_metrics["year"] = unit_year_mapping.get(uid)
                gk_metrics["question_type"] = qt
                gk_metrics["unit_id"] = uid
                gk_metrics_list.append(gk_metrics)
                gk_metrics_by_qtype[qt].append(gk_metrics)  # [2025-12-28 Added] Collect by question type
            if cs_metrics is not None:
                # [2025-12-31 Added] Add year and question type information
                cs_metrics["year"] = unit_year_mapping.get(uid)
                cs_metrics["question_type"] = qt
                cs_metrics["unit_id"] = uid
                cs_metrics_list.append(cs_metrics)
                cs_metrics_by_qtype[qt].append(cs_metrics)  # [2025-12-28 Added] Collect by question type

            # [2025-12 Added] Collect missing dimension records
            missing_dims_item = _extract_missing_dimensions(evaluation_state, uid, qt)
            if missing_dims_item:
                all_missing_dims_records.append(missing_dims_item)

            ai_disp = f"{ai_score:.1f}" if isinstance(ai_score, (int, float)) else "N/A"
            ped_disp = f"F1={ped_metrics['f1']:.3f}" if ped_metrics else "N/A"
            print(f"    [OK] Question type={qt}, AI={ai_disp}, Ped={ped_disp}")

        except Exception as e:
            print(f"    [EXCEPTION] Generation exception: {e}")
            agent_errors["agent1"] += 1
            result_item["stage1_status"] = "error"
            result_item["skip_reason"] = f"stage1_exception: {e}"

        all_results.append(result_item)

    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    # [2025-12 Refactored] Calculate pedagogical dimension micro/macro P/R/F1 aggregation
    def _compute_ped_round_metrics(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk", success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Calculate pedagogical dimension round aggregation metrics

        [Important Note]
        This function's calculation logic duplicates pedagogical_eval.aggregate_round().
        High-frequency dimension definition unified source: src.shared.dimension_config

        This helper intentionally keeps the report-local summary shape aligned
        with PedagogicalRoundAggregation while preserving existing output fields.
        """
        from src.shared.dimension_config import get_high_freq_dims_by_mode
        from output_analysis.core.bootstrap_ci import (
            bootstrap_micro_metrics,
            bootstrap_macro_metrics_dimension_view
        )

        if not metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # [2025-12-31 Added] Filter out questions with empty original gold dimensions
        skipped_no_dims = 0
        valid_metrics_list = []
        for m in metrics_list:
            gold = m.get("gold_dimensions", [])
            if not gold:
                skipped_no_dims += 1
                continue
            valid_metrics_list.append(m)

        if not valid_metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "skipped_no_dims": skipped_no_dims,
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # Micro average: aggregate TP/FP/FN then calculate
        total_tp = sum(m["tp"] for m in valid_metrics_list)
        total_fp = sum(m["fp"] for m in valid_metrics_list)
        total_fn = sum(m["fn"] for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # [2026-01 Added] Off-target and Success@k
        off_target = 1.0 - micro_precision
        success_count = sum(1 for m in valid_metrics_list if m.get("recall", 0) >= success_threshold)
        success_at_k = success_count / len(valid_metrics_list) if valid_metrics_list else 0.0

        # [2026-01-07 Fix] Macro average: dimension perspective (calculate P/R/F1 for each dimension, then average)
        # Collect statistics for all dimensions
        dim_stats = {}  # {dim_code: {"tp": 0, "fp": 0, "fn": 0}}
        for m in valid_metrics_list:
            gold = set(m.get("gold_dimensions", []))
            pred = set(m.get("predicted_dimensions", []))
            all_dims = gold | pred
            for d in all_dims:
                if d not in dim_stats:
                    dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats[d]["tp"] += 1
                elif d in pred and d not in gold:
                    dim_stats[d]["fp"] += 1
                elif d in gold and d not in pred:
                    dim_stats[d]["fn"] += 1

        # Calculate P/R/F1 for each dimension, then average
        dim_p, dim_r, dim_f1 = [], [], []
        for d, st in dim_stats.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p.append(p_d)
            dim_r.append(r_d)
            dim_f1.append(f1_d)
        macro_precision = _avg(dim_p)
        macro_recall = _avg(dim_r)
        macro_f1 = _avg(dim_f1)

        result = {
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "skipped_no_dims": skipped_no_dims,  # [2025-12-31] Number of questions with no original dimensions skipped
            "off_target": off_target,
            "success_at_k": success_at_k,
            "success_threshold": success_threshold,
        }

        # [2026-01-17 Added] Bootstrap confidence interval calculation
        if len(valid_metrics_list) >= 2:  # At least 2 samples needed to calculate CI
            n_bootstrap = 1000
            bootstrap_seed = 42
            micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            result["bootstrap_ci"] = {
                "micro_precision": micro_ci["micro_precision"],
                "micro_recall": micro_ci["micro_recall"],
                "micro_f1": micro_ci["micro_f1"],
                "macro_f1": macro_ci["macro_f1"],
                "n_bootstrap": n_bootstrap,
                "ci_level": 0.95,
            }

        # [2025-12 Added] Metrics after excluding high-frequency dimensions
        # [2025-12-31 Updated] If gold is empty after removing high-frequency dimensions, skip the question (not included in statistics)
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0  # Number of questions skipped containing only high-frequency dimensions
            excl_valid_results = []  # For calculating Success@k
            for m in valid_metrics_list:  # [2025-12-31] Use filtered list
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # Check if gold dimensions remain after removing high-frequency dimensions
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    # This question only has high-frequency dimensions, skip from statistics
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, p_ex, r_ex, f1_ex = _calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                excl_valid_results.append({'recall': r_ex})

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # [2026-01 Added] Off-target and Success@k after excluding high-frequency dimensions
            excl_off_target = 1.0 - excl_micro_p
            excl_success_count = sum(1 for r in excl_valid_results if r['recall'] >= success_threshold)
            excl_success_at_k = excl_success_count / len(excl_valid_results) if excl_valid_results else 0.0

            result["exclude_high_freq"] = {
                "excluded_dims": sorted(list(high_freq_dims)),
                "micro_precision": excl_micro_p,
                "micro_recall": excl_micro_r,
                "micro_f1": excl_micro_f1,
                "total_tp": excl_total_tp,
                "total_fp": excl_total_fp,
                "total_fn": excl_total_fn,
                "skipped_only_high_freq": skipped_only_high_freq,  # Number of questions skipped containing only high-frequency dimensions
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            # [2026-01-17 Added] Bootstrap confidence interval after excluding high-frequency dimensions
            if len(valid_metrics_list) >= 2:
                excl_micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                excl_macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                result["exclude_high_freq"]["bootstrap_ci"] = {
                    "excl_hf_micro_f1": excl_micro_ci["micro_f1"],
                    "excl_hf_macro_f1": excl_macro_ci["macro_f1"],
                }

        return result

    ped_round_metrics = _compute_ped_round_metrics(ped_metrics_list, dim_mode="gk")
    # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
    gk_round_metrics = _compute_ped_round_metrics(gk_metrics_list, dim_mode="gk")
    cs_round_metrics = _compute_ped_round_metrics(cs_metrics_list, dim_mode="cs")

    # [2025-12-28 Added] Pedagogical metrics aggregation grouped by question type
    ped_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in ped_metrics_by_qtype.items()
        if metrics_list  # Only include question types with data
    }
    gk_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in gk_metrics_by_qtype.items()
        if metrics_list
    }
    cs_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="cs")
        for qt, metrics_list in cs_metrics_by_qtype.items()
        if metrics_list
    }

    # [2025-12-29 Added] Save iteration trigger log
    iteration_trigger_log = generation_orchestrator.save_iteration_trigger_summary()

    # [2025-12-31 Added] Print PRF statistics summary (similar to recalc_prf_with_voted_gold.py style)
    from src.shared.report_generator import print_prf_summary
    dim_mode = getattr(config.pipeline.agent1, "dimension_mode", "gk_only")

    # Select which metrics to print based on dimension mode
    if dim_mode in ("gk_only", "gk") and gk_metrics_list:
        print_prf_summary(gk_metrics_list, dim_mode="gk", title=f"Total {run_total} questions")
    elif dim_mode in ("cs_only", "cs") and cs_metrics_list:
        print_prf_summary(cs_metrics_list, dim_mode="cs", title=f"Total {run_total} questions")

    return {
        "run_total": run_total,
        "generation_success": generation_success,
        "evaluation_success": evaluation_success,
        "stage2_skipped": stage2_skipped,
        "skipped_no_dim": skipped_no_dim,  # [2025-12-31 Added] Number of questions skipped due to no corresponding dimensions
        "avg_ai_score": _avg(ai_scores),
        # [2025-12 Refactored] Pedagogical evaluation uses P/R/F1 aggregation
        "ped_round_metrics": ped_round_metrics,
        # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
        "gk_round_metrics": gk_round_metrics if gk_metrics_list else None,
        "cs_round_metrics": cs_round_metrics if cs_metrics_list else None,
        # [2025-12-28 Added] Pedagogical metrics aggregation grouped by question type
        "ped_round_metrics_by_question_type": ped_round_metrics_by_qtype if ped_round_metrics_by_qtype else None,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype if gk_round_metrics_by_qtype else None,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype if cs_round_metrics_by_qtype else None,
        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_buckets.items()},
        "question_type_distribution": question_type_counts,
        "agent_errors": agent_errors,
        "results": all_results,
        "eval_models": list(getattr(evaluation_orchestrator, "eval_model_names", None) or router.get_eval_model_names()),
        "model_family_filter": dict(getattr(evaluation_orchestrator, "model_family_filter", {}) or {}),
        # [2025-12 Added] Missing dimension aggregation
        "missing_dims_records": all_missing_dims_records,
        # [2025-12-29 Added] iteration trigger log path
        "iteration_trigger_log": iteration_trigger_log,
        # [2026-01 Added] generation_state iteration information（for good/bad case）
        "generation_states": all_generation_states,
    }


# ============================================================================
# Single question mode
# ============================================================================

def run_single_mode(config: "ExperimentConfig", unit_id: str):
    """
    [2025-12 Architecture Refactored] Single question mode.

    - Use unified run_units pipeline.
    - CLI does not perform DataLoader pre-validation.
    - Statistics only depend on schema-stable fields.
    """
    from src.shared.llm_router import LLMRouter
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator

    unit_id = str(unit_id)

    print("\n" + "=" * 80)
    print(f"[Single Mode] unit_id = {unit_id}")
    print("=" * 80)

    router = LLMRouter.from_config(config)

    # CLI drives unit_id
    config.pipeline.agent1.material_selection_strategy = "manual"

    generation_orchestrator = GenerationOrchestrator(config, llm_router=router)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router, eval_mode=getattr(config, "eval_mode", None))

    # [2025-12 Architecture Refactored] Use unified run_units
    stats = run_units(
        config=config,
        unit_ids=[unit_id],
        router=router,
        generation_orchestrator=generation_orchestrator,
        evaluation_orchestrator=evaluation_orchestrator,
    )

    # Convert to single question result format (compatible with old interface)
    result_item = stats["results"][0] if stats["results"] else {}

    result = {
        "unit_id": unit_id,
        "question_type": result_item.get("question_type"),
        "stage2_ready": result_item.get("stage1_status") == "success",
        "generation_success": result_item.get("stage1_status") == "success",
        "evaluation_success": result_item.get("stage2_status") == "success",
        "ai_score": result_item.get("ai_overall_score"),
        "pedagogical_score": result_item.get("ped_overall_score"),
        "eval_models": stats.get("eval_models", []),
        "skip_reason": result_item.get("skip_reason"),
    }

    # DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)
    return result


# ============================================================================
# [2026-01 Added] Experiment type auto-classification
# ============================================================================

def _classify_experiment_type(config: ExperimentConfig, args: argparse.Namespace) -> str:
    """
    Auto-classify experiment type, return experiment type identifier.

    Args:
        config: Experiment configuration
        args: CLI arguments

    Returns:
        Experiment type ID, such as "GOLD_DIM_DEEPSEEK", "NEG_CTRL", "TEMP_1_5", etc.
    """
    # 1. Ablation experiment
    skip_agent = config.pipeline.stage1_ablation.skip_agent
    if skip_agent == "agent2":
        return "ABLATION_SKIP_AGENT2"
    if skip_agent == "agent4":
        return "ABLATION_SKIP_AGENT4"

    # 2. Baseline experiment
    if getattr(config, "is_baseline", False):
        return "BASELINE"

    # 3. Negative control experiment
    if getattr(config, "is_negative_control", False):
        return "NEG_CTRL"

    # 4. Low-frequency experiment (supports all models, consistent with gold_dim model list)
    if getattr(config, "is_lowfreq", False):
        lowfreq_k = getattr(config, "lowfreq_k", 1)
        model_name = config.llm.model_name.lower()
        if "deepseek-chat" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_DEEPSEEK_CHAT"
        elif "doubao" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_DOUBAO"
        elif "gemini" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_GEMINI"
        elif "gpt-5-mini" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_GPT5_MINI"
        elif "openai-5" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_OPENAI_5"
        elif "qwen" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_QWEN"
        elif "deepseek" in model_name:
            return f"LOW_FREQ_{lowfreq_k}_DEEPSEEK"
        else:
            return f"LOW_FREQ_{lowfreq_k}_UNKNOWN"

    # 5. Temperature experiment (non-default temperature).
    temperature = args.temperature
    if temperature is not None and temperature != 1.0:
        if temperature == 0.5:
            return "TEMP_0_5"
        elif temperature == 1.5:
            return "TEMP_1_5"
        elif temperature == 2.0:
            return "TEMP_2_0"

    # 6. Gold-dimension validation experiment, grouped by model.
    model_name = config.llm.model_name.lower()

    if "deepseek-chat" in model_name:
        return "GOLD_DIM_DEEPSEEK_CHAT"
    if "doubao" in model_name:
        return "GOLD_DIM_DOUBAO"
    if "gemini" in model_name:
        return "GOLD_DIM_GEMINI"
    if "gpt-5-mini" in model_name:
        return "GOLD_DIM_GPT5_MINI"
    if "openai-5" in model_name:
        return "GOLD_DIM_OPENAI_5"
    if "qwen" in model_name:
        return "GOLD_DIM_QWEN"
    if "deepseek" in model_name:
        return "GOLD_DIM_DEEPSEEK"

    return "UNKNOWN"


# ============================================================================
# Resume result merge.
# ============================================================================

def _merge_resume_results(
    config: ExperimentConfig,
    new_stats: Dict[str, Any],
    start_from: int,
    all_unit_ids: List[str],
) -> Dict[str, Any]:
    """
    Merge existing results with newly generated results in resume mode.

    Args:
        config: Experiment configuration whose output_dir points to the resume directory.
        new_stats: Stats from the new run.
        start_from: Starting unit_id.
        all_unit_ids: Full unit_id list.

    Returns:
        Merged stats dict.
    """
    stage2_dir = config.output_dir / "stage2"

    # Collect existing unit results before start_from.
    existing_unit_ids = [uid for uid in all_unit_ids if int(uid) < start_from]
    print(f"[Resume] Loaded existing unit results: {len(existing_unit_ids)} (units 1-{start_from-1})")

    existing_results = []
    existing_ai_scores = []
    existing_ped_metrics = []
    existing_gk_metrics = []
    existing_cs_metrics = []
    existing_question_types = {"single-choice": 0, "essay": 0, "other": 0}

    # Pedagogical metrics grouped by question type.
    existing_ped_by_qtype = {"single-choice": [], "essay": [], "other": []}
    existing_gk_by_qtype = {"single-choice": [], "essay": [], "other": []}
    existing_cs_by_qtype = {"single-choice": [], "essay": [], "other": []}

    for uid in existing_unit_ids:
        unit_dir = stage2_dir / f"unit_{uid}"
        eval_state_path = unit_dir / "evaluation_state.json"

        if not eval_state_path.exists():
            print(f"  [WARN] unit_{uid} missing evaluation_state.json, skipping")
            continue

        try:
            with open(eval_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Extract result.
            result_item = {
                "unit_id": uid,
                "stage2_status": "success",
            }

            # Question type.
            qt = "other"
            core_input = state.get("input_record", {}) or state.get("core_input", {}) or state.get("input", {})
            if core_input:
                qt = core_input.get("question_type", "other")
            result_item["question_type"] = qt
            existing_question_types[qt] = existing_question_types.get(qt, 0) + 1

            # AI score, supporting both old and new JSON structures.
            ai_result = state.get("ai_result", {}) or (state.get("ai_eval", {}) or {}).get("result", {})
            ai_overall = ai_result.get("overall_score") if ai_result else None
            if ai_overall is not None:
                existing_ai_scores.append(float(ai_overall))
                result_item["ai_overall_score"] = float(ai_overall)

            # Pedagogical P/R/F1 metrics, supporting both old and new structures.
            ped_result = state.get("pedagogical_result", {}) or (state.get("pedagogical_eval", {}) or {}).get("result", {})
            gk_result = state.get("gk_result", {}) or (state.get("gk_eval", {}) or {}).get("result", {})
            cs_result = state.get("cs_result", {}) or (state.get("cs_eval", {}) or {}).get("result", {})

            if ped_result:
                metrics = {
                    "precision": ped_result.get("precision", 0),
                    "recall": ped_result.get("recall", 0),
                    "f1": ped_result.get("f1", 0),
                }
                existing_ped_metrics.append(metrics)
                existing_ped_by_qtype[qt].append(metrics)
                result_item["ped_metrics"] = metrics

            if gk_result:
                gk_metrics = {
                    "precision": gk_result.get("precision", 0),
                    "recall": gk_result.get("recall", 0),
                    "f1": gk_result.get("f1", 0),
                }
                existing_gk_metrics.append(gk_metrics)
                existing_gk_by_qtype[qt].append(gk_metrics)

            if cs_result:
                cs_metrics = {
                    "precision": cs_result.get("precision", 0),
                    "recall": cs_result.get("recall", 0),
                    "f1": cs_result.get("f1", 0),
                }
                existing_cs_metrics.append(cs_metrics)
                existing_cs_by_qtype[qt].append(cs_metrics)

            existing_results.append(result_item)

        except Exception as e:
            print(f"  [ERROR] Failed to load unit_{uid}: {e}")

    print(f"[Resume] Successfully loaded {len(existing_results)} existing unit results")

    # Merge results.
    all_results = existing_results + new_stats.get("results", [])
    all_ai_scores = existing_ai_scores + new_stats.get("ai_scores", [])
    all_ped_metrics = existing_ped_metrics + new_stats.get("ped_metrics_list", [])
    all_gk_metrics = existing_gk_metrics + new_stats.get("gk_metrics_list", [])
    all_cs_metrics = existing_cs_metrics + new_stats.get("cs_metrics_list", [])

    # Merge question type distribution.
    merged_question_types = dict(existing_question_types)
    for qt, count in new_stats.get("question_type_distribution", {}).items():
        merged_question_types[qt] = merged_question_types.get(qt, 0) + count

    # Merge grouped pedagogical metrics.
    merged_ped_by_qtype = {}
    merged_gk_by_qtype = {}
    merged_cs_by_qtype = {}

    for qt in ["single-choice", "essay", "other"]:
        merged_ped_by_qtype[qt] = existing_ped_by_qtype[qt] + new_stats.get("ped_metrics_by_qtype", {}).get(qt, [])
        merged_gk_by_qtype[qt] = existing_gk_by_qtype[qt] + new_stats.get("gk_metrics_by_qtype", {}).get(qt, [])
        merged_cs_by_qtype[qt] = existing_cs_by_qtype[qt] + new_stats.get("cs_metrics_by_qtype", {}).get(qt, [])

    # Recompute summary stats.
    def compute_round_metrics(metrics_list):
        if not metrics_list:
            return {"micro": {"precision": 0, "recall": 0, "f1": 0}, "macro": {"precision": 0, "recall": 0, "f1": 0}}
        precisions = [m.get("precision", 0) for m in metrics_list if m.get("precision") is not None]
        recalls = [m.get("recall", 0) for m in metrics_list if m.get("recall") is not None]
        f1s = [m.get("f1", 0) for m in metrics_list if m.get("f1") is not None]
        macro = {
            "precision": sum(precisions) / len(precisions) if precisions else 0,
            "recall": sum(recalls) / len(recalls) if recalls else 0,
            "f1": sum(f1s) / len(f1s) if f1s else 0,
        }
        return {"micro": macro.copy(), "macro": macro}

    def compute_metrics_by_qtype(metrics_by_qtype):
        result = {}
        for qt, metrics_list in metrics_by_qtype.items():
            result[qt] = compute_round_metrics(metrics_list)
        return result

    # Recompute AI average by question type.
    merged_ai_by_qtype = {}
    for qt in ["single-choice", "essay", "other"]:
        scores = [r.get("ai_overall_score") for r in all_results if r.get("question_type") == qt and r.get("ai_overall_score") is not None]
        merged_ai_by_qtype[qt] = sum(scores) / len(scores) if scores else 0

    # Build merged stats.
    merged_stats = {
        "run_total": len(all_results),
        "generation_success": len(existing_results) + new_stats.get("generation_success", 0),
        "evaluation_success": len(existing_results) + new_stats.get("evaluation_success", 0),
        "stage2_skipped": new_stats.get("stage2_skipped", 0),
        "skipped_no_dim": new_stats.get("skipped_no_dim", 0),

        "avg_ai_score": sum(all_ai_scores) / len(all_ai_scores) if all_ai_scores else 0,
        "avg_ai_score_by_question_type": merged_ai_by_qtype,

        "ped_round_metrics": compute_round_metrics(all_ped_metrics),
        "ped_round_metrics_by_question_type": compute_metrics_by_qtype(merged_ped_by_qtype),
        "gk_round_metrics_by_question_type": compute_metrics_by_qtype(merged_gk_by_qtype),
        "cs_round_metrics_by_question_type": compute_metrics_by_qtype(merged_cs_by_qtype),

        "question_type_distribution": merged_question_types,

        "eval_models": new_stats.get("eval_models", []),
        "agent_errors": new_stats.get("agent_errors", []),
        "results": all_results,

        # Keep raw metric lists for downstream use.
        "ai_scores": all_ai_scores,
        "ped_metrics_list": all_ped_metrics,
        "gk_metrics_list": all_gk_metrics,
        "cs_metrics_list": all_cs_metrics,
        "ped_metrics_by_qtype": merged_ped_by_qtype,
        "gk_metrics_by_qtype": merged_gk_by_qtype,
        "cs_metrics_by_qtype": merged_cs_by_qtype,

        # Resume metadata.
        "resume_info": {
            "is_resumed": True,
            "start_from": start_from,
            "existing_units": len(existing_results),
            "new_units": len(new_stats.get("results", [])),
        },
    }

    return merged_stats


# ============================================================================
# Full/subset mode.
# ============================================================================

def run_full_mode(
    config: ExperimentConfig,
    args: argparse.Namespace,
    subset_size: Optional[int] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    subset_file: Optional[str] = None,
    resume_dir: Optional[str] = None,
    start_from: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full/subset execution uses the same pipeline for all sizes.

    The CLI only decides the unit_id list. Execution uses the unified run_units pipeline.
    Summary stats depend only on stable schema fields.

    Resume mode:
    - resume_dir: existing experiment directory.
    - start_from: run units with id >= start_from.
    - output is written back to resume_dir.
    - final summary merges existing and new unit results.
    """
    from src.shared.data_loader import DataLoader
    from src.shared.llm_router import LLMRouter
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file

    # Resume mode.
    is_resume_mode = resume_dir is not None and start_from is not None

    if is_resume_mode:
        print("\n" + "=" * 80)
        print("[Resume Mode] Continuing experiment from specified position")
        print(f"  Target directory: {resume_dir}")
        print(f"  Starting unit: {start_from}")
        print("=" * 80)

        # Override output_dir with the existing resume directory.
        config.output_dir = Path(resume_dir)
    else:
        print("\n" + "=" * 80)
        if subset_size or subset_file:
            print("[Subset Mode] Starting subset unit_id run")
        else:
            print("[Full Mode] Starting all unit_id run")
        print("=" * 80)

    router = LLMRouter.from_config(config)

    # Decide the unit_id list here; the CLI only performs routing.
    # DataLoader is used only to fetch unit IDs, not to pre-validate Stage1.
    data_loader = DataLoader()
    mappings = data_loader.load_question_dimension_mappings()

    total_units = len(mappings)
    print(f"[Dataset] Total units: {total_units}")

    subset_result = None
    all_unit_ids = [str(m.unit_id) for m in mappings]

    # Resume mode only runs units >= start_from.
    if is_resume_mode:
        unit_ids_to_run = [uid for uid in all_unit_ids if int(uid) >= start_from]
        print(f"[Resume] Running units {start_from}-{max(int(u) for u in all_unit_ids)} ({len(unit_ids_to_run)} total)")
    elif subset_file:
        unit_ids_to_run = [str(x) for x in load_subset_from_file(Path(subset_file))]
        print(f"[Subset mode] Loaded {len(unit_ids_to_run)} unit_ids from file")
    elif subset_size:
        # Subset sampling needs materials.
        materials = data_loader.load_materials()
        subset_result = build_subset_unit_ids(
            materials=materials,
            mappings=mappings,
            subset_size=subset_size,
            seed=subset_seed,
            strategy=subset_strategy,
        )
        unit_ids_to_run = [str(x) for x in subset_result.unit_ids]
        print(f"[Subset mode] Sampled {len(unit_ids_to_run)} unit_ids (strategy={subset_strategy}, seed={subset_seed})")
    else:
        unit_ids_to_run = all_unit_ids
        print(f"[Full mode] Running {len(unit_ids_to_run)} unit_ids")

    # Unit IDs are driven by the CLI.
    config.pipeline.agent1.material_selection_strategy = "manual"

    generation_orchestrator = GenerationOrchestrator(config, llm_router=router)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router, eval_mode=getattr(config, "eval_mode", None))

    # Use the unified run_units pipeline.
    stats = run_units(
        config=config,
        unit_ids=unit_ids_to_run,
        router=router,
        generation_orchestrator=generation_orchestrator,
        evaluation_orchestrator=evaluation_orchestrator,
    )

    # Resume mode: merge existing and newly generated results.
    if is_resume_mode:
        print(f"\n[Resume] Merging existing results with newly generated results...")
        stats = _merge_resume_results(
            config=config,
            new_stats=stats,
            start_from=start_from,
            all_unit_ids=all_unit_ids,
        )
        print(f"[Resume] Merge complete, total {stats['run_total']} units")

    # Current timestamp for summary output.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "total_units": total_units,
        "run_total": stats["run_total"],
        "subset_size": stats["run_total"] if (subset_size or subset_file) else None,
        "subset_strategy": subset_strategy if subset_size else None,
        "subset_seed": subset_seed if subset_size else None,
        "subset_file": subset_file,

        "generation_success": stats["generation_success"],
        "evaluation_success": stats["evaluation_success"],
        "stage2_skipped": stats["stage2_skipped"],
        "skipped_no_dim": stats.get("skipped_no_dim", 0),

        # Generation failures are excluded from average-score calculations.
        "generation_failure": {
            "failure_count": stats["run_total"] - stats["generation_success"],
            "failure_rate": round((stats["run_total"] - stats["generation_success"]) / stats["run_total"], 3) if stats["run_total"] > 0 else 0,
            "failure_unit_ids": [r.get("unit_id") for r in stats["results"] if r.get("stage1_status") != "success"],
            "success_count": stats["generation_success"],
        },

        # Iteration statistics.
        "iteration_stats": _compute_iteration_stats(stats["results"]),

        "avg_ai_score": stats["avg_ai_score"],
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        "ped_round_metrics": stats["ped_round_metrics"],
        # Pedagogical metrics grouped by question type.
        "ped_round_metrics_by_question_type": stats.get("ped_round_metrics_by_question_type"),
        "gk_round_metrics_by_question_type": stats.get("gk_round_metrics_by_question_type"),
        "cs_round_metrics_by_question_type": stats.get("cs_round_metrics_by_question_type"),

        "avg_ai_score_by_question_type": stats["avg_ai_score_by_question_type"],

        "question_type_distribution": stats["question_type_distribution"],

        "config": {
            "generator_model": config.llm.model_name,
            "dim_mode": config.pipeline.agent1.dimension_mode,
            "prompt_level": config.pipeline.prompt_extraction.prompt_level,
            "stage1_skip_agent": config.pipeline.stage1_ablation.skip_agent,
        },
        # Full LLM config snapshot for auto-discovery workflows.
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # Experiment type markers for auto-discovery workflows.
        "experiment_type": {
            "is_baseline": getattr(config, "is_baseline", False),
            "is_negative_control": getattr(config, "is_negative_control", False),
            "is_hard_negative_control": getattr(config, "is_hard_negative_control", False),
            "is_lowfreq": getattr(config, "is_lowfreq", False),
            "lowfreq_k": getattr(config, "lowfreq_k", None),
            "is_gold_dim": not getattr(config, "is_baseline", False) and not getattr(config, "is_negative_control", False) and not getattr(config, "is_hard_negative_control", False) and not getattr(config, "is_lowfreq", False),
            "is_temperature_exp": args.temperature is not None and args.temperature != 1.0,
            "is_ablation": config.pipeline.stage1_ablation.skip_agent != "none",
            "experiment_category": _classify_experiment_type(config, args),
        },
        # Temperature config snapshot.
        "temperature": {
            "stage1": api_config.STAGE1_TEMPERATURE,
            "stage2": api_config.STAGE2_TEMPERATURE,
            "cli_override": args.temperature,
        },
        # Ablation config.
        "ablation": {
            "skip_agent": config.pipeline.stage1_ablation.skip_agent,
            "is_ablation": config.pipeline.stage1_ablation.skip_agent != "none",
        },
        # Stage1 ablation field.
        "stage1_skip_agent": config.pipeline.stage1_ablation.skip_agent,
        "eval_models": stats["eval_models"],
        "agent_errors": stats["agent_errors"],
        "timestamp": timestamp,
        "results": stats["results"],
    }

    # Add LLM retry audit info.
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM calls had retries or final failures; check network and API service status.",
        }
        print(f"\n[WARN] LLM calls had {len(retry_audit.retry_records)} retries and {len(retry_audit.failure_records)} final failures")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "All LLM calls succeeded on the first attempt; no network instability detected.",
        }

    # Calculate bottom-20% low-quality metrics.
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # Collect and print good/bad cases.
    generation_states = stats.get("generation_states", {})
    good_bad_cases = collect_good_bad_cases(
        results=stats["results"],
        generation_states=generation_states,
        max_cases=3,
    )
    summary["good_cases"] = good_bad_cases.get("good_cases", [])
    summary["bad_cases"] = good_bad_cases.get("bad_cases", [])

    # Print Good/Bad Cases before saving summary.
    print_good_bad_cases(good_bad_cases)

    # Export full Good/Bad Case showcase documents.
    if good_bad_cases.get("good_cases") or good_bad_cases.get("bad_cases"):
        try:
            from src.showcase import CaseCollector, CaseExporter, PromptHighlighter

            print("\n[Good/Bad Cases] Generating detailed showcase documents...")

            collector = CaseCollector(
                output_dir=config.output_dir,
                experiment_id=config.experiment_id,
            )
            highlighter = PromptHighlighter()
            exporter = CaseExporter(
                output_dir=config.output_dir,
                highlighter=highlighter,
            )

            # Collect full case data.
            full_good_cases = []
            for c in good_bad_cases.get("good_cases", []):
                try:
                    full_case = collector.collect_full_case(
                        unit_id=c["unit_id"],
                        case_type="good",
                        basic_info=c,
                    )
                    full_good_cases.append(full_case)
                except Exception as e:
                    print(f"    [WARN] Failed to collect good case unit_{c['unit_id']}: {e}")

            full_bad_cases = []
            for c in good_bad_cases.get("bad_cases", []):
                try:
                    full_case = collector.collect_full_case(
                        unit_id=c["unit_id"],
                        case_type="bad",
                        basic_info=c,
                    )
                    full_bad_cases.append(full_case)
                except Exception as e:
                    print(f"    [WARN] Failed to collect bad case unit_{c['unit_id']}: {e}")

            # Export.
            if full_good_cases or full_bad_cases:
                index_path = exporter.export_all_cases(
                    good_cases=full_good_cases,
                    bad_cases=full_bad_cases,
                    experiment_id=config.experiment_id,
                    prompt_level=config.pipeline.prompt_extraction.prompt_level or "C",
                )
                print(f"[Good/Bad Cases] Detailed showcase generated: {index_path}")
            else:
                print("[Good/Bad Cases] No valid data to export")

        except Exception as e:
            print(f"[Good/Bad Cases] Detailed export failed: {e}")

    summary_path = config.output_dir / "summary.json"
    # Normalize float precision to 3 decimals.
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate human-readable Markdown report.
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[Markdown Report] Generated: {md_result['md_report']}")
    except Exception as e:
        print(f"[Markdown Report] Generation failed: {e}")

    # Save missing-dimension aggregation report.
    missing_dims_records = stats.get("missing_dims_records", [])
    if missing_dims_records:
        _save_missing_dimensions_report(
            missing_dims_records=missing_dims_records,
            output_dir=config.output_dir,
            experiment_id=config.experiment_id,
        )

    if subset_result:
        subset_unit_ids_path = config.output_dir / "subset_unit_ids.json"
        subset_stats_path = config.output_dir / "subset_stats.json"
        subset_result.save_unit_ids_json(subset_unit_ids_path)
        subset_result.save_stats_json(subset_stats_path)
        print(f"\n[Subset Sampling] Audit files saved:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")

    mode_label = "Subset mode" if (subset_size or subset_file) else "Full mode"
    print(f"\n[{mode_label} Complete] Summary saved to: {summary_path}")

    # Write round_manifest if round-id is enabled.
    round_root = getattr(config, "round_root", None)
    if round_root is not None:
        manifest_record = {
            "round_id": config.round_id,
            "run_folder": config.run_folder,
            "run_id": config.experiment_id,
            "subset_size": subset_size,
            "subset_file": subset_file,
            "dim_mode": config.pipeline.agent1.dimension_mode,
            "prompt_level": config.pipeline.prompt_extraction.prompt_level,
            "generator_model": config.llm.model_name,
            "summary_path": str(Path(config.run_folder) / "summary.json"),
            "timestamp": timestamp,
            "avg_ai_score": summary["avg_ai_score"],
            # Use macro F1 as the pedagogical summary metric.
            "ped_macro_f1": summary["ped_round_metrics"]["macro"]["f1"],
        }
        append_round_manifest(round_root, manifest_record)
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)

    # Auto-extract generated questions as JSON and Markdown.
    try:
        from scripts.extract_questions import auto_extract_and_save
        print(f"\n[Question Extract] Auto-extracting all generated questions...")
        extract_result = auto_extract_and_save(config.output_dir, config.experiment_id)
        if extract_result["success"]:
            print(f"[Question Extract] Extracted {extract_result['total_questions']} questions")
            print(f"  - JSON: {extract_result['json_path']}")
            print(f"  - Markdown: {extract_result['markdown_path']}")
        else:
            print(f"[Question Extract] Extraction failed: {extract_result.get('error', 'unknown error')}")
    except Exception as e:
        print(f"[Question Extract] Extraction raised an error: {e}")

    return summary


# ============================================================================
# Baseline evaluation mode for original questions.
# ============================================================================

def run_baseline_mode(
    config: ExperimentConfig,
    unit_ids: Optional[List[str]] = None,
    *,
    subset_result: Optional[Any] = None,
    selection_metadata: Optional[Dict[str, Any]] = None,
    input_dim_source: str = "random",
) -> Dict[str, Any]:
    """
    Baseline evaluation mode: evaluate original exam questions directly.

    This mode:
    1. Uses original exam-question data from raw_material.json.
    2. Combines it with dimensions from merged_kaocha_jk_cs.json.
    3. Skips Stage1 generation.
    4. Builds Stage2Record objects directly for evaluation.
    5. Produces baseline scores for original questions.

    Args:
        config: Experiment configuration
        unit_ids: Optional unit_id list; None means all units.
        subset_result: Subset sampling result when --subset-size is used.
        selection_metadata: Unit selection metadata.
        input_dim_source: Baseline input dimension source, "random" or "gold".

    Returns the full summary report.
    """
    from src.shared.llm_router import LLMRouter
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.evaluation.experiment_stats_reporter import ExperimentStatsReporter

    # Baseline mode defaults to GK pedagogical evaluation only.
    eval_mode = getattr(config, "eval_mode", None) or "gk"

    # Infer the dimension mode used to build Stage2Record from eval_mode.
    if "gk" in eval_mode and "cs" in eval_mode:
        record_dim_mode = "gk+cs"
    elif "gk" in eval_mode:
        record_dim_mode = "gk"
    elif "cs" in eval_mode:
        record_dim_mode = "cs"
    else:
        record_dim_mode = "gk"

    # Evaluation mode label.
    eval_mode_label = {
        "ai": "AI only",
        "gk": "GK pedagogical only",
        "cs": "CS pedagogical only",
        "gk+cs": "GK + CS pedagogical",
        "ai+gk": "AI + GK pedagogical",
        "ai+cs": "AI + CS pedagogical",
        "ai+gk+cs": "AI + GK + CS pedagogical",
    }.get(eval_mode, eval_mode)

    print("\n" + "=" * 80)
    print("[Baseline evaluation mode] Direct original-question evaluation (Stage1 skipped)")
    print(f"  Eval mode: {eval_mode_label}")
    print("=" * 80)

    router = LLMRouter.from_config(config)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router, eval_mode=eval_mode)

    input_dim_source = (input_dim_source or "random").strip().lower()
    if input_dim_source not in ("random", "gold"):
        raise ValueError(f"Unsupported baseline input_dim_source: {input_dim_source}")

    # Baseline supports random input dimensions or gold input dimensions.
    baseline_evaluator = BaselineEvaluator(baseline_mode=(input_dim_source == "random"))
    if input_dim_source == "random":
        print("[Baseline] Input dimensions: random; gold dimensions: original-question dimensions")
    else:
        print("[Baseline] Input dimensions: original-question dimensions; gold dimensions: original-question dimensions (human oracle)")

    selection_metadata = dict(selection_metadata or {})
    selection_mode = selection_metadata.get("selection_mode", "full")

    # Build the unit_id list for evaluation.
    available_unit_ids = baseline_evaluator.get_all_unit_ids()
    if unit_ids:
        # Keep only valid unit_ids.
        all_unit_ids = [uid for uid in unit_ids if uid in available_unit_ids]
        if not all_unit_ids:
            print(f"[WARN] None of the specified unit_ids exist in the original-question data: {unit_ids}")
            all_unit_ids = available_unit_ids
    else:
        all_unit_ids = available_unit_ids
    total_units = len(all_unit_ids)

    print(f"[Baseline] Total original questions: {total_units}, eval mode: {eval_mode_label}")

    # Initialize stats reporter.
    stats_reporter = ExperimentStatsReporter(
        experiment_id=config.experiment_id,
        run_mode="baseline",
        config_info={
            "eval_mode": eval_mode,
            "eval_models": list(getattr(evaluation_orchestrator, "eval_model_names", None) or router.get_eval_model_names()),
        },
    )

    # Counters.
    evaluation_success = 0
    evaluation_failed = 0

    ai_scores: List[float] = []
    # Pedagogical evaluation uses P/R/F1 metrics.
    ped_metrics_list: List[Dict[str, Any]] = []
    # Independent GK/CS metric collection.
    gk_metrics_list: List[Dict[str, Any]] = []
    cs_metrics_list: List[Dict[str, Any]] = []

    # Load year mapping for grouped stats.
    from src.shared.data_loader import DataLoader
    data_loader = DataLoader()
    unit_year_mapping = data_loader.load_unit_year_mapping()

    # Group stats by question type.
    question_type_counts = {"single-choice": 0, "essay": 0, "other": 0}
    score_buckets = {
        "single-choice": {"ai": [], "ped_f1": []},
        "essay": {"ai": [], "ped_f1": []},
        "other": {"ai": [], "ped_f1": []},
    }
    ped_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    gk_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    cs_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }

    all_results: List[Dict[str, Any]] = []

    # Missing-dimension records.
    all_missing_dims_records: List[Dict[str, Any]] = []

    for i, unit_id in enumerate(all_unit_ids):
        print(f"\n>>> [{i+1}/{total_units}] Evaluating original question unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "question_type": None,
            "material_type": None,
            "ai_score": None,
            "ped_metrics": None,
            "error": None,
        }

        try:
            # Build Stage2Record directly and skip Stage1.
            stage2_record = baseline_evaluator.build_stage2_record(
                unit_id=unit_id,
                experiment_id=config.experiment_id,
                dim_mode=record_dim_mode,
            )

            if stage2_record is None:
                evaluation_failed += 1
                result_item["error"] = "failed to build Stage2Record"
                print("    [SKIP] Failed to build Stage2Record")
                all_results.append(result_item)
                continue

            # In random-dimension baseline mode, evaluate P/R/F1 against gold dimensions.
            if baseline_evaluator.is_baseline_mode():
                gold_dim_ids, gold_gk_dims, gold_cs_dims = baseline_evaluator.get_gold_dimensions(
                    unit_id=unit_id,
                    dim_mode=record_dim_mode
                )
                # Keep input dimensions for audit.
                input_dim_ids = stage2_record.core_input.dimension_ids.copy() if stage2_record.core_input.dimension_ids else []
                # Replace with gold dimensions for metric evaluation.
                stage2_record.core_input.dimension_ids = gold_dim_ids

            # Read question type and material type.
            bq = baseline_evaluator.get_baseline_question(unit_id)
            qt = bq.question_type if bq else "unknown"
            mt = bq.material_type if bq else "unknown"
            dim_count = len(bq.dimension_ids) if bq else 0

            result_item["question_type"] = qt
            result_item["material_type"] = mt

            # Run Stage2 evaluation.
            evaluation_state = evaluation_orchestrator.run(stage2_record)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["error"] = "evaluation not completed"
                print("    [FAIL] Evaluation not completed")
                all_results.append(result_item)

                stats_reporter.add_result(
                    unit_id=unit_id,
                    question_type=qt,
                    material_type=mt,
                    source="baseline",
                    dimension_count=dim_count,
                    stage1_success=True,
                    stage2_success=False,
                    error_info="evaluation not completed",
                )
                continue

            # Extract scores.
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            # Use the active pedagogical metric as the ped_metrics alias.
            if ped_metrics is None:
                if "cs" in eval_mode and "gk" not in eval_mode:
                    ped_metrics = cs_metrics
                else:
                    ped_metrics = gk_metrics

            result_item["ai_score"] = ai_score
            result_item["ped_metrics"] = ped_metrics
            result_item["gk_metrics"] = gk_metrics
            result_item["cs_metrics"] = cs_metrics

            # Extract dimension-level evaluation details for summary transparency.
            model_weights = dict(getattr(evaluation_orchestrator, "eval_model_weights", {}) or {})
            eval_details = _extract_eval_details(evaluation_state, model_weights)
            result_item["eval_details"] = eval_details

            evaluation_success += 1

            # Normalize question type for grouped stats.
            qt_normalized = qt if qt in question_type_counts else "other"
            question_type_counts[qt_normalized] += 1

            if ai_score is not None:
                ai_scores.append(ai_score)
                score_buckets[qt_normalized]["ai"].append(ai_score)
            if ped_metrics is not None:
                # Add year and question type for grouped stats.
                ped_metrics["year"] = unit_year_mapping.get(unit_id)
                ped_metrics["question_type"] = qt_normalized
                ped_metrics["unit_id"] = unit_id
                ped_metrics_list.append(ped_metrics)
                score_buckets[qt_normalized]["ped_f1"].append(ped_metrics["f1"])
                ped_metrics_by_qtype[qt_normalized].append(ped_metrics)
            # Collect independent GK/CS metrics.
            if gk_metrics is not None:
                # Add year and question type.
                gk_metrics["year"] = unit_year_mapping.get(unit_id)
                gk_metrics["question_type"] = qt_normalized
                gk_metrics["unit_id"] = unit_id
                gk_metrics_list.append(gk_metrics)
                gk_metrics_by_qtype[qt_normalized].append(gk_metrics)
            if cs_metrics is not None:
                # Add year and question type.
                cs_metrics["year"] = unit_year_mapping.get(unit_id)
                cs_metrics["question_type"] = qt_normalized
                cs_metrics["unit_id"] = unit_id
                cs_metrics_list.append(cs_metrics)
                cs_metrics_by_qtype[qt_normalized].append(cs_metrics)

            # Add to stats reporter; ped_f1 is converted to a 0-100 score.
            ped_f1_score = ped_metrics["f1"] * 100 if ped_metrics else None
            stats_reporter.add_result(
                unit_id=unit_id,
                question_type=qt,
                material_type=mt,
                ai_score=ai_score,
                pedagogical_score=ped_f1_score,
                source="baseline",
                dimension_count=dim_count,
                stage1_success=True,
                stage2_success=True,
            )

            # Collect missing-dimension records.
            missing_dims_item = _extract_missing_dimensions(evaluation_state, unit_id, qt)
            if missing_dims_item:
                all_missing_dims_records.append(missing_dims_item)

            ai_disp = f"{ai_score:.1f}" if isinstance(ai_score, (int, float)) else "N/A"
            ped_disp = f"F1={ped_metrics['f1']:.3f}" if ped_metrics else "N/A"
            print(f"    [OK] question_type={qt}, material={mt}, AI={ai_disp}, Ped={ped_disp}")

        except Exception as e:
            evaluation_failed += 1
            result_item["error"] = str(e)
            print(f"    [EXCEPTION] Evaluation exception: {e}")

            stats_reporter.add_result(
                unit_id=unit_id,
                question_type=result_item.get("question_type", "unknown"),
                material_type=result_item.get("material_type", "unknown"),
                source="baseline",
                stage1_success=True,
                stage2_success=False,
                error_info=str(e),
            )

        all_results.append(result_item)

    # Average helper.
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # Save stats report.
    stats_json_path = config.output_dir / "baseline_stats_report.json"
    stats_csv_path = config.output_dir / "baseline_stats_report.csv"
    stats_reporter.save_json(stats_json_path)
    stats_reporter.save_csv(stats_csv_path)
    stats_reporter.print_summary()

    # Build summary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get model weights.
    model_weights = router.get_eval_model_weights()

    # Compute pedagogical micro/macro P/R/F1 summary.
    def _compute_ped_round_metrics(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk", success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Compute round-level pedagogical metrics.

        [Important Note]
        This duplicates part of pedagogical_eval.aggregate_round().
        The high-frequency dimension definition is shared via src.shared.dimension_config.

        This helper keeps the round summary compatible with the existing
        baseline output schema.
        """
        from src.shared.dimension_config import get_high_freq_dims_by_mode
        from output_analysis.core.bootstrap_ci import (
            bootstrap_micro_metrics,
            bootstrap_macro_metrics_dimension_view
        )

        if not metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # Skip questions with empty gold dimensions.
        skipped_no_dims = 0
        valid_metrics_list = []
        for m in metrics_list:
            gold = m.get("gold_dimensions", [])
            if not gold:
                skipped_no_dims += 1
                continue
            valid_metrics_list.append(m)

        if not valid_metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "skipped_no_dims": skipped_no_dims,
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # Micro average: sum TP/FP/FN first, then compute P/R/F1.
        total_tp = sum(m["tp"] for m in valid_metrics_list)
        total_fp = sum(m["fp"] for m in valid_metrics_list)
        total_fn = sum(m["fn"] for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Off-target and Success@k.
        off_target = 1.0 - micro_precision
        success_count = sum(1 for m in valid_metrics_list if m.get("recall", 0) >= success_threshold)
        success_at_k = success_count / len(valid_metrics_list) if valid_metrics_list else 0.0

        # Macro average from the dimension view: compute P/R/F1 per dimension, then average.
        dim_stats = {}  # {dim_code: {"tp": 0, "fp": 0, "fn": 0}}
        for m in valid_metrics_list:
            gold = set(m.get("gold_dimensions", []))
            pred = set(m.get("predicted_dimensions", []))
            all_dims = gold | pred
            for d in all_dims:
                if d not in dim_stats:
                    dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats[d]["tp"] += 1
                elif d in pred and d not in gold:
                    dim_stats[d]["fp"] += 1
                elif d in gold and d not in pred:
                    dim_stats[d]["fn"] += 1

        # Compute P/R/F1 per dimension, then average.
        dim_p, dim_r, dim_f1 = [], [], []
        for d, st in dim_stats.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p.append(p_d)
            dim_r.append(r_d)
            dim_f1.append(f1_d)
        macro_precision = _avg(dim_p)
        macro_recall = _avg(dim_r)
        macro_f1 = _avg(dim_f1)

        result = {
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "skipped_no_dims": skipped_no_dims,
            "off_target": off_target,
            "success_at_k": success_at_k,
            "success_threshold": success_threshold,
        }

        # Bootstrap confidence intervals require at least two samples.
        if len(valid_metrics_list) >= 2:
            n_bootstrap = 1000
            bootstrap_seed = 42
            micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            result["bootstrap_ci"] = {
                "micro_precision": micro_ci["micro_precision"],
                "micro_recall": micro_ci["micro_recall"],
                "micro_f1": micro_ci["micro_f1"],
                "macro_f1": macro_ci["macro_f1"],
                "n_bootstrap": n_bootstrap,
                "ci_level": 0.95,
            }

        # Metrics after excluding high-frequency dimensions.
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0
            excl_valid_results = []
            for m in valid_metrics_list:
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # Skip if no gold dimensions remain after exclusion.
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, p_ex, r_ex, f1_ex = _calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                excl_valid_results.append({'recall': r_ex})

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # Off-target and Success@k after exclusion.
            excl_off_target = 1.0 - excl_micro_p
            excl_success_count = sum(1 for r in excl_valid_results if r['recall'] >= success_threshold)
            excl_success_at_k = excl_success_count / len(excl_valid_results) if excl_valid_results else 0.0

            result["exclude_high_freq"] = {
                "excluded_dims": sorted(list(high_freq_dims)),
                "micro_precision": excl_micro_p,
                "micro_recall": excl_micro_r,
                "micro_f1": excl_micro_f1,
                "total_tp": excl_total_tp,
                "total_fp": excl_total_fp,
                "total_fn": excl_total_fn,
                "skipped_only_high_freq": skipped_only_high_freq,
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            # Bootstrap CI after excluding high-frequency dimensions.
            if len(valid_metrics_list) >= 2:
                excl_micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                excl_macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                result["exclude_high_freq"]["bootstrap_ci"] = {
                    "excl_hf_micro_f1": excl_micro_ci["micro_f1"],
                    "excl_hf_macro_f1": excl_macro_ci["macro_f1"],
                }

        return result

    ped_round_metrics = _compute_ped_round_metrics(ped_metrics_list, dim_mode="gk")
    # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
    gk_round_metrics = _compute_ped_round_metrics(gk_metrics_list, dim_mode="gk")
    cs_round_metrics = _compute_ped_round_metrics(cs_metrics_list, dim_mode="cs")

    # Pedagogical metrics grouped by question type.
    ped_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in ped_metrics_by_qtype.items()
        if metrics_list
    }
    gk_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in gk_metrics_by_qtype.items()
        if metrics_list
    }
    cs_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="cs")
        for qt, metrics_list in cs_metrics_by_qtype.items()
        if metrics_list
    }

    summary = {
        "experiment_id": config.experiment_id,
        "run_mode": "baseline",
        "total_questions": total_units,
        "evaluation_success": evaluation_success,
        "evaluation_failed": evaluation_failed,
        "avg_ai_score": _avg(ai_scores),
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        "ped_round_metrics": ped_round_metrics,
        # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
        "gk_round_metrics": gk_round_metrics if gk_metrics_list else None,
        "cs_round_metrics": cs_round_metrics if cs_metrics_list else None,
        # Pedagogical metrics grouped by question type.
        "ped_round_metrics_by_question_type": ped_round_metrics_by_qtype if ped_round_metrics_by_qtype else None,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype if gk_round_metrics_by_qtype else None,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype if cs_round_metrics_by_qtype else None,
        # Average scores grouped by question type.
        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_buckets.items() if v["ai"]},
        "question_type_distribution": question_type_counts,
        "eval_models": list(getattr(evaluation_orchestrator, "eval_model_names", None) or router.get_eval_model_names()),
        "model_family_filter": dict(getattr(evaluation_orchestrator, "model_family_filter", {}) or {}),
        # Expose model weights.
        "model_weights": model_weights,
        "score_calculation": {
            "ai_eval": {
                "method": "weighted_average",
                "description": "AI dimension scores are weighted by dimension weights; model scores are weighted by model_weights.",
                "formula": "overall = Σ(dimension_score × dimension_weight) / Σ(dimension_weight)",
                "dimension_details": "Each dimension contains score, weight, contribution=score*weight, and level.",
                "model_details": "Each model contains average_score, weight, and contribution=avg*weight."
            },
            "pedagogical_eval": {
                "method": "hit_based_prf",
                "description": "Precision/Recall/F1 evaluation based on dimension hits.",
                "micro_average": "Compute global P/R/F1 after summing TP/FP/FN across all questions.",
                "macro_average": "Arithmetic average of per-question P/R/F1.",
                "metrics": {
                    "precision": "TP / (TP + FP): how many predicted dimensions are correct.",
                    "recall": "TP / (TP + FN): how many gold dimensions are recovered.",
                    "f1": "2 * P * R / (P + R): harmonic mean of P/R."
                }
            }
        },
        # Full LLM config snapshot for auto-discovery workflows.
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # Experiment type markers for auto-discovery workflows.
        "experiment_type": {
            "is_baseline": True,
            "is_negative_control": False,
            "is_hard_negative_control": False,
            "is_lowfreq": False,
            "lowfreq_k": None,
        },
        "selection_mode": selection_mode,
        "subset_size": total_units if selection_mode == "subset_size" else None,
        "subset_strategy": selection_metadata.get("subset_strategy") if selection_mode == "subset_size" else None,
        "subset_seed": selection_metadata.get("subset_seed") if selection_mode == "subset_size" else None,
        "subset_file": selection_metadata.get("subset_file") if selection_mode == "subset_file" else None,
        "specified_unit_ids": selection_metadata.get("specified_unit_ids") if selection_mode == "unit_id" else None,
        "exam_type": selection_metadata.get("exam_type", "all"),
        "baseline_input_dim_source": input_dim_source,
        "dimension_sources": {
            "input": "merged_mix_dimension_jk_cs.json" if input_dim_source == "random" else "merged_kaocha_jk_cs.json",
            "gold": "merged_kaocha_jk_cs.json",
        },
        # Config field kept aligned with full mode.
        "config": {
            "generator_model": None,
            "dim_mode": eval_mode,
            "prompt_level": "Baseline",
            "baseline_input_dim_source": input_dim_source,
        },
        "timestamp": timestamp,
        "results": all_results,
    }

    # Add LLM retry audit info.
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM calls had retries or final failures; check network and API service status.",
        }
        print(f"\n[WARN] LLM calls had {len(retry_audit.retry_records)} retries and {len(retry_audit.failure_records)} final failures")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "All LLM calls succeeded on the first attempt; no network instability detected.",
        }

    # Calculate bottom-20% low-quality metrics.
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # Save summary.json.
    summary_path = config.output_dir / "summary.json"
    # Normalize float precision to 3 decimals.
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    subset_unit_ids_path = config.output_dir / "subset_unit_ids.json"
    if subset_result:
        subset_stats_path = config.output_dir / "subset_stats.json"
        subset_result.save_unit_ids_json(subset_unit_ids_path)
        subset_result.save_stats_json(subset_stats_path)
        print(f"\n[Subset Sampling] Audit files saved:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")
    else:
        unit_selection_payload = {
            "selection_mode": selection_mode,
            "unit_ids": all_unit_ids,
            "subset_file": selection_metadata.get("subset_file"),
            "specified_unit_ids": selection_metadata.get("specified_unit_ids"),
            "exam_type": selection_metadata.get("exam_type", "all"),
            "baseline_input_dim_source": input_dim_source,
        }
        with open(subset_unit_ids_path, "w", encoding="utf-8") as f:
            json.dump(unit_selection_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Unit Selection] Saved unit_id list for this run: {subset_unit_ids_path}")

    # Generate human-readable Markdown report.
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[Markdown Report] Generated: {md_result['md_report']}")
    except Exception as e:
        print(f"[Markdown Report] Generation failed: {e}")

    # Save missing-dimension aggregation report.
    if all_missing_dims_records:
        _save_missing_dimensions_report(
            missing_dims_records=all_missing_dims_records,
            output_dir=config.output_dir,
            experiment_id=config.experiment_id,
        )

    # Print PRF statistics summary.
    from src.shared.report_generator import print_prf_summary

    # Print metrics selected by eval_mode.
    if "gk" in eval_mode and gk_metrics_list:
        print_prf_summary(gk_metrics_list, dim_mode="gk", title=f"Baseline Evaluation - Total {total_units} questions")
    if "cs" in eval_mode and cs_metrics_list:
        print_prf_summary(cs_metrics_list, dim_mode="cs", title=f"Baseline Evaluation - Total {total_units} questions")

    print(f"\n[Baseline Evaluation Complete] Summary saved to: {summary_path}")

    return summary


# ============================================================================
# [2025-12 Added] questionsextract mode
# ============================================================================

def run_extract_mode(args):
    """
    Extract generated questions from an experiment directory.

    Examples:
        python run.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427
        python run.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427 --unit-id 10
        python run.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427 --extract-format markdown --extract-output questions.md
    """
    from scripts.extract_questions import (
        extract_questions,
        format_question_text,
        format_question_markdown,
        print_summary,
    )

    exp_dir = Path(args.extract_dir)
    if not exp_dir.exists():
        print(f"[ERROR] Directory not found: {exp_dir}")
        sys.exit(1)

    # Check whether the stage2 directory exists.
    stage2_dir = exp_dir / "stage2"
    if not stage2_dir.exists():
        print(f"[ERROR] stage2 directory not found: {stage2_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Question extract mode")
    print("=" * 60)
    print(f"  Experiment directory: {exp_dir}")
    print(f"  Output format:        {args.extract_format}")
    if args.unit_id:
        print(f"  Unit:                 {args.unit_id}")
    if args.extract_output:
        print(f"  Output file:          {args.extract_output}")
    print("=" * 60 + "\n")

    try:
        # Parse unit_id.
        unit_id = None
        if args.unit_id:
            try:
                unit_id = int(args.unit_id)
            except ValueError:
                print(f"[ERROR] unit-id must be numeric: {args.unit_id}")
                sys.exit(1)

        # Extract questions.
        questions = extract_questions(exp_dir, unit_id)

        if not questions:
            print("[INFO] No questions found")
            return

        # Print stats.
        print_summary(questions)

        # Format output.
        include_scores = not args.no_scores

        if args.extract_format == "json":
            output = json.dumps(questions, ensure_ascii=False, indent=2)
        elif args.extract_format == "markdown":
            output = "\n".join(format_question_markdown(q, include_scores) for q in questions)
        else:  # text
            output = "\n".join(format_question_text(q, include_scores) for q in questions)

        # Output.
        if args.extract_output:
            output_path = Path(args.extract_output)
            # Ensure output directory exists.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"[SUCCESS] Saved to: {output_path}")
        else:
            # Handle Windows terminal encoding issues.
            try:
                print(output)
            except UnicodeEncodeError:
                # Retry after replacing special characters.
                safe_output = (
                    output
                    .replace("\u2713", "[v]")
                    .replace("\u2717", "[x]")
                    .replace("\u2705", "[PASS]")
                    .replace("\u274c", "[FAIL]")
                    .replace("\u23f3", "[...]")
                )
                print(safe_output)

    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Stage1-only mode.
# ============================================================================

def run_stage1_only_mode(
    config: ExperimentConfig,
    subset_size: Optional[int] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    subset_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Stage1 generation only, without Stage2 evaluation.

    Useful when Stage1 and Stage2 need different network environments.

    Output structure:
    - stage2/ contains per-question generation_state.json files with stage2_record.
    - stage1_summary.json records the Stage1 run summary.
    - subset_unit_ids.json stores unit_ids for stage2-only mode.
    """
    from src.shared.data_loader import DataLoader
    from src.shared.llm_router import LLMRouter
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file

    print("\n" + "=" * 80)
    print("[Stage1-Only Mode] Run question generation only (Stage2 skipped)")
    if subset_size or subset_file:
        print("[Subset mode] Starting subset unit_id run")
    else:
        print("[Full mode] Starting all unit_id run")
    print("=" * 80)

    router = LLMRouter.from_config(config)

    # ---------- decide unit_id list ----------
    data_loader = DataLoader()
    mappings = data_loader.load_question_dimension_mappings()

    total_units = len(mappings)
    print(f"[Dataset] Total units: {total_units}")

    subset_result = None

    if subset_file:
        unit_ids_to_run = [str(x) for x in load_subset_from_file(Path(subset_file))]
        print(f"[Subset mode] Loaded {len(unit_ids_to_run)} unit_ids from file")
    elif subset_size:
        materials = data_loader.load_materials()
        subset_result = build_subset_unit_ids(
            materials=materials,
            mappings=mappings,
            subset_size=subset_size,
            seed=subset_seed,
            strategy=subset_strategy,
        )
        unit_ids_to_run = [str(x) for x in subset_result.unit_ids]
        print(f"[Subset mode] Sampled {len(unit_ids_to_run)} unit_ids (strategy={subset_strategy}, seed={subset_seed})")
    else:
        unit_ids_to_run = [str(m.unit_id) for m in mappings]
        print(f"[Full mode] Running {len(unit_ids_to_run)} unit_ids")

    # Unit IDs are driven by the CLI.
    config.pipeline.agent1.material_selection_strategy = "manual"

    generation_orchestrator = GenerationOrchestrator(config, llm_router=router)

    # Counters.
    generation_success = 0
    generation_failed = 0
    stage2_record_count = 0

    question_type_counts = {"single-choice": 0, "essay": 0, "other": 0}
    agent_errors = {"agent1": 0, "agent2": 0, "agent3": 0, "agent4": 0, "agent5": 0}

    all_results: List[Dict[str, Any]] = []
    run_total = len(unit_ids_to_run)

    for i, uid in enumerate(unit_ids_to_run):
        uid = str(uid)
        print(f"\n>>> [{i+1}/{run_total}] Processing unit_id={uid}...")

        result_item = {
            "unit_id": uid,
            "stage1_status": "pending",
            "question_type": None,
            "has_stage2_record": False,
            "dim_mode": config.pipeline.agent1.dimension_mode,
            "skip_reason": None,
        }

        try:
            # -------- Stage1 only --------
            generation_state = generation_orchestrator.run_single(unit_id=uid)
            stage2_record = getattr(generation_state, "stage2_record", None)

            if stage2_record is None:
                generation_failed += 1
                result_item["stage1_status"] = "no_stage2_record"
                result_item["has_stage2_record"] = False

                # Count agent errors.
                if not getattr(generation_state, "agent1_success", False):
                    agent_errors["agent1"] += 1
                elif not getattr(generation_state, "agent2_success", False):
                    agent_errors["agent2"] += 1
                elif not getattr(generation_state, "agent3_success", False):
                    agent_errors["agent3"] += 1
                elif not getattr(generation_state, "agent4_success", False):
                    agent_errors["agent4"] += 1
                elif not getattr(generation_state, "agent5_success", False):
                    agent_errors["agent5"] += 1
                else:
                    agent_errors["agent5"] += 1

                print("    [SKIP] Stage1 did not produce stage2_record")
                all_results.append(result_item)
                continue

            generation_success += 1
            stage2_record_count += 1
            result_item["stage1_status"] = "success"
            result_item["has_stage2_record"] = True

            # Extract question type.
            qt = getattr(stage2_record.core_input, "question_type", None) or "other"
            if qt not in question_type_counts:
                qt = "other"
            question_type_counts[qt] += 1
            result_item["question_type"] = qt

            # Extract dimension info.
            dimension_ids = getattr(stage2_record.core_input, "dimension_ids", []) or []
            result_item["dimension_ids"] = dimension_ids
            result_item["dimension_count"] = len(dimension_ids)

            print(f"    [OK] question_type={qt}, dimensions={len(dimension_ids)}, dim_mode={config.pipeline.agent1.dimension_mode}")

        except Exception as e:
            print(f"    [EXCEPTION] Generation exception: {e}")
            agent_errors["agent1"] += 1
            result_item["stage1_status"] = "error"
            result_item["skip_reason"] = f"stage1_exception: {e}"
            generation_failed += 1

        all_results.append(result_item)

    # Build summary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "run_mode": "stage1-only",
        "total_units": total_units,
        "run_total": run_total,
        "subset_size": run_total if (subset_size or subset_file) else None,
        "subset_strategy": subset_strategy if subset_size else None,
        "subset_seed": subset_seed if subset_size else None,
        "subset_file": subset_file,

        "generation_success": generation_success,
        "generation_failed": generation_failed,
        "stage2_record_count": stage2_record_count,

        "question_type_distribution": question_type_counts,

        "config": {
            "generator_model": config.llm.model_name,
            "dim_mode": config.pipeline.agent1.dimension_mode,
            "prompt_level": config.pipeline.prompt_extraction.prompt_level,
            "stage1_skip_agent": config.pipeline.stage1_ablation.skip_agent,
        },
        # Full LLM config snapshot for auto-discovery workflows.
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # Experiment type markers for auto-discovery workflows.
        "experiment_type": {
            "is_baseline": False,
            "is_negative_control": getattr(config, "is_negative_control", False),
            "is_hard_negative_control": getattr(config, "is_hard_negative_control", False),
            "is_lowfreq": getattr(config, "is_lowfreq", False),
            "lowfreq_k": getattr(config, "lowfreq_k", None),
        },

        "agent_errors": agent_errors,
        "timestamp": timestamp,
        "results": all_results,

        # Mark this as stage1-only output for stage2-only mode.
        "stage1_only_output": True,
        "stage2_pending": True,
    }

    # Add LLM retry audit info.
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM calls had retries or final failures; check network and API service status.",
        }
        print(f"\n[WARN] LLM calls had {len(retry_audit.retry_records)} retries and {len(retry_audit.failure_records)} final failures")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "All LLM calls succeeded on the first attempt; no network instability detected.",
        }

    # Save stage1_summary.json.
    summary_path = config.output_dir / "stage1_summary.json"
    # Normalize float precision to 3 decimals.
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate human-readable Markdown report.
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=True,
        )
        if "md_report" in md_result:
            print(f"[Markdown Report] Generated: {md_result['md_report']}")
    except Exception as e:
        print(f"[Markdown Report] Generation failed: {e}")

    # Save unit_id list for stage2-only mode.
    if subset_result:
        subset_unit_ids_path = config.output_dir / "subset_unit_ids.json"
        subset_stats_path = config.output_dir / "subset_stats.json"
        subset_result.save_unit_ids_json(subset_unit_ids_path)
        subset_result.save_stats_json(subset_stats_path)
        print(f"\n[Subset Sampling] Audit files saved:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")
    else:
        # Full mode also saves unit_ids.
        unit_ids_path = config.output_dir / "subset_unit_ids.json"
        with open(unit_ids_path, "w", encoding="utf-8") as f:
            json.dump(unit_ids_to_run, f, ensure_ascii=False, indent=2)

    mode_label = "Subset mode" if (subset_size or subset_file) else "Full mode"
    print(f"\n[{mode_label} Stage1-Only Complete] Summary saved to: {summary_path}")

    # Write round_manifest if round-id is enabled.
    round_root = getattr(config, "round_root", None)
    if round_root is not None:
        manifest_record = {
            "round_id": config.round_id,
            "run_folder": config.run_folder,
            "run_id": config.experiment_id,
            "run_mode": "stage1-only",
            "subset_size": subset_size,
            "subset_file": subset_file,
            "dim_mode": config.pipeline.agent1.dimension_mode,
            "prompt_level": config.pipeline.prompt_extraction.prompt_level,
            "generator_model": config.llm.model_name,
            "summary_path": str(Path(config.run_folder) / "stage1_summary.json"),
            "timestamp": timestamp,
            "generation_success": generation_success,
            "stage2_record_count": stage2_record_count,
            "stage2_pending": True,
        }
        append_round_manifest(round_root, manifest_record)

    return summary


# ============================================================================
# Stage1 no-dimension-prompt ablation mode.
# Reuses BaselineEvaluator to build Stage2Record objects.
# ============================================================================

def run_ablation_nodim_mode(
    config: ExperimentConfig,
    subset_size: Optional[int] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    subset_file: Optional[str] = None,
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Stage1 ablation experiment: generate directly without dimension prompts.

    Purpose: quantify the value of the Stage1 multi-agent workflow.

    Flow:
    1. Select unit_ids based on CLI selection priority.
    2. Stage1 ablation: generate one question with a simple type-specific prompt.
    3. Stage2 normal path: reuse original dimension labels via BaselineEvaluator.build_stage2_record.
    4. Replace only generated stem/options/answer points; keep dimension metadata.

    Difference from normal full mode:
    - Stage1 skips dimension matching, prompt synthesis, anchor discovery, and verification.
    - AblationGenerator generates directly.
    - Stage2 reuses the existing evaluation pipeline.

    Unit selection priority:
    1. unit_ids from CLI --unit-id
    2. subset_file
    3. subset_size
    4. full dataset
    """
    from src.shared.data_loader import DataLoader
    from src.shared.llm_router import LLMRouter
    from src.generation.ablation_generator import AblationGenerator, AblationGeneratorConfig
    from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.shared.schemas import Stage2CoreInput, Stage2Record, Stage1Meta
    from dataclasses import asdict
    import json

    # Default eval mode does not enable AI evaluation.
    eval_mode = getattr(config, "eval_mode", None) or "gk"
    eval_mode_label = {
        "ai": "AI only",
        "gk": "GK pedagogical only",
        "cs": "CS pedagogical only",
        "gk+cs": "GK + CS pedagogical",
        "ai+gk": "AI + GK pedagogical",
        "ai+cs": "AI + CS pedagogical",
        "ai+gk+cs": "AI + GK + CS pedagogical",
    }.get(eval_mode, eval_mode)

    # Infer record_dim_mode for loading dimension data.
    if "gk" in eval_mode and "cs" in eval_mode:
        record_dim_mode = "gk+cs"
    elif "gk" in eval_mode:
        record_dim_mode = "gk"
    elif "cs" in eval_mode:
        record_dim_mode = "cs"
    else:
        record_dim_mode = "gk"

    print("\n" + "=" * 80)
    print("[Ablation Mode] Stage1 no-dimension prompt generation")
    print("[Mode] Use fixed type-specific prompts, skip all Stage1 agents, and reuse original dimension labels in Stage2")
    print(f"[Eval Mode] {eval_mode_label}")
    print("=" * 80)

    router = LLMRouter.from_config(config)

    # ---------- initialize BaselineEvaluator for original question metadata and dimensions ----------
    baseline_evaluator = BaselineEvaluator()
    data_loader = DataLoader()
    mappings = data_loader.load_question_dimension_mappings()
    total_units = len(mappings)
    all_unit_ids = [str(m.unit_id) for m in mappings]
    available_unit_id_set = set(all_unit_ids)
    requested_unit_ids = [str(uid).strip() for uid in (unit_ids or []) if str(uid).strip()]
    subset_result = None
    selection_mode = "full"

    if requested_unit_ids:
        selection_mode = "unit_id"
        invalid_unit_ids = [uid for uid in requested_unit_ids if uid not in available_unit_id_set]
        if invalid_unit_ids:
            print(f"[Ablation] [WARN] Ignoring invalid unit_ids: {invalid_unit_ids}")
        unit_ids_to_run = [uid for uid in requested_unit_ids if uid in available_unit_id_set]
        if not unit_ids_to_run:
            raise ValueError("ablation-nodim received --unit-id, but no valid unit_id was found; aborting to avoid accidental full run")
        print(f"[Ablation] Using specified unit_ids: {unit_ids_to_run}")
    elif subset_file:
        selection_mode = "subset_file"
        loaded_unit_ids = [str(x) for x in load_subset_from_file(Path(subset_file))]
        invalid_unit_ids = [uid for uid in loaded_unit_ids if uid not in available_unit_id_set]
        if invalid_unit_ids:
            print(f"[Ablation] [WARN] subset_file contains invalid unit_ids, ignored: {invalid_unit_ids}")
        unit_ids_to_run = [uid for uid in loaded_unit_ids if uid in available_unit_id_set]
        if not unit_ids_to_run:
            raise ValueError(f"ablation-nodim subset_file has no valid unit_id: {subset_file}")
        print(f"[Subset mode] Loaded {len(unit_ids_to_run)} unit_ids from file")
    elif subset_size:
        selection_mode = "subset_size"
        materials = data_loader.load_materials()
        subset_result = build_subset_unit_ids(
            materials=materials,
            mappings=mappings,
            subset_size=subset_size,
            seed=subset_seed,
            strategy=subset_strategy,
        )
        unit_ids_to_run = [str(x) for x in subset_result.unit_ids]
        print(f"[Subset mode] Sampled {len(unit_ids_to_run)} unit_ids (strategy={subset_strategy}, seed={subset_seed})")
    else:
        unit_ids_to_run = all_unit_ids
        print(f"[Full mode] Running {len(unit_ids_to_run)} unit_ids")

    run_total = len(unit_ids_to_run)

    print(f"[Dataset] Total questions: {total_units}")
    print(f"[Run] Units in this run: {run_total}")

    # ---------- initialize ablation generator ----------
    gen_client = router.get_generator_client()
    ablation_config = AblationGeneratorConfig()
    generator = AblationGenerator(
        llm_client=gen_client,
        config=ablation_config,
    )

    # ---------- initialize evaluator ----------
    evaluation_orchestrator = EvaluationOrchestrator(
        config,
        llm_router=router,
        eval_mode=eval_mode,
    )

    # ---------- counters ----------
    generation_success = 0
    generation_failed = 0
    evaluation_success = 0
    evaluation_failed = 0
    question_type_distribution: Dict[str, int] = {"single-choice": 0, "essay": 0, "other": 0}
    ai_scores: List[float] = []
    ai_scores_by_qtype: Dict[str, List[float]] = {"single-choice": [], "essay": [], "other": []}
    ped_gk_results = []
    ped_cs_results = []
    ped_gk_by_qtype: Dict[str, List] = {"single-choice": [], "essay": [], "other": []}
    ped_cs_by_qtype: Dict[str, List] = {"single-choice": [], "essay": [], "other": []}

    all_results: List[Dict[str, Any]] = []

    # Load year mapping for grouped stats.
    unit_year_mapping = data_loader.load_unit_year_mapping()

    # ---------- iterate over unit_ids ----------
    for i, unit_id in enumerate(unit_ids_to_run, 1):
        print(f"\n>>> [{i}/{run_total}] Processing unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "question_type": None,
            "stage1_status": None,
            "stage2_status": None,
            "ai_overall_score": None,
            "ped_metrics": None,
            "gk_metrics": None,
            "cs_metrics": None,
            "skip_reason": None,
        }

        # Get original question metadata.
        bq = baseline_evaluator.get_baseline_question(unit_id)
        if not bq:
            generation_failed += 1
            result_item["skip_reason"] = "failed to load original question metadata"
            result_item["stage1_status"] = "skip"
            print("    [SKIP] Failed to load original question metadata")
            all_results.append(result_item)
            continue

        # Original question type controls the ablation generation prompt.
        original_qtype = bq.question_type
        material_text = bq.material_text

        result_item["question_type"] = original_qtype
        question_type_distribution[original_qtype] = question_type_distribution.get(original_qtype, 0) + 1

        # ========== Stage1 ablation: direct generation ==========
        try:
            generated_question = generator.generate(
                material=material_text,
                question_type=original_qtype,
                unit_id=str(unit_id),
            )

            if "[GENERATION_FAILED]" in generated_question.stem or "[生成失败]" in generated_question.stem:
                generation_failed += 1
                result_item["stage1_status"] = "fail"
                result_item["skip_reason"] = "generation failed"
                print("    [Stage1] Generation failed")
                all_results.append(result_item)
                continue

            generation_success += 1
            result_item["stage1_status"] = "success"
            print(f"    [Stage1] Generation succeeded: stem length={len(generated_question.stem)}")

        except Exception as e:
            generation_failed += 1
            result_item["stage1_status"] = "error"
            result_item["skip_reason"] = f"stage1_exception: {e}"
            print(f"    [Stage1] Exception: {e}")
            all_results.append(result_item)
            continue

        # ========== Stage2: reuse BaselineEvaluator to build Stage2Record ==========
        try:
            # Get the original Stage2Record with correct dimension metadata.
            original_stage2_record = baseline_evaluator.build_stage2_record(
                unit_id=str(unit_id),
                experiment_id=config.experiment_id,
                dim_mode=record_dim_mode,
            )

            if original_stage2_record is None:
                evaluation_failed += 1
                result_item["stage2_status"] = "skip"
                result_item["skip_reason"] = "failed to build Stage2Record"
                print("    [Stage2] Failed to build Stage2Record")
                all_results.append(result_item)
                continue

            # Replace stem/options/answer points with ablation-generated content.
            core = original_stage2_record.core_input

            # Build options list.
            options_list = None
            if generated_question.options:
                options_list = [
                    {"label": opt.label, "content": opt.content, "is_correct": opt.is_correct}
                    for opt in generated_question.options
                ]

            # Build answer-points list.
            answer_points_list = None
            if generated_question.answer_points:
                answer_points_list = [
                    {"point": pt.point, "score": pt.score}
                    for pt in generated_question.answer_points
                ]

            # Create a new Stage2CoreInput with original dimensions and generated content.
            new_core_input = Stage2CoreInput(
                experiment_id=core.experiment_id,
                unit_id=core.unit_id,
                material_text=core.material_text,
                question_type=core.question_type,
                # Generated content.
                stem=generated_question.stem,
                explanation=generated_question.explanation or "",
                # Original dimension metadata.
                gk_dims=core.gk_dims,
                cs_dims=core.cs_dims,
                exam_skill=core.exam_skill,
                dimension_ids=core.dimension_ids,
                # Generated options/answers.
                options=options_list,
                correct_answer=generated_question.correct_answer,
                answer_points=answer_points_list,
                total_score=generated_question.total_score,
            )

            # Mark Stage1 metadata as ablation mode.
            new_stage1_meta = Stage1Meta(
                ablation_skip_agent="all",
            )

            # Build new Stage2Record.
            stage2_record = Stage2Record(
                core_input=new_core_input,
                stage1_meta=new_stage1_meta,
            )

            # Run evaluation.
            evaluation_state = evaluation_orchestrator.run(stage2_record)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["stage2_status"] = "fail"
                result_item["skip_reason"] = f"evaluation not completed: {getattr(evaluation_state, 'current_stage', 'unknown')}"
                print("    [Stage2] Evaluation not completed")
                all_results.append(result_item)
                continue

            evaluation_success += 1
            result_item["stage2_status"] = "success"

            # Extract scores.
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            # Keep field names aligned with full mode.
            result_item["ai_overall_score"] = ai_score
            # Select the active pedagogical metric by eval_mode.
            if "cs" in eval_mode and "gk" not in eval_mode:
                result_item["ped_metrics"] = cs_metrics
            else:
                result_item["ped_metrics"] = gk_metrics
            result_item["gk_metrics"] = gk_metrics
            result_item["cs_metrics"] = cs_metrics

            if ai_score is not None:
                ai_scores.append(float(ai_score))
                ai_scores_by_qtype[original_qtype].append(float(ai_score))

            if gk_metrics:
                # Add year and question type.
                gk_metrics["year"] = unit_year_mapping.get(unit_id)
                gk_metrics["question_type"] = original_qtype
                gk_metrics["unit_id"] = unit_id
                ped_gk_results.append(gk_metrics)
                ped_gk_by_qtype[original_qtype].append(gk_metrics)
            if cs_metrics:
                # Add year and question type.
                cs_metrics["year"] = unit_year_mapping.get(unit_id)
                cs_metrics["question_type"] = original_qtype
                cs_metrics["unit_id"] = unit_id
                ped_cs_results.append(cs_metrics)
                ped_cs_by_qtype[original_qtype].append(cs_metrics)

            ai_display = f"{ai_score:.1f}" if ai_score is not None else "N/A"
            gk_f1 = gk_metrics.get("f1", 0) if gk_metrics else 0
            cs_f1 = cs_metrics.get("f1", 0) if cs_metrics else 0
            print(f"    [Stage2] Success: AI={ai_display}, GK_F1={gk_f1:.2f}, CS_F1={cs_f1:.2f}")

            # evaluation_orchestrator.run() already saves evaluation_state.json.

        except Exception as e:
            evaluation_failed += 1
            result_item["stage2_status"] = "error"
            result_item["skip_reason"] = f"stage2_exception: {e}"
            import traceback
            print(f"    [Stage2] Exception: {e}")
            traceback.print_exc()

        all_results.append(result_item)

    # ---------- compute summary stats ----------
    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    def _compute_ped_round_metrics_local(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk", success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Compute round-level pedagogical metrics using the same logic as full mode.

        [Important Note]
        This duplicates part of pedagogical_eval.aggregate_round().
        The high-frequency dimension definition is shared via src.shared.dimension_config.
        """
        from src.shared.dimension_config import get_high_freq_dims_by_mode
        from output_analysis.core.bootstrap_ci import (
            bootstrap_micro_metrics,
            bootstrap_macro_metrics_dimension_view
        )

        if not metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # Skip questions with empty gold dimensions.
        skipped_no_dims = 0
        valid_metrics_list = []
        for m in metrics_list:
            gold = m.get("gold_dimensions", [])
            if not gold:
                skipped_no_dims += 1
                continue
            valid_metrics_list.append(m)

        if not valid_metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "skipped_no_dims": skipped_no_dims,
                "off_target": 0.0,
                "success_at_k": 0.0,
                "success_threshold": success_threshold,
            }

        # Micro average: sum TP/FP/FN first, then compute P/R/F1.
        total_tp = sum(m.get("tp", 0) for m in valid_metrics_list)
        total_fp = sum(m.get("fp", 0) for m in valid_metrics_list)
        total_fn = sum(m.get("fn", 0) for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Off-target and Success@k.
        off_target = 1.0 - micro_precision
        success_count = sum(1 for m in valid_metrics_list if m.get("recall", 0) >= success_threshold)
        success_at_k = success_count / len(valid_metrics_list) if valid_metrics_list else 0.0

        # Macro average from the dimension view: compute P/R/F1 per dimension, then average.
        dim_stats = {}
        for m in valid_metrics_list:
            gold = set(m.get("gold_dimensions", []))
            pred = set(m.get("predicted_dimensions", []))
            all_dims = gold | pred
            for d in all_dims:
                if d not in dim_stats:
                    dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats[d]["tp"] += 1
                elif d in pred and d not in gold:
                    dim_stats[d]["fp"] += 1
                elif d in gold and d not in pred:
                    dim_stats[d]["fn"] += 1

        dim_p, dim_r, dim_f1 = [], [], []
        for d, st in dim_stats.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p.append(p_d)
            dim_r.append(r_d)
            dim_f1.append(f1_d)
        macro_precision = _avg(dim_p)
        macro_recall = _avg(dim_r)
        macro_f1 = _avg(dim_f1)

        result = {
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "skipped_no_dims": skipped_no_dims,
            # Off-target and Success@k.
            "off_target": off_target,
            "success_at_k": success_at_k,
            "success_threshold": success_threshold,
        }

        # Bootstrap confidence intervals require at least two samples.
        if len(valid_metrics_list) >= 2:
            n_bootstrap = 1000
            bootstrap_seed = 42
            micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            result["bootstrap_ci"] = {
                "micro_precision": micro_ci["micro_precision"],
                "micro_recall": micro_ci["micro_recall"],
                "micro_f1": micro_ci["micro_f1"],
                "macro_f1": macro_ci["macro_f1"],
                "n_bootstrap": n_bootstrap,
                "ci_level": 0.95,
            }

        # Metrics after excluding high-frequency dimensions.
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0
            excl_valid_results = []
            for m in valid_metrics_list:
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # Skip if no gold dimensions remain after exclusion.
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, _, r_ex, _ = _calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                excl_valid_results.append({'recall': r_ex})

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # Off-target and Success@k after exclusion.
            excl_off_target = 1.0 - excl_micro_p
            excl_success_count = sum(1 for r in excl_valid_results if r['recall'] >= success_threshold)
            excl_success_at_k = excl_success_count / len(excl_valid_results) if excl_valid_results else 0.0

            result["exclude_high_freq"] = {
                "excluded_dims": sorted(list(high_freq_dims)),
                "micro_precision": excl_micro_p,
                "micro_recall": excl_micro_r,
                "micro_f1": excl_micro_f1,
                "total_tp": excl_total_tp,
                "total_fp": excl_total_fp,
                "total_fn": excl_total_fn,
                "skipped_only_high_freq": skipped_only_high_freq,
                # Off-target and Success@k.
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            # Bootstrap CI after excluding high-frequency dimensions.
            if len(valid_metrics_list) >= 2:
                excl_micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                excl_macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                result["exclude_high_freq"]["bootstrap_ci"] = {
                    "excl_hf_micro_f1": excl_micro_ci["micro_f1"],
                    "excl_hf_macro_f1": excl_macro_ci["macro_f1"],
                }

        return result

    avg_ai_score = _avg(ai_scores)
    avg_ai_by_qtype = {
        qt: _avg(scores)
        for qt, scores in ai_scores_by_qtype.items()
        if scores
    }

    # Use the same micro/macro calculation logic as full mode.
    gk_round_metrics = _compute_ped_round_metrics_local(ped_gk_results, dim_mode="gk")
    cs_round_metrics = _compute_ped_round_metrics_local(ped_cs_results, dim_mode="cs")

    gk_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics_local(results, dim_mode="gk")
        for qt, results in ped_gk_by_qtype.items()
        if results
    }
    cs_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics_local(results, dim_mode="cs")
        for qt, results in ped_cs_by_qtype.items()
        if results
    }

    # Current timestamp for summary output.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "ablation_mode": "nodim",

        "total_units": total_units,
        "run_total": run_total,
        "selection_mode": selection_mode,
        "subset_size": run_total if (subset_size or subset_file) else None,
        "subset_strategy": subset_strategy if subset_size else None,
        "subset_seed": subset_seed if subset_size else None,
        "subset_file": subset_file,
        "specified_unit_ids": unit_ids_to_run if requested_unit_ids else None,
        "generation_success": generation_success,
        "generation_failed": generation_failed,
        "evaluation_success": evaluation_success,
        "evaluation_failed": evaluation_failed,
        "stage2_skipped": generation_failed,

        "avg_ai_score": avg_ai_score,
        "avg_ai_score_by_question_type": avg_ai_by_qtype,

        # Select the active pedagogical metrics according to eval_mode.
        "ped_round_metrics": cs_round_metrics if ("cs" in eval_mode and "gk" not in eval_mode) else gk_round_metrics,
        "gk_round_metrics": gk_round_metrics,
        "cs_round_metrics": cs_round_metrics,

        "ped_round_metrics_by_question_type": cs_round_metrics_by_qtype if ("cs" in eval_mode and "gk" not in eval_mode) else gk_round_metrics_by_qtype,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype,

        "question_type_distribution": question_type_distribution,

        "config": {
            "generator_model": config.llm.model_name,
            "eval_mode": eval_mode,
            "dim_mode": record_dim_mode,
            "prompt_level": "N/A (ablation-nodim; fixed type-specific prompt)",
            "stage1_skip_agent": "all",
        },
        # Full LLM config snapshot for auto-discovery workflows.
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # Experiment type markers for auto-discovery workflows.
        "experiment_type": {
            "is_baseline": False,
            "is_negative_control": False,
            "is_hard_negative_control": False,
            "is_lowfreq": False,
            "lowfreq_k": None,
            "is_ablation": True,
        },
        "stage1_skip_agent": "all",
        "eval_models": list(getattr(evaluation_orchestrator, "eval_model_names", None) or STAGE2_EVAL_MODELS),
        "model_family_filter": dict(getattr(evaluation_orchestrator, "model_family_filter", {}) or {}),
        "timestamp": timestamp,
        "results": all_results,
    }

    # Calculate bottom-20% low-quality metrics.
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), record_dim_mode
    )

    # Save summary.
    summary_path = config.output_dir / "summary.json"
    # Normalize float precision to 3 decimals.
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    subset_unit_ids_path = config.output_dir / "subset_unit_ids.json"
    if subset_result:
        subset_stats_path = config.output_dir / "subset_stats.json"
        subset_result.save_unit_ids_json(subset_unit_ids_path)
        subset_result.save_stats_json(subset_stats_path)
        print(f"\n[Subset Sampling] Audit files saved:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")
    else:
        unit_selection_payload = {
            "selection_mode": selection_mode,
            "unit_ids": unit_ids_to_run,
            "subset_file": subset_file,
            "specified_unit_ids": requested_unit_ids or None,
        }
        with open(subset_unit_ids_path, "w", encoding="utf-8") as f:
            json.dump(unit_selection_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Unit Selection] Saved unit_id list for this run: {subset_unit_ids_path}")

    # Print PRF statistics summary.
    from src.shared.report_generator import print_prf_summary

    # Print metrics selected by eval_mode.
    if "gk" in eval_mode and ped_gk_results:
        print_prf_summary(ped_gk_results, dim_mode="gk", title=f"ablation experiment - Total {run_total} questions")
    if "cs" in eval_mode and ped_cs_results:
        print_prf_summary(ped_cs_results, dim_mode="cs", title=f"ablation experiment - Total {run_total} questions")

    print(f"\n[Ablation Experiment] Summary saved to: {summary_path}")

    # Generate human-readable Markdown report.
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[Markdown Report] Generated: {md_result['md_report']}")
    except Exception as e:
        print(f"[Markdown Report] Generation failed: {e}")

    # Print summary selected by eval_mode.
    print("\n" + "=" * 80)
    print(f"[Ablation Experiment Complete] Summary (ablation-nodim, eval_mode={eval_mode})")
    print("=" * 80)
    print(f"  Total dataset questions: {total_units}")
    print(f"  Questions in this run: {run_total}")
    print(f"  Generation success: {generation_success}, failed: {generation_failed}")
    print(f"  Evaluation success: {evaluation_success}, failed: {evaluation_failed}")

    # Display only the evaluator groups selected by eval_mode.
    if "ai" in eval_mode:
        print()
        print("  [AI Dimension Evaluation]")
        print(f"    Average score: {avg_ai_score:.2f}")
        for qt, score in avg_ai_by_qtype.items():
            print(f"      - {qt}: {score:.2f}")

    if "gk" in eval_mode:
        print()
        print("  [GK Pedagogical Dimension Evaluation]")
        print(f"    Micro: P={gk_round_metrics['micro']['precision']:.4f}, R={gk_round_metrics['micro']['recall']:.4f}, F1={gk_round_metrics['micro']['f1']:.4f}")
        print(f"    Macro: P={gk_round_metrics['macro']['precision']:.4f}, R={gk_round_metrics['macro']['recall']:.4f}, F1={gk_round_metrics['macro']['f1']:.4f}")
        print(f"    TP={gk_round_metrics['micro']['total_tp']}, FP={gk_round_metrics['micro']['total_fp']}, FN={gk_round_metrics['micro']['total_fn']}")
        # Metrics after excluding high-frequency dimensions.
        if gk_round_metrics.get("exclude_high_freq"):
            excl = gk_round_metrics["exclude_high_freq"]
            print(f"    After excluding high-frequency dimensions {excl['excluded_dims']}:")
            print(f"      Micro: P={excl['micro_precision']:.4f}, R={excl['micro_recall']:.4f}, F1={excl['micro_f1']:.4f}")

    if "cs" in eval_mode:
        print()
        print("  [CS Pedagogical Dimension Evaluation]")
        print(f"    Micro: P={cs_round_metrics['micro']['precision']:.4f}, R={cs_round_metrics['micro']['recall']:.4f}, F1={cs_round_metrics['micro']['f1']:.4f}")
        print(f"    Macro: P={cs_round_metrics['macro']['precision']:.4f}, R={cs_round_metrics['macro']['recall']:.4f}, F1={cs_round_metrics['macro']['f1']:.4f}")
        print(f"    TP={cs_round_metrics['micro']['total_tp']}, FP={cs_round_metrics['micro']['total_fp']}, FN={cs_round_metrics['micro']['total_fn']}")
        # Metrics after excluding high-frequency dimensions.
        if cs_round_metrics.get("exclude_high_freq"):
            excl = cs_round_metrics["exclude_high_freq"]
            print(f"    After excluding high-frequency dimensions {excl['excluded_dims']}:")
            print(f"      Micro: P={excl['micro_precision']:.4f}, R={excl['micro_recall']:.4f}, F1={excl['micro_f1']:.4f}")

    print("=" * 80)

    return summary


def _save_ablation_unit_result(
    output_dir: Path,
    unit_id: str,
    baseline_question,
    generated_question: "GeneratedQuestion",
    evaluation_state: "EvaluationPipelineState",
):
    """Save one ablation unit result."""
    import json
    from dataclasses import asdict

    stage2_dir = output_dir / "stage2" / f"unit_{unit_id}"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Extract evaluation result.
    ai_result = None
    gk_result = None
    cs_result = None

    if hasattr(evaluation_state, "ai_eval_result") and evaluation_state.ai_eval_result:
        ai_res = evaluation_state.ai_eval_result
        if isinstance(ai_res, dict):
            ai_result = ai_res
        elif hasattr(ai_res, "model_dump"):
            ai_result = ai_res.model_dump()
        else:
            ai_result = {"overall_score": getattr(ai_res, "overall_score", None)}

    if hasattr(evaluation_state, "gk_eval_result") and evaluation_state.gk_eval_result:
        gk_res = evaluation_state.gk_eval_result
        if isinstance(gk_res, dict):
            gk_result = gk_res
        else:
            gk_result = {
                "f1": getattr(gk_res, "f1", None),
                "precision": getattr(gk_res, "precision", None),
                "recall": getattr(gk_res, "recall", None),
            }

    if hasattr(evaluation_state, "cs_eval_result") and evaluation_state.cs_eval_result:
        cs_res = evaluation_state.cs_eval_result
        if isinstance(cs_res, dict):
            cs_result = cs_res
        else:
            cs_result = {
                "f1": getattr(cs_res, "f1", None),
                "precision": getattr(cs_res, "precision", None),
                "recall": getattr(cs_res, "recall", None),
            }

    result = {
        "unit_id": unit_id,
        "ablation_mode": True,
        "original_question": {
            "stem": baseline_question.stem,
            "question_type": baseline_question.question_type,
        },
        "generated_question": asdict(generated_question),
        "evaluation": {
            "current_stage": getattr(evaluation_state, "current_stage", None),
            "ai_eval_result": ai_result,
            "gk_eval_result": gk_result,
            "cs_eval_result": cs_result,
        },
    }

    result_path = stage2_dir / "evaluation_state.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _dict_to_generated_question(d: Dict) -> "GeneratedQuestion":
    """Convert a dict to a GeneratedQuestion object."""
    from src.shared.schemas import GeneratedQuestion, OptionItem, AnswerPoint

    options = None
    if d.get("options"):
        options = [
            OptionItem(
                label=opt.get("label", ""),
                content=opt.get("content", ""),
                is_correct=opt.get("is_correct", False),
            )
            for opt in d["options"]
        ]

    answer_points = None
    if d.get("answer_points"):
        answer_points = [
            AnswerPoint(
                point=pt.get("point", ""),
                score=pt.get("score", 0),
            )
            for pt in d["answer_points"]
        ]

    return GeneratedQuestion(
        stem=d.get("stem", ""),
        question_type=d.get("question_type", ""),
        options=options,
        correct_answer=d.get("correct_answer"),
        answer_points=answer_points,
        total_score=d.get("total_score"),
        explanation=d.get("explanation", ""),
        material_text=d.get("material_text", ""),
    )


def _save_ablation_eval_result(output_dir: Path, unit_id: str, gen_result: Dict, eval_result: Dict):
    """Save ablation experiment evaluation result."""
    import json

    stage2_dir = output_dir / "stage2" / f"unit_{unit_id}"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "unit_id": unit_id,
        "ablation_mode": True,
        "generation": gen_result,
        "evaluation": eval_result,
    }

    result_path = stage2_dir / "evaluation_state.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _aggregate_ped_metrics(results: List[Dict]) -> Dict:
    """Aggregate pedagogical P/R/F1 metrics."""
    if not results:
        return {"micro": {}, "macro": {}}

    # Collect P/R/F1 values.
    precisions = []
    recalls = []
    f1s = []

    for r in results:
        if isinstance(r, dict):
            p = r.get("precision", 0)
            rec = r.get("recall", 0)
            f = r.get("f1", 0)
            if p is not None:
                precisions.append(float(p))
            if rec is not None:
                recalls.append(float(rec))
            if f is not None:
                f1s.append(float(f))

    # Compute macro average.
    macro = {
        "precision": sum(precisions) / len(precisions) if precisions else 0,
        "recall": sum(recalls) / len(recalls) if recalls else 0,
        "f1": sum(f1s) / len(f1s) if f1s else 0,
    }

    # Use macro as a placeholder micro until detailed TP/FP/FN is available.
    micro = macro.copy()

    return {"micro": micro, "macro": macro}


# ============================================================================
# Stage2-only mode.
# ============================================================================

def run_stage2_only_mode(
    config: ExperimentConfig,
    stage1_dir: str,
    stage2_output_dir: Optional[str] = None,
    incremental_ai: bool = False,
) -> Dict[str, Any]:
    """
    Run Stage2 evaluation only from existing Stage1 output.

    Useful when Stage1 and Stage2 need different network environments.

    Supports cross-dimension-mode evaluation by backfilling missing dimension
    metadata from the configured merged dimension file.

    Args:
        config: Experiment configuration
        stage1_dir: Stage1 output directory.
        stage2_output_dir: Optional Stage2 output directory. If omitted, results are appended in stage1_dir.

    Outputs:
    - Updated evaluation_state.json files.
    - summary.json.
    """
    import shutil
    from src.shared.llm_router import LLMRouter
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.shared.schemas import Stage2Record, Stage2CoreInput, Stage1Meta

    stage1_path = Path(stage1_dir)
    stage2_src_subdir = stage1_path / "stage2"

    # If a separate output directory is specified, copy question data into it.
    if stage2_output_dir:
        output_path = Path(stage2_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stage2_subdir = output_path / "stage2"
        stage2_subdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("[Stage2-Only Mode] Evaluation only (separate output directory)")
        print(f"  Stage1 source directory: {stage1_path}")
        print(f"  Stage2 output directory: {output_path}")
        print("=" * 80)

        # Copy question data to the new directory.
        print(f"[Copy] Copying question data from {stage2_src_subdir} to {stage2_subdir}...")
        copied_count = 0
        for unit_dir in stage2_src_subdir.iterdir():
            if unit_dir.is_dir() and unit_dir.name.startswith("unit_"):
                dest_unit_dir = stage2_subdir / unit_dir.name
                dest_unit_dir.mkdir(parents=True, exist_ok=True)
                # Support both data sources.
                gen_state_src = unit_dir / "generation_state.json"
                eval_state_src = unit_dir / "evaluation_state.json"
                if gen_state_src.exists():
                    shutil.copy2(gen_state_src, dest_unit_dir / "generation_state.json")
                    copied_count += 1
                elif eval_state_src.exists():
                    # Full-mode output uses evaluation_state.json as the question data source.
                    shutil.copy2(eval_state_src, dest_unit_dir / "evaluation_state.json")
                    copied_count += 1
        print(f"[Copy] Question data copied for {copied_count} units")

        # Copy optional metadata files.
        for meta_file in ["stage1_summary.json", "subset_unit_ids.json", "subset_stats.json"]:
            src_meta = stage1_path / meta_file
            if src_meta.exists():
                shutil.copy2(src_meta, output_path / meta_file)

        # Save source info.
        source_info = {
            "source_stage1_dir": str(stage1_path),
            "eval_mode": getattr(config, "eval_mode", None),
            "experiment_id": config.experiment_id,
        }
        with open(output_path / "stage2_source_info.json", "w", encoding="utf-8") as f:
            json.dump(source_info, f, ensure_ascii=False, indent=2)
    else:
        output_path = stage1_path
        stage2_subdir = stage2_src_subdir
        print("\n" + "=" * 80)
        print("[Stage2-Only Mode] Evaluation only (read existing Stage1 output)")
        print(f"  Stage1 directory: {stage1_path}")
        print("=" * 80)

    # Read stage1_summary.json for config metadata.
    stage1_summary_path = stage1_path / "stage1_summary.json"
    stage1_summary = None
    if stage1_summary_path.exists():
        with open(stage1_summary_path, "r", encoding="utf-8") as f:
            stage1_summary = json.load(f)
        print(f"[Stage1 Info] Experiment ID: {stage1_summary.get('experiment_id')}")
        print(f"[Stage1 Info] Generation success: {stage1_summary.get('generation_success')}")
        print(f"[Stage1 Info] Stage2Record count: {stage1_summary.get('stage2_record_count')}")

    full_summary = None
    full_summary_path = stage1_path / "summary.json"
    if full_summary_path.exists():
        try:
            with open(full_summary_path, "r", encoding="utf-8") as f:
                full_summary = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read summary.json for model metadata: {e}")

    detected_generation_model = _detect_stage1_generation_model(stage1_summary, full_summary)
    if detected_generation_model:
        config.stage1_generation_model = detected_generation_model
        config.llm.model_name = detected_generation_model
        print(f"[Stage1 Info] Generation model: {detected_generation_model}")

    router = LLMRouter.from_config(config)
    # Supports incremental AI evaluation mode.
    evaluation_orchestrator = EvaluationOrchestrator(
        config,
        llm_router=router,
        eval_mode=getattr(config, "eval_mode", None),
        incremental_ai=incremental_ai
    )

    # Initialize BaselineEvaluator for cross-dimension-mode evaluation.
    # Missing dimension families can be reloaded from the configured dimension source.
    eval_mode = getattr(config, "eval_mode", None) or "gk"
    need_gk_dims = "gk" in eval_mode
    need_cs_dims = "cs" in eval_mode

    # Ablation control: read use_random_dims.
    use_random_dims = getattr(config.pipeline.stage1_ablation, "use_random_dims", False)
    dimension_source = "merged_mix_dimension_jk_cs.json (random dimensions)" if use_random_dims else "merged_kaocha_jk_cs.json (original dimensions)"

    baseline_evaluator = BaselineEvaluator(use_random_dims=use_random_dims) if (need_gk_dims or need_cs_dims) else None

    if baseline_evaluator:
        print(f"[Stage2-Only] eval_mode={eval_mode}; BaselineEvaluator initialized for dimension backfill")
        print(f"[Stage2-Only] Dimension source: {dimension_source}")

    if incremental_ai:
        print("[Stage2-Only] Incremental AI mode: skip completed units (success=True and overall_score>0)")

    # Iterate over unit directories under stage2.
    unit_dirs = sorted([d for d in stage2_subdir.iterdir() if d.is_dir() and d.name.startswith("unit_")])

    if not unit_dirs:
        print(f"[ERROR] No unit directories found under stage2: {stage2_subdir}")
        return {"error": "no_unit_dirs"}

    print(f"[Stage2-Only] Found {len(unit_dirs)} unit directories")

    # Counters.
    evaluation_success = 0
    evaluation_failed = 0

    ai_scores: List[float] = []
    # Pedagogical evaluation uses P/R/F1 metrics.
    ped_metrics_list: List[Dict[str, Any]] = []
    # Independent GK/CS metric collection.
    gk_metrics_list: List[Dict[str, Any]] = []
    cs_metrics_list: List[Dict[str, Any]] = []

    question_type_counts = {"single-choice": 0, "essay": 0, "other": 0}
    score_buckets = {
        "single-choice": {"ai": [], "ped_f1": []},
        "essay": {"ai": [], "ped_f1": []},
        "other": {"ai": [], "ped_f1": []},
    }

    # Pedagogical metrics grouped by question type.
    ped_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    gk_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }
    cs_metrics_by_qtype: Dict[str, List[Dict[str, Any]]] = {
        "single-choice": [],
        "essay": [],
        "other": [],
    }

    all_results: List[Dict[str, Any]] = []

    # Missing-dimension records.
    all_missing_dims_records: List[Dict[str, Any]] = []

    run_total = len(unit_dirs)

    for i, unit_dir in enumerate(unit_dirs):
        unit_id = unit_dir.name.replace("unit_", "")
        print(f"\n>>> [{i+1}/{run_total}] Evaluating unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "stage2_status": "pending",
            "question_type": None,
            "ai_overall_score": None,
            "ped_metrics": None,
            "error": None,
        }

        try:
            # Support two data sources: generation_state.json or evaluation_state.json.
            generation_state_path = unit_dir / "generation_state.json"
            evaluation_state_path = unit_dir / "evaluation_state.json"

            core_input_data = None
            stage1_meta_data = {}
            # Existing evaluation state for incremental append evaluation.
            existing_eval_state = None

            if generation_state_path.exists():
                # Source 1: stage1-only output.
                with open(generation_state_path, "r", encoding="utf-8") as f:
                    gen_state_data = json.load(f)
                stage2_record_data = gen_state_data.get("stage2_record")
                if stage2_record_data:
                    core_input_data = stage2_record_data.get("core_input", {})
                    stage1_meta_data = stage2_record_data.get("stage1_meta", {})

            # Always read evaluation_state.json if available to reuse existing evaluation results.
            if evaluation_state_path.exists():
                with open(evaluation_state_path, "r", encoding="utf-8") as f:
                    eval_state_data = json.load(f)
                # Keep existing evaluation state for incremental mode.
                existing_eval_state = eval_state_data
                # If core_input_data is not available yet, read it from evaluation_state.json.
                if core_input_data is None:
                    # Full-mode question data is stored in the input field.
                    core_input_data = eval_state_data.get("input", {})
                    stage1_meta_data = eval_state_data.get("stage1_meta", {})

            if not core_input_data:
                evaluation_failed += 1
                result_item["stage2_status"] = "no_question_data"
                result_item["error"] = "question data not found (generation_state.json or evaluation_state.json)"
                print("    [SKIP] Question data not found")
                all_results.append(result_item)
                continue

            # Cross-dimension-mode evaluation: check and backfill missing dimension metadata.
            gk_dims = core_input_data.get("gk_dims", {})
            cs_dims = core_input_data.get("cs_dims", {})
            dimension_ids = core_input_data.get("dimension_ids", [])

            dims_reloaded = False
            if baseline_evaluator:
                # Cross-dimension mode may require rebuilding dimension_ids entirely.
                bq = baseline_evaluator.get_baseline_question(unit_id)
                if bq:
                    # Select dimension family by eval_mode.
                    if need_cs_dims and not need_gk_dims:
                        # CS-only mode: use CS dimensions and clear GK dimensions.
                        cs_dims = bq.cs_dims
                        gk_dims = {}
                        # Filter CS dimensions from bq.dimension_ids.
                        cs_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心素养（四维）-") or d.startswith("学习任务群-") or d.startswith("语文学科能力要求-")]
                        dimension_ids = cs_dim_ids
                        dims_reloaded = True
                        print(f"    [Dimension Replace] CS-only mode: using {len(cs_dim_ids)} CS dimensions")
                    elif need_gk_dims and not need_cs_dims:
                        # GK-only mode: use GK dimensions and clear CS dimensions.
                        if not gk_dims:
                            gk_dims = bq.gk_dims
                        cs_dims = {}
                        # Filter GK dimensions from bq.dimension_ids.
                        gk_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心价值-") or d.startswith("学科素养-") or d.startswith("关键能力-") or d.startswith("必备知识-") or d.startswith("四翼要求-") or d.startswith("情境-")]
                        dimension_ids = gk_dim_ids
                        dims_reloaded = True
                        print(f"    [Dimension Replace] GK-only mode: using {len(gk_dim_ids)} GK dimensions")
                    else:
                        # GK+CS mode: backfill missing dimension families.
                        if not cs_dims:
                            cs_dims = bq.cs_dims
                            cs_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心素养（四维）-") or d.startswith("学习任务群-") or d.startswith("语文学科能力要求-")]
                            if cs_dim_ids:
                                dimension_ids = list(dimension_ids) + cs_dim_ids
                                dims_reloaded = True
                                print(f"    [Dimension Backfill] Loaded {len(cs_dim_ids)} CS dimensions from merged source")
                        if not gk_dims:
                            gk_dims = bq.gk_dims
                            gk_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心价值-") or d.startswith("学科素养-") or d.startswith("关键能力-") or d.startswith("必备知识-") or d.startswith("四翼要求-") or d.startswith("情境-")]
                            if gk_dim_ids:
                                dimension_ids = list(dimension_ids) + gk_dim_ids
                                dims_reloaded = True
                                print(f"    [Dimension Backfill] Loaded {len(gk_dim_ids)} GK dimensions from merged source")

            # Deduplicate dimension_ids.
            dimension_ids = list(dict.fromkeys(dimension_ids))

            core_input = Stage2CoreInput(
                experiment_id=core_input_data.get("experiment_id", config.experiment_id),
                unit_id=core_input_data.get("unit_id", unit_id),
                material_text=core_input_data.get("material_text", ""),
                question_type=core_input_data.get("question_type", "essay"),
                stem=core_input_data.get("stem", ""),
                explanation=core_input_data.get("explanation", ""),
                gk_dims=gk_dims,
                cs_dims=cs_dims,
                exam_skill=core_input_data.get("exam_skill", {}),
                dimension_ids=dimension_ids,
                options=core_input_data.get("options"),
                correct_answer=core_input_data.get("correct_answer"),
                answer_points=core_input_data.get("answer_points"),
                total_score=core_input_data.get("total_score"),
                # Anchor metadata.
                anchors=core_input_data.get("anchors"),
                anchor_count=core_input_data.get("anchor_count", 0),
            )

            stage1_meta = Stage1Meta(
                agent5_overall_score=stage1_meta_data.get("agent5_overall_score"),
                agent5_layer_scores=stage1_meta_data.get("agent5_layer_scores", {}),
                agent5_need_revision=stage1_meta_data.get("agent5_need_revision"),
                agent5_is_reject=stage1_meta_data.get("agent5_is_reject"),
                agent5_issue_types=stage1_meta_data.get("agent5_issue_types", []),
                ablation_skip_agent=stage1_meta_data.get("ablation_skip_agent", "none"),
            )

            stage2_record = Stage2Record(
                core_input=core_input,
                stage1_meta=stage1_meta,
            )

            # Get question type.
            qt = core_input.question_type or "other"
            if qt not in question_type_counts:
                qt = "other"
            question_type_counts[qt] += 1
            result_item["question_type"] = qt

            # Extract anchor info when available.
            anchor_count = core_input.anchor_count or 0
            anchors = core_input.anchors
            result_item["anchor_count"] = anchor_count
            if anchors:
                result_item["anchors"] = anchors

            # Show existing evaluation status for incremental mode.
            if existing_eval_state:
                ai_done = existing_eval_state.get("ai_eval", {}).get("success", False)
                ped_done = existing_eval_state.get("pedagogical_eval", {}).get("success", False)
                gk_done = existing_eval_state.get("gk_eval", {}).get("success", False)
                cs_done = existing_eval_state.get("cs_eval", {}).get("success", False)
                print(f"    [Existing Evaluation] AI={ai_done}, Ped={ped_done}, GK={gk_done}, CS={cs_done}")

            # Run Stage2 evaluation; pass existing state for incremental append.
            evaluation_state = evaluation_orchestrator.run(stage2_record, existing_eval_state=existing_eval_state)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["stage2_status"] = "fail"
                result_item["error"] = "evaluation not completed"
                print("    [FAIL] Evaluation not completed")
                all_results.append(result_item)
                continue

            evaluation_success += 1
            result_item["stage2_status"] = "success"

            # Extract scores.
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            result_item["ai_overall_score"] = ai_score
            result_item["ped_metrics"] = ped_metrics
            result_item["gk_metrics"] = gk_metrics
            result_item["cs_metrics"] = cs_metrics

            if ai_score is not None:
                ai_scores.append(ai_score)
                score_buckets[qt]["ai"].append(ai_score)
            if ped_metrics is not None:
                ped_metrics_list.append(ped_metrics)
                score_buckets[qt]["ped_f1"].append(ped_metrics["f1"])
                ped_metrics_by_qtype[qt].append(ped_metrics)
            # Collect independent GK/CS metrics.
            if gk_metrics is not None:
                gk_metrics_list.append(gk_metrics)
                gk_metrics_by_qtype[qt].append(gk_metrics)
            if cs_metrics is not None:
                cs_metrics_list.append(cs_metrics)
                cs_metrics_by_qtype[qt].append(cs_metrics)

            # Collect missing-dimension records.
            missing_dims_item = _extract_missing_dimensions(evaluation_state, unit_id, qt)
            if missing_dims_item:
                all_missing_dims_records.append(missing_dims_item)

            ai_disp = f"{ai_score:.1f}" if isinstance(ai_score, (int, float)) else "N/A"
            ped_disp = f"F1={ped_metrics['f1']:.3f}" if ped_metrics else "N/A"
            print(f"    [OK] question_type={qt}, AI={ai_disp}, Ped={ped_disp}")

        except Exception as e:
            evaluation_failed += 1
            result_item["stage2_status"] = "error"
            result_item["error"] = str(e)
            print(f"    [EXCEPTION] Evaluation exception: {e}")

        all_results.append(result_item)

    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    # Compute pedagogical micro/macro P/R/F1 summary.
    def _compute_ped_round_metrics(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk") -> Dict[str, Any]:
        """
        Compute round-level pedagogical metrics.

        [Important Note]
        This duplicates part of pedagogical_eval.aggregate_round().
        The high-frequency dimension definition is shared via src.shared.dimension_config.
        """
        from src.shared.dimension_config import get_high_freq_dims_by_mode
        from output_analysis.core.bootstrap_ci import (
            bootstrap_micro_metrics,
            bootstrap_macro_metrics_dimension_view
        )

        if not metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # Skip questions with empty gold dimensions.
        skipped_no_dims = 0
        valid_metrics_list = []
        for m in metrics_list:
            gold = m.get("gold_dimensions", [])
            if not gold:
                skipped_no_dims += 1
                continue
            valid_metrics_list.append(m)

        if not valid_metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "skipped_no_dims": skipped_no_dims,
            }

        total_tp = sum(m["tp"] for m in valid_metrics_list)
        total_fp = sum(m["fp"] for m in valid_metrics_list)
        total_fn = sum(m["fn"] for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Macro average from the dimension view: compute P/R/F1 per dimension, then average.
        dim_stats = {}
        for m in valid_metrics_list:
            gold = set(m.get("gold_dimensions", []))
            pred = set(m.get("predicted_dimensions", []))
            all_dims = gold | pred
            for d in all_dims:
                if d not in dim_stats:
                    dim_stats[d] = {"tp": 0, "fp": 0, "fn": 0}
                if d in gold and d in pred:
                    dim_stats[d]["tp"] += 1
                elif d in pred and d not in gold:
                    dim_stats[d]["fp"] += 1
                elif d in gold and d not in pred:
                    dim_stats[d]["fn"] += 1

        dim_p, dim_r, dim_f1 = [], [], []
        for d, st in dim_stats.items():
            tp_d, fp_d, fn_d = st["tp"], st["fp"], st["fn"]
            p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
            r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
            f1_d = 2 * p_d * r_d / (p_d + r_d) if (p_d + r_d) > 0 else 0.0
            dim_p.append(p_d)
            dim_r.append(r_d)
            dim_f1.append(f1_d)
        macro_precision = _avg(dim_p)
        macro_recall = _avg(dim_r)
        macro_f1 = _avg(dim_f1)

        result = {
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1, "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn},
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
            "skipped_no_dims": skipped_no_dims,
        }

        # Bootstrap confidence intervals require at least two samples.
        if len(valid_metrics_list) >= 2:
            n_bootstrap = 1000
            bootstrap_seed = 42
            micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=None, seed=bootstrap_seed)
            result["bootstrap_ci"] = {
                "micro_precision": micro_ci["micro_precision"],
                "micro_recall": micro_ci["micro_recall"],
                "micro_f1": micro_ci["micro_f1"],
                "macro_f1": macro_ci["macro_f1"],
                "n_bootstrap": n_bootstrap,
                "ci_level": 0.95,
            }

        # Metrics after excluding high-frequency dimensions.
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0
            for m in valid_metrics_list:
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # Skip if no gold dimensions remain after exclusion.
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, _, _, _ = _calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            result["exclude_high_freq"] = {
                "excluded_dims": sorted(list(high_freq_dims)),
                "micro_precision": excl_micro_p,
                "micro_recall": excl_micro_r,
                "micro_f1": excl_micro_f1,
                "total_tp": excl_total_tp,
                "total_fp": excl_total_fp,
                "total_fn": excl_total_fn,
                "skipped_only_high_freq": skipped_only_high_freq,
            }

            # Bootstrap CI after excluding high-frequency dimensions.
            if len(valid_metrics_list) >= 2:
                excl_micro_ci = bootstrap_micro_metrics(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                excl_macro_ci = bootstrap_macro_metrics_dimension_view(valid_metrics_list, n_bootstrap, exclude_dims=high_freq_dims, seed=bootstrap_seed)
                result["exclude_high_freq"]["bootstrap_ci"] = {
                    "excl_hf_micro_f1": excl_micro_ci["micro_f1"],
                    "excl_hf_macro_f1": excl_macro_ci["macro_f1"],
                }

        return result

    ped_round_metrics = _compute_ped_round_metrics(ped_metrics_list, dim_mode="gk")
    # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
    gk_round_metrics = _compute_ped_round_metrics(gk_metrics_list, dim_mode="gk")
    cs_round_metrics = _compute_ped_round_metrics(cs_metrics_list, dim_mode="cs")

    # Pedagogical metrics grouped by question type.
    ped_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in ped_metrics_by_qtype.items()
        if metrics_list
    }
    gk_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="gk")
        for qt, metrics_list in gk_metrics_by_qtype.items()
        if metrics_list
    }
    cs_round_metrics_by_qtype = {
        qt: _compute_ped_round_metrics(metrics_list, dim_mode="cs")
        for qt, metrics_list in cs_metrics_by_qtype.items()
        if metrics_list
    }

    # Preserve metadata from an existing summary because stage2-only is an append/update mode.
    existing_summary = None
    existing_config = {}
    existing_exp_type = {}
    existing_llm_config = {}

    summary_path = output_path / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
            existing_config = existing_summary.get("config", {})
            existing_exp_type = existing_summary.get("experiment_type", {})
            existing_llm_config = existing_summary.get("llm_config", {})
        except Exception as e:
            print(f"[WARN] Failed to read existing summary.json: {e}; defaults will be used")

    # Build summary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read original config metadata from stage1_summary.
    stage1_config = stage1_summary.get("config", {}) if stage1_summary else {}

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "run_mode": "stage2-only",
        "stage1_dir": str(stage1_path),
        "output_dir": str(output_path),
        "stage1_experiment_id": stage1_summary.get("experiment_id") if stage1_summary else None,

        "total_units": run_total,
        "run_total": run_total,

        "evaluation_success": evaluation_success,
        "evaluation_failed": evaluation_failed,

        "avg_ai_score": _avg(ai_scores),
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        "ped_round_metrics": ped_round_metrics,
        # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
        "gk_round_metrics": gk_round_metrics if gk_metrics_list else None,
        "cs_round_metrics": cs_round_metrics if cs_metrics_list else None,
        # Pedagogical metrics grouped by question type.
        "ped_round_metrics_by_question_type": ped_round_metrics_by_qtype if ped_round_metrics_by_qtype else None,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype if gk_round_metrics_by_qtype else None,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype if cs_round_metrics_by_qtype else None,

        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_buckets.items()},

        "question_type_distribution": question_type_counts,

        # Prefer existing summary config, then stage1_config, then defaults.
        "config": {
            "generator_model": existing_config.get("generator_model")
                              or stage1_config.get("generator_model", "unknown"),
            "dim_mode": existing_config.get("dim_mode")
                       or stage1_config.get("dim_mode", config.pipeline.agent1.dimension_mode),
            "prompt_level": existing_config.get("prompt_level")
                           or stage1_config.get("prompt_level", config.pipeline.prompt_extraction.prompt_level),
            "stage1_skip_agent": existing_config.get("stage1_skip_agent")
                                or stage1_config.get("stage1_skip_agent", "none"),
            # Preserve ablation control metadata.
            "use_random_dims": existing_config.get("use_random_dims", use_random_dims),
            "dimension_source": existing_config.get("dimension_source", dimension_source),
        },
        # Preserve existing Stage1 LLM metadata when possible; Stage2 uses current config.
        "llm_config": {
            "stage1_model": existing_llm_config.get("stage1_model")
                           or existing_config.get("generator_model")
                           or stage1_config.get("generator_model", api_config.STAGE1_MODEL),
            "stage1_temperature": existing_llm_config.get("stage1_temperature", api_config.STAGE1_TEMPERATURE),
            "stage1_preset": existing_llm_config.get("stage1_preset", api_config.STAGE1_PRESET),
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # Preserve existing experiment-type metadata.
        "experiment_type": {
            "is_baseline": existing_exp_type.get("is_baseline", False),
            "is_negative_control": existing_exp_type.get("is_negative_control", use_random_dims),
            "is_hard_negative_control": existing_exp_type.get("is_hard_negative_control", getattr(config, "is_hard_negative_control", False)),
            "is_lowfreq": existing_exp_type.get("is_lowfreq", getattr(config, "is_lowfreq", False)),
            "lowfreq_k": existing_exp_type.get("lowfreq_k") or getattr(config, "lowfreq_k", None),
        },

        "eval_models": list(getattr(evaluation_orchestrator, "eval_model_names", None) or router.get_eval_model_names()),
        "model_family_filter": dict(getattr(evaluation_orchestrator, "model_family_filter", {}) or {}),
        "timestamp": timestamp,
        "results": all_results,
    }

    # Add LLM retry audit info.
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM calls had retries or final failures; check network and API service status.",
        }
        print(f"\n[WARN] LLM calls had {len(retry_audit.retry_records)} retries and {len(retry_audit.failure_records)} final failures")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "All LLM calls succeeded on the first attempt; no network instability detected.",
        }

    # Calculate bottom-20% low-quality metrics.
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # Print PRF statistics summary.
    from src.shared.report_generator import print_prf_summary

    # Print metrics selected by eval_mode.
    if "gk" in eval_mode and gk_metrics_list:
        print_prf_summary(gk_metrics_list, dim_mode="gk", title=f"Stage2-Only - Total {run_total} questions")
    if "cs" in eval_mode and cs_metrics_list:
        print_prf_summary(cs_metrics_list, dim_mode="cs", title=f"Stage2-Only - Total {run_total} questions")

    # Save summary.json to output_path.
    summary_path = output_path / "summary.json"
    # Normalize float precision to 3 decimals.
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate human-readable Markdown report.
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=output_path,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[Markdown Report] Generated: {md_result['md_report']}")
    except Exception as e:
        print(f"[Markdown Report] Generation failed: {e}")

    # Save missing-dimension aggregation report.
    if all_missing_dims_records:
        _save_missing_dimensions_report(
            missing_dims_records=all_missing_dims_records,
            output_dir=output_path,
            experiment_id=config.experiment_id,
        )

    print(f"\n[Stage2-Only Complete] Summary saved to: {summary_path}")

    # Auto-extract generated questions.
    try:
        from scripts.extract_questions import auto_extract_and_save
        print(f"\n[Question Extract] Auto-extracting all generated questions...")
        extract_result = auto_extract_and_save(output_path, config.experiment_id)
        if extract_result["success"]:
            print(f"[Question Extract] Extracted {extract_result['total_questions']} questions")
            print(f"  - JSON: {extract_result['json_path']}")
            print(f"  - Markdown: {extract_result['markdown_path']}")
        else:
            print(f"[Question Extract] Extraction failed: {extract_result.get('error', 'unknown error')}")
    except Exception as e:
        print(f"[Question Extract] Extraction raised an error: {e}")

    return summary


# ============================================================================
# Main entry point.
# ============================================================================

def main():
    args = parse_arguments()

    # Extract mode runs independently and does not initialize LLM clients.
    if args.run_mode == "extract":
        run_extract_mode(args)
        return

    # Print actual runtime file paths.
    _print_runtime_files()

    # Clear LLM retry audit records so each experiment is independent.
    clear_retry_audit()

    print("\n" + "=" * 80)
    print("Experiment configuration (data-driven question type)")
    print("=" * 80)
    print(f"  Run mode:      {args.run_mode}")
    if args.run_mode == "single":
        print(f"  unit_id:      {args.unit_id}")
    elif args.run_mode == "baseline":
        print("  Mode:          Original-question baseline evaluation (Stage1 skipped)")
        # Show exam-type filter.
        exam_type = getattr(args, "exam_type", "all")
        exam_type_label = {"national": "national only", "local": "local only", "all": "all"}.get(exam_type, exam_type)
        print(f"  Exam filter:   {exam_type_label}")
        # Baseline mode defaults to GK pedagogical evaluation only.
        eval_mode = getattr(args, "eval_mode", None) or "gk"
        eval_mode_label = {
            "ai": "AI only",
            "gk": "GK pedagogical only",
            "cs": "CS pedagogical only",
            "gk+cs": "GK + CS pedagogical",
            "ai+gk": "AI + GK pedagogical",
            "ai+cs": "AI + CS pedagogical",
            "ai+gk+cs": "AI + GK + CS pedagogical",
        }.get(eval_mode, eval_mode)
        print(f"  Eval mode:     {eval_mode_label}")
        baseline_dim_source = getattr(args, "baseline_dim_source", "random") or "random"
        baseline_dim_label = "gold dimensions / human oracle" if baseline_dim_source == "gold" else "random input dimensions / original-question gold"
        print(f"  Dimension source: {baseline_dim_label}")
        if args.unit_id:
            print(f"  Unit:          {args.unit_id}")
        elif args.subset_size:
            print(f"  Subset size:   {args.subset_size}")
            print(f"  Strategy:      {args.subset_strategy}")
            print(f"  Seed:          {args.subset_seed}")
        elif args.subset_file:
            print(f"  Subset file:   {args.subset_file}")
        else:
            print("  Subset:        disabled; using full dataset")
        print(f"  Eval models:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "stage1-only":
        print("  Mode:          Stage1 generation only; evaluation skipped")
        if args.subset_size:
            print(f"  Subset size:   {args.subset_size}")
            print(f"  Strategy:      {args.subset_strategy}")
            print(f"  Seed:          {args.subset_seed}")
        elif args.subset_file:
            print(f"  Subset file:   {args.subset_file}")
        else:
            print("  Subset:        disabled; using full dataset")
    elif args.run_mode == "stage2-only":
        print("  Mode:          Stage2 evaluation only; reads existing Stage1 output")
        print(f"  Stage1 dir:    {args.stage1_dir}")
        print(f"  Eval models:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "ablation-nodim":
        print("  Mode:          Stage1 ablation experiment (no dimension prompts)")
        print("  Purpose:       test the value of the dimension-prompt system")
        if args.unit_id:
            print(f"  Unit:          {args.unit_id}")
        elif args.subset_size:
            print(f"  Subset size:   {args.subset_size}")
            print(f"  Strategy:      {args.subset_strategy}")
            print(f"  Seed:          {args.subset_seed}")
        elif args.subset_file:
            print(f"  Subset file:   {args.subset_file}")
        else:
            print("  Subset:        disabled; using full dataset")
        # Default eval mode does not enable AI evaluation.
        ablation_eval_mode = getattr(args, "eval_mode", None) or "gk"
        ablation_eval_mode_label = {
            "ai": "AI only",
            "gk": "GK pedagogical only",
            "cs": "CS pedagogical only",
            "gk+cs": "GK + CS pedagogical",
            "ai+gk": "AI + GK pedagogical",
            "ai+cs": "AI + CS pedagogical",
            "ai+gk+cs": "AI + GK + CS pedagogical",
        }.get(ablation_eval_mode, ablation_eval_mode)
        print(f"  Eval mode:     {ablation_eval_mode_label}")
        print(f"  Generator:     {args.generator_model}")
        print(f"  Eval models:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "full":
        if args.subset_size:
            print(f"  Subset size:   {args.subset_size}")
            print(f"  Strategy:      {args.subset_strategy}")
            print(f"  Seed:          {args.subset_seed}")
        elif args.subset_file:
            print(f"  Subset file:   {args.subset_file}")
        else:
            print("  Subset:        disabled; using full dataset")

    # Show generation parameters for modes that run Stage1.
    # [2026-01 Deprecated] Removed baseline-recent
    if args.run_mode not in ("baseline", "stage2-only", "ablation-nodim"):
        print(f"  Dim mode:      {args.dim_mode}")
        print(f"  Prompt level:  {args.prompt_level}")
        print(f"  Generator:     {args.generator_model}")
        if args.run_mode != "stage1-only":
            print(f"  Eval models:   {STAGE2_EVAL_MODELS}")

    # Show round-id info.
    if args.round_id:
        print(f"  round-id:     {args.round_id}")

    # Show Stage1 ablation info.
    stage1_skip = getattr(args, "stage1_skip", "none") or "none"
    if stage1_skip != "none":
        print(f"  Stage1 ablation: skip {stage1_skip}")
    print("=" * 80 + "\n")

    config = create_experiment_config(args)

    # Show final output directory.
    print(f"[Config] experiment_id: {config.experiment_id}")
    if args.round_id:
        print(f"[Config] round_id: {config.round_id}")
        print(f"[Config] run_folder: {config.run_folder}")
        print(f"[Config] round_root: {config.round_root}")
    print(f"[Config] output_dir: {config.output_dir}")
    print("")

    llm_log_dir = config.output_dir / "llm_logs"
    llm_logger = init_global_logger(
        output_dir=str(llm_log_dir),
        experiment_id=config.experiment_id,
        enabled=True,
        format="jsonl",
    )

    def _fmt_score(x):
        return f"{x:.1f}" if isinstance(x, (int, float)) else "N/A"

    if args.run_mode == "single":
        # CLI does not pre-validate DataLoader here; Stage1 decides question type.

        result = run_single_mode(config, args.unit_id)

        print("\n" + "=" * 80)
        print("Single question mode result")
        print("=" * 80)
        print(f"  experiment_id: {config.experiment_id}")
        print(f"  unit_id:       {result.get('unit_id')}")
        print(f"  question_type: {result.get('question_type')}")
        print(f"  stage1_status: {result.get('generation_success')}")
        print(f"  stage2_status: {result.get('evaluation_success')}")

        if result.get("evaluation_success"):
            print(f"  AI score:      {_fmt_score(result.get('ai_score'))}")
            print(f"  Ped score:     {_fmt_score(result.get('pedagogical_score'))}")
        else:
            if not result.get("stage2_ready"):
                skip_reason = result.get("skip_reason", "stage1_no_stage2_record")
                print(f"  Skip reason:   {skip_reason}")

        print(f"  Eval models:   {result.get('eval_models')}")
        print(f"  Output dir:    {config.output_dir}")
        print("=" * 80)

    elif args.run_mode == "baseline":
        # Baseline evaluation mode for original questions.
        # Priority: unit_id > subset_file > subset_size > exam_type/full.
        # Deprecated year_start/year_end filtering was removed.
        from src.evaluation.baseline_evaluator import BaselineEvaluator
        from src.shared.data_loader import DataLoader
        from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file

        baseline_unit_ids = None
        subset_result = None
        selection_metadata = {
            "selection_mode": "full",
            "subset_strategy": args.subset_strategy,
            "subset_seed": args.subset_seed,
            "subset_file": args.subset_file,
            "specified_unit_ids": None,
            "exam_type": getattr(args, "exam_type", "all") or "all",
        }
        filter_label_parts = []
        baseline_dim_source = getattr(args, "baseline_dim_source", "random") or "random"
        baseline_evaluator = BaselineEvaluator(baseline_mode=(baseline_dim_source == "random"))
        available_unit_ids = [str(uid) for uid in baseline_evaluator.get_all_unit_ids()]
        available_unit_id_set = set(available_unit_ids)

        # Exam-type filtering.
        exam_type = getattr(args, "exam_type", "all") or "all"
        has_exam_filter = exam_type != "all"
        if has_exam_filter:
            candidate_unit_ids = [
                str(uid) for uid in baseline_evaluator.get_unit_ids_by_exam_type(
                    exam_type=exam_type,
                    start_year=None,
                    end_year=None
                )
            ]
            exam_label = {"national": "national exams", "local": "local exams"}.get(exam_type, exam_type)
            filter_label_parts.append(exam_label)
        else:
            candidate_unit_ids = available_unit_ids
            filter_label_parts.append("all original questions")
        candidate_unit_id_set = set(candidate_unit_ids)

        if args.unit_id:
            selection_metadata["selection_mode"] = "unit_id"
            requested_unit_ids = [u.strip() for u in args.unit_id.split(",") if u.strip()]
            selection_metadata["specified_unit_ids"] = requested_unit_ids
            invalid_unit_ids = [uid for uid in requested_unit_ids if uid not in available_unit_id_set]
            if invalid_unit_ids:
                print(f"[Baseline] [WARN] Ignoring invalid unit_ids: {invalid_unit_ids}")
            baseline_unit_ids = [uid for uid in requested_unit_ids if uid in candidate_unit_id_set]
            if not baseline_unit_ids:
                raise ValueError("baseline received --unit-id, but no valid unit_id was found in the filtered candidate pool")
            filter_label_parts.append(f"specified questions ({len(baseline_unit_ids)})")
        elif args.subset_file:
            selection_metadata["selection_mode"] = "subset_file"
            loaded_unit_ids = [str(x) for x in load_subset_from_file(Path(args.subset_file))]
            invalid_unit_ids = [uid for uid in loaded_unit_ids if uid not in available_unit_id_set]
            if invalid_unit_ids:
                print(f"[Baseline] [WARN] subset_file contains invalid unit_ids, ignored: {invalid_unit_ids}")
            baseline_unit_ids = [uid for uid in loaded_unit_ids if uid in candidate_unit_id_set]
            if not baseline_unit_ids:
                raise ValueError(f"baseline subset_file has no valid unit_id in current candidate pool: {args.subset_file}")
            filter_label_parts.append(f"subset file ({len(baseline_unit_ids)})")
        elif args.subset_size:
            selection_metadata["selection_mode"] = "subset_size"
            data_loader = DataLoader()
            materials = data_loader.load_materials()
            mappings = data_loader.load_question_dimension_mappings()
            if has_exam_filter:
                materials = [m for m in materials if str(m.material_id) in candidate_unit_id_set]
                mappings = [m for m in mappings if str(m.unit_id) in candidate_unit_id_set]
            subset_result = build_subset_unit_ids(
                materials=materials,
                mappings=mappings,
                subset_size=args.subset_size,
                seed=args.subset_seed,
                strategy=args.subset_strategy,
            )
            baseline_unit_ids = [str(x) for x in subset_result.unit_ids]
            filter_label_parts.append(f"sampled {len(baseline_unit_ids)} questions")
        else:
            baseline_unit_ids = candidate_unit_ids if has_exam_filter else None

        filter_label = ", ".join(filter_label_parts) if filter_label_parts else "all original questions"

        summary = run_baseline_mode(
            config,
            unit_ids=baseline_unit_ids,
            subset_result=subset_result,
            selection_metadata=selection_metadata,
            input_dim_source=baseline_dim_source,
        )

        print("\n" + "=" * 80)
        print(f"Baseline evaluation summary ({filter_label})")
        print("=" * 80)
        print(f"  Experiment ID: {summary['experiment_id']}")
        print(f"  Total original questions: {summary['total_questions']}")
        print(f"  Evaluation success: {summary['evaluation_success']}")
        print(f"  Evaluation failed: {summary['evaluation_failed']}")
        print(f"  Dimension source: {summary.get('baseline_input_dim_source', 'random')}")
        print(f"  Average AI score: {summary['avg_ai_score']:.2f}")
        if summary.get('subset_size'):
            print(f"  Subset size: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  Strategy: {summary['subset_strategy']}")
                print(f"  Seed: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  Subset file: {summary['subset_file']}")
        if summary.get('specified_unit_ids'):
            print(f"  Unit: {summary['specified_unit_ids']}")
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  Pedagogical Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  Pedagogical Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")
        print(f"  Eval models: {summary['eval_models']}")
        print(f"  Output dir: {config.output_dir}")
        print("=" * 80)

    elif args.run_mode == "stage1-only":
        # Run Stage1 only.
        summary = run_stage1_only_mode(
            config,
            subset_size=args.subset_size,
            subset_strategy=args.subset_strategy,
            subset_seed=args.subset_seed,
            subset_file=args.subset_file,
        )

        mode_label = "Subset mode" if (args.subset_size or args.subset_file) else "Full mode"
        print("\n" + "=" * 80)
        print(f"[Stage1-Only {mode_label}] Summary")
        print("=" * 80)
        print(f"  Experiment ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print(f"  Total units: {summary['total_units']}")
        print(f"  Units in this run: {summary.get('run_total')}")
        if summary.get('subset_size'):
            print(f"  Subset size: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  Strategy: {summary['subset_strategy']}")
                print(f"  Seed: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  Subset file: {summary['subset_file']}")

        print(f"  Generation success: {summary['generation_success']}")
        print(f"  Generation failed: {summary['generation_failed']}")
        print(f"  Stage2Record count: {summary['stage2_record_count']}")

        print(f"\n  Question type distribution:")
        for qt, count in summary['question_type_distribution'].items():
            print(f"    - {qt}: {count}")

        print(f"\n  Agent error stats: {summary['agent_errors']}")
        print(f"\n  Output location: {config.output_dir}")
        print("\n  [Tip] Stage1 is complete. After switching network, run Stage2 with:")
        print(f"    python run.py --run-mode stage2-only --stage1-dir \"{config.output_dir}\"")
        print("=" * 80)

    elif args.run_mode == "stage2-only":
        # Run Stage2 only.
        incremental_ai = getattr(args, "incremental_ai", False)
        summary = run_stage2_only_mode(config, args.stage1_dir, args.stage2_output_dir, incremental_ai)

        if "error" not in summary:
            print("\n" + "=" * 80)
            print("[Stage2-Only] Summary")
            print("=" * 80)
            print(f"  Experiment ID: {summary['experiment_id']}")
            print(f"  Stage1 directory: {summary['stage1_dir']}")
            print(f"  Stage1 experiment ID: {summary.get('stage1_experiment_id', 'N/A')}")
            print(f"  Total units: {summary['total_units']}")

            print(f"  Evaluation success: {summary['evaluation_success']}")
            print(f"  Evaluation failed: {summary['evaluation_failed']}")
            print(f"  Average AI score: {summary['avg_ai_score']:.2f}")
            # Pedagogical evaluation uses P/R/F1 summary metrics.
            ped_metrics = summary.get('ped_round_metrics', {})
            micro = ped_metrics.get('micro', {})
            macro = ped_metrics.get('macro', {})
            print(f"  Pedagogical Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
            print(f"  Pedagogical Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

            print(f"\n  Question type distribution:")
            for qt, count in summary['question_type_distribution'].items():
                print(f"    - {qt}: {count}")

            if isinstance(summary.get("avg_ai_score_by_question_type"), dict):
                print(f"\n  Stats by question type:")
                ped_by_qt = summary.get("ped_round_metrics_by_question_type") or {}
                for qt in sorted(summary["avg_ai_score_by_question_type"].keys()):
                    ai_score = summary['avg_ai_score_by_question_type'].get(qt, 0)
                    ped_f1 = ped_by_qt.get(qt, {}).get("macro", {}).get("f1", 0)
                    print(f"    - {qt}: AI={ai_score:.2f}, Ped_F1={ped_f1:.4f}")

            print(f"  Eval models: {summary['eval_models']}")
            output_dir = summary.get('output_dir') or args.stage1_dir
            print(f"\n  Output location: {output_dir}")
            print("=" * 80)

    elif args.run_mode == "ablation-nodim":
        # Stage1 no-dimension-prompt ablation.
        specified_ablation_unit_ids = None
        if args.unit_id:
            specified_ablation_unit_ids = [u.strip() for u in args.unit_id.split(",") if u.strip()]

        summary = run_ablation_nodim_mode(
            config,
            subset_size=args.subset_size,
            subset_strategy=args.subset_strategy,
            subset_seed=args.subset_seed,
            subset_file=args.subset_file,
            unit_ids=specified_ablation_unit_ids,
        )

        selection_mode = summary.get("selection_mode", "full")
        mode_label = {
            "unit_id": "Specified-Unit mode",
            "subset_file": "Subset mode",
            "subset_size": "Subset mode",
            "full": "Full mode",
        }.get(selection_mode, "Full mode")
        print("\n" + "=" * 80)
        print(f"[Ablation Experiment - No Dimension Prompt] {mode_label} Summary")
        print("=" * 80)
        print(f"  Experiment ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print("  Ablation: skip dimension matching and prompt system; generate directly from original material")
        print(f"  Total units: {summary['total_units']}")
        print(f"  Units in this run: {summary.get('run_total')}")
        if summary.get('specified_unit_ids'):
            print(f"  Unit: {summary['specified_unit_ids']}")
        if summary.get('subset_size'):
            print(f"  Subset size: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  Strategy: {summary['subset_strategy']}")
                print(f"  Seed: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  Subset file: {summary['subset_file']}")

        print(f"  Generation success: {summary['generation_success']}")
        print(f"  Generation failed: {summary['generation_failed']}")
        print(f"  Evaluation success: {summary['evaluation_success']}")
        print(f"  Average AI score: {summary['avg_ai_score']:.2f}")
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  Pedagogical (GK) Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  Pedagogical (GK) Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

        # CS pedagogical evaluation.
        ped_cs_metrics = summary.get('cs_round_metrics', {})
        summary_eval_mode = summary.get('config', {}).get('eval_mode', '') or ''
        if "cs" in summary_eval_mode and ped_cs_metrics:
            micro_cs = ped_cs_metrics.get('micro', {})
            macro_cs = ped_cs_metrics.get('macro', {})
            print(f"  Pedagogical (CS) Micro: P={micro_cs.get('precision', 0):.4f}, R={micro_cs.get('recall', 0):.4f}, F1={micro_cs.get('f1', 0):.4f}")
            print(f"  Pedagogical (CS) Macro: P={macro_cs.get('precision', 0):.4f}, R={macro_cs.get('recall', 0):.4f}, F1={macro_cs.get('f1', 0):.4f}")

        print(f"\n  Question type distribution:")
        for qt, count in summary.get('question_type_distribution', {}).items():
            print(f"    - {qt}: {count}")

        print(f"  Eval models: {summary['eval_models']}")
        print(f"\n  Output location: {config.output_dir}")
        print("=" * 80)

    elif args.run_mode == "full":
        summary = run_full_mode(
            config,
            args,
            subset_size=args.subset_size,
            subset_strategy=args.subset_strategy,
            subset_seed=args.subset_seed,
            subset_file=args.subset_file,
            resume_dir=getattr(args, "resume_dir", None),
            start_from=getattr(args, "start_from", None),
        )

        # Choose mode label.
        if getattr(args, "resume_dir", None):
            mode_label = "Resume mode"
        elif args.subset_size or args.subset_file:
            mode_label = "Subset mode"
        else:
            mode_label = "Full mode"
        print("\n" + "=" * 80)
        print(f"{mode_label} Summary")
        print("=" * 80)
        print(f"  Experiment ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print(f"  Total units: {summary['total_units']}")
        print(f"  Units in this run: {summary.get('run_total')}")
        if summary.get('subset_size'):
            print(f"  Subset size: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  Strategy: {summary['subset_strategy']}")
                print(f"  Seed: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  Subset file: {summary['subset_file']}")

        print(f"  Generation success (stage2_record produced): {summary['generation_success']}")
        print(f"  Stage2 skipped (empty stage2_record): {summary.get('stage2_skipped', 0)}")
        print(f"  Evaluation success: {summary['evaluation_success']}")
        print(f"  Average AI score: {summary['avg_ai_score']:.2f}")
        # Pedagogical evaluation uses P/R/F1 summary metrics.
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  Pedagogical Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  Pedagogical Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

        print(f"\n  Question type distribution:")
        for qt, count in summary['question_type_distribution'].items():
            print(f"    - {qt}: {count}")

        if isinstance(summary.get("avg_ai_score_by_question_type"), dict):
            print(f"\n  Stats by question type:")
            ped_by_qt = summary.get("ped_round_metrics_by_question_type") or {}
            for qt in sorted(summary["avg_ai_score_by_question_type"].keys()):
                ai_score = summary['avg_ai_score_by_question_type'].get(qt, 0)
                ped_f1 = ped_by_qt.get(qt, {}).get("macro", {}).get("f1", 0)
                print(f"    - {qt}: AI={ai_score:.2f}, Ped_F1={ped_f1:.4f}")

        print(f"\n  Agent error stats: {summary['agent_errors']}")
        print(f"  Eval models: {summary['eval_models']}")
        print(f"\n  Output location: {config.output_dir}")
        print("=" * 80)

    if llm_logger:
        llm_logger.save()
        llm_logger.print_summary()
        print(f"[LLMLogger] LLM call logs saved to: {llm_log_dir}")

    print("\n[SUCCESS] Experiment completed!")


if __name__ == "__main__":
    main()
