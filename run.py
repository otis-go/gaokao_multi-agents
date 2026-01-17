#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cli.py
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
   python cli.py --run-mode single --unit-id 10 --dim-mode gk --prompt-level C

2. Full mode (run all 181 questions):
   python cli.py --run-mode full --dim-mode gk --prompt-level C

3. Subset sampling mode (40 or 60 questions proportional stratified):
   python cli.py --run-mode full --subset-size 40 --subset-strategy proportional_stratified --subset-seed 42

4. Load subset from file:
   python cli.py --run-mode full --subset-file outputs/exp1/subset_unit_ids.json

5. Use round-id to organize experiments in the same round (recommended for subset40/60 comparison):
   # Step 1: Run subset40
   python cli.py --run-mode full --subset-size 40 --round-id ROUND_20251208_A_deepseek --dim-mode gk --prompt-level C

   # Step 2: Run subset60 (same round-id)
   python cli.py --run-mode full --subset-size 60 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

   # Output structure:
   # outputs/ROUND_20251207_A/
   #   subset40_stratified_seed42_gk_C_20251207_120000/
   #     summary.json, subset_unit_ids.json, subset_stats.json, stage2/, llm_logs/
   #   subset60_stratified_seed42_gk_C_20251207_121500/
   #     summary.json, etc.
   #   round_manifest.jsonl

6. View help:
   python cli.py --help

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
- AI-centric evaluation results (three-model ensemble)
- Pedagogical evaluation results
- summary.json (full mode)
- subset_unit_ids.json / subset_stats.json (subset mode)
- round_manifest.jsonl (records each run index when round-id enabled)

Core Changes:
1. Removed manual question type: auto-determined from data
2. Unified run modes: single or full
3. Unified LLM control via LLMRouter
4. Stage 2 dual evaluation shares same three-model group
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
        # Baseline evaluation mode (all real exam questions): needs dim_mode, no prompt_level
        parts.append("baseline")
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
  python cli.py --run-mode single --unit-id 1 --dim-mode gk --prompt-level C

  # Full mode (run all questions)
  python cli.py --run-mode full --dim-mode gk --prompt-level C

  # Subset sampling mode (40 or 60 questions proportional stratified)
  python cli.py --run-mode full --subset-size 40 --subset-strategy proportional_stratified --subset-seed 42

  # Load subset from file
  python cli.py --run-mode full --subset-file outputs/exp1/subset_unit_ids.json

  # Use round-id to organize experiments in the same round (recommended for subset40/60 comparison)
  # Step 1: Run subset40
  python cli.py --run-mode full --subset-size 40 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

  # Step 2: Run subset60 (same round-id)
  python cli.py --run-mode full --subset-size 60 --round-id ROUND_20251207_A --dim-mode gk --prompt-level C

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
        help="Run mode: single (single question) / full (all or subset) / baseline (baseline evaluation: random dimension input + real exam gold dimension for pedagogical metrics) / extract (extract questions) / stage1-only (Stage1 only) / stage2-only (Stage2 only) / ablation-nodim (ablation experiment: no dimension prompts, direct generation)"
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

    # ========== [2025-12 Added] Self-check trace parameters ==========
    parser.add_argument(
        "--enable-self-check-trace",
        action="store_true",
        help="Enable runtime call trace (for self-check suite)"
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
        choices=["ai", "gk", "cs", "ai+gk", "ai+cs"],
        help="Evaluation mode (control which evaluations Stage2 runs):\n"
             "  ai: AI evaluation only\n"
             "  gk: GK pedagogical evaluation only (default)\n"
             "  cs: CS pedagogical evaluation only\n"
             "  ai+gk: AI + GK pedagogical evaluation\n"
             "  ai+cs: AI + CS pedagogical evaluation\n"
             "AI evaluation not enabled by default, specify ai or ai+gk/ai+cs if needed"
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

    args = parser.parse_args()

    # ========== Parameter validation ==========
    if args.run_mode == "single" and args.unit_id is None:
        parser.error("Single mode (--run-mode single) must specify --unit-id")

    if args.run_mode == "full" and args.unit_id is not None:
        print(f"[WARN] Full/subset mode ignores --unit-id parameter")
        args.unit_id = None

    if args.run_mode == "baseline" and args.unit_id is not None:
        print(f"[INFO] Baseline mode will only evaluate specified --unit-id (omit this param for all questions)")

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
        if args.subset_size is not None or args.subset_file is not None:
            print(f"[WARN] Baseline mode ignores --subset-size / --subset-file parameters")
            args.subset_size = None
            args.subset_file = None
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
        # [2026-01 Modified] AI evaluation not enabled by default
        if args.eval_mode is None:
            args.eval_mode = "gk"
            print(f"[TIP] ablation-nodim mode default eval_mode=gk (GK pedagogical evaluation only, specify --eval-mode ai+gk for AI evaluation)")
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
            # Infer dimension tag from eval_mode (prioritize gk)
            if "gk" in eval_mode:
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

        if args.run_mode == "single":
            experiment_id = f"EXP_{mode_tag}_U{args.unit_id}_{dim_tag}_{args.prompt_level}{ablation_tag}_{timestamp}"
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
        # [2026-01 Refactored] Remove gk+cs mode, only support separate gk or cs
        if "gk" in eval_mode:
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


def _get_overall_score_number(x, default=None):
    """把 overall_score 转成 float（避免出现字符串导致格式化崩溃）。"""
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
    【2025-12 架构Refactored】统一抽取 Stage2 分数。

    读取字段：
    - ai_eval_result.overall_score -> AI 综合得分
    - pedagogical_eval_result -> 教育学 P/R/F1 指标（新版 hit-based 评估）
    - gk_eval_result -> GK 维度教育学 P/R/F1 指标（独立评估模式）
    - cs_eval_result -> CS 维度教育学 P/R/F1 指标（独立评估模式）

    返回值：
    - ai_score: float | None
    - ped_metrics: dict | None  包含 {f1, precision, recall, tp, fp, fn}
    - gk_metrics: dict | None  包含 {f1, precision, recall, tp, fp, fn}（独立评估模式）
    - cs_metrics: dict | None  包含 {f1, precision, recall, tp, fp, fn}（独立评估模式）
    """
    def _get_overall(x):
        if not x:
            return None
        if isinstance(x, dict):
            return x.get("overall_score")
        return getattr(x, "overall_score", None)

    def _extract_ped_metrics(ped_result):
        """从教育学评估结果中提取 P/R/F1 指标"""
        if ped_result is None:
            return None
        # 从 PedagogicalHitBasedResult 提取指标
        if isinstance(ped_result, dict):
            f1 = ped_result.get("f1")
            precision = ped_result.get("precision")
            recall = ped_result.get("recall")
            tp = ped_result.get("tp")
            fp = ped_result.get("fp")
            fn = ped_result.get("fn")
            # [2025-12-31 Added] 提取维度信息（for排除高频后的统计）
            gold_dimensions = ped_result.get("gold_dimensions", [])
            predicted_dimensions = ped_result.get("predicted_dimensions", [])
        else:
            f1 = getattr(ped_result, "f1", None)
            precision = getattr(ped_result, "precision", None)
            recall = getattr(ped_result, "recall", None)
            tp = getattr(ped_result, "tp", None)
            fp = getattr(ped_result, "fp", None)
            fn = getattr(ped_result, "fn", None)
            # [2025-12-31 Added] 提取维度信息（for排除高频后的统计）
            gold_dimensions = getattr(ped_result, "gold_dimensions", [])
            predicted_dimensions = getattr(ped_result, "predicted_dimensions", [])

        # 只要有任何一items指标，就构建 metrics dict
        if any(x is not None for x in [f1, precision, recall, tp, fp, fn]):
            return {
                "f1": float(f1) if f1 is not None else 0.0,
                "precision": float(precision) if precision is not None else 0.0,
                "recall": float(recall) if recall is not None else 0.0,
                "tp": int(tp) if tp is not None else 0,
                "fp": int(fp) if fp is not None else 0,
                "fn": int(fn) if fn is not None else 0,
                # [2025-12-31 Added] 包含维度信息
                "gold_dimensions": list(gold_dimensions) if gold_dimensions else [],
                "predicted_dimensions": list(predicted_dimensions) if predicted_dimensions else [],
            }
        return None

    # AI 评分（保持不变）
    ai = _get_overall(getattr(evaluation_state, "ai_eval_result", None))

    try:
        ai = float(ai) if ai is not None else None
    except Exception:
        ai = None

    # [2025-12 Refactored] 教育学评估使用 hit-based P/R/F1 指标
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    ped_metrics = _extract_ped_metrics(ped_result)

    # [2025-12 Added] 独立的 GK/CS 教育学评估结果
    gk_result = getattr(evaluation_state, "gk_eval_result", None)
    gk_metrics = _extract_ped_metrics(gk_result)

    cs_result = getattr(evaluation_state, "cs_eval_result", None)
    cs_metrics = _extract_ped_metrics(cs_result)

    return ai, ped_metrics, gk_metrics, cs_metrics


def _extract_high_variance_dims(evaluation_state, unit_id: str, question_type: str) -> List[Dict[str, Any]]:
    """
    [2025-12 Added] 从评估状态中提取高方差维度记录。

    高方差维度定义：同一维度下不同模型给出的评分相差过大（≥40分）

    返回结构:
    [
        {
            "unit_id": str,
            "question_type": str,
            "eval_type": "ai_centric" | "pedagogical",
            "dimension_id": str,
            "dimension_name": str,
            "score_diff": float,
            "max_score": float,
            "max_model": str,
            "min_score": float,
            "min_model": str,
            "all_scores": {model_name: score},
        },
        ...
    ]
    """
    result: List[Dict[str, Any]] = []

    def _get_audit(eval_result):
        if eval_result is None:
            return None
        if hasattr(eval_result, "audit"):
            return eval_result.audit
        if isinstance(eval_result, dict):
            return eval_result.get("audit")
        return None

    # 从 AI 评估结果中提取
    ai_result = getattr(evaluation_state, "ai_eval_result", None)
    ai_audit = _get_audit(ai_result)
    if isinstance(ai_audit, dict):
        ai_high_variance = ai_audit.get("high_variance_dims", [])
        for item in ai_high_variance:
            if isinstance(item, dict):
                record = {
                    "unit_id": unit_id,
                    "question_type": question_type,
                    "eval_type": "ai_centric",
                    "dimension_id": item.get("dimension_id", ""),
                    "dimension_name": item.get("dimension_name", item.get("dimension_id", "")),
                    "score_diff": item.get("score_diff", 0),
                    "max_score": item.get("max_score", 0),
                    "max_model": item.get("max_model", ""),
                    "min_score": item.get("min_score", 0),
                    "min_model": item.get("min_model", ""),
                    "all_scores": item.get("all_scores", {}),
                }
                result.append(record)

    # 从教育学评估结果中提取
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    ped_audit = _get_audit(ped_result)
    if isinstance(ped_audit, dict):
        ped_high_variance = ped_audit.get("high_variance_dims", [])
        for item in ped_high_variance:
            if isinstance(item, dict):
                record = {
                    "unit_id": unit_id,
                    "question_type": question_type,
                    "eval_type": "pedagogical",
                    "dimension_id": item.get("dimension_id", ""),
                    "dimension_name": item.get("dimension_name", item.get("dimension_id", "")),
                    "score_diff": item.get("score_diff", 0),
                    "max_score": item.get("max_score", 0),
                    "max_model": item.get("max_model", ""),
                    "min_score": item.get("min_score", 0),
                    "min_model": item.get("min_model", ""),
                    "all_scores": item.get("all_scores", {}),
                }
                result.append(record)

    return result


def _round_floats(obj: Any, decimals: int = 3) -> Any:
    """
    [2026-01 Added] 递归地将字典/列表中的浮点数四舍五入到指定小数位。

    Args:
        obj: 待处理的对象（字典、列表或其他）
        decimals: 保留的小数位数，默认3位

    Returns:
        处理后的对象
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(item, decimals) for item in obj]
    else:
        return obj


def _compute_iteration_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    [2026-01 Added] 计算迭代相关统计。

    统计内容：
    - iteration_failed_count: 因迭代超限失败的questions数
    - iteration_failed_examples: 具体失败的 unit_id 列表（最多5items）
    - iteration_success_rate: 触发迭代的questions中成功完成的比例
    - iteration_triggered_count: 触发了迭代（iteration_count > 1）的questions数
    - iteration_triggered_rate: 触发迭代的概率

    Args:
        results: stats["results"] 列表，每项包含 iteration_count, max_iteration_exceeded

    Returns:
        迭代统计字典
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

        # 触发迭代：iteration_count > 1（即进行了至少一次重试）
        if iteration_count > 1:
            triggered_count += 1

        # 迭代失败：max_iteration_exceeded = True
        if max_exceeded:
            failed_count += 1
            if len(failed_examples) < 5:
                failed_examples.append(r.get("unit_id", "unknown"))

    # 迭代成功率 = (触发迭代但没失败的questions数) / 触发迭代的questions数
    # 即：1 - (迭代失败数 / 触发迭代数)
    if triggered_count > 0:
        iteration_success_rate = round((triggered_count - failed_count) / triggered_count, 3)
    else:
        iteration_success_rate = 1.0  # 没有触发迭代，视为100%成功

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
    [2025-12 Added] 提取评估维度明细，for summary.json 透明化。

    返回结构:
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

    # 提取 AI 评估明细
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

        # 维度明细
        # 【2025-12 优化】添加每items维度的贡献值
        for dim_id, dim_data in (dimensions if isinstance(dimensions, dict) else {}).items():
            if isinstance(dim_data, dict):
                dim_score = dim_data.get("score")
                dim_weight = dim_data.get("weight", 1.0)
                ai_details["dimensions"][dim_id] = {
                    "score": dim_score,
                    "weight": dim_weight,
                    "contribution": round(dim_score * dim_weight, 2) if dim_score is not None else None,  # 对最终得分的贡献
                    "level": dim_data.get("level"),
                }

        # 模型摘要
        # 【2025-12 优化】添加每items模型对最终得分的贡献
        for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
            if isinstance(model_dims, dict):
                scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                if scores:
                    model_weight = model_weights.get(model_name, 0.0)
                    model_avg = round(sum(scores) / len(scores), 2)
                    ai_details["model_summary"][model_name] = {
                        "average_score": model_avg,
                        "weight": model_weight,
                        "contribution": round(model_avg * model_weight, 2),  # 对最终得分的贡献
                        "dimension_count": len(scores),
                    }

        result["ai_eval"] = ai_details

    # 提取教育学评估明细
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

        # 维度明细
        # 【2025-12 优化】添加每items维度的权重和贡献值
        # 教育学评估使用etc权重（简单平均），权重 = 1/维度数
        dim_count = len(dimension_results) if dimension_results else 1
        dim_weight = round(1.0 / dim_count, 4) if dim_count > 0 else 0

        for dim_res in dimension_results:
            if hasattr(dim_res, "dimension_name"):
                dim_name = dim_res.dimension_name
                dim_score = float(getattr(dim_res, "score", 0))
                ped_details["dimensions"][dim_name] = {
                    "score": dim_score,
                    "weight": dim_weight,
                    "contribution": round(dim_score * dim_weight, 2),  # 对最终得分的贡献
                    "hit_level": getattr(dim_res, "hit_level", ""),
                }
            elif isinstance(dim_res, dict):
                dim_name = dim_res.get("dimension_name", "")
                if dim_name:
                    dim_score = dim_res.get("score", 0)
                    ped_details["dimensions"][dim_name] = {
                        "score": dim_score,
                        "weight": dim_weight,
                        "contribution": round(dim_score * dim_weight, 2),  # 对最终得分的贡献
                        "hit_level": dim_res.get("hit_level", ""),
                    }

        # 模型摘要
        # 【2025-12 优化】添加每items模型对最终得分的贡献
        for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
            if isinstance(model_dims, dict):
                scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                if scores:
                    model_weight = model_weights.get(model_name, 0.0)
                    model_avg = round(sum(scores) / len(scores), 2)
                    ped_details["model_summary"][model_name] = {
                        "average_score": model_avg,
                        "weight": model_weight,
                        "contribution": round(model_avg * model_weight, 2),  # 对最终得分的贡献
                        "dimension_count": len(scores),
                    }

        result["pedagogical_eval"] = ped_details

    return result


def _extract_missing_dimensions(evaluation_state, unit_id: str, question_type: str = "") -> Optional[Dict[str, Any]]:
    """
    [2025-12 Added] 从评估状态中提取缺失维度信息。

    缺失维度定义：gold_dimensions - predicted_dimensions（即 False Negative）

    返回结构:
    {
        "unit_id": str,
        "question_type": str,
        "gold_dimensions": List[str],       # 预期维度
        "predicted_dimensions": List[str],  # 实际命中维度
        "missing_dimensions": List[str],    # 缺失维度（FN）
        "extra_dimensions": List[str],      # 多余维度（FP）
        "tp": int,
        "fp": int,
        "fn": int,
        "precision": float,
        "recall": float,
        "f1": float,
    }
    such as果没有教育学评估结果，返回 None
    """
    ped_result = getattr(evaluation_state, "pedagogical_eval_result", None)
    if ped_result is None:
        return None

    # 从 PedagogicalHitBasedResult 中提取字段
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

    # such as果没有显式的 missing_dimensions，手动计算
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
    [2025-12 Added] 保存教育学维度评估汇总报告。

    输出两items文件：
    1. missing_dimensions_report.json - JSON 格式，包含完整信息
    2. missing_dimensions_report.txt - 文本格式，方便人工查看

    【2025-12-19 Updated】现在输出每questions的完整命中情况：
    - TP（正确命中）：预测的维度也在金标准中
    - FP（多余预测）：预测的维度不在金标准中
    - FN（漏掉）：金标准中的维度未被预测到

    返回: (json_path, txt_path)
    """
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 过滤出有效记录
    valid_records = [r for r in missing_dims_records if r]

    # 统计缺失维度频率（FN）
    missing_freq: Dict[str, int] = {}
    for r in valid_records:
        for dim in r.get("missing_dimensions", []):
            missing_freq[dim] = missing_freq.get(dim, 0) + 1

    # 统计多余维度频率（FP）
    extra_freq: Dict[str, int] = {}
    for r in valid_records:
        for dim in r.get("extra_dimensions", []):
            extra_freq[dim] = extra_freq.get(dim, 0) + 1

    # 按频率排序
    sorted_missing_freq = sorted(missing_freq.items(), key=lambda x: -x[1])
    sorted_extra_freq = sorted(extra_freq.items(), key=lambda x: -x[1])

    # 统计总体指标
    total_tp = sum(r.get("tp", 0) for r in valid_records)
    total_fp = sum(r.get("fp", 0) for r in valid_records)
    total_fn = sum(r.get("fn", 0) for r in valid_records)
    avg_precision = sum(r.get("precision", 0) for r in valid_records) / len(valid_records) if valid_records else 0
    avg_recall = sum(r.get("recall", 0) for r in valid_records) / len(valid_records) if valid_records else 0
    avg_f1 = sum(r.get("f1", 0) for r in valid_records) / len(valid_records) if valid_records else 0

    # 构建汇总
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

    # 保存 JSON
    json_path = output_dir / "missing_dimensions_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存 TXT（人工友好格式）
    txt_path = output_dir / "missing_dimensions_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"教育学维度评估汇总报告\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"实验ID: {experiment_id}\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"总questions数: {len(valid_records)}\n")
        f.write(f"\n")

        # 总体指标
        f.write(f"【总体评估指标】\n")
        f.write(f"-" * 60 + "\n")
        f.write(f"  总 TP（正确命中）: {total_tp}\n")
        f.write(f"  总 FP（多余预测）: {total_fp}\n")
        f.write(f"  总 FN（漏掉维度）: {total_fn}\n")
        f.write(f"  平均 Precision: {avg_precision:.4f}\n")
        f.write(f"  平均 Recall:    {avg_recall:.4f}\n")
        f.write(f"  平均 F1:        {avg_f1:.4f}\n")
        f.write(f"\n")

        # 缺失维度频率（FN）
        f.write(f"【漏掉维度统计 (FN)】（按频率降序）\n")
        f.write(f"-" * 60 + "\n")
        if sorted_missing_freq:
            for dim, freq in sorted_missing_freq:
                f.write(f"  {dim}: {freq} 次\n")
        else:
            f.write(f"  （无漏掉维度）\n")
        f.write(f"\n")

        # 多余维度频率（FP）
        f.write(f"【多余预测统计 (FP)】（按频率降序）\n")
        f.write(f"-" * 60 + "\n")
        if sorted_extra_freq:
            for dim, freq in sorted_extra_freq:
                f.write(f"  {dim}: {freq} 次\n")
        else:
            f.write(f"  （无多余预测）\n")
        f.write(f"\n")

        # 每questions详细情况
        f.write(f"【各questions维度命中详情】\n")
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

            # 计算命中的维度（TP）
            gold_set = set(gold)
            pred_set = set(predicted)
            hit_dims = sorted(gold_set & pred_set)

            f.write(f"\nquestions unit_id={unit_id} (questions型: {qt})\n")
            f.write(f"  金标准维度({len(gold)}items): {gold}\n")
            f.write(f"  预测维度({len(predicted)}items): {predicted}\n")
            f.write(f"  ---\n")
            f.write(f"  ✓ 命中 TP({len(hit_dims)}items): {hit_dims}\n")
            f.write(f"  ✗ 漏掉 FN({len(missing)}items): {missing}\n")
            f.write(f"  + 多余 FP({len(extra)}items): {extra}\n")
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
    [2026-01 Added] 收集 Good Case 和 Bad Case for展示。

    Bad Case 优先级:
    1. 迭代超限失败的案例（max_iteration_exceeded=True）
    2. F1=0 的案例
    3. F1 最低的案例

    Good Case 优先级:
    1. F1=1.0 的案例
    2. F1 最高的案例

    Args:
        results: 评估结果列表
        generation_states: unit_id -> generation_state 的映射（可选，for获取iteration information）
        max_cases: 每类最多收集的案例数

    Returns:
        {"good_cases": [...], "bad_cases": [...]}
    """
    generation_states = generation_states or {}

    # 收集 bad cases
    bad_cases = []

    # 1. 首先收集迭代超限失败的案例
    for uid, gen_state in generation_states.items():
        if gen_state.get("max_iteration_exceeded", False):
            bad_cases.append({
                "unit_id": uid,
                "fail_type": "iteration_exceeded",
                "iteration_count": gen_state.get("iteration_count", 0),
                "fail_reason": gen_state.get("iteration_fail_reason", "迭代超限"),
                "f1": None,
                "stem_preview": gen_state.get("stem_preview", ""),
            })

    # 2. 从评估结果中收集低 F1 案例
    # 【注意】results 中的 ped_metrics 包含 f1, precision, recall, gold_dimensions, predicted_dimensions
    def _get_f1(r):
        """从 result_item 中提取 F1 值"""
        ped = r.get("ped_metrics") or r.get("gk_metrics") or r.get("cs_metrics")
        if ped:
            return ped.get("f1")
        return r.get("f1")  # 兼容直接存储 f1 的情况

    def _get_metrics(r):
        """从 result_item 中提取评估指标"""
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
        # 跳过已经在迭代超限中的
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
            "stem_preview": "",  # 无法从 result 获取questions干，需要从 generation_state 获取
            "question_type": r.get("question_type", ""),
        })

    # 收集 good cases
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
            "stem_preview": "",  # 无法从 result 获取questions干
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

    # [2025-12 Added] High variance dimension collection
    all_high_variance_records: List[Dict[str, Any]] = []

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

            # [2025-12 Added] Collect high variance dimension records
            high_variance_items = _extract_high_variance_dims(evaluation_state, uid, qt)
            if high_variance_items:
                all_high_variance_records.extend(high_variance_items)

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

        TODO: Ideally should directly use PedagogicalRoundAggregation results
        instead of recalculating here. This requires larger refactoring work.
        """
        from src.evaluation.pedagogical_eval import get_high_freq_dims_by_mode, calculate_prf
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
                tp_ex, fp_ex, fn_ex, p_ex, r_ex, f1_ex = calculate_prf(gold, pred, exclude_dims=high_freq_dims)
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
        "eval_models": router.get_eval_model_names(),
        # [2025-12 Added] High variance dimension aggregation
        "high_variance_records": all_high_variance_records,
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

    ✅ Use unified run_units pipeline
    ✅ CLI does not perform DataLoader pre-validation
    ✅ Statistics only depend on schema stable fields
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

    # 5. 温度实验（非默认温度）
    temperature = args.temperature
    if temperature is not None and temperature != 1.0:
        if temperature == 0.5:
            return "TEMP_0_5"
        elif temperature == 1.5:
            return "TEMP_1_5"
        elif temperature == 2.0:
            return "TEMP_2_0"

    # 6. 真questions维度验证实验（默认类型，按模型分类）
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
# [2026-01 Added] 断点续传结果合并
# ============================================================================

def _merge_resume_results(
    config: ExperimentConfig,
    new_stats: Dict[str, Any],
    start_from: int,
    all_unit_ids: List[str],
) -> Dict[str, Any]:
    """
    合并断点续传模式下的已有结果和新生成结果。

    Args:
        config: Experiment configuration（output_dir 指向断点续传目录）
        new_stats: 新运行的 unit 统计结果
        start_from: 起始 unit_id
        all_unit_ids: 全量 unit_id 列表

    Returns:
        合并后的 stats 字典
    """
    stage2_dir = config.output_dir / "stage2"

    # 收集已有 unit 的结果（1 到 start_from-1）
    existing_unit_ids = [uid for uid in all_unit_ids if int(uid) < start_from]
    print(f"[Resume] Loaded existing unit results: {len(existing_unit_ids)} (units 1-{start_from-1})")

    existing_results = []
    existing_ai_scores = []
    existing_ped_metrics = []
    existing_gk_metrics = []
    existing_cs_metrics = []
    existing_question_types = {"single-choice": 0, "essay": 0, "other": 0}

    # 按questions型分组的教育学指标
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

            # 提取结果
            result_item = {
                "unit_id": uid,
                "stage2_status": "success",
            }

            # questions型
            qt = "other"
            core_input = state.get("input_record", {}) or state.get("core_input", {}) or state.get("input", {})
            if core_input:
                qt = core_input.get("question_type", "other")
            result_item["question_type"] = qt
            existing_question_types[qt] = existing_question_types.get(qt, 0) + 1

            # AI 分数 - 兼容两种结构: ai_result 或 ai_eval.result
            ai_result = state.get("ai_result", {}) or (state.get("ai_eval", {}) or {}).get("result", {})
            ai_overall = ai_result.get("overall_score") if ai_result else None
            if ai_overall is not None:
                existing_ai_scores.append(float(ai_overall))
                result_item["ai_overall_score"] = float(ai_overall)

            # 教育学指标（P/R/F1）- 兼容两种结构
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

    # 合并结果
    all_results = existing_results + new_stats.get("results", [])
    all_ai_scores = existing_ai_scores + new_stats.get("ai_scores", [])
    all_ped_metrics = existing_ped_metrics + new_stats.get("ped_metrics_list", [])
    all_gk_metrics = existing_gk_metrics + new_stats.get("gk_metrics_list", [])
    all_cs_metrics = existing_cs_metrics + new_stats.get("cs_metrics_list", [])

    # 合并questions型分布
    merged_question_types = dict(existing_question_types)
    for qt, count in new_stats.get("question_type_distribution", {}).items():
        merged_question_types[qt] = merged_question_types.get(qt, 0) + count

    # 合并按questions型分组的教育学指标
    merged_ped_by_qtype = {}
    merged_gk_by_qtype = {}
    merged_cs_by_qtype = {}

    for qt in ["single-choice", "essay", "other"]:
        merged_ped_by_qtype[qt] = existing_ped_by_qtype[qt] + new_stats.get("ped_metrics_by_qtype", {}).get(qt, [])
        merged_gk_by_qtype[qt] = existing_gk_by_qtype[qt] + new_stats.get("gk_metrics_by_qtype", {}).get(qt, [])
        merged_cs_by_qtype[qt] = existing_cs_by_qtype[qt] + new_stats.get("cs_metrics_by_qtype", {}).get(qt, [])

    # 重新计算汇总统计
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

    # 计算合并后的 AI 平均分（按questions型）
    merged_ai_by_qtype = {}
    for qt in ["single-choice", "essay", "other"]:
        scores = [r.get("ai_overall_score") for r in all_results if r.get("question_type") == qt and r.get("ai_overall_score") is not None]
        merged_ai_by_qtype[qt] = sum(scores) / len(scores) if scores else 0

    # 构建合并后的 stats
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

        # 保留原始指标列表供后续使用
        "ai_scores": all_ai_scores,
        "ped_metrics_list": all_ped_metrics,
        "gk_metrics_list": all_gk_metrics,
        "cs_metrics_list": all_cs_metrics,
        "ped_metrics_by_qtype": merged_ped_by_qtype,
        "gk_metrics_by_qtype": merged_gk_by_qtype,
        "cs_metrics_by_qtype": merged_cs_by_qtype,

        # 断点续传元信息
        "resume_info": {
            "is_resumed": True,
            "start_from": start_from,
            "existing_units": len(existing_results),
            "new_units": len(new_stats.get("results", [])),
        },
    }

    return merged_stats


# ============================================================================
# 全量/Subset mode
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
    【2025-12 架构Refactored】full / 40 / 60 都走同一套逻辑。

    ✅ CLI 只负责：决定 unit_id 列表
    ✅ 使用统一的 run_units 管线
    ✅ 统计只依赖 schema 稳定字段

    [2026-01 Added] 断点续传模式：
    - resume_dir: 已有的实验目录
    - start_from: 从指定 unit_id 开始运行
    - 只运行 >= start_from 的 unit，结果写入 resume_dir
    - 最后合并所有 unit 结果生成新的 summary
    """
    from src.shared.data_loader import DataLoader
    from src.shared.llm_router import LLMRouter
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file

    # [2026-01 Added] 断点续传模式
    is_resume_mode = resume_dir is not None and start_from is not None

    if is_resume_mode:
        print("\n" + "=" * 80)
        print("[Resume Mode] Continuing experiment from specified position")
        print(f"  Target directory: {resume_dir}")
        print(f"  Starting unit: {start_from}")
        print("=" * 80)

        # 覆盖 config.output_dir 为已有目录
        config.output_dir = Path(resume_dir)
    else:
        print("\n" + "=" * 80)
        if subset_size or subset_file:
            print("[Subset Mode] Starting subset unit_id run")
        else:
            print("[Full Mode] Starting all unit_id run")
        print("=" * 80)

    router = LLMRouter.from_config(config)

    # ---------- 只在这里决定 unit_id 列表（CLI 职责） ----------
    # 【注意】这里使用 DataLoader 只是为了获取 unit_id 列表，不做任何前置校验
    data_loader = DataLoader()
    mappings = data_loader.load_question_dimension_mappings()

    total_units = len(mappings)
    print(f"[数据集] 总 unit 数量: {total_units}")

    subset_result = None
    all_unit_ids = [str(m.unit_id) for m in mappings]  # 全量 unit_id 列表

    # [2026-01 Added] 断点续传模式：只运行 >= start_from 的 unit
    if is_resume_mode:
        unit_ids_to_run = [uid for uid in all_unit_ids if int(uid) >= start_from]
        print(f"[Resume] Running units {start_from}-{max(int(u) for u in all_unit_ids)} ({len(unit_ids_to_run)} total)")
    elif subset_file:
        unit_ids_to_run = [str(x) for x in load_subset_from_file(Path(subset_file))]
        print(f"[Subset mode] 从文件加载 {len(unit_ids_to_run)} items unit_id")
    elif subset_size:
        # 子集采样需要 materials
        materials = data_loader.load_materials()
        subset_result = build_subset_unit_ids(
            materials=materials,
            mappings=mappings,
            subset_size=subset_size,
            seed=subset_seed,
            strategy=subset_strategy,
        )
        unit_ids_to_run = [str(x) for x in subset_result.unit_ids]
        print(f"[Subset mode] 采样 {len(unit_ids_to_run)} items unit_id (strategy={subset_strategy}, seed={subset_seed})")
    else:
        unit_ids_to_run = all_unit_ids
        print(f"[Full mode] 运行 {len(unit_ids_to_run)} items unit_id")

    # CLI 驱动 unit_id
    config.pipeline.agent1.material_selection_strategy = "manual"

    generation_orchestrator = GenerationOrchestrator(config, llm_router=router)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router, eval_mode=getattr(config, "eval_mode", None))

    # 【2025-12 架构Refactored】使用统一的 run_units
    stats = run_units(
        config=config,
        unit_ids=unit_ids_to_run,
        router=router,
        generation_orchestrator=generation_orchestrator,
        evaluation_orchestrator=evaluation_orchestrator,
    )

    # [2026-01 Added] 断点续传模式：合并已有结果和新结果
    if is_resume_mode:
        print(f"\n[Resume] Merging existing results with newly generated results...")
        stats = _merge_resume_results(
            config=config,
            new_stats=stats,
            start_from=start_from,
            all_unit_ids=all_unit_ids,
        )
        print(f"[Resume] Merge complete, total {stats['run_total']} units")

    # 获取当前时间戳for summary
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
        "skipped_no_dim": stats.get("skipped_no_dim", 0),  # [2025-12-31 Added] 因无对应维度而跳过的questions数

        # [2026-01 Added] 生成失败统计（失败questions不参with平均分计算）
        "generation_failure": {
            "failure_count": stats["run_total"] - stats["generation_success"],
            "failure_rate": round((stats["run_total"] - stats["generation_success"]) / stats["run_total"], 3) if stats["run_total"] > 0 else 0,
            "failure_unit_ids": [r.get("unit_id") for r in stats["results"] if r.get("stage1_status") != "success"],
            "success_count": stats["generation_success"],
        },

        # [2026-01 Added] 迭代统计（with生成失败区分）
        "iteration_stats": _compute_iteration_stats(stats["results"]),

        "avg_ai_score": stats["avg_ai_score"],
        # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
        "ped_round_metrics": stats["ped_round_metrics"],
        # [2025-12-28 Added] 按questions型分组的教育学指标汇总
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
        # [2026-01 Added] 完整 LLM 配置记录（for auto-discover 模式）
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # [2026-01 Added] 实验类型标记（for auto-discover 模式分类）
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
        # [2026-01 Added] 温度配置记录
        "temperature": {
            "stage1": api_config.STAGE1_TEMPERATURE,
            "stage2": api_config.STAGE2_TEMPERATURE,  # 固定 0.0
            "cli_override": args.temperature,
        },
        # [2026-01 Added] ablation experiment配置
        "ablation": {
            "skip_agent": config.pipeline.stage1_ablation.skip_agent,
            "is_ablation": config.pipeline.stage1_ablation.skip_agent != "none",
        },
        # [2025-12 Added] Stage1 ablation experiment字段
        "stage1_skip_agent": config.pipeline.stage1_ablation.skip_agent,
        "eval_models": stats["eval_models"],
        "agent_errors": stats["agent_errors"],
        "timestamp": timestamp,
        "results": stats["results"],
    }

    # [2025-12 Added] 添加 LLM 调用重试审计信息
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM 调用过程中发生了重试或失败，请检查网络连接和 API 服务状态",
        }
        print(f"\n[警告] LLM 调用出现 {len(retry_audit.retry_records)} 次重试, {len(retry_audit.failure_records)} 次最终失败")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "所有 LLM 调用均一次成功，无网络波动questions",
        }

    # [2026-01 Added] 计算后20%低质量questions指标
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # [2026-01 Added] 收集并输出 Good/Bad Cases
    generation_states = stats.get("generation_states", {})
    good_bad_cases = collect_good_bad_cases(
        results=stats["results"],
        generation_states=generation_states,
        max_cases=3,
    )
    summary["good_cases"] = good_bad_cases.get("good_cases", [])
    summary["bad_cases"] = good_bad_cases.get("bad_cases", [])

    # 打印 Good/Bad Cases（在保存 summary 之前）
    print_good_bad_cases(good_bad_cases)

    # [2026-01 Added] 导出完整的 Good/Bad Case 展示文档
    if good_bad_cases.get("good_cases") or good_bad_cases.get("bad_cases"):
        try:
            from src.showcase import CaseCollector, CaseExporter, PromptHighlighter

            print("\n[Good/Bad Cases] 正在生成详细展示文档...")

            collector = CaseCollector(
                output_dir=config.output_dir,
                experiment_id=config.experiment_id,
            )
            highlighter = PromptHighlighter()
            exporter = CaseExporter(
                output_dir=config.output_dir,
                highlighter=highlighter,
            )

            # 收集完整数据
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
                    print(f"    [警告] 收集 good case unit_{c['unit_id']} 失败: {e}")

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
                    print(f"    [警告] 收集 bad case unit_{c['unit_id']} 失败: {e}")

            # 导出
            if full_good_cases or full_bad_cases:
                index_path = exporter.export_all_cases(
                    good_cases=full_good_cases,
                    bad_cases=full_bad_cases,
                    experiment_id=config.experiment_id,
                    prompt_level=config.pipeline.prompt_extraction.prompt_level or "C",
                )
                print(f"[Good/Bad Cases] 详细展示已生成: {index_path}")
            else:
                print("[Good/Bad Cases] 无有效数据可导出")

        except Exception as e:
            print(f"[Good/Bad Cases] 详细导出失败: {e}")

    summary_path = config.output_dir / "summary.json"
    # [2026-01 Added] 统一浮点数精度为3位小数
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # [2025-12 Added] 生成人类可读的 MD 报告
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[MD报告] 已生成: {md_result['md_report']}")
    except Exception as e:
        print(f"[MD报告] 生成失败: {e}")

    # [2025-12 Added] 保存High variance dimension aggregation报告
    high_variance_records = stats.get("high_variance_records", [])
    if high_variance_records:
        high_variance_report = {
            "experiment_id": config.experiment_id,
            "timestamp": timestamp,
            "total_high_variance_count": len(high_variance_records),
            "threshold_used": 50.0,  # with eval 模块中的 SCORE_DIFF_THRESHOLD 保持一致
            "description": "以下维度在多模型评估中出现显著评分差异（≥50分），建议人工复核",
            "summary": {
                "affected_units": len(set(r.get("unit_id") for r in high_variance_records)),
                "ai_centric_count": len([r for r in high_variance_records if r.get("eval_type") == "ai_centric"]),
                "pedagogical_count": len([r for r in high_variance_records if r.get("eval_type") == "pedagogical"]),
            },
            "by_eval_type": {
                "ai_centric": [r for r in high_variance_records if r.get("eval_type") == "ai_centric"],
                "pedagogical": [r for r in high_variance_records if r.get("eval_type") == "pedagogical"],
            },
            "by_question_type": {},
            "by_unit_id": {},
            "records": high_variance_records,
        }

        # 按questions型分组
        for r in high_variance_records:
            qt = r.get("question_type", "other")
            if qt not in high_variance_report["by_question_type"]:
                high_variance_report["by_question_type"][qt] = []
            high_variance_report["by_question_type"][qt].append(r)

        # 按unit_id分组
        for r in high_variance_records:
            uid = r.get("unit_id", "unknown")
            if uid not in high_variance_report["by_unit_id"]:
                high_variance_report["by_unit_id"][uid] = []
            high_variance_report["by_unit_id"][uid].append(r)

        high_variance_path = config.output_dir / "high_variance_report.json"
        with open(high_variance_path, "w", encoding="utf-8") as f:
            json.dump(high_variance_report, f, ensure_ascii=False, indent=2)

        # 打印详细的高方差汇总信息
        print(f"\n" + "=" * 60)
        print(f"[High variance dimension aggregation] Total {len(high_variance_records)} 条记录")
        print(f"=" * 60)
        print(f"  阈值: ≥50分差距")
        print(f"  涉及questions数: {high_variance_report['summary']['affected_units']}")
        print(f"  AI评估: {high_variance_report['summary']['ai_centric_count']} 条")
        print(f"  教育学评估: {high_variance_report['summary']['pedagogical_count']} 条")
        print(f"\n[按questions分组明细]")
        for uid, items in high_variance_report["by_unit_id"].items():
            print(f"  unit_id={uid} ({len(items)}items维度):")
            for item in items:
                eval_type_short = "AI" if item.get("eval_type") == "ai_centric" else "Ped"
                dim_name = item.get("dimension_name", item.get("dimension_id", "?"))
                print(f"    - [{eval_type_short}] {dim_name}: {item.get('max_model')}={item.get('max_score'):.0f} vs {item.get('min_model')}={item.get('min_score'):.0f} (差{item.get('score_diff'):.0f}分)")
        print(f"\n  报告已保存: {high_variance_path}")
        print(f"=" * 60)

    # [2025-12 Added] 保存Missing dimension aggregation报告
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
        print(f"\n[子集采样] 审计文件已保存:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")

    mode_label = "Subset mode" if (subset_size or subset_file) else "Full mode"
    print(f"\n[{mode_label}完成] 汇总已保存到: {summary_path}")

    # 写入 round_manifest（such as果启用了 round-id）
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
            # [2025-12 Refactored] 使用 macro F1 作为教育学汇总指标
            "ped_macro_f1": summary["ped_round_metrics"]["macro"]["f1"],
        }
        append_round_manifest(round_root, manifest_record)
# DEAD_CODE_CANDIDATE(reason=runtime_miss,trace=miss,refs=1)

    # [2025-12 Added] 自动提取questions并保存为JSON和Markdown格式
    try:
        from scripts.extract_questions import auto_extract_and_save
        print(f"\n[questions提取] 正在自动提取所有生成的questions...")
        extract_result = auto_extract_and_save(config.output_dir, config.experiment_id)
        if extract_result["success"]:
            print(f"[questions提取] 成功提取 {extract_result['total_questions']} 道questions")
            print(f"  - JSON格式: {extract_result['json_path']}")
            print(f"  - Markdown格式: {extract_result['markdown_path']}")
        else:
            print(f"[questions提取] 提取失败: {extract_result.get('error', '未知错误')}")
    except Exception as e:
        print(f"[questions提取] 提取过程出错: {e}")

    return summary


# ============================================================================
# Baseline evaluation mode（真questions直评）
# ============================================================================

def run_baseline_mode(config: ExperimentConfig, unit_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    [2025-12 Added] Baseline evaluation mode - 真questions直接评估

    功能：
    1. 使用 raw_material.json 中的真questions数据
    2. 结合 merged_kaocha_jk_cs.json 中的维度信息
    3. 跳过 Stage1 生成阶段
    4. 直接构造 Stage2Record 进行评估
    5. 获取真questions的基准评分

    Args:
        config: Experiment configuration
        unit_ids: 指定要评估的 unit_id 列表（None 表示评估全部）

    返回完整的统计报告。
    """
    from src.shared.llm_router import LLMRouter
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.evaluation.experiment_stats_reporter import ExperimentStatsReporter

    # [2026-01 Modified] baseline 模式默认不启用AI评估
    eval_mode = getattr(config, "eval_mode", None) or "gk"  # 默认仅 GK 教育学评估

    # 从 eval_mode 推断for构建 Stage2Record 的维度模式
    # 这决定了真questions数据中使用哪些维度信息（gk_dims / cs_dims）
    # [2026-01 Refactored] Removed gk+cs 模式，默认使用 gk
    if "gk" in eval_mode:
        record_dim_mode = "gk"
    elif "cs" in eval_mode:
        record_dim_mode = "cs"
    else:
        record_dim_mode = "gk"  # 仅 AI 评估时，默认使用 gk 维度

    # 评估模式标签
    eval_mode_label = {
        "ai": "仅 AI 评估",
        "gk": "仅 GK 教育学评估",
        "cs": "仅 CS 教育学评估",
        "ai+gk": "AI + GK 教育学评估",
        "ai+cs": "AI + CS 教育学评估",
    }.get(eval_mode, eval_mode)

    print("\n" + "=" * 80)
    print("[Baseline evaluation mode] 真questions直接评估（跳过 Stage1）")
    print(f"  评估模式: {eval_mode_label}")
    print("=" * 80)

    router = LLMRouter.from_config(config)
    evaluation_orchestrator = EvaluationOrchestrator(config, llm_router=router, eval_mode=eval_mode)

    # [2026-01 Refactored] baseline模式：输入用随机维度，gold用真questions维度
    # baseline_mode=True 表示：
    # - dimension_mapping_path 使用 merged_mix_dimension_jk_cs.json（随机维度）
    # - gold_dimension_path 使用 merged_kaocha_jk_cs.json（真questions维度）
    baseline_evaluator = BaselineEvaluator(baseline_mode=True)
    print(f"[baseline模式] 输入维度: 随机维度, Gold维度: 真questions维度")

    # 获取要评估的 unit_id 列表
    available_unit_ids = baseline_evaluator.get_all_unit_ids()
    if unit_ids:
        # 筛选出有效的 unit_ids
        all_unit_ids = [uid for uid in unit_ids if uid in available_unit_ids]
        if not all_unit_ids:
            print(f"[警告] 指定的 unit_id {unit_ids} 均不存在于真questions数据中")
            all_unit_ids = available_unit_ids
    else:
        all_unit_ids = available_unit_ids
    total_units = len(all_unit_ids)

    print(f"[基准评估] Total {total_units} 道真questions，评估模式: {eval_mode_label}")

    # 初始化统计报告器
    stats_reporter = ExperimentStatsReporter(
        experiment_id=config.experiment_id,
        run_mode="baseline",
        config_info={
            "eval_mode": eval_mode,
            "eval_models": router.get_eval_model_names(),
        },
    )

    # 统计变量
    evaluation_success = 0
    evaluation_failed = 0

    ai_scores: List[float] = []
    # [2025-12 Refactored] 教育学使用 P/R/F1 指标收集
    ped_metrics_list: List[Dict[str, Any]] = []
    # [2025-12 Added] 独立 GK/CS 评估指标收集
    gk_metrics_list: List[Dict[str, Any]] = []
    cs_metrics_list: List[Dict[str, Any]] = []

    # [2025-12-31 Added] 加载年份映射（for分年份统计）
    from src.shared.data_loader import DataLoader
    data_loader = DataLoader()
    unit_year_mapping = data_loader.load_unit_year_mapping()

    # [2025-12-28 Added] 按questions型分组统计
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

    # [2025-12 Added] 缺失维度收集
    all_missing_dims_records: List[Dict[str, Any]] = []

    for i, unit_id in enumerate(all_unit_ids):
        print(f"\n>>> [{i+1}/{total_units}] 评估真questions unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "question_type": None,
            "material_type": None,
            "ai_score": None,
            "ped_metrics": None,  # [2025-12 Refactored] 改为 P/R/F1 指标
            "error": None,
        }

        try:
            # 构建 Stage2Record（跳过 Stage1，根据 eval_mode 推断的维度模式筛选维度）
            # [2026-01 Refactored] baseline模式下，这里使用的是随机维度（merged_mix_dimension）
            stage2_record = baseline_evaluator.build_stage2_record(
                unit_id=unit_id,
                experiment_id=config.experiment_id,
                dim_mode=record_dim_mode,
            )

            if stage2_record is None:
                evaluation_failed += 1
                result_item["error"] = "无法构建 Stage2Record"
                print(f"    [SKIP] 无法构建 Stage2Record")
                all_results.append(result_item)
                continue

            # [2026-01 Refactored] baseline模式：用gold维度（真questions维度）替换dimension_idsfor教育学评估
            # 这样：输入给模型的是随机维度，但计算P/R/F1时用的是真questionsgold维度
            if baseline_evaluator.is_baseline_mode():
                gold_dim_ids, gold_gk_dims, gold_cs_dims = baseline_evaluator.get_gold_dimensions(
                    unit_id=unit_id,
                    dim_mode=record_dim_mode
                )
                # 保存输入维度信息（for记录）
                input_dim_ids = stage2_record.core_input.dimension_ids.copy() if stage2_record.core_input.dimension_ids else []
                # 替换为gold维度for评估
                stage2_record.core_input.dimension_ids = gold_dim_ids

            # 获取questions型和材料类型
            bq = baseline_evaluator.get_baseline_question(unit_id)
            qt = bq.question_type if bq else "unknown"
            mt = bq.material_type if bq else "未知"
            dim_count = len(bq.dimension_ids) if bq else 0

            result_item["question_type"] = qt
            result_item["material_type"] = mt

            # 执行 Stage2 评估
            evaluation_state = evaluation_orchestrator.run(stage2_record)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["error"] = "评估未完成"
                print(f"    [FAIL] 评估未完成")
                all_results.append(result_item)

                stats_reporter.add_result(
                    unit_id=unit_id,
                    question_type=qt,
                    material_type=mt,
                    source="baseline",
                    dimension_count=dim_count,
                    stage1_success=True,
                    stage2_success=False,
                    error_info="评估未完成",
                )
                continue

            # 提取评分
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            result_item["ai_score"] = ai_score
            result_item["ped_metrics"] = ped_metrics  # [2025-12 Refactored] P/R/F1 指标
            result_item["gk_metrics"] = gk_metrics  # [2025-12 Added] 独立 GK 评估指标
            result_item["cs_metrics"] = cs_metrics  # [2025-12 Added] 独立 CS 评估指标

            # [2025-12 Added] 提取评估维度明细（for summary.json 透明化）
            model_weights = dict(getattr(evaluation_orchestrator, "eval_model_weights", {}) or {})
            eval_details = _extract_eval_details(evaluation_state, model_weights)
            result_item["eval_details"] = eval_details

            evaluation_success += 1

            # [2025-12-28 Added] 标准化questions型for按questions型统计
            qt_normalized = qt if qt in question_type_counts else "other"
            question_type_counts[qt_normalized] += 1

            if ai_score is not None:
                ai_scores.append(ai_score)
                score_buckets[qt_normalized]["ai"].append(ai_score)
            if ped_metrics is not None:
                # [2025-12-31 Added] 添加年份和questions型信息（for分组统计）
                ped_metrics["year"] = unit_year_mapping.get(unit_id)
                ped_metrics["question_type"] = qt_normalized
                ped_metrics["unit_id"] = unit_id
                ped_metrics_list.append(ped_metrics)
                score_buckets[qt_normalized]["ped_f1"].append(ped_metrics["f1"])
                ped_metrics_by_qtype[qt_normalized].append(ped_metrics)
            # [2025-12 Added] 收集独立 GK/CS 评估指标
            if gk_metrics is not None:
                # [2025-12-31 Added] 添加年份和questions型信息
                gk_metrics["year"] = unit_year_mapping.get(unit_id)
                gk_metrics["question_type"] = qt_normalized
                gk_metrics["unit_id"] = unit_id
                gk_metrics_list.append(gk_metrics)
                gk_metrics_by_qtype[qt_normalized].append(gk_metrics)
            if cs_metrics is not None:
                # [2025-12-31 Added] 添加年份和questions型信息
                cs_metrics["year"] = unit_year_mapping.get(unit_id)
                cs_metrics["question_type"] = qt_normalized
                cs_metrics["unit_id"] = unit_id
                cs_metrics_list.append(cs_metrics)
                cs_metrics_by_qtype[qt_normalized].append(cs_metrics)

            # 添加到统计报告器（ped_f1 作为教育学分数指标）
            ped_f1_score = ped_metrics["f1"] * 100 if ped_metrics else None  # 转换为百分制
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

            # [2025-12 Added] 收集缺失维度记录
            missing_dims_item = _extract_missing_dimensions(evaluation_state, unit_id, qt)
            if missing_dims_item:
                all_missing_dims_records.append(missing_dims_item)

            ai_disp = f"{ai_score:.1f}" if isinstance(ai_score, (int, float)) else "N/A"
            ped_disp = f"F1={ped_metrics['f1']:.3f}" if ped_metrics else "N/A"
            print(f"    [OK] questions型={qt}, 材料={mt}, AI={ai_disp}, Ped={ped_disp}")

        except Exception as e:
            evaluation_failed += 1
            result_item["error"] = str(e)
            print(f"    [EXCEPTION] 评估异常: {e}")

            stats_reporter.add_result(
                unit_id=unit_id,
                question_type=result_item.get("question_type", "unknown"),
                material_type=result_item.get("material_type", "未知"),
                source="baseline",
                stage1_success=True,
                stage2_success=False,
                error_info=str(e),
            )

        all_results.append(result_item)

    # 计算平均分
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # 保存统计报告
    stats_json_path = config.output_dir / "baseline_stats_report.json"
    stats_csv_path = config.output_dir / "baseline_stats_report.csv"
    stats_reporter.save_json(stats_json_path)
    stats_reporter.save_csv(stats_csv_path)
    stats_reporter.print_summary()

    # 构建汇总
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 获取模型权重
    model_weights = router.get_eval_model_weights()

    # [2025-12 Refactored] 计算教育学维度的 micro/macro P/R/F1 汇总
    def _compute_ped_round_metrics(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk", success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        计算教育学维度的轮次汇总指标

        [Important Note]
        此函数的计算逻辑with pedagogical_eval.aggregate_round() 存在重复。
        高频维度定义统一来源: src.shared.dimension_config

        TODO: 理想情况下应该直接使用 PedagogicalRoundAggregation 的结果，
        而不是在这里重新计算。这需要更大的Refactored工作。
        """
        from src.evaluation.pedagogical_eval import get_high_freq_dims_by_mode, calculate_prf
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

        # [2025-12-31 Added] 过滤掉原始gold维度为空的questions
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

        # Micro 平均：汇总 TP/FP/FN 后计算
        total_tp = sum(m["tp"] for m in valid_metrics_list)
        total_fp = sum(m["fp"] for m in valid_metrics_list)
        total_fn = sum(m["fn"] for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # [2026-01 Added] Off-target 和 Success@k
        off_target = 1.0 - micro_precision
        success_count = sum(1 for m in valid_metrics_list if m.get("recall", 0) >= success_threshold)
        success_at_k = success_count / len(valid_metrics_list) if valid_metrics_list else 0.0

        # [2026-01-07 Fix] Macro 平均：维度视角（对每items维度计算P/R/F1，再取平均）
        # 收集所有维度的统计
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

        # 计算每items维度的 P/R/F1，再取平均
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
            "skipped_no_dims": skipped_no_dims,  # 【2025-12-31】原始无维度被跳过的questions数
            "off_target": off_target,
            "success_at_k": success_at_k,
            "success_threshold": success_threshold,
        }

        # [2026-01-17 Added] Bootstrap 置信区间计算
        if len(valid_metrics_list) >= 2:  # 至少需要2items样本才能计算CI
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

        # [2025-12 Added] 排除高频维度后的指标
        # [2025-12-31 Updated] such as果questions去除高频维度后gold为空，则跳过该questions（不计入统计）
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0  # 仅含高频维度被跳过的questions数
            excl_valid_results = []  # for计算 Success@k
            for m in valid_metrics_list:  # 【2025-12-31】使用过滤后的列表
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # 检查去除高频维度后是否还有剩余gold维度
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    # 该questions只有高频维度，跳过不计入统计
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, p_ex, r_ex, f1_ex = calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                excl_valid_results.append({'recall': r_ex})

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # [2026-01 Added] 排除高频后的 Off-target 和 Success@k
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
                "skipped_only_high_freq": skipped_only_high_freq,  # 被跳过的仅含高频维度questions数
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            # [2026-01-17 Added] 排除高频后的 Bootstrap 置信区间
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

    # [2025-12-28 Added] 按questions型分组的教育学指标汇总
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
        # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
        "ped_round_metrics": ped_round_metrics,
        # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
        "gk_round_metrics": gk_round_metrics if gk_metrics_list else None,
        "cs_round_metrics": cs_round_metrics if cs_metrics_list else None,
        # [2025-12-28 Added] 按questions型分组的教育学指标汇总
        "ped_round_metrics_by_question_type": ped_round_metrics_by_qtype if ped_round_metrics_by_qtype else None,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype if gk_round_metrics_by_qtype else None,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype if cs_round_metrics_by_qtype else None,
        # [2025-12-28 Added] 按questions型分组的平均分
        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_buckets.items() if v["ai"]},
        "question_type_distribution": question_type_counts,
        "eval_models": router.get_eval_model_names(),
        # [2025-12 Added] 模型权重透明化
        "model_weights": model_weights,
        "score_calculation": {
            "ai_eval": {
                "method": "weighted_average",
                "description": "各AI维度得分按维度权重加权平均，模型间按model_weights加权",
                "formula": "overall = Σ(dimension_score × dimension_weight) / Σ(dimension_weight)",
                "dimension_details": "每items维度包含: score(得分), weight(权重), contribution(贡献值=score×weight), level(etc级)",
                "model_details": "每items模型包含: average_score(维度平均分), weight(模型权重), contribution(贡献值=avg×weight)"
            },
            "pedagogical_eval": {
                "method": "hit_based_prf",
                "description": "基于20维度命中（hit=true/false）的 Precision/Recall/F1 评估",
                "micro_average": "汇总所有questions的 TP/FP/FN 后计算全局 P/R/F1",
                "macro_average": "各questions P/R/F1 的简单算术平均",
                "metrics": {
                    "precision": "TP / (TP + FP) - 预测命中的维度中有多少是正确的",
                    "recall": "TP / (TP + FN) - 金标准维度中有多少被正确预测",
                    "f1": "2 * P * R / (P + R) - P/R 的调和平均"
                }
            }
        },
        # [2026-01 Added] 完整 LLM 配置记录（for auto-discover 模式）
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # [2026-01 Added] 实验类型标记（for auto-discover 模式分类）
        "experiment_type": {
            "is_baseline": True,
            "is_negative_control": False,
            "is_hard_negative_control": False,
            "is_lowfreq": False,
            "lowfreq_k": None,
        },
        # [2026-01 Added] config 字段（with full mode 保持一致）
        "config": {
            "generator_model": None,  # baseline 模式不使用生成模型
            "dim_mode": eval_mode,
            "prompt_level": "Baseline",
        },
        "timestamp": timestamp,
        "results": all_results,
    }

    # [2025-12 Added] 添加 LLM 调用重试审计信息
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM 调用过程中发生了重试或失败，请检查网络连接和 API 服务状态",
        }
        print(f"\n[警告] LLM 调用出现 {len(retry_audit.retry_records)} 次重试, {len(retry_audit.failure_records)} 次最终失败")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "所有 LLM 调用均一次成功，无网络波动questions",
        }

    # [2026-01 Added] 计算后20%低质量questions指标
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # 保存汇总到 summary.json
    summary_path = config.output_dir / "summary.json"
    # [2026-01 Added] 统一浮点数精度为3位小数
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # [2025-12 Added] 生成人类可读的 MD 报告
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[MD报告] 已生成: {md_result['md_report']}")
    except Exception as e:
        print(f"[MD报告] 生成失败: {e}")

    # [2025-12 Added] 保存Missing dimension aggregation报告
    if all_missing_dims_records:
        _save_missing_dimensions_report(
            missing_dims_records=all_missing_dims_records,
            output_dir=config.output_dir,
            experiment_id=config.experiment_id,
        )

    # [2025-12-31 Added] Print PRF statistics summary（similar to recalc_prf_with_voted_gold.py style）
    from src.shared.report_generator import print_prf_summary

    # 根据 eval_mode 选择打印哪些指标
    if "gk" in eval_mode and gk_metrics_list:
        print_prf_summary(gk_metrics_list, dim_mode="gk", title=f"基准评估 - Total {total_units} questions")
    if "cs" in eval_mode and cs_metrics_list:
        print_prf_summary(cs_metrics_list, dim_mode="cs", title=f"基准评估 - Total {total_units} questions")

    print(f"\n[基准评估完成] 汇总已保存到: {summary_path}")

    return summary


# ============================================================================
# [2025-12 Added] questionsextract mode
# ============================================================================

def run_extract_mode(args):
    """
    从实验目录提取生成的questions。

    用法示例:
        python cli.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427
        python cli.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427 --unit-id 10
        python cli.py --run-mode extract --extract-dir outputs/EXP_BASELINE_gk_C_20251209_213427 --extract-format markdown --extract-output questions.md
    """
    from scripts.extract_questions import (
        extract_questions,
        format_question_text,
        format_question_markdown,
        print_summary,
    )

    exp_dir = Path(args.extract_dir)
    if not exp_dir.exists():
        print(f"[ERROR] 目录不存在: {exp_dir}")
        sys.exit(1)

    # 检查 stage2 目录是否存在
    stage2_dir = exp_dir / "stage2"
    if not stage2_dir.exists():
        print(f"[ERROR] stage2 目录不存在: {stage2_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("questionsextract mode")
    print("=" * 60)
    print(f"  实验目录:   {exp_dir}")
    print(f"  输出格式:   {args.extract_format}")
    if args.unit_id:
        print(f"  指定 Unit:  {args.unit_id}")
    if args.extract_output:
        print(f"  输出文件:   {args.extract_output}")
    print("=" * 60 + "\n")

    try:
        # 解析 unit_id
        unit_id = None
        if args.unit_id:
            try:
                unit_id = int(args.unit_id)
            except ValueError:
                print(f"[ERROR] unit-id 必须是数字: {args.unit_id}")
                sys.exit(1)

        # 提取questions
        questions = extract_questions(exp_dir, unit_id)

        if not questions:
            print("[INFO] 未找到任何questions")
            return

        # 打印统计
        print_summary(questions)

        # 格式化输出
        include_scores = not args.no_scores

        if args.extract_format == "json":
            output = json.dumps(questions, ensure_ascii=False, indent=2)
        elif args.extract_format == "markdown":
            output = "\n".join(format_question_markdown(q, include_scores) for q in questions)
        else:  # text
            output = "\n".join(format_question_text(q, include_scores) for q in questions)

        # 输出
        if args.extract_output:
            output_path = Path(args.extract_output)
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"[SUCCESS] 已保存到: {output_path}")
        else:
            # 处理 Windows 终端编码questions
            try:
                print(output)
            except UnicodeEncodeError:
                # Removed特殊字符后重试
                safe_output = output.replace("✓", "[v]").replace("✗", "[x]").replace("✅", "[PASS]").replace("❌", "[FAIL]").replace("⏳", "[...]")
                print(safe_output)

    except Exception as e:
        print(f"[ERROR] 提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# [2025-12 Added] Stage1 单独运行模式
# ============================================================================

def run_stage1_only_mode(
    config: ExperimentConfig,
    subset_size: Optional[int] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    subset_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [2025-12 Added] 仅运行 Stage1（生成阶段），不进行 Stage2 评估。

    for需要切换网络环境的场景：
    - Stage1 使用国内网络（such as豆包模型）
    - Stage2 稍后使用国外网络（such as DMX 海外版）

    输出结构：
    - stage2/ 目录下保存每道questions的 generation_state.json（含 stage2_record）
    - stage1_summary.json 记录本次 Stage1 运行的汇总信息
    - subset_unit_ids.json 保存运行的 unit_id 列表（for stage2-only 模式加载）
    """
    from src.shared.data_loader import DataLoader
    from src.shared.llm_router import LLMRouter
    from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
    from src.generation.utils.subset_sampler import build_subset_unit_ids, load_subset_from_file

    print("\n" + "=" * 80)
    print("[Stage1-Only 模式] 仅运行出questions阶段（跳过 Stage2 评估）")
    if subset_size or subset_file:
        print("[Subset mode] 开始运行子集 unit_id")
    else:
        print("[Full mode] 开始运行所有 unit_id")
    print("=" * 80)

    router = LLMRouter.from_config(config)

    # ---------- 决定 unit_id 列表 ----------
    data_loader = DataLoader()
    mappings = data_loader.load_question_dimension_mappings()

    total_units = len(mappings)
    print(f"[数据集] 总 unit 数量: {total_units}")

    subset_result = None

    if subset_file:
        unit_ids_to_run = [str(x) for x in load_subset_from_file(Path(subset_file))]
        print(f"[Subset mode] 从文件加载 {len(unit_ids_to_run)} items unit_id")
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
        print(f"[Subset mode] 采样 {len(unit_ids_to_run)} items unit_id (strategy={subset_strategy}, seed={subset_seed})")
    else:
        unit_ids_to_run = [str(m.unit_id) for m in mappings]
        print(f"[Full mode] 运行 {len(unit_ids_to_run)} items unit_id")

    # CLI 驱动 unit_id
    config.pipeline.agent1.material_selection_strategy = "manual"

    generation_orchestrator = GenerationOrchestrator(config, llm_router=router)

    # 统计变量
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
            # -------- 仅 Stage 1 --------
            generation_state = generation_orchestrator.run_single(unit_id=uid)
            stage2_record = getattr(generation_state, "stage2_record", None)

            if stage2_record is None:
                generation_failed += 1
                result_item["stage1_status"] = "no_stage2_record"
                result_item["has_stage2_record"] = False

                # 统计 agent 错误
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

                print(f"    [SKIP] Stage1 未产出 stage2_record")
                all_results.append(result_item)
                continue

            generation_success += 1
            stage2_record_count += 1
            result_item["stage1_status"] = "success"
            result_item["has_stage2_record"] = True

            # 提取questions型
            qt = getattr(stage2_record.core_input, "question_type", None) or "other"
            if qt not in question_type_counts:
                qt = "other"
            question_type_counts[qt] += 1
            result_item["question_type"] = qt

            # 提取维度信息
            dimension_ids = getattr(stage2_record.core_input, "dimension_ids", []) or []
            result_item["dimension_ids"] = dimension_ids
            result_item["dimension_count"] = len(dimension_ids)

            print(f"    [OK] questions型={qt}, 维度数={len(dimension_ids)}, dim_mode={config.pipeline.agent1.dimension_mode}")

        except Exception as e:
            print(f"    [EXCEPTION] 生成异常: {e}")
            agent_errors["agent1"] += 1
            result_item["stage1_status"] = "error"
            result_item["skip_reason"] = f"stage1_exception: {e}"
            generation_failed += 1

        all_results.append(result_item)

    # 构建汇总
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
        # [2026-01 Added] 完整 LLM 配置记录（for auto-discover 模式）
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # [2026-01 Added] 实验类型标记（for auto-discover 模式分类）
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

        # 【重要】标记这是 stage1-only 输出，供 stage2-only 识别
        "stage1_only_output": True,
        "stage2_pending": True,
    }

    # [2025-12 Added] 添加 LLM 调用重试审计信息
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM 调用过程中发生了重试或失败，请检查网络连接和 API 服务状态",
        }
        print(f"\n[警告] LLM 调用出现 {len(retry_audit.retry_records)} 次重试, {len(retry_audit.failure_records)} 次最终失败")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "所有 LLM 调用均一次成功，无网络波动questions",
        }

    # 保存 stage1_summary.json
    summary_path = config.output_dir / "stage1_summary.json"
    # [2026-01 Added] 统一浮点数精度为3位小数
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # [2025-12 Added] 生成人类可读的 MD 报告
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=True,
        )
        if "md_report" in md_result:
            print(f"[MD报告] 已生成: {md_result['md_report']}")
    except Exception as e:
        print(f"[MD报告] 生成失败: {e}")

    # 保存 unit_ids 列表（供 stage2-only 使用）
    if subset_result:
        subset_unit_ids_path = config.output_dir / "subset_unit_ids.json"
        subset_stats_path = config.output_dir / "subset_stats.json"
        subset_result.save_unit_ids_json(subset_unit_ids_path)
        subset_result.save_stats_json(subset_stats_path)
        print(f"\n[子集采样] 审计文件已保存:\n  - {subset_unit_ids_path}\n  - {subset_stats_path}")
    else:
        # Full mode也保存 unit_ids
        unit_ids_path = config.output_dir / "subset_unit_ids.json"
        with open(unit_ids_path, "w", encoding="utf-8") as f:
            json.dump(unit_ids_to_run, f, ensure_ascii=False, indent=2)

    mode_label = "Subset mode" if (subset_size or subset_file) else "Full mode"
    print(f"\n[{mode_label} Stage1-Only 完成] 汇总已保存到: {summary_path}")

    # 写入 round_manifest（such as果启用了 round-id）
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
# [2025-12 Added] Stage1 ablation experiment模式（无维度提示词）
# 【2025-12-28 Refactored】还原为 181 questions模式，复用 BaselineEvaluator 构建 Stage2Record
# ============================================================================

def run_ablation_nodim_mode(
    config: ExperimentConfig,
    subset_size: Optional[int] = None,
    subset_strategy: str = "stratified",
    subset_seed: int = 42,
    subset_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    【2025-12-28 Refactored】Stage1 ablation experiment - 无维度提示词直接生成。

    消融目的：验证 Stage1 多 Agent 流程（维度匹配、提示词合成、证据锚点、验证）的价值。

    【流程】
    1. 遍历原始 181 items unit_id
    2. Stage1 消融：对每items unit_id，根据原questions型标签，用简单提示词生成一道questions
    3. Stage2 正常：使用该questions原有的维度标签，复用 BaselineEvaluator.build_stage2_record
    4. 只替换生成的questions干/选项/答案要点，维度信息完全保留

    with正常 full 模式的区别：
    - Stage1 跳过所有 Agent（维度匹配、提示词合成、证据锚点、验证）
    - 使用 AblationGenerator 直接生成questions
    - Stage2 完全复用现有流程
    """
    from src.shared.llm_router import LLMRouter
    from src.generation.ablation_generator import AblationGenerator, AblationGeneratorConfig
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.shared.schemas import Stage2CoreInput, Stage2Record, Stage1Meta
    from dataclasses import asdict
    import json

    # [2026-01 Modified] 默认不启用AI评估
    eval_mode = getattr(config, "eval_mode", None) or "gk"
    eval_mode_label = {
        "ai": "仅 AI 评估",
        "gk": "仅 GK 教育学评估",
        "cs": "仅 CS 教育学评估",
        "ai+gk": "AI + GK 教育学评估",
        "ai+cs": "AI + CS 教育学评估",
    }.get(eval_mode, eval_mode)

    # 根据 eval_mode 推断 record_dim_mode（for加载维度数据）
    # [2026-01 Refactored] Removedgk+cs模式，优先使用gk
    if "gk" in eval_mode:
        record_dim_mode = "gk"
    elif "cs" in eval_mode:
        record_dim_mode = "cs"
    else:
        record_dim_mode = "gk"  # 仅 AI 评估时，默认使用gk维度

    print("\n" + "=" * 80)
    print("[ablation experiment模式] Stage1 消融 - 无维度提示词直接生成")
    print("[模式说明] 181questions模式，Stage1跳过所有Agent，Stage2复用原有维度标签")
    print(f"[评估模式] {eval_mode_label}")
    print("=" * 80)

    router = LLMRouter.from_config(config)

    # ---------- 初始化 BaselineEvaluator（获取原始questions信息和维度）----------
    baseline_evaluator = BaselineEvaluator()
    all_unit_ids = baseline_evaluator.get_all_unit_ids()
    total_units = len(all_unit_ids)

    print(f"[数据集] 总questions数: {total_units}")

    # ---------- 初始化消融生成器 ----------
    gen_client = router.get_generator_client()
    ablation_config = AblationGeneratorConfig()
    generator = AblationGenerator(
        llm_client=gen_client,
        config=ablation_config,
    )

    # ---------- 初始化评估器（使用前面获取的 eval_mode）----------
    evaluation_orchestrator = EvaluationOrchestrator(
        config,
        llm_router=router,
        eval_mode=eval_mode,
    )

    # ---------- 统计变量 ----------
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

    # [2025-12-31 Added] 加载年份映射（for分年份统计）
    from src.shared.data_loader import DataLoader
    data_loader = DataLoader()
    unit_year_mapping = data_loader.load_unit_year_mapping()

    # ---------- 遍历每items unit_id ----------
    for i, unit_id in enumerate(all_unit_ids, 1):
        print(f"\n>>> [{i}/{total_units}] 处理 unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "question_type": None,
            "stage1_status": None,
            "stage2_status": None,
            "ai_overall_score": None,  # with full 模式字段名一致
            "ped_metrics": None,  # with full 模式一致（使用 gk 作为默认）
            "gk_metrics": None,
            "cs_metrics": None,
            "skip_reason": None,  # with full 模式一致
        }

        # 获取原始questions信息
        bq = baseline_evaluator.get_baseline_question(unit_id)
        if not bq:
            generation_failed += 1
            result_item["skip_reason"] = "无法获取原始questions信息"
            result_item["stage1_status"] = "skip"
            print(f"    [SKIP] 无法获取原始questions信息")
            all_results.append(result_item)
            continue

        # 原questions型（for生成对应类型的questions）
        original_qtype = bq.question_type  # "single-choice" 或 "essay"
        qtype_cn = "选择questions" if original_qtype == "single-choice" else "简答questions"
        material_text = bq.material_text

        result_item["question_type"] = original_qtype
        question_type_distribution[original_qtype] = question_type_distribution.get(original_qtype, 0) + 1

        # ========== Stage1 消融：简单生成 ==========
        try:
            generated_question = generator.generate(
                material=material_text,
                question_type=qtype_cn,
                unit_id=str(unit_id),
            )

            if "[生成失败]" in generated_question.stem:
                generation_failed += 1
                result_item["stage1_status"] = "fail"
                result_item["skip_reason"] = "生成失败"
                print(f"    [Stage1] 生成失败")
                all_results.append(result_item)
                continue

            generation_success += 1
            result_item["stage1_status"] = "success"
            print(f"    [Stage1] 生成成功: questions干长度={len(generated_question.stem)}")

        except Exception as e:
            generation_failed += 1
            result_item["stage1_status"] = "error"
            result_item["skip_reason"] = f"stage1_exception: {e}"
            print(f"    [Stage1] 异常: {e}")
            all_results.append(result_item)
            continue

        # ========== Stage2：复用 BaselineEvaluator 构建 Stage2Record ==========
        try:
            # 先获取原始的 Stage2Record（包含正确的维度信息）
            # record_dim_mode 已在函数开头根据 eval_mode 计算
            original_stage2_record = baseline_evaluator.build_stage2_record(
                unit_id=str(unit_id),
                experiment_id=config.experiment_id,
                dim_mode=record_dim_mode,
            )

            if original_stage2_record is None:
                evaluation_failed += 1
                result_item["stage2_status"] = "skip"
                result_item["skip_reason"] = "无法构建 Stage2Record"
                print(f"    [Stage2] 无法构建 Stage2Record")
                all_results.append(result_item)
                continue

            # 替换questions干/选项/答案要点为消融生成的内容
            core = original_stage2_record.core_input

            # 构建选项列表
            options_list = None
            if generated_question.options:
                options_list = [
                    {"label": opt.label, "content": opt.content, "is_correct": opt.is_correct}
                    for opt in generated_question.options
                ]

            # 构建答案要点列表
            answer_points_list = None
            if generated_question.answer_points:
                answer_points_list = [
                    {"point": pt.point, "score": pt.score}
                    for pt in generated_question.answer_points
                ]

            # 创建新的 Stage2CoreInput，保留原维度，替换生成内容
            new_core_input = Stage2CoreInput(
                experiment_id=core.experiment_id,
                unit_id=core.unit_id,
                material_text=core.material_text,
                question_type=core.question_type,
                # 替换为消融生成的内容
                stem=generated_question.stem,
                explanation=generated_question.explanation or "",
                # 保留原有维度信息
                gk_dims=core.gk_dims,
                cs_dims=core.cs_dims,
                exam_skill=core.exam_skill,
                dimension_ids=core.dimension_ids,
                # 替换为消融生成的选项/答案
                options=options_list,
                correct_answer=generated_question.correct_answer,
                answer_points=answer_points_list,
                total_score=generated_question.total_score,
            )

            # 构建新的 Stage1Meta（标记为消融模式）
            new_stage1_meta = Stage1Meta(
                ablation_skip_agent="all",
            )

            # 构建新的 Stage2Record
            stage2_record = Stage2Record(
                core_input=new_core_input,
                stage1_meta=new_stage1_meta,
            )

            # 执行评估
            evaluation_state = evaluation_orchestrator.run(stage2_record)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["stage2_status"] = "fail"
                result_item["skip_reason"] = f"评估未完成: {getattr(evaluation_state, 'current_stage', 'unknown')}"
                print(f"    [Stage2] 评估未完成")
                all_results.append(result_item)
                continue

            evaluation_success += 1
            result_item["stage2_status"] = "success"

            # 提取评分
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            # with full 模式字段名一致（根据 eval_mode 动态选择）
            result_item["ai_overall_score"] = ai_score
            # 【Fix】根据 eval_mode 选择正确的 ped_metrics
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
                # [2025-12-31 Added] 添加年份和questions型信息
                gk_metrics["year"] = unit_year_mapping.get(unit_id)
                gk_metrics["question_type"] = original_qtype
                gk_metrics["unit_id"] = unit_id
                ped_gk_results.append(gk_metrics)
                ped_gk_by_qtype[original_qtype].append(gk_metrics)
            if cs_metrics:
                # [2025-12-31 Added] 添加年份和questions型信息
                cs_metrics["year"] = unit_year_mapping.get(unit_id)
                cs_metrics["question_type"] = original_qtype
                cs_metrics["unit_id"] = unit_id
                ped_cs_results.append(cs_metrics)
                ped_cs_by_qtype[original_qtype].append(cs_metrics)

            ai_display = f"{ai_score:.1f}" if ai_score is not None else "N/A"
            gk_f1 = gk_metrics.get("f1", 0) if gk_metrics else 0
            cs_f1 = cs_metrics.get("f1", 0) if cs_metrics else 0
            print(f"    [Stage2] 成功: AI={ai_display}, GK_F1={gk_f1:.2f}, CS_F1={cs_f1:.2f}")

            # 【注意】evaluation_orchestrator.run() 已经自动保存了 evaluation_state.json
            # 不再手动调用 _save_ablation_unit_result，避免覆盖为不同格式

        except Exception as e:
            evaluation_failed += 1
            result_item["stage2_status"] = "error"
            result_item["skip_reason"] = f"stage2_exception: {e}"
            import traceback
            print(f"    [Stage2] 异常: {e}")
            traceback.print_exc()

        all_results.append(result_item)

    # ---------- 计算汇总统计 ----------
    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    def _compute_ped_round_metrics_local(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk", success_threshold: float = 0.8) -> Dict[str, Any]:
        """
        计算教育学维度的轮次汇总指标（with full 模式相同的逻辑）

        [Important Note]
        此函数的计算逻辑with pedagogical_eval.aggregate_round() 存在重复。
        高频维度定义统一来源: src.shared.dimension_config
        """
        from src.evaluation.pedagogical_eval import get_high_freq_dims_by_mode, calculate_prf
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

        # [2025-12-31 Added] 过滤掉原始gold维度为空的questions
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

        # Micro 平均：汇总 TP/FP/FN 后计算
        total_tp = sum(m.get("tp", 0) for m in valid_metrics_list)
        total_fp = sum(m.get("fp", 0) for m in valid_metrics_list)
        total_fn = sum(m.get("fn", 0) for m in valid_metrics_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # [2026-01 Added] Off-target 和 Success@k
        off_target = 1.0 - micro_precision
        success_count = sum(1 for m in valid_metrics_list if m.get("recall", 0) >= success_threshold)
        success_at_k = success_count / len(valid_metrics_list) if valid_metrics_list else 0.0

        # [2026-01-07 Fix] Macro 平均：维度视角（对每items维度计算P/R/F1，再取平均）
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
            "skipped_no_dims": skipped_no_dims,  # 【2025-12-31】原始无维度被跳过的questions数
            # [2026-01 Added] Off-target 和 Success@k
            "off_target": off_target,
            "success_at_k": success_at_k,
            "success_threshold": success_threshold,
        }

        # [2026-01-17 Added] Bootstrap 置信区间计算
        if len(valid_metrics_list) >= 2:  # 至少需要2items样本才能计算CI
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

        # [2025-12 Added] 排除高频维度后的指标
        # [2025-12-31 Updated] such as果questions去除高频维度后gold为空，则跳过该questions（不计入统计）
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0  # 仅含高频维度被跳过的questions数
            excl_valid_results = []  # [2026-01 Added] for计算 Success@k
            for m in valid_metrics_list:  # 【2025-12-31】使用过滤后的列表
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # 检查去除高频维度后是否还有剩余gold维度
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    # 该questions只有高频维度，跳过不计入统计
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, _, r_ex, _ = calculate_prf(gold, pred, exclude_dims=high_freq_dims)
                excl_total_tp += tp_ex
                excl_total_fp += fp_ex
                excl_total_fn += fn_ex
                excl_valid_results.append({'recall': r_ex})

            excl_micro_p = excl_total_tp / (excl_total_tp + excl_total_fp) if (excl_total_tp + excl_total_fp) > 0 else 0.0
            excl_micro_r = excl_total_tp / (excl_total_tp + excl_total_fn) if (excl_total_tp + excl_total_fn) > 0 else 0.0
            excl_micro_f1 = 2 * excl_micro_p * excl_micro_r / (excl_micro_p + excl_micro_r) if (excl_micro_p + excl_micro_r) > 0 else 0.0

            # [2026-01 Added] 计算排除高频后的 Off-target 和 Success@k
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
                "skipped_only_high_freq": skipped_only_high_freq,  # 被跳过的仅含高频维度questions数
                # [2026-01 Added] Off-target 和 Success@k
                "off_target": excl_off_target,
                "success_at_k": excl_success_at_k,
                "success_threshold": success_threshold,
            }

            # [2026-01-17 Added] 排除高频后的 Bootstrap 置信区间
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

    # 【Fix】使用正确的 micro/macro 计算方法（with full 模式一致）
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

    # 获取当前时间戳for summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "ablation_mode": "nodim",

        "total_units": total_units,
        "run_total": total_units,
        "generation_success": generation_success,
        "generation_failed": generation_failed,
        "evaluation_success": evaluation_success,
        "evaluation_failed": evaluation_failed,
        "stage2_skipped": generation_failed,  # with full 模式兼容

        "avg_ai_score": avg_ai_score,
        "avg_ai_score_by_question_type": avg_ai_by_qtype,

        # [2026-01 Refactored] 根据 eval_mode 动态选择 ped_round_metrics
        # - such as果有 gk 评估，使用 gk_round_metrics
        # - such as果只有 cs 评估，使用 cs_round_metrics
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
            "prompt_level": "N/A (ablation-nodim)",
            "stage1_skip_agent": "all",
        },
        # [2026-01 Added] 完整 LLM 配置记录（for auto-discover 模式）
        "llm_config": {
            "stage1_model": api_config.STAGE1_MODEL,
            "stage1_temperature": api_config.STAGE1_TEMPERATURE,
            "stage1_preset": api_config.STAGE1_PRESET,
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # [2026-01 Added] 实验类型标记（for auto-discover 模式分类）
        "experiment_type": {
            "is_baseline": False,
            "is_negative_control": False,
            "is_hard_negative_control": False,
            "is_lowfreq": False,
            "lowfreq_k": None,
            "is_ablation": True,
        },
        "stage1_skip_agent": "all",
        "eval_models": STAGE2_EVAL_MODELS,
        "timestamp": timestamp,
        "results": all_results,
    }

    # [2026-01 Added] 计算后20%低质量questions指标
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), record_dim_mode
    )

    # 保存汇总
    summary_path = config.output_dir / "summary.json"
    # [2026-01 Added] 统一浮点数精度为3位小数
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # [2025-12-31 Added] Print PRF statistics summary（similar to recalc_prf_with_voted_gold.py style）
    from src.shared.report_generator import print_prf_summary

    # 根据 eval_mode 选择打印哪些指标
    if "gk" in eval_mode and ped_gk_results:
        print_prf_summary(ped_gk_results, dim_mode="gk", title=f"ablation experiment - Total {total_units} questions")
    if "cs" in eval_mode and ped_cs_results:
        print_prf_summary(ped_cs_results, dim_mode="cs", title=f"ablation experiment - Total {total_units} questions")

    print(f"\n[ablation experiment] 汇总已保存到: {summary_path}")

    # 【Added】生成人类可读的 MD 报告（with full 模式一致）
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=config.output_dir,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[MD报告] 已生成: {md_result['md_report']}")
    except Exception as e:
        print(f"[MD报告] 生成失败: {e}")

    # 打印汇总（根据 eval_mode 动态显示相关评估结果）
    print("\n" + "=" * 80)
    print(f"[ablation experiment完成] 汇总统计（ablation-nodim, eval_mode={eval_mode}）")
    print("=" * 80)
    print(f"  总questions数: {total_units}")
    print(f"  生成成功: {generation_success}, 失败: {generation_failed}")
    print(f"  评估成功: {evaluation_success}, 失败: {evaluation_failed}")

    # 【2025-12 优化】根据 eval_mode 动态显示相关评估结果
    if "ai" in eval_mode:
        print()
        print(f"  【AI 维度评估】")
        print(f"    平均分: {avg_ai_score:.2f}")
        for qt, score in avg_ai_by_qtype.items():
            print(f"      - {qt}: {score:.2f}")

    if "gk" in eval_mode:
        print()
        print(f"  【GK 教育学维度评估】")
        print(f"    Micro: P={gk_round_metrics['micro']['precision']:.4f}, R={gk_round_metrics['micro']['recall']:.4f}, F1={gk_round_metrics['micro']['f1']:.4f}")
        print(f"    Macro: P={gk_round_metrics['macro']['precision']:.4f}, R={gk_round_metrics['macro']['recall']:.4f}, F1={gk_round_metrics['macro']['f1']:.4f}")
        print(f"    TP={gk_round_metrics['micro']['total_tp']}, FP={gk_round_metrics['micro']['total_fp']}, FN={gk_round_metrics['micro']['total_fn']}")
        # [2025-12 Added] 排除高频维度后的指标
        if gk_round_metrics.get("exclude_high_freq"):
            excl = gk_round_metrics["exclude_high_freq"]
            print(f"    排除高频维度 {excl['excluded_dims']} 后:")
            print(f"      Micro: P={excl['micro_precision']:.4f}, R={excl['micro_recall']:.4f}, F1={excl['micro_f1']:.4f}")

    if "cs" in eval_mode:
        print()
        print(f"  【CS 教育学维度评估】")
        print(f"    Micro: P={cs_round_metrics['micro']['precision']:.4f}, R={cs_round_metrics['micro']['recall']:.4f}, F1={cs_round_metrics['micro']['f1']:.4f}")
        print(f"    Macro: P={cs_round_metrics['macro']['precision']:.4f}, R={cs_round_metrics['macro']['recall']:.4f}, F1={cs_round_metrics['macro']['f1']:.4f}")
        print(f"    TP={cs_round_metrics['micro']['total_tp']}, FP={cs_round_metrics['micro']['total_fp']}, FN={cs_round_metrics['micro']['total_fn']}")
        # [2025-12 Added] 排除高频维度后的指标
        if cs_round_metrics.get("exclude_high_freq"):
            excl = cs_round_metrics["exclude_high_freq"]
            print(f"    排除高频维度 {excl['excluded_dims']} 后:")
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
    """保存ablation experiment单items unit 的结果"""
    import json
    from dataclasses import asdict

    stage2_dir = output_dir / "stage2" / f"unit_{unit_id}"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # 提取评估结果
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
    """将字典转换为 GeneratedQuestion 对象"""
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
    """保存ablation experiment的评估结果"""
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
    """聚合教育学评估的 P/R/F1 指标"""
    if not results:
        return {"micro": {}, "macro": {}}

    # 收集所有结果的 P/R/F1
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

    # 计算 macro 平均
    macro = {
        "precision": sum(precisions) / len(precisions) if precisions else 0,
        "recall": sum(recalls) / len(recalls) if recalls else 0,
        "f1": sum(f1s) / len(f1s) if f1s else 0,
    }

    # micro 暂时使用 macro（需要更详细的 TP/FP/FN 信息来计算真正的 micro）
    micro = macro.copy()

    return {"micro": micro, "macro": macro}


# ============================================================================
# [2025-12 Added] Stage2 单独运行模式
# ============================================================================

def run_stage2_only_mode(
    config: ExperimentConfig,
    stage1_dir: str,
    stage2_output_dir: Optional[str] = None,
    incremental_ai: bool = False,
) -> Dict[str, Any]:
    """
    [2025-12 Added] 仅运行 Stage2（评估阶段），读取已有的 Stage1 输出。

    for需要切换网络环境的场景：
    - Stage1 已使用国内网络完成（such as豆包模型）
    - Stage2 使用国外网络进行评估（such as DMX 海外版）

    【2026-01 增强】支持跨维度模式评估：
    - such as果原始数据中缺少当前 eval_mode 需要的维度（such as cs_dims 为空）
    - 自动从 merged_kaocha_jk_cs.json 中根据 unit_id 重新加载维度信息

    Args:
        config: Experiment configuration
        stage1_dir: Stage1 输出目录路径
        stage2_output_dir: [2025-12 Added] 评估结果输出目录（可选）
            - such as不指定，则在 stage1_dir 原目录追加评估结果
            - such as指定，则将questions复制到新目录并在新目录中进行评估

    输出结构：
    - 在 stage1_dir 原有结构上追加评估结果
    - Updated evaluation_state.json 文件
    - 生成 summary.json（完整的实验汇总）
    """
    import shutil
    from src.shared.llm_router import LLMRouter
    from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.evaluation.baseline_evaluator import BaselineEvaluator
    from src.shared.schemas import Stage2Record, Stage2CoreInput, Stage1Meta

    stage1_path = Path(stage1_dir)
    stage2_src_subdir = stage1_path / "stage2"

    # [2025-12 Added] such as果指定了独立输出目录，则复制questions数据到新目录
    if stage2_output_dir:
        output_path = Path(stage2_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stage2_subdir = output_path / "stage2"
        stage2_subdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("[Stage2-Only 模式] 仅运行评估阶段（独立输出目录）")
        print(f"  Stage1 源目录: {stage1_path}")
        print(f"  Stage2 输出目录: {output_path}")
        print("=" * 80)

        # 复制questions数据到新目录
        print(f"[复制] 正在将questions数据从 {stage2_src_subdir} 复制到 {stage2_subdir}...")
        copied_count = 0
        for unit_dir in stage2_src_subdir.iterdir():
            if unit_dir.is_dir() and unit_dir.name.startswith("unit_"):
                dest_unit_dir = stage2_subdir / unit_dir.name
                dest_unit_dir.mkdir(parents=True, exist_ok=True)
                # [2025-12 Fix] 支持两种数据源
                gen_state_src = unit_dir / "generation_state.json"
                eval_state_src = unit_dir / "evaluation_state.json"
                if gen_state_src.exists():
                    shutil.copy2(gen_state_src, dest_unit_dir / "generation_state.json")
                    copied_count += 1
                elif eval_state_src.exists():
                    # full 模式的输出，复制 evaluation_state.json 作为questions数据源
                    shutil.copy2(eval_state_src, dest_unit_dir / "evaluation_state.json")
                    copied_count += 1
        print(f"[复制] questions数据复制完成，Total {copied_count} items unit")

        # 复制其他元信息文件（可选）
        for meta_file in ["stage1_summary.json", "subset_unit_ids.json", "subset_stats.json"]:
            src_meta = stage1_path / meta_file
            if src_meta.exists():
                shutil.copy2(src_meta, output_path / meta_file)

        # 保存来源信息
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
        print("[Stage2-Only 模式] 仅运行评估阶段（读取已有 Stage1 输出）")
        print(f"  Stage1 目录: {stage1_path}")
        print("=" * 80)

    # 读取 stage1_summary.json 获取配置信息
    stage1_summary_path = stage1_path / "stage1_summary.json"
    stage1_summary = None
    if stage1_summary_path.exists():
        with open(stage1_summary_path, "r", encoding="utf-8") as f:
            stage1_summary = json.load(f)
        print(f"[Stage1信息] 实验ID: {stage1_summary.get('experiment_id')}")
        print(f"[Stage1信息] 生成成功: {stage1_summary.get('generation_success')}")
        print(f"[Stage1信息] Stage2Record数: {stage1_summary.get('stage2_record_count')}")

    router = LLMRouter.from_config(config)
    # [2026-01 Added] 支持增量 AI 评估模式（参数已从函数参数传入）
    evaluation_orchestrator = EvaluationOrchestrator(
        config,
        llm_router=router,
        eval_mode=getattr(config, "eval_mode", None),
        incremental_ai=incremental_ai
    )

    # 【2026-01 增强】初始化 BaselineEvaluator for跨维度模式评估
    # 当原始数据缺少某些维度时，根据 use_random_dims 配置从对应文件重新加载
    eval_mode = getattr(config, "eval_mode", None) or "gk"
    need_gk_dims = "gk" in eval_mode
    need_cs_dims = "cs" in eval_mode

    # [2026-01 Added] 消融控制：获取 use_random_dims 配置
    use_random_dims = getattr(config.pipeline.stage1_ablation, "use_random_dims", False)
    dimension_source = "merged_mix_dimension_jk_cs.json（随机维度）" if use_random_dims else "merged_kaocha_jk_cs.json（原始维度）"

    baseline_evaluator = BaselineEvaluator(use_random_dims=use_random_dims) if (need_gk_dims or need_cs_dims) else None

    if baseline_evaluator:
        print(f"[Stage2-Only] eval_mode={eval_mode}, 已初始化 BaselineEvaluator for维度补充")
        print(f"[Stage2-Only] 维度文件: {dimension_source}")

    if incremental_ai:
        print(f"[Stage2-Only] 增量 AI 评估模式：跳过已完成的单元（success=True 且 overall_score>0）")

    # 遍历 stage2 目录下的所有 unit 子目录
    unit_dirs = sorted([d for d in stage2_subdir.iterdir() if d.is_dir() and d.name.startswith("unit_")])

    if not unit_dirs:
        print(f"[错误] stage2 目录下没有找到任何 unit 子目录: {stage2_subdir}")
        return {"error": "no_unit_dirs"}

    print(f"[Stage2-Only] 找到 {len(unit_dirs)} items unit 目录")

    # 统计变量
    evaluation_success = 0
    evaluation_failed = 0

    ai_scores: List[float] = []
    # [2025-12 Refactored] 教育学使用 P/R/F1 指标收集
    ped_metrics_list: List[Dict[str, Any]] = []
    # [2025-12 Added] 独立 GK/CS 评估指标收集
    gk_metrics_list: List[Dict[str, Any]] = []
    cs_metrics_list: List[Dict[str, Any]] = []

    question_type_counts = {"single-choice": 0, "essay": 0, "other": 0}
    score_buckets = {
        "single-choice": {"ai": [], "ped_f1": []},
        "essay": {"ai": [], "ped_f1": []},
        "other": {"ai": [], "ped_f1": []},
    }

    # [2025-12-28 Added] 按questions型分组的教育学指标收集
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
    all_high_variance_records: List[Dict[str, Any]] = []

    # [2025-12 Added] 缺失维度收集
    all_missing_dims_records: List[Dict[str, Any]] = []

    run_total = len(unit_dirs)

    for i, unit_dir in enumerate(unit_dirs):
        unit_id = unit_dir.name.replace("unit_", "")
        print(f"\n>>> [{i+1}/{run_total}] 评估 unit_id={unit_id}...")

        result_item = {
            "unit_id": unit_id,
            "stage2_status": "pending",
            "question_type": None,
            "ai_overall_score": None,
            "ped_metrics": None,  # [2025-12 Refactored] 改为 P/R/F1 指标
            "error": None,
        }

        try:
            # [2025-12 Fix] 支持两种数据源：generation_state.json 或 evaluation_state.json
            generation_state_path = unit_dir / "generation_state.json"
            evaluation_state_path = unit_dir / "evaluation_state.json"

            core_input_data = None
            stage1_meta_data = {}
            # [2026-01 Added] for智能追加评估的已有评估状态
            existing_eval_state = None

            if generation_state_path.exists():
                # 来源1: stage1-only 模式的输出（generation_state.json）
                with open(generation_state_path, "r", encoding="utf-8") as f:
                    gen_state_data = json.load(f)
                stage2_record_data = gen_state_data.get("stage2_record")
                if stage2_record_data:
                    core_input_data = stage2_record_data.get("core_input", {})
                    stage1_meta_data = stage2_record_data.get("stage1_meta", {})

            # [2026-01 Modified] 始终尝试读取 evaluation_state.json 以获取已有评估结果
            if evaluation_state_path.exists():
                with open(evaluation_state_path, "r", encoding="utf-8") as f:
                    eval_state_data = json.load(f)
                # 保存已有评估状态（for智能追加）
                existing_eval_state = eval_state_data
                # such as果还没有 core_input_data，从 evaluation_state.json 中获取
                if core_input_data is None:
                    # full 模式的questions数据在 input 字段中
                    core_input_data = eval_state_data.get("input", {})
                    stage1_meta_data = eval_state_data.get("stage1_meta", {})

            if not core_input_data:
                evaluation_failed += 1
                result_item["stage2_status"] = "no_question_data"
                result_item["error"] = "无法找到questions数据（generation_state.json 或 evaluation_state.json）"
                print(f"    [SKIP] 无法找到questions数据")
                all_results.append(result_item)
                continue

            # 【2026-01 增强】跨维度模式评估：检查并补充缺失的维度信息
            # such as果当前 eval_mode 需要某类维度但原始数据中缺失，则从 merged_kaocha_jk_cs.json 重新加载
            gk_dims = core_input_data.get("gk_dims", {})
            cs_dims = core_input_data.get("cs_dims", {})
            dimension_ids = core_input_data.get("dimension_ids", [])

            dims_reloaded = False
            if baseline_evaluator:
                # [2026-01 Fix] 跨维度模式评估时，需要完全重建 dimension_ids
                # 原始数据可能只有 GK 维度，运行 CS 评估时需要用 CS 维度替换
                bq = baseline_evaluator.get_baseline_question(unit_id)
                if bq:
                    # 根据 eval_mode 决定使用哪些维度
                    if need_cs_dims and not need_gk_dims:
                        # 仅 CS 模式：完全使用 CS 维度，清除 GK 维度
                        cs_dims = bq.cs_dims
                        gk_dims = {}  # 清除 GK 维度
                        # 从 bq.dimension_ids 中筛选出 CS 维度
                        cs_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心素养（四维）-") or d.startswith("学习任务群-") or d.startswith("语文学科能力要求-")]
                        dimension_ids = cs_dim_ids  # 完全替换，不是追加
                        dims_reloaded = True
                        print(f"    [维度替换] CS-only 模式: 使用 {len(cs_dim_ids)} items CS 维度")
                    elif need_gk_dims and not need_cs_dims:
                        # 仅 GK 模式：完全使用 GK 维度，清除 CS 维度
                        if not gk_dims:
                            gk_dims = bq.gk_dims
                        cs_dims = {}  # 清除 CS 维度
                        # 从 bq.dimension_ids 中筛选出 GK 维度
                        gk_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心价值-") or d.startswith("学科素养-") or d.startswith("关键能力-") or d.startswith("必备知识-") or d.startswith("四翼要求-") or d.startswith("情境-")]
                        dimension_ids = gk_dim_ids  # 完全替换，不是追加
                        dims_reloaded = True
                        print(f"    [维度替换] GK-only 模式: 使用 {len(gk_dim_ids)} items GK 维度")
                    else:
                        # GK+CS 模式：补充缺失的维度
                        if not cs_dims:
                            cs_dims = bq.cs_dims
                            cs_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心素养（四维）-") or d.startswith("学习任务群-") or d.startswith("语文学科能力要求-")]
                            if cs_dim_ids:
                                dimension_ids = list(dimension_ids) + cs_dim_ids
                                dims_reloaded = True
                                print(f"    [维度补充] 从 merged 加载了 {len(cs_dim_ids)} items CS 维度")
                        if not gk_dims:
                            gk_dims = bq.gk_dims
                            gk_dim_ids = [d for d in bq.dimension_ids if d.startswith("核心价值-") or d.startswith("学科素养-") or d.startswith("关键能力-") or d.startswith("必备知识-") or d.startswith("四翼要求-") or d.startswith("情境-")]
                            if gk_dim_ids:
                                dimension_ids = list(dimension_ids) + gk_dim_ids
                                dims_reloaded = True
                                print(f"    [维度补充] 从 merged 加载了 {len(gk_dim_ids)} items GK 维度")

            # 去重 dimension_ids
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
                # [2026-01 Added] 锚点信息
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

            # 获取questions型
            qt = core_input.question_type or "other"
            if qt not in question_type_counts:
                qt = "other"
            question_type_counts[qt] += 1
            result_item["question_type"] = qt

            # [2026-01 Added] 提取锚点信息（只有使用了Stage1且没有做agent2消融才有）
            anchor_count = core_input.anchor_count or 0
            anchors = core_input.anchors
            result_item["anchor_count"] = anchor_count
            if anchors:
                result_item["anchors"] = anchors

            # [2026-01 Added] 显示已有评估状态（for智能追加）
            if existing_eval_state:
                ai_done = existing_eval_state.get("ai_eval", {}).get("success", False)
                ped_done = existing_eval_state.get("pedagogical_eval", {}).get("success", False)
                gk_done = existing_eval_state.get("gk_eval", {}).get("success", False)
                cs_done = existing_eval_state.get("cs_eval", {}).get("success", False)
                print(f"    [检测已有评估] AI={ai_done}, Ped={ped_done}, GK={gk_done}, CS={cs_done}")

            # 执行 Stage2 评估（传递已有评估状态支持智能追加）
            evaluation_state = evaluation_orchestrator.run(stage2_record, existing_eval_state=existing_eval_state)

            if getattr(evaluation_state, "current_stage", None) != "completed":
                evaluation_failed += 1
                result_item["stage2_status"] = "fail"
                result_item["error"] = "评估未完成"
                print(f"    [FAIL] 评估未完成")
                all_results.append(result_item)
                continue

            evaluation_success += 1
            result_item["stage2_status"] = "success"

            # 提取分数
            ai_score, ped_metrics, gk_metrics, cs_metrics = _extract_eval_scores(evaluation_state)

            result_item["ai_overall_score"] = ai_score
            result_item["ped_metrics"] = ped_metrics  # [2025-12 Refactored] P/R/F1 指标
            result_item["gk_metrics"] = gk_metrics  # [2025-12 Added] 独立 GK 评估指标
            result_item["cs_metrics"] = cs_metrics  # [2025-12 Added] 独立 CS 评估指标

            if ai_score is not None:
                ai_scores.append(ai_score)
                score_buckets[qt]["ai"].append(ai_score)
            if ped_metrics is not None:
                ped_metrics_list.append(ped_metrics)
                score_buckets[qt]["ped_f1"].append(ped_metrics["f1"])
                ped_metrics_by_qtype[qt].append(ped_metrics)  # [2025-12-28 Added] 按questions型收集
            # [2025-12 Added] 收集独立 GK/CS 评估指标
            if gk_metrics is not None:
                gk_metrics_list.append(gk_metrics)
                gk_metrics_by_qtype[qt].append(gk_metrics)  # [2025-12-28 Added] 按questions型收集
            if cs_metrics is not None:
                cs_metrics_list.append(cs_metrics)
                cs_metrics_by_qtype[qt].append(cs_metrics)  # [2025-12-28 Added] 按questions型收集

            # 收集高方差维度记录
            high_variance_items = _extract_high_variance_dims(evaluation_state, unit_id, qt)
            if high_variance_items:
                all_high_variance_records.extend(high_variance_items)

            # [2025-12 Added] 收集缺失维度记录
            missing_dims_item = _extract_missing_dimensions(evaluation_state, unit_id, qt)
            if missing_dims_item:
                all_missing_dims_records.append(missing_dims_item)

            ai_disp = f"{ai_score:.1f}" if isinstance(ai_score, (int, float)) else "N/A"
            ped_disp = f"F1={ped_metrics['f1']:.3f}" if ped_metrics else "N/A"
            print(f"    [OK] questions型={qt}, AI={ai_disp}, Ped={ped_disp}")

        except Exception as e:
            evaluation_failed += 1
            result_item["stage2_status"] = "error"
            result_item["error"] = str(e)
            print(f"    [EXCEPTION] 评估异常: {e}")

        all_results.append(result_item)

    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    # [2025-12 Refactored] 计算教育学维度的 micro/macro P/R/F1 汇总
    def _compute_ped_round_metrics(metrics_list: List[Dict[str, Any]], dim_mode: str = "gk") -> Dict[str, Any]:
        """
        计算教育学维度的轮次汇总指标

        [Important Note]
        此函数的计算逻辑with pedagogical_eval.aggregate_round() 存在重复。
        高频维度定义统一来源: src.shared.dimension_config
        """
        from src.evaluation.pedagogical_eval import get_high_freq_dims_by_mode, calculate_prf
        from output_analysis.core.bootstrap_ci import (
            bootstrap_micro_metrics,
            bootstrap_macro_metrics_dimension_view
        )

        if not metrics_list:
            return {
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # [2025-12-31 Added] 过滤掉原始gold维度为空的questions
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

        # [2026-01-07 Fix] Macro 平均：维度视角（对每items维度计算P/R/F1，再取平均）
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
            "skipped_no_dims": skipped_no_dims,  # 【2025-12-31】原始无维度被跳过的questions数
        }

        # [2026-01-17 Added] Bootstrap 置信区间计算
        if len(valid_metrics_list) >= 2:  # 至少需要2items样本才能计算CI
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

        # [2025-12 Added] 排除高频维度后的指标
        # [2025-12-31 Updated] such as果questions去除高频维度后gold为空，则跳过该questions（不计入统计）
        high_freq_dims = get_high_freq_dims_by_mode(dim_mode)
        if high_freq_dims:
            excl_total_tp, excl_total_fp, excl_total_fn = 0, 0, 0
            skipped_only_high_freq = 0  # 仅含高频维度被跳过的questions数
            for m in valid_metrics_list:  # 【2025-12-31】使用过滤后的列表
                gold = m.get("gold_dimensions", [])
                pred = m.get("predicted_dimensions", [])
                # 检查去除高频维度后是否还有剩余gold维度
                gold_after_excl = set(gold) - high_freq_dims
                if not gold_after_excl:
                    # 该questions只有高频维度，跳过不计入统计
                    skipped_only_high_freq += 1
                    continue
                tp_ex, fp_ex, fn_ex, _, _, _ = calculate_prf(gold, pred, exclude_dims=high_freq_dims)
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
                "skipped_only_high_freq": skipped_only_high_freq,  # 被跳过的仅含高频维度questions数
            }

            # [2026-01-17 Added] 排除高频后的 Bootstrap 置信区间
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

    # [2025-12-28 Added] 按questions型分组的教育学指标汇总
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

    # ============ [2026-01 Fix] 读取现有 summary 保留元数据 ============
    # stage2-only 模式应该是"追加/Updated"模式，不是"重建"模式
    # 需要保留现有 summary.json 中的 config、experiment_type、llm_config etc元数据
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
            print(f"[警告] 读取现有 summary.json 失败: {e}，将使用默认值")

    # 构建汇总
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从 stage1_summary 获取原始配置信息
    stage1_config = stage1_summary.get("config", {}) if stage1_summary else {}

    summary = {
        "experiment_id": config.experiment_id,
        "round_id": getattr(config, "round_id", None),
        "run_folder": getattr(config, "run_folder", None),
        "run_mode": "stage2-only",
        "stage1_dir": str(stage1_path),
        "output_dir": str(output_path),  # [2025-12 Added] 评估结果输出目录
        "stage1_experiment_id": stage1_summary.get("experiment_id") if stage1_summary else None,

        "total_units": run_total,
        "run_total": run_total,

        "evaluation_success": evaluation_success,
        "evaluation_failed": evaluation_failed,

        "avg_ai_score": _avg(ai_scores),
        # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
        "ped_round_metrics": ped_round_metrics,
        # [2025-12 Added] Independent GK/CS evaluation metrics aggregation
        "gk_round_metrics": gk_round_metrics if gk_metrics_list else None,
        "cs_round_metrics": cs_round_metrics if cs_metrics_list else None,
        # [2025-12-28 Added] 按questions型分组的教育学指标汇总
        "ped_round_metrics_by_question_type": ped_round_metrics_by_qtype if ped_round_metrics_by_qtype else None,
        "gk_round_metrics_by_question_type": gk_round_metrics_by_qtype if gk_round_metrics_by_qtype else None,
        "cs_round_metrics_by_question_type": cs_round_metrics_by_qtype if cs_round_metrics_by_qtype else None,

        "avg_ai_score_by_question_type": {k: _avg(v["ai"]) for k, v in score_buckets.items()},

        "question_type_distribution": question_type_counts,

        # [2026-01 Fix] 优先使用现有 summary 中的配置，然后是 stage1_config，最后是默认值
        "config": {
            "generator_model": existing_config.get("generator_model")
                              or stage1_config.get("generator_model", "unknown"),
            "dim_mode": existing_config.get("dim_mode")
                       or stage1_config.get("dim_mode", config.pipeline.agent1.dimension_mode),
            "prompt_level": existing_config.get("prompt_level")
                           or stage1_config.get("prompt_level", config.pipeline.prompt_extraction.prompt_level),
            "stage1_skip_agent": existing_config.get("stage1_skip_agent")
                                or stage1_config.get("stage1_skip_agent", "none"),
            # [2026-01 Added] 消融控制记录（保留现有值）
            "use_random_dims": existing_config.get("use_random_dims", use_random_dims),
            "dimension_source": existing_config.get("dimension_source", dimension_source),
        },
        # [2026-01 Fix] 完整 LLM 配置记录 - 保留现有值
        "llm_config": {
            "stage1_model": existing_llm_config.get("stage1_model")
                           or existing_config.get("generator_model")
                           or stage1_config.get("generator_model", api_config.STAGE1_MODEL),
            "stage1_temperature": existing_llm_config.get("stage1_temperature", api_config.STAGE1_TEMPERATURE),
            "stage1_preset": existing_llm_config.get("stage1_preset", api_config.STAGE1_PRESET),
            "stage2_eval_models": api_config.STAGE2_EVAL_MODELS,  # 评估模型可以Updated
            "stage2_temperature": api_config.STAGE2_TEMPERATURE,
            "stage2_network": api_config.STAGE2_NETWORK,
        },
        # [2026-01 Fix] 实验类型标记 - 保留现有值，不要覆盖
        "experiment_type": {
            "is_baseline": existing_exp_type.get("is_baseline", False),
            "is_negative_control": existing_exp_type.get("is_negative_control", use_random_dims),
            "is_hard_negative_control": existing_exp_type.get("is_hard_negative_control", getattr(config, "is_hard_negative_control", False)),
            "is_lowfreq": existing_exp_type.get("is_lowfreq", getattr(config, "is_lowfreq", False)),
            "lowfreq_k": existing_exp_type.get("lowfreq_k") or getattr(config, "lowfreq_k", None),
        },

        "eval_models": router.get_eval_model_names(),
        "timestamp": timestamp,
        "results": all_results,
    }

    # 添加 LLM 调用重试审计信息
    retry_audit = get_retry_audit()
    if retry_audit.has_issues():
        summary["llm_retry_audit"] = {
            "has_issues": True,
            "total_retries": len(retry_audit.retry_records),
            "total_failures": len(retry_audit.failure_records),
            "retry_records": retry_audit.retry_records,
            "failure_records": retry_audit.failure_records,
            "note": "LLM 调用过程中发生了重试或失败，请检查网络连接和 API 服务状态",
        }
        print(f"\n[警告] LLM 调用出现 {len(retry_audit.retry_records)} 次重试, {len(retry_audit.failure_records)} 次最终失败")
    else:
        summary["llm_retry_audit"] = {
            "has_issues": False,
            "total_retries": 0,
            "total_failures": 0,
            "note": "所有 LLM 调用均一次成功，无网络波动questions",
        }

    # [2026-01 Added] 计算后20%低质量questions指标
    dim_mode = summary.get("config", {}).get("dim_mode", "gk")
    summary["bottom_20_metrics"] = calculate_bottom_20_metrics(
        summary.get("results", []), dim_mode
    )

    # [2026-01 Added] Print PRF statistics summary（包含高频维度排除统计）
    from src.shared.report_generator import print_prf_summary

    # 根据 eval_mode 决定打印哪些指标
    if "gk" in eval_mode and gk_metrics_list:
        print_prf_summary(gk_metrics_list, dim_mode="gk", title=f"Stage2-Only - Total {run_total} questions")
    if "cs" in eval_mode and cs_metrics_list:
        print_prf_summary(cs_metrics_list, dim_mode="cs", title=f"Stage2-Only - Total {run_total} questions")

    # 保存 summary.json 到输出目录（output_path，such as指定了独立输出目录则为新目录，否则为原目录）
    summary_path = output_path / "summary.json"
    # [2026-01 Added] 统一浮点数精度为3位小数
    summary = _round_floats(summary, decimals=3)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # [2025-12 Added] 生成人类可读的 MD 报告
    try:
        md_result = generate_reports_from_summary(
            summary_path=summary_path,
            output_dir=output_path,
            is_stage1_only=False,
        )
        if "md_report" in md_result:
            print(f"[MD报告] 已生成: {md_result['md_report']}")
    except Exception as e:
        print(f"[MD报告] 生成失败: {e}")

    # 保存High variance dimension aggregation报告
    if all_high_variance_records:
        high_variance_report = {
            "experiment_id": config.experiment_id,
            "timestamp": timestamp,
            "total_high_variance_count": len(all_high_variance_records),
            "threshold_used": 50.0,
            "description": "以下维度在多模型评估中出现显著评分差异（≥50分），建议人工复核",
            "summary": {
                "affected_units": len(set(r.get("unit_id") for r in all_high_variance_records)),
                "ai_centric_count": len([r for r in all_high_variance_records if r.get("eval_type") == "ai_centric"]),
                "pedagogical_count": len([r for r in all_high_variance_records if r.get("eval_type") == "pedagogical"]),
            },
            "records": all_high_variance_records,
        }

        high_variance_path = output_path / "high_variance_report.json"
        with open(high_variance_path, "w", encoding="utf-8") as f:
            json.dump(high_variance_report, f, ensure_ascii=False, indent=2)
        print(f"\n[High variance dimension aggregation] Total {len(all_high_variance_records)} 条记录，已保存到: {high_variance_path}")

    # [2025-12 Added] 保存Missing dimension aggregation报告
    if all_missing_dims_records:
        _save_missing_dimensions_report(
            missing_dims_records=all_missing_dims_records,
            output_dir=output_path,
            experiment_id=config.experiment_id,
        )

    print(f"\n[Stage2-Only 完成] 汇总已保存到: {summary_path}")

    # 自动提取questions
    try:
        from scripts.extract_questions import auto_extract_and_save
        print(f"\n[questions提取] 正在自动提取所有生成的questions...")
        extract_result = auto_extract_and_save(output_path, config.experiment_id)
        if extract_result["success"]:
            print(f"[questions提取] 成功提取 {extract_result['total_questions']} 道questions")
            print(f"  - JSON格式: {extract_result['json_path']}")
            print(f"  - Markdown格式: {extract_result['markdown_path']}")
        else:
            print(f"[questions提取] 提取失败: {extract_result.get('error', '未知错误')}")
    except Exception as e:
        print(f"[questions提取] 提取过程出错: {e}")

    return summary


# ============================================================================
# 主入口
# ============================================================================

def main():
    args = parse_arguments()

    # ========== [2025-12 Added] extract 模式：提取questions（独立处理，无需初始化 LLM） ==========
    if args.run_mode == "extract":
        run_extract_mode(args)
        return

    # 【2025-12 架构Refactored】打印运行时实际使用的文件路径
    _print_runtime_files()

    # [2025-12 Added] 自检跟踪支持
    _self_check_trace_enabled = getattr(args, "enable_self_check_trace", False)
    _self_check_finalize = None
    if _self_check_trace_enabled:
        print("[self_check] Trace mode enabled")

    # [2025-12 Added] 清空 LLM 重试审计记录（确保每次实验独立）
    clear_retry_audit()

    print("\n" + "=" * 80)
    print("Experiment configuration（数据驱动questions型）")
    print("=" * 80)
    print(f"  运行模式:     {args.run_mode}")
    if args.run_mode == "single":
        print(f"  unit_id:      {args.unit_id}")
    elif args.run_mode == "baseline":
        print(f"  模式说明:     真questions基准评估（跳过 Stage1）")
        # [2025-12 Added] 显示卷别筛选选项
        exam_type = getattr(args, "exam_type", "all")
        exam_type_label = {"national": "仅全国卷", "local": "仅地方卷", "all": "全部"}.get(exam_type, exam_type)
        print(f"  卷别筛选:     {exam_type_label}")
        # [2026-01 Deprecated] 年份筛选不再使用
        # if args.year_start is not None or args.year_end is not None:
        #     year_range = ""
        #     if args.year_start and args.year_end:
        #         year_range = f"{args.year_start}-{args.year_end}年"
        #     elif args.year_start:
        #         year_range = f"{args.year_start}年及之后"
        #     elif args.year_end:
        #         year_range = f"{args.year_end}年及之前"
        #     print(f"  年份范围:     {year_range}")
        # [2026-01 Modified] baseline 模式默认不启用AI评估
        eval_mode = getattr(args, "eval_mode", None) or "gk"  # 默认仅 GK 教育学评估
        eval_mode_label = {
            "ai": "仅 AI 评估",
            "gk": "仅 GK 教育学评估",
            "cs": "仅 CS 教育学评估",
            "ai+gk": "AI + GK 教育学评估",
            "ai+cs": "AI + CS 教育学评估",
        }.get(eval_mode, eval_mode)
        print(f"  评估模式:     {eval_mode_label}")
        print(f"  评估模型组:   {STAGE2_EVAL_MODELS}")
    # [2026-01 Deprecated] baseline-recent 不再使用
    # elif args.run_mode == "baseline-recent":
    #     print(f"  模式说明:     近五年真questions基准评估（2021-2025，跳过 Stage1）")
    #     # [2026-01 Refactored] baseline-recent 模式也显示 eval_mode 而非 dim_mode
    #     eval_mode = getattr(args, "eval_mode", None) or "ai+gk"
    #     eval_mode_label_recent = {
    #         "ai": "仅 AI 评估",
    #         "gk": "仅 GK 教育学评估",
    #         "cs": "仅 CS 教育学评估",
    #         "ai+gk": "AI + GK 教育学评估",
    #         "ai+cs": "AI + CS 教育学评估",
    #     }.get(eval_mode, eval_mode)
    #     print(f"  评估模式:     {eval_mode_label_recent}")
    #     print(f"  评估模型组:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "stage1-only":
        print(f"  模式说明:     仅运行 Stage1（出questions阶段），跳过评估")
        if args.subset_size:
            print(f"  子集大小:     {args.subset_size}")
            print(f"  采样策略:     {args.subset_strategy}")
            print(f"  随机种子:     {args.subset_seed}")
        elif args.subset_file:
            print(f"  子集文件:     {args.subset_file}")
        else:
            print(f"  子集采样:     (未启用，使用全量)")
    elif args.run_mode == "stage2-only":
        print(f"  模式说明:     仅运行 Stage2（评估阶段），读取已有 Stage1 输出")
        print(f"  Stage1目录:   {args.stage1_dir}")
        print(f"  评估模型组:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "ablation-nodim":
        print(f"  模式说明:     Stage1 ablation experiment（无维度提示词，直接生成）")
        print(f"  消融目的:     验证维度提示词系统的价值")
        if args.subset_size:
            print(f"  子集大小:     {args.subset_size}")
            print(f"  采样策略:     {args.subset_strategy}")
            print(f"  随机种子:     {args.subset_seed}")
        elif args.subset_file:
            print(f"  子集文件:     {args.subset_file}")
        else:
            print(f"  子集采样:     (未启用，使用全量)")
        # [2026-01 Modified] 默认不启用AI评估
        ablation_eval_mode = getattr(args, "eval_mode", None) or "gk"
        ablation_eval_mode_label = {
            "ai": "仅 AI 评估",
            "gk": "仅 GK 教育学评估",
            "cs": "仅 CS 教育学评估",
            "ai+gk": "AI + GK 教育学评估",
            "ai+cs": "AI + CS 教育学评估",
        }.get(ablation_eval_mode, ablation_eval_mode)
        print(f"  评估模式:     {ablation_eval_mode_label}")
        print(f"  生成模型:     {args.generator_model}")
        print(f"  评估模型组:   {STAGE2_EVAL_MODELS}")
    elif args.run_mode == "full":
        if args.subset_size:
            print(f"  子集大小:     {args.subset_size}")
            print(f"  采样策略:     {args.subset_strategy}")
            print(f"  随机种子:     {args.subset_seed}")
        elif args.subset_file:
            print(f"  子集文件:     {args.subset_file}")
        else:
            print(f"  子集采样:     (未启用，使用全量)")

    # 非 baseline/stage2-only/ablation-nodim 模式显示生成相关参数
    # [2026-01 Deprecated] Removed baseline-recent
    if args.run_mode not in ("baseline", "stage2-only", "ablation-nodim"):
        print(f"  维度模式:     {args.dim_mode}")
        print(f"  提示词档次:   {args.prompt_level}")
        print(f"  生成模型:     {args.generator_model}")
        if args.run_mode != "stage1-only":
            print(f"  评估模型组:   {STAGE2_EVAL_MODELS}")

    # 显示 round-id 信息
    if args.round_id:
        print(f"  round-id:     {args.round_id}")

    # 显示 Stage1 ablation experiment信息
    stage1_skip = getattr(args, "stage1_skip", "none") or "none"
    if stage1_skip != "none":
        print(f"  Stage1 消融:   跳过 {stage1_skip}")
    print("=" * 80 + "\n")

    config = create_experiment_config(args)

    # [2025-12 Added] 启用自检跟踪（config 创建后）
    if _self_check_trace_enabled:
        try:
            from tools.self_check import enable_trace, finalize_trace
            enable_trace(str(config.output_dir))
            _self_check_finalize = finalize_trace
        except Exception as e:
            print(f"[self_check] Warning: Failed to enable trace: {e}")
            _self_check_trace_enabled = False

    # 显示最终输出目录
    print(f"[配置] experiment_id: {config.experiment_id}")
    if args.round_id:
        print(f"[配置] round_id: {config.round_id}")
        print(f"[配置] run_folder: {config.run_folder}")
        print(f"[配置] round_root: {config.round_root}")
    print(f"[配置] output_dir: {config.output_dir}")
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
        # 【2025-12 架构Refactored】CLI 不做 DataLoader 前置校验
        # 不再预读 expected_qt，questions型完全由 Stage1 orchestrator 内部决定

        result = run_single_mode(config, args.unit_id)

        print("\n" + "=" * 80)
        print("Single question mode结果")
        print("=" * 80)
        print(f"  experiment_id: {config.experiment_id}")
        print(f"  unit_id:       {result.get('unit_id')}")
        print(f"  实际questions型:      {result.get('question_type')}")
        print(f"  stage1_status: {result.get('generation_success')}")
        print(f"  stage2_status: {result.get('evaluation_success')}")

        if result.get("evaluation_success"):
            print(f"  AI 评分:       {_fmt_score(result.get('ai_score'))}")
            print(f"  教育学评分:    {_fmt_score(result.get('pedagogical_score'))}")
        else:
            if not result.get("stage2_ready"):
                skip_reason = result.get("skip_reason", "stage1_no_stage2_record")
                print(f"  跳过原因:      {skip_reason}")

        print(f"  评估模型组:    {result.get('eval_models')}")
        print(f"  输出目录:      {config.output_dir}")
        print("=" * 80)

    elif args.run_mode == "baseline":
        # Baseline evaluation mode（真questions直评）
        # 优先级：unit_id > exam_type > 全部
        # [2026-01 Deprecated] Removed year_start/year_end 筛选
        from src.evaluation.baseline_evaluator import BaselineEvaluator

        baseline_unit_ids = None
        filter_label_parts = []  # for构建筛选条件描述

        if args.unit_id:
            # such as果指定了 unit_id，只评估该questions
            baseline_unit_ids = [u.strip() for u in args.unit_id.split(",")]
            filter_label_parts.append(f"指定questions({len(baseline_unit_ids)}道)")
        else:
            # 使用 exam_type 筛选
            baseline_evaluator = BaselineEvaluator()

            # [2025-12 Added] 支持卷别类型筛选
            exam_type = getattr(args, "exam_type", "all") or "all"
            has_exam_filter = exam_type != "all"

            if has_exam_filter:
                # 使用卷别筛选方法
                baseline_unit_ids = baseline_evaluator.get_unit_ids_by_exam_type(
                    exam_type=exam_type,
                    start_year=None,
                    end_year=None
                )

                # 构建筛选条件描述
                exam_label = {"national": "全国卷", "local": "地方卷"}.get(exam_type, exam_type)
                filter_label_parts.append(exam_label)
            else:
                filter_label_parts.append("全部真questions")

        filter_label = "，".join(filter_label_parts) if filter_label_parts else "全部真questions"

        summary = run_baseline_mode(config, unit_ids=baseline_unit_ids)

        print("\n" + "=" * 80)
        print(f"Baseline evaluation mode汇总报告（{filter_label}）")
        print("=" * 80)
        print(f"  实验 ID: {summary['experiment_id']}")
        print(f"  总真questions数: {summary['total_questions']}")
        print(f"  评估成功: {summary['evaluation_success']}")
        print(f"  评估失败: {summary['evaluation_failed']}")
        print(f"  平均 AI 评分: {summary['avg_ai_score']:.2f}")
        # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  教育学维度 Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  教育学维度 Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")
        print(f"  评估模型组: {summary['eval_models']}")
        print(f"  输出目录: {config.output_dir}")
        print("=" * 80)

    # [2026-01 Deprecated] baseline-recent 不再使用
    # elif args.run_mode == "baseline-recent":
    #     # [2025-12 Added] 近五年真questionsBaseline evaluation mode（2021-2025）
    #     from src.evaluation.baseline_evaluator import BaselineEvaluator
    #
    #     # 获取近五年真questions的 unit_id 列表
    #     baseline_evaluator = BaselineEvaluator()
    #     recent_unit_ids = baseline_evaluator.get_recent_years_unit_ids(start_year=2021, end_year=2025)
    #
    #     if not recent_unit_ids:
    #         print("[错误] 未找到2021-2025年的真questions数据")
    #     else:
    #         print(f"[近五年基线] 筛选到 {len(recent_unit_ids)} 道真questions（2021-2025年）")
    #         summary = run_baseline_mode(config, unit_ids=recent_unit_ids)
    #
    #         print("\n" + "=" * 80)
    #         print("近五年真questions基准评估汇总报告（2021-2025）")
    #         print("=" * 80)
    #         print(f"  实验 ID: {summary['experiment_id']}")
    #         print(f"  年份范围: 2021-2025（含）")
    #         print(f"  总真questions数: {summary['total_questions']}")
    #         print(f"  评估成功: {summary['evaluation_success']}")
    #         print(f"  评估失败: {summary['evaluation_failed']}")
    #         print(f"  平均 AI 评分: {summary['avg_ai_score']:.2f}")
    #         # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
    #         ped_metrics = summary.get('ped_round_metrics', {})
    #         micro = ped_metrics.get('micro', {})
    #         macro = ped_metrics.get('macro', {})
    #         print(f"  教育学维度 Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
    #         print(f"  教育学维度 Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")
    #         print(f"  评估模型组: {summary['eval_models']}")
    #         print(f"  输出目录: {config.output_dir}")
    #         print("=" * 80)

    elif args.run_mode == "stage1-only":
        # [2025-12 Added] 仅运行 Stage1（出questions阶段）
        summary = run_stage1_only_mode(
            config,
            subset_size=args.subset_size,
            subset_strategy=args.subset_strategy,
            subset_seed=args.subset_seed,
            subset_file=args.subset_file,
        )

        mode_label = "Subset mode" if (args.subset_size or args.subset_file) else "Full mode"
        print("\n" + "=" * 80)
        print(f"[Stage1-Only {mode_label}] 汇总报告")
        print("=" * 80)
        print(f"  实验 ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print(f"  总 unit 数: {summary['total_units']}")
        print(f"  本次运行数量: {summary.get('run_total')}")
        if summary.get('subset_size'):
            print(f"  子集大小: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  采样策略: {summary['subset_strategy']}")
                print(f"  随机种子: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  子集文件: {summary['subset_file']}")

        print(f"  生成成功: {summary['generation_success']}")
        print(f"  生成失败: {summary['generation_failed']}")
        print(f"  Stage2Record数量: {summary['stage2_record_count']}")

        print(f"\n  questions型分布:")
        for qt, count in summary['question_type_distribution'].items():
            print(f"    - {qt}: {count}")

        print(f"\n  Agent 错误统计: {summary['agent_errors']}")
        print(f"\n  结果保存位置: {config.output_dir}")
        print(f"\n  [提示] Stage1 完成！切换网络后可使用以下命令运行 Stage2：")
        print(f"    python cli.py --run-mode stage2-only --stage1-dir \"{config.output_dir}\"")
        print("=" * 80)

    elif args.run_mode == "stage2-only":
        # [2025-12 Added] 仅运行 Stage2（评估阶段）
        incremental_ai = getattr(args, "incremental_ai", False)
        summary = run_stage2_only_mode(config, args.stage1_dir, args.stage2_output_dir, incremental_ai)

        if "error" not in summary:
            print("\n" + "=" * 80)
            print("[Stage2-Only] 汇总报告")
            print("=" * 80)
            print(f"  实验 ID: {summary['experiment_id']}")
            print(f"  Stage1 目录: {summary['stage1_dir']}")
            print(f"  Stage1 实验ID: {summary.get('stage1_experiment_id', 'N/A')}")
            print(f"  总 unit 数: {summary['total_units']}")

            print(f"  评估成功: {summary['evaluation_success']}")
            print(f"  评估失败: {summary['evaluation_failed']}")
            print(f"  平均 AI 评分: {summary['avg_ai_score']:.2f}")
            # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
            ped_metrics = summary.get('ped_round_metrics', {})
            micro = ped_metrics.get('micro', {})
            macro = ped_metrics.get('macro', {})
            print(f"  教育学维度 Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
            print(f"  教育学维度 Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

            print(f"\n  questions型分布:")
            for qt, count in summary['question_type_distribution'].items():
                print(f"    - {qt}: {count}")

            if isinstance(summary.get("avg_ai_score_by_question_type"), dict):
                print(f"\n  分questions型统计:")
                ped_by_qt = summary.get("ped_round_metrics_by_question_type") or {}
                for qt in sorted(summary["avg_ai_score_by_question_type"].keys()):
                    ai_score = summary['avg_ai_score_by_question_type'].get(qt, 0)
                    ped_f1 = ped_by_qt.get(qt, {}).get("macro", {}).get("f1", 0)
                    print(f"    - {qt}: AI={ai_score:.2f}, Ped_F1={ped_f1:.4f}")

            print(f"  评估模型组: {summary['eval_models']}")
            output_dir = summary.get('output_dir') or args.stage1_dir
            print(f"\n  结果保存位置: {output_dir}")
            print("=" * 80)

    elif args.run_mode == "ablation-nodim":
        # [2025-12 Added] Stage1 ablation experiment（无维度提示词直接生成）
        summary = run_ablation_nodim_mode(
            config,
            subset_size=args.subset_size,
            subset_strategy=args.subset_strategy,
            subset_seed=args.subset_seed,
            subset_file=args.subset_file,
        )

        mode_label = "Subset mode" if (args.subset_size or args.subset_file) else "Full mode"
        print("\n" + "=" * 80)
        print(f"[ablation experiment - 无维度提示词] {mode_label}汇总报告")
        print("=" * 80)
        print(f"  实验 ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print(f"  消融说明: 跳过维度匹配和提示词系统，直接使用原始材料生成questions")
        print(f"  总 unit 数: {summary['total_units']}")
        print(f"  本次运行数量: {summary.get('run_total')}")
        if summary.get('subset_size'):
            print(f"  子集大小: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  采样策略: {summary['subset_strategy']}")
                print(f"  随机种子: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  子集文件: {summary['subset_file']}")

        print(f"  生成成功: {summary['generation_success']}")
        print(f"  生成失败: {summary['generation_failed']}")
        print(f"  评估成功: {summary['evaluation_success']}")
        print(f"  平均 AI 评分: {summary['avg_ai_score']:.2f}")
        # 教育学评估使用 P/R/F1 汇总
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  教育学(GK) Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  教育学(GK) Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

        # CS 教育学评估
        ped_cs_metrics = summary.get('ped_cs_round_metrics', {})
        if ped_cs_metrics:
            micro_cs = ped_cs_metrics.get('micro', {})
            macro_cs = ped_cs_metrics.get('macro', {})
            print(f"  教育学(CS) Micro: P={micro_cs.get('precision', 0):.4f}, R={micro_cs.get('recall', 0):.4f}, F1={micro_cs.get('f1', 0):.4f}")
            print(f"  教育学(CS) Macro: P={macro_cs.get('precision', 0):.4f}, R={macro_cs.get('recall', 0):.4f}, F1={macro_cs.get('f1', 0):.4f}")

        print(f"\n  questions型分布:")
        for qt, count in summary.get('question_type_distribution', {}).items():
            print(f"    - {qt}: {count}")

        print(f"  评估模型组: {summary['eval_models']}")
        print(f"\n  结果保存位置: {config.output_dir}")
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

        # 判断模式标签
        if getattr(args, "resume_dir", None):
            mode_label = "断点续传模式"
        elif args.subset_size or args.subset_file:
            mode_label = "Subset mode"
        else:
            mode_label = "Full mode"
        print("\n" + "=" * 80)
        print(f"{mode_label}汇总报告")
        print("=" * 80)
        print(f"  实验 ID: {summary['experiment_id']}")
        if summary.get('round_id'):
            print(f"  round_id: {summary['round_id']}")
            print(f"  run_folder: {summary['run_folder']}")
        print(f"  总 unit 数: {summary['total_units']}")
        print(f"  本次运行数量: {summary.get('run_total')}")
        if summary.get('subset_size'):
            print(f"  子集大小: {summary['subset_size']}")
            if summary.get('subset_strategy'):
                print(f"  采样策略: {summary['subset_strategy']}")
                print(f"  随机种子: {summary['subset_seed']}")
        if summary.get('subset_file'):
            print(f"  子集文件: {summary['subset_file']}")

        print(f"  生成成功(stage2_record产出): {summary['generation_success']}")
        print(f"  Stage2跳过(stage2_record为空): {summary.get('stage2_skipped', 0)}")
        print(f"  评估成功(completed): {summary['evaluation_success']}")
        print(f"  平均 AI 评分: {summary['avg_ai_score']:.2f}")
        # [2025-12 Refactored] 教育学评估使用 P/R/F1 汇总
        ped_metrics = summary.get('ped_round_metrics', {})
        micro = ped_metrics.get('micro', {})
        macro = ped_metrics.get('macro', {})
        print(f"  教育学维度 Micro: P={micro.get('precision', 0):.4f}, R={micro.get('recall', 0):.4f}, F1={micro.get('f1', 0):.4f}")
        print(f"  教育学维度 Macro: P={macro.get('precision', 0):.4f}, R={macro.get('recall', 0):.4f}, F1={macro.get('f1', 0):.4f}")

        print(f"\n  questions型分布:")
        for qt, count in summary['question_type_distribution'].items():
            print(f"    - {qt}: {count}")

        if isinstance(summary.get("avg_ai_score_by_question_type"), dict):
            print(f"\n  分questions型统计:")
            ped_by_qt = summary.get("ped_round_metrics_by_question_type") or {}
            for qt in sorted(summary["avg_ai_score_by_question_type"].keys()):
                ai_score = summary['avg_ai_score_by_question_type'].get(qt, 0)
                ped_f1 = ped_by_qt.get(qt, {}).get("macro", {}).get("f1", 0)
                print(f"    - {qt}: AI={ai_score:.2f}, Ped_F1={ped_f1:.4f}")

        print(f"\n  Agent 错误统计: {summary['agent_errors']}")
        print(f"  评估模型组: {summary['eval_models']}")
        print(f"\n  结果保存位置: {config.output_dir}")
        print("=" * 80)

    if llm_logger:
        llm_logger.save()
        llm_logger.print_summary()
        print(f"[LLMLogger] LLM 调用日志已保存到: {llm_log_dir}")

    # [2025-12 Added] 结束自检跟踪
    if _self_check_trace_enabled and _self_check_finalize:
        try:
            trace_summary = _self_check_finalize()
            print(f"\n[self_check] Trace finalized:")
            print(f"  - Records: {trace_summary.get('trace_records', 0)}")
            print(f"  - Hit functions: {trace_summary.get('hit_functions', 0)}")
            print(f"  - Call edges: {trace_summary.get('call_edges', 0)}")
            print(f"  - Dead candidates: {trace_summary.get('dead_candidates', 0)}")
            print(f"  - Trace file: {trace_summary.get('trace_file', 'N/A')}")
        except Exception as e:
            print(f"[self_check] Warning: Failed to finalize trace: {e}")

    print("\n[SUCCESS] 实验执行完成!")


if __name__ == "__main__":
    main()
