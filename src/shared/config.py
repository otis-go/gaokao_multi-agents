from __future__ import annotations

"""
src/shared/config.py

Centralized configuration data structures for the experiment project:
- ExperimentConfig: Global config for one experiment run
- PipelineConfig: Config for Stage 1 four Agents
- EvaluationConfig: Config for Stage 2 evaluation modules

Note:
- This file has no dependencies on other project modules, only provides pure data structures
- Used by run_full_experiment.py, Orchestrators and Agents

LLM Parameter Design (Stage 1 vs Stage 2):
- Stage 1 (Generation): Uses STAGE1_TEMPERATURE (api_config.py), current value 0.5
- Stage 2 (Evaluation): Uses STAGE2_TEMPERATURE (api_config.py), current value 0.0
- To modify temperature, edit api_config.py directly
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


# ====== Data Path Configuration ======

@dataclass
class DataConfig:
    """
    Configuration for all data file paths.
    Default settings provided here, modify according to your project structure.
    """
    raw_material_path: str = "data/raw_material.json"
    abc_prompt_path: str = "data/ABC_prompt.json"
    dimension_mapping_path: str = "data/merged_kaocha_jk_cs.json"


# ====== Stage 1: Configuration for Four Agents ======

@dataclass
class Agent1Config:
    """Agent 1: Material & Dimension Selector Configuration."""
    # Whether to enable this Agent
    enabled: bool = True
    # Material selection strategy: sequential / random / manual
    material_selection_strategy: str = "sequential"
    # Dimension selection strategy: from_mapping / random etc.
    dimension_selection_strategy: str = "from_mapping"
    # Dimension mode (gk+cs compound mode removed): gk_only / cs_only
    dimension_mode: str = "gk_only"

@dataclass
class PromptExtractionConfig:
    """
    Prompt extraction config (formerly Agent2's prompt level settings)

    [2025-12-27] Refactoring:
    - Original Agent2 (prompt synthesizer) removed
    - Prompts extracted directly from ABC_prompt.json
    - This config only specifies which prompt level to extract

    [2025-12-28] Added ablation option:
    - skip_question_type_prompt: Skip question type dimension prompt extraction
    """
    # Which ABC prompt level to use: A / B / C
    prompt_level: str = "C"
    # [Ablation option] Whether to skip question type dimension prompt extraction
    # True: Use only pedagogical dimensions (gk/cs)
    # False: Use both pedagogical and question type dimensions (default)
    skip_question_type_prompt: bool = False


@dataclass
class Agent2Config:
    """Agent 2: Anchor Finder Configuration (formerly Agent3)"""
    # Whether to enable this Agent
    enabled: bool = True
    min_anchors: int = 2
    max_anchors: int = 6
    min_anchor_length: int = 8


@dataclass
class Agent3Config:
    """Agent 3: Question Generator Configuration (formerly Agent4)"""
    # Whether to enable this Agent
    enabled: bool = True
    # generation_strategy distinguishes single-choice / essay generation details
    generation_strategy: str = "auto"


@dataclass
class Agent4Config:
    """
    Agent 4: Lightweight Verifier Configuration (formerly Agent5)

    Design Note:
    Agent4 = gatekeeper, only blocks obvious garbage, no fine-grained judgment.
    Fine-grained evaluation delegated to Stage2 AI-centric and Pedagogical evaluation.

    Fixed 2-layer check:
    - Layer1: Structure/format hard check (rules, no LLM)
    - Layer2: Single solver consistency check (1 LLM call)
    """
    # Whether to enable this Agent
    enabled: bool = True
    revision_threshold: float = 0.8
    reject_threshold: float = 0.4
    # Ratio for judging unbalanced answer point scores
    unbalanced_score_ratio: float = 3.0
    # Max iteration rounds (max regeneration rounds when verification fails)
    verifier_max_rounds: int = 3

    # Layer2 parameters
    # l2_stable_disagreement_count > 0 enables Layer2 check
    l2_stable_disagreement_count: int = 2
    # LLM confidence threshold (only high-confidence disagreements count)
    # [2026-01] Changed from 0.8 to 0.6
    l2_confidence_threshold: float = 0.6


@dataclass
class Stage1AblationConfig:
    """
    Stage 1 ablation study configuration.

    skip_agent options:
    - "none": Don't skip any Agent (default)
    - "agent2": Skip Agent2 (anchor finder), provide empty anchor set
    - "agent4": Skip Agent4 (verifier), provide pass placeholder
    - "agent3": Cannot skip (otherwise no question generated)

    use_random_dims options:
    - False: Use original dimension file merged_kaocha_jk_cs.json (default)
    - True: Use random dimension file merged_mix_dimension_jk_cs.json
      - Dimension count per question unchanged
      - But dimension names randomly selected from ABC_prompt.json
      - For validating controlled dimension mechanism effectiveness

    use_low_freq_random options (added 2026-01-05):
    - False: Don't use low-frequency random dimensions (default)
    - True: Use merged_low_freq_random_jk_cs.json
      - Assign fixed count of rare dimensions per question
      - Dimensions randomly selected from rare dimension pool
      - For validating on rare dimensions

    low_freq_ablation options (added 2026-01-05):
    - False: Provide dimension prompts normally (default)
    - True: Skip dimension prompts for questions with rare dimensions
      - For comparing hit results with/without rare dimension prompts

    skip_dims options (added 2026-01):
    - False: Use dimension control normally (default)
    - True: No-control Baseline - Skip Stage1 dimension selection and constraints
      - Generate questions from material without dimension prompts
      - For comparison with dimension-controlled pipeline

    Usage:
    CLI: python cli.py --run-mode full --stage1-skip agent2
    CLI: python cli.py --run-mode full --use-random-dims  # Random dimension ablation
    CLI: python cli.py --run-mode full --low-freq-random  # Low-frequency random dimensions
    CLI: python cli.py --run-mode full --low-freq-ablation  # Low-frequency ablation
    CLI: python cli.py --run-mode full --skip-stage1-dims  # No-control Baseline
    """
    skip_agent: str = "none"  # "none" / "agent2" / "agent4"
    use_random_dims: bool = False  # Whether to use random dimension file
    # [2026-01-05] Low-frequency dimension experiment config
    use_low_freq_random: bool = False  # Whether to use low-freq random dimension file
    low_freq_random_count: int = 3  # Rare dimensions per question
    low_freq_ablation: bool = False  # Skip prompts for rare dim questions
    # [2026-01] No-control Baseline
    skip_dims: bool = False  # Whether to skip dimension control


@dataclass
class PipelineConfig:
    """
    Stage 1 overall pipeline config, aggregating four Agent sub-configs.

    [2025-12-27] Refactoring:
    - Agent1: Material & Dimension Selection
    - Agent2: Anchor Finding (formerly Agent3)
    - Agent3: Question Generation (formerly Agent4)
    - Agent4: Lightweight Verification (formerly Agent5)
    - prompt_extraction: Prompt extraction config (formerly Agent2's prompt_level)
    """
    agent1: Agent1Config = field(default_factory=Agent1Config)
    agent2: Agent2Config = field(default_factory=Agent2Config)
    agent3: Agent3Config = field(default_factory=Agent3Config)
    agent4: Agent4Config = field(default_factory=Agent4Config)
    prompt_extraction: PromptExtractionConfig = field(default_factory=PromptExtractionConfig)

    # Ablation study configuration
    stage1_ablation: Stage1AblationConfig = field(default_factory=Stage1AblationConfig)


# ============================================================================
# [2025-12] All model and API configs migrated to api_config.py
# Variables below serve as compatibility layer, actual values read from api_config.py
# To modify model configuration, edit src/shared/api_config.py directly
# ============================================================================

def _get_stage2_eval_models():
    """Retrieve evaluation model list from api_config.py"""
    try:
        from src.shared.api_config import get_stage2_eval_models
        return get_stage2_eval_models()
    except ImportError:
        # Fallback defaults
        return ["gpt-4.1", "claude-opus-4-5-20251101-thinking", "deepseek-v3.2-exp-thinking"]

def _get_model_weights():
    """Retrieve model weights from api_config.py"""
    try:
        from src.shared.api_config import get_stage2_model_weights
        return get_stage2_model_weights()
    except ImportError:
        return {"gpt-4.1": 0.35, "claude-opus-4-5-20251101-thinking": 0.35, "deepseek-v3.2-exp-thinking": 0.30}

def _get_stage1_model():
    """Retrieve Stage1 model name from api_config.py"""
    try:
        from src.shared.api_config import STAGE1_CONFIG
        return STAGE1_CONFIG.model
    except ImportError:
        return "deepseek-reasoner"

def _get_stage1_temperature():
    """Retrieve Stage1 temperature from api_config.py"""
    try:
        from src.shared.api_config import STAGE1_CONFIG
        return STAGE1_CONFIG.temperature
    except ImportError:
        return 1.0  # 2025-12-26: default temperature set to 1.0

def _get_stage1_max_tokens():
    """Retrieve Stage1 max_tokens from api_config.py"""
    try:
        from src.shared.api_config import STAGE1_CONFIG
        return STAGE1_CONFIG.max_tokens
    except ImportError:
        return 4096

# Stage 2 multi-model list for evaluation (read from api_config.py)
STAGE2_EVAL_MODELS: List[str] = _get_stage2_eval_models()

# Model weight configuration (read from api_config.py)
MODEL_WEIGHT: Dict[str, float] = _get_model_weights()

# Stage 1 LLM parameters (read from api_config.py)
STAGE1_LLM_PARAMS: Dict[str, any] = {
    "model": _get_stage1_model(),
    "temperature": _get_stage1_temperature(),
    "max_tokens": _get_stage1_max_tokens(),
    "top_p": 0.95,
}

# Stage 2 LLM parameters
# [2025-12] Increased max_tokens to 4096 to avoid truncated evaluation output
STAGE2_LLM_PARAMS: Dict[str, any] = {
    "temperature": 0.0,
    "max_tokens": 4096,
    "top_p": 1.0,
}

# ====== Stage 2: Evaluation Config (for EvaluationOrchestrator) ======
# According to the pipeline diagram, Stage 2 contains two independent evaluation systems:
# 1. AI-centric Evaluation - multi-model ensemble scoring
# 2. Pedagogical Evaluation - prompt-based evaluation using ABC_evaluation_prompt.json

@dataclass
class ModelWeightConfig:
    """
    Weight configuration for a single model.

    Pipeline Mapping:
    Stage 2 -> AI-centric Evaluation -> weight config in multi-model ensemble scoring

    Field Description:
    - model_name: Model name (e.g., "GLM-4.5-Flash", "Qwen3-8B", "hunyuan-lite")
    - api_type: API type ("openai" / "anthropic" / "azure" etc.)
    - weight: Weight in ensemble scoring (all weights will be normalized)
    - api_key_env: Environment variable name for reading API Key
    - base_url: Optional custom API endpoint
    """
    model_name: str = STAGE1_LLM_PARAMS["model"]  # Default: use Stage1 configured model
    api_type: str = "openai"
    weight: float = 1.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None


def _create_default_eval_models() -> list:
    """Create default Stage 2 multi-model evaluation configuration"""
    return [
        ModelWeightConfig(
            model_name=model_name,
            api_type="openai",
            weight=MODEL_WEIGHT.get(model_name, 1.0 / len(STAGE2_EVAL_MODELS))
        )
        for model_name in STAGE2_EVAL_MODELS
    ]


@dataclass
class AiCentricEvalConfig:
    """
    AI-centric evaluation configuration.

    Pipeline Mapping:
    Stage 2 -> AI-centric Evaluation

    Design Notes:
    - Supports multi-model ensemble scoring: configurable LLMs with weights
    - Reads evaluation dimension config from ai_eval_prompt.json
    - Full score 100, configurable pass threshold
    - Uses STAGE2_EVAL_MODELS and MODEL_WEIGHT for parallel multi-model evaluation

    Field Description:
    - enabled: Whether to enable AI-centric evaluation
    - models: Model list with weights for evaluation (defaults to STAGE2_EVAL_MODELS)
    - ai_eval_prompt_path: Path to AI evaluation dimension config
    - pass_threshold: Pass threshold (0-100)
    - aggregate_method: Multi-model aggregation method ("weighted_average" / "majority_vote")
    """
    enabled: bool = False  # [2026-01] Disabled by default, enable via --eval-mode ai
    # Multi-model config: use models from STAGE2_EVAL_MODELS for ensemble scoring
    models: list = field(default_factory=_create_default_eval_models)
    ai_eval_prompt_path: str = "data/ai_eval_prompt.json"
    pass_threshold: float = 70.0
    aggregate_method: str = "weighted_average"
    # Legacy field compatibility
    prompt_level: str = "C"
    model_name: str = STAGE1_LLM_PARAMS["model"]  # Default: use Stage1 configured model


@dataclass
class PedagogicalEvalConfig:
    """
    Pedagogical evaluation configuration.

    [2025-12] Refactoring Notes:
    - Uses new PedagogicalHitBasedEval (hit/not-hit based P/R/F1 evaluation)
    - Evaluates using pedagogical dimensions from ABC_evaluation_prompt.json
    - Supports independent gk/cs dimension evaluation

    Field Description:
    - enabled: Whether to enable pedagogical evaluation
    - dim_mode: Dimension mode for pedagogical evaluation ("gk" or "cs")
      - Default None inherits from Stage1's dimension_mode
      - [2026-01] Removed gk+cs composite mode
    """
    enabled: bool = True
    dim_mode: Optional[str] = None  # None inherits from Stage1, can be "gk" / "cs"


@dataclass
class LLMRuntimeConfig:
    """
    Runtime LLM invocation configuration.

    Default Model Notes:
    - Default: GLM-4.5-Flash for generation and evaluation
    - Can switch to other models via CLI args or config for formal experiments
    - Stage 1/2 temperature etc. configured via STAGE1_LLM_PARAMS / STAGE2_LLM_PARAMS

    [2025-12] Stage1 Routing Isolation:
    Stage1 (generation) auto-routes to real API endpoints based on model_name,
    instead of using the DMX proxy configured in OPENAI_BASE_URL.

    Routing Rules:
    - OpenAI models (gpt-*, o*, chatgpt*) -> Official OpenAI API, STAGE1_OPENAI_API_KEY
    - DeepSeek models (deepseek*) -> https://api.deepseek.com, STAGE1_DEEPSEEK_API_KEY
    - Google Gemini models (gemini*) -> google_genai API, STAGE1_GOOGLE_API_KEY
    - Other models -> Fallback to DMX (OPENAI_BASE_URL + OPENAI_API_KEY)

    Stage2 (evaluation) continues using DMX proxy, config unchanged.

    Environment Variables:
    Stage1 real endpoints (in .env):
    - STAGE1_OPENAI_API_KEY    # Official OpenAI API Key
    - STAGE1_DEEPSEEK_API_KEY  # DeepSeek API Key
    - STAGE1_GOOGLE_API_KEY    # Google Gemini API Key

    Stage2 DMX proxy (unchanged):
    - OPENAI_API_KEY           # DMX proxy Key
    - OPENAI_BASE_URL          # DMX proxy URL (https://www.dmxapi.cn/v1)
    """
    api_type: str = "openai"                      # "openai" / "google_genai" / "anthropic" / "azure" / "dummy"
    model_name: str = STAGE1_LLM_PARAMS["model"]  # Default: use Stage1 configured model
    temperature: float = STAGE1_LLM_PARAMS["temperature"]
    max_tokens: int = STAGE1_LLM_PARAMS["max_tokens"]
    top_p: float = STAGE1_LLM_PARAMS["top_p"]
    verbose: bool = False
    api_key_env: str = "OPENAI_API_KEY"           # Default, overridden by LLMRouter Stage1 routing
    base_url: Optional[str] = None                # Default, overridden by LLMRouter Stage1 routing

    def __post_init__(self) -> None:
        # If base_url not explicitly set, read from environment variable
        # Note: This value will be overridden by resolve_stage1_backend in LLMRouter.from_config
        import os
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")

@dataclass
class EvaluationConfig:
    """
    Stage 2 evaluation configuration.

    Pipeline Mapping:
    According to the pipeline diagram, Stage 2 contains two independent evaluation systems:
    1. ai_centric: AI-centric Evaluation (multi-model ensemble scoring)
    2. pedagogical: Pedagogical Evaluation (dimension-based prompt_eval)

    Both systems run independently, final decision combines both results.
    """
    ai_centric: AiCentricEvalConfig = field(default_factory=AiCentricEvalConfig)
    pedagogical: PedagogicalEvalConfig = field(default_factory=PedagogicalEvalConfig)


# ====== Top-level ExperimentConfig ======

@dataclass
class ExperimentConfig:
    """
    Global configuration object for a single experiment run.

    Used by:
    - scripts/run_full_experiment.py
    - GenerationOrchestrator
    - EvaluationOrchestrator
    and other modules.
    """
    experiment_id: str
    question_type: str = "single-choice"
    num_questions: int = 1
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Whether to save intermediate state
    save_intermediate: bool = True

    # Convenience access: top-level aliases for four Agent sub-configs
    agent1: Agent1Config = field(init=False)
    agent2: Agent2Config = field(init=False)
    agent3: Agent3Config = field(init=False)
    agent4: Agent4Config = field(init=False)

    def __post_init__(self) -> None:
        # Expose pipeline sub-configs to top-level aliases
        self.agent1 = self.pipeline.agent1
        self.agent2 = self.pipeline.agent2
        self.agent3 = self.pipeline.agent3
        self.agent4 = self.pipeline.agent4

    def get_output_path(self, filename: str) -> Path:
        """
        Return output file path, ensuring parent directory exists.
        - filename can be relative path with subdirs like "stage1/unit_xxx/xxx.json"
        - Also compatible with legacy usage (single filename only)
        """
        from pathlib import Path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        rel = Path(filename)
        out_path = rel if rel.is_absolute() else (self.output_dir / rel)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path


# ====== Factory Function: Create ExperimentConfig with Defaults ======

def create_default_config(experiment_id: str) -> ExperimentConfig:
    """
    Create an ExperimentConfig with default settings, ensuring output directory exists.

    Called at the start of run_full_experiment(experiment_id=...).
    """
    base_output_dir = Path("outputs") / experiment_id
    base_output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        output_dir=base_output_dir,
    )

    # You can also modify some default strategies here, for example:
    # config.pipeline.agent1.dimension_mode = "gk"  # or "cs"
    # config.evaluation.ai_centric.prompt_level = "C"

    return config
