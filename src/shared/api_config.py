"""
src/shared/api_config.py

[2025-12] Simplified API configuration for quick experiment switching

================================================================================
                              Configuration Guide
================================================================================

[API Key Config] - Configure all API keys in the .env file in project root
  Environment variables from .env are auto-loaded at startup

[Stage1 Generation] - Select preset via STAGE1_PRESET variable in this file
  Available presets:
  - "openai_official"     : OpenAI official (overseas network)
  - "google_official"     : Google official (overseas network)
  - "deepseek_official"   : DeepSeek official (any network)
  - "doubao_official"     : Doubao official (domestic network)
  - "qwen_official"       : Alibaba Qwen (any network)
  - "dmx_overseas"        : DMX overseas proxy (overseas network)
  - "dmx_domestic"        : DMX domestic proxy (domestic network)

[Stage2 Evaluation] - Select network environment via STAGE2_NETWORK variable
  Available options:
  - "overseas"  : Overseas network (DMX overseas + OpenAI official hybrid)
  - "domestic"  : Domestic network (DMX domestic unified proxy)

================================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    # Find .env file in project root
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # Skip if python-dotenv not installed
    pass


# ============================================================================
#            Experiment Config - Modify Here Only
# ============================================================================

# ==================== Stage1 Generation Model Selection ====================
# Options: "openai_official", "google_official", "deepseek_official",
#          "doubao_official", "qwen_official", "dmx_overseas", "dmx_domestic"
STAGE1_PRESET = "google_official"

# Stage1 model name (choose appropriate model for the preset type)
# - OpenAI: "gpt-5.2", "gpt-5-mini-2025-08-07", "o1-preview" etc.
# - Google: "gemini-3-flash-preview", "gemini-1.5-pro", "gemini-1.5-flash" etc.
# - DeepSeek: "deepseek-reasoner", "deepseek-chat" etc.
# - Doubao: "doubao-pro-32k", "doubao-seed-1-6-251015" etc.
# - Qwen: "qwen-max", "qwen3-max", "qwen-turbo", "qwen-long" etc.
# - DMX: Supports all above models
STAGE1_MODEL = "gemini-3-flash-preview"

# ==================== Stage2 Evaluation Network Selection ====================
# Options: "overseas" (overseas network), "domestic" (domestic network)
STAGE2_NETWORK = "overseas"


# ============================================================================
#            API Key Config - Read from .env
# ============================================================================

# ==================== Official API Keys (from env) ====================

# OpenAI official API Key (overseas network)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Google Gemini API Key (overseas network)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# DeepSeek official API Key (any network)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Doubao (Volcengine ARK) API Key (domestic network)
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")

# Alibaba Qwen API Key (any network)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# ==================== DMX Proxy API Keys (from env) ====================

# DMX overseas version (for overseas network)
DMX_OVERSEAS_API_KEY = os.getenv("DMX_OVERSEAS_API_KEY", "")
DMX_OVERSEAS_BASE_URL = os.getenv("DMX_OVERSEAS_BASE_URL", "https://www.dmxapi.com/v1")

# DMX domestic version (for domestic network)
DMX_DOMESTIC_API_KEY = os.getenv("DMX_DOMESTIC_API_KEY", "")
DMX_DOMESTIC_BASE_URL = os.getenv("DMX_DOMESTIC_BASE_URL", "https://www.dmxapi.cn/v1")


# ============================================================================
#                    Model Capability Config
# ============================================================================

# Model prefixes that don't support temperature parameter
NO_TEMPERATURE_MODEL_PREFIXES = [
    "gpt-5-mini", "o1-mini", "o1-preview", "o1", "o3-mini", "o3", "o4-mini",
    "doubao-seed",  # Doubao Seed series (reasoning models)
    # Note: deepseek-chat / deepseek-reasoner support temperature, not in this list
]

# Model prefixes that don't support max_tokens (use max_completion_tokens instead)
NO_MAX_TOKENS_MODEL_PREFIXES = [
    "gpt-5", "gpt-4.1", "o1", "o3", "o4", "doubao-seed",
]

# Model prefixes that support reasoning_effort parameter
REASONING_EFFORT_MODEL_PREFIXES = [
    "doubao-seed", "o1", "o3",
]

# Model max_tokens limit configuration
MODEL_MAX_TOKENS_LIMITS = {
    "deepseek-chat": 8192,
    "deepseek-coder": 8192,
}


# ============================================================================
#                Stage2 Evaluation Model Config
# ============================================================================

# Stage2 evaluation model list
STAGE2_EVAL_MODELS = [
    "doubao-seed-1-6-251015",
    "gpt-5-mini-2025-08-07",
    "deepseek-v3.2-exp-thinking",
]

# Stage2 model weights
STAGE2_MODEL_WEIGHTS = {
    "doubao-seed-1-6-251015": 0.33,
    "gpt-5-mini-2025-08-07": 0.33,
    "deepseek-v3.2-exp-thinking": 0.34,
}

# ============================================================================
#      Unified Temperature Config
#  Modify params below to adjust all Stage1/Stage2 temperature settings
# ============================================================================

# Stage1 generation params - controls creativity in question design
STAGE1_TEMPERATURE = 1.0  # Medium temp, balance creativity and stability

# Stage2 evaluation params - controls stability in evaluation
STAGE2_TEMPERATURE = 0.0  # Fixed zero temperature for stability
STAGE2_MAX_TOKENS = 4096


# ============================================================================
#                        Helper Functions
# ============================================================================

def is_no_temperature_model(model_name: str) -> bool:
    """
    Check if model doesn't support temperature parameter
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(prefix.lower() in model_lower for prefix in NO_TEMPERATURE_MODEL_PREFIXES)


def is_no_max_tokens_model(model_name: str) -> bool:
    """
    Check if model doesn't support max_tokens parameter
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(prefix.lower() in model_lower for prefix in NO_MAX_TOKENS_MODEL_PREFIXES)


def get_model_max_tokens_limit(model_name: str) -> Optional[int]:
    """
    Get max_tokens limit for a model
    """
    if not model_name:
        return None
    model_lower = model_name.lower()
    for prefix, limit in MODEL_MAX_TOKENS_LIMITS.items():
        if prefix.lower() in model_lower:
            return limit
    return None


def clamp_max_tokens(model_name: str, requested_max_tokens: int) -> int:
    """
    Adjust max_tokens value based on model limits
    """
    limit = get_model_max_tokens_limit(model_name)
    if limit is not None and requested_max_tokens > limit:
        return limit
    return requested_max_tokens


def is_reasoning_effort_model(model_name: str) -> bool:
    """
    Check if model supports reasoning_effort parameter
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(prefix.lower() in model_lower for prefix in REASONING_EFFORT_MODEL_PREFIXES)


def is_ark_model(model_name: str) -> bool:
    """
    Check if model is a Doubao (ARK) model
    """
    if not model_name:
        return False
    return "doubao" in model_name.lower()


# ============================================================================
#              Stage1 Config Getter Functions
# ============================================================================

# Stage1 preset configuration table
_STAGE1_PRESETS = {
    "openai_official": {
        "api_type": "openai",
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "provider": "openai",
    },
    "google_official": {
        "api_type": "google_genai",
        "api_key": GOOGLE_API_KEY,
        "base_url": None,
        "provider": "google",
    },
    "deepseek_official": {
        "api_type": "openai",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "provider": "deepseek",
    },
    "doubao_official": {
        "api_type": "openai",
        "api_key": DOUBAO_API_KEY,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "provider": "ark",
    },
    "qwen_official": {
        "api_type": "openai",
        "api_key": QWEN_API_KEY,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "provider": "qwen",
    },
    "dmx_overseas": {
        "api_type": "openai",
        "api_key": DMX_OVERSEAS_API_KEY,
        "base_url": DMX_OVERSEAS_BASE_URL,
        "provider": "dmx",
    },
    "dmx_domestic": {
        "api_type": "openai",
        "api_key": DMX_DOMESTIC_API_KEY,
        "base_url": DMX_DOMESTIC_BASE_URL,
        "provider": "dmx",
    },
}


def get_stage1_client_params() -> dict:
    """
    Get Stage1 LLMClient initialization parameters

    Auto-generate config based on STAGE1_PRESET and STAGE1_MODEL
    """
    preset = _STAGE1_PRESETS.get(STAGE1_PRESET)
    if not preset:
        raise ValueError(f"Unknown Stage1 preset: {STAGE1_PRESET}")

    model = STAGE1_MODEL
    no_temp = is_no_temperature_model(model)
    use_reasoning_effort = is_reasoning_effort_model(model)

    # Set default params based on model type
    if is_ark_model(model):
        default_temperature = STAGE1_TEMPERATURE  # Use unified Stage1 temperature
        default_max_tokens = 65535
        reasoning_effort = "medium" if use_reasoning_effort else None
    else:
        default_temperature = STAGE1_TEMPERATURE  # Use unified Stage1 temperature
        default_max_tokens = 16384
        reasoning_effort = None

    return {
        "api_type": preset["api_type"],
        "model_name": model,
        "api_key": preset["api_key"],
        "base_url": preset["base_url"],
        "temperature": None if no_temp else default_temperature,
        "max_tokens": default_max_tokens,
        "no_temperature": no_temp,
        "reasoning_effort": reasoning_effort,
    }


# ============================================================================
#              Stage2 Config Getter Functions
# ============================================================================

def get_stage2_client_params(model_name: str = None) -> dict:
    """
    Get Stage2 LLMClient initialization parameters

    Auto-select config based on STAGE2_NETWORK:
    - "overseas": Overseas network, GPT uses OpenAI official, others (deepseek/doubao etc.) use DMX overseas
    - "domestic": Domestic network, all models use DMX domestic
    """
    if STAGE2_NETWORK == "domestic":
        # Domestic network: all via DMX domestic
        return {
            "api_type": "openai",
            "api_key": DMX_DOMESTIC_API_KEY,
            "base_url": DMX_DOMESTIC_BASE_URL,
            "temperature": STAGE2_TEMPERATURE,
            "max_tokens": STAGE2_MAX_TOKENS,
        }
    else:
        # Overseas network: GPT uses official, others use DMX overseas
        is_gpt = model_name and model_name.lower().startswith("gpt")
        if is_gpt:
            return {
                "api_type": "openai",
                "api_key": OPENAI_API_KEY,
                "base_url": None,  # OpenAI official endpoint
                "temperature": STAGE2_TEMPERATURE,
                "max_tokens": STAGE2_MAX_TOKENS,
            }
        else:
            return {
                "api_type": "openai",
                "api_key": DMX_OVERSEAS_API_KEY,
                "base_url": DMX_OVERSEAS_BASE_URL,
                "temperature": STAGE2_TEMPERATURE,
                "max_tokens": STAGE2_MAX_TOKENS,
            }


def get_stage2_eval_models() -> List[str]:
    """
    Get Stage2 evaluation model list
    """
    return STAGE2_EVAL_MODELS


def get_stage2_model_weights() -> Dict[str, float]:
    """
    Get Stage2 model weights
    """
    return STAGE2_MODEL_WEIGHTS


# ============================================================================
#      Compatibility Interface for Legacy Code
# ============================================================================

@dataclass
class Stage1Config:
    """
    Stage 1 configuration (legacy compatibility)
    """
    provider: str = "openai"
    model: str = "gpt-5.2"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = STAGE1_TEMPERATURE  # Use unified Stage1 temperature
    max_tokens: int = 16384
    reasoning_effort: str = "medium"


@dataclass
class Stage2Config:
    """
    Stage 2 configuration (legacy compatibility)
    """
    dmx_api_key: str = DMX_OVERSEAS_API_KEY
    dmx_base_url: str = DMX_OVERSEAS_BASE_URL
    openai_api_key: str = OPENAI_API_KEY
    eval_models: List[str] = field(default_factory=lambda: STAGE2_EVAL_MODELS)
    model_weights: Dict[str, float] = field(default_factory=lambda: STAGE2_MODEL_WEIGHTS)
    temperature: float = STAGE2_TEMPERATURE
    max_tokens: int = STAGE2_MAX_TOKENS


@dataclass
class Stage2DomesticConfig:
    """
    Stage 2 domestic network configuration (legacy compatibility)
    """
    dmx_api_key: str = DMX_DOMESTIC_API_KEY
    dmx_base_url: str = DMX_DOMESTIC_BASE_URL
    eval_models: List[str] = field(default_factory=lambda: STAGE2_EVAL_MODELS)
    model_weights: Dict[str, float] = field(default_factory=lambda: STAGE2_MODEL_WEIGHTS)
    temperature: float = STAGE2_TEMPERATURE
    max_tokens: int = STAGE2_MAX_TOKENS


# Compatibility variables
STAGE1_USE_DMX = STAGE1_PRESET in ["dmx_overseas", "dmx_domestic"]
STAGE2_NETWORK_MODE = STAGE2_NETWORK
STAGE2_GPT_USE_OPENAI_OFFICIAL = True  # GPT always uses official in overseas mode

# Compatibility instances
STAGE1_CONFIG = Stage1Config(
    provider=_STAGE1_PRESETS.get(STAGE1_PRESET, {}).get("provider", "openai"),
    model=STAGE1_MODEL,
    api_key=_STAGE1_PRESETS.get(STAGE1_PRESET, {}).get("api_key", ""),
    base_url=_STAGE1_PRESETS.get(STAGE1_PRESET, {}).get("base_url"),
    temperature=STAGE1_TEMPERATURE,  # Use unified Stage1 temperature
)
# [Compatibility alias] STAGE1_DMX_CONFIG equals STAGE1_CONFIG (unified, no longer distinguished)
STAGE1_DMX_CONFIG = STAGE1_CONFIG
STAGE2_CONFIG = Stage2Config()
STAGE2_DOMESTIC_CONFIG = Stage2DomesticConfig()


def get_stage1_config() -> Stage1Config:
    """
    Get Stage1 config (legacy compatibility)
    """
    return STAGE1_CONFIG


def get_stage2_config() -> Stage2Config:
    """
    Get Stage2 config (legacy compatibility)
    """
    return STAGE2_CONFIG


def get_stage2_domestic_config() -> Stage2DomesticConfig:
    """
    Get Stage2 domestic config (legacy compatibility)
    """
    return STAGE2_DOMESTIC_CONFIG


def get_stage1_dmx_config():
    """
    Legacy compatibility
    """
    return STAGE1_CONFIG


def get_stage1_dmx_client_params() -> dict:
    """
    Legacy compatibility
    """
    return get_stage1_client_params()


# ============================================================================
#                        Debug Output
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("                    API Configuration Check")
    print("=" * 70)

    print("\n[Stage1 Generation Config]")
    print(f"  Preset: {STAGE1_PRESET}")
    print(f"  Model: {STAGE1_MODEL}")
    params = get_stage1_client_params()
    print(f"  API Type: {params['api_type']}")
    print(f"  Base URL: {params['base_url'] or 'Official endpoint'}")
    print(f"  Temperature: {params['temperature'] if params['temperature'] is not None else 'N/A'}")

    print("\n[Stage2 Evaluation Config]")
    print(f"  Network: {STAGE2_NETWORK} ({'Domestic' if STAGE2_NETWORK == 'domestic' else 'Overseas'})")
    print(f"  Eval Models: {STAGE2_EVAL_MODELS}")
    print(f"  Model Weights: {STAGE2_MODEL_WEIGHTS}")

    print("\n[Network Restrictions]")
    print("  - Doubao: Domestic only")
    print("  - OpenAI/Google: Overseas only")
    print("  - DeepSeek: Any network")
    print("  - DMX Domestic: Domestic")
    print("  - DMX Overseas: Overseas")

    print("\n[Quick Switch Guide]")
    print("  1. Stage1 switch model: Modify STAGE1_PRESET and STAGE1_MODEL")
    print("  2. Stage2 switch network: Modify STAGE2_NETWORK")
    print("=" * 70)
