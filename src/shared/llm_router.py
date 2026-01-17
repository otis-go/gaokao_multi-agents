# src/shared/llm_router.py
# Unified LLM Client Router

"""
Module Description:
LLMRouter is the unified control point for all LLM calls in the project.

Design Principles:
1. All Agents and evaluation modules must obtain LLMClient from Router, not create them directly
2. Single control point: switching models only requires modifying api_config.py
3. Stage 1 (generation) and Stage 2 (evaluation) use different models and parameters
4. Stage 2 dual evaluation system shares the same fixed three-model evaluator

Configuration Method - 2025-12 Update:
All API configs are centralized in src/shared/api_config.py:
- STAGE1_CONFIG: Generation stage config (select one model for experiment)
- STAGE2_CONFIG: Evaluation stage config (fixed three-model via DMX interface)

Users only need to modify api_config.py, no environment variables needed.

Usage:
```python
from src.shared.llm_router import LLMRouter

# Initialize (usually at program entry)
router = LLMRouter.from_config(experiment_config)

# Get generation model
gen_client = router.get_generator_client()

# Get evaluation model group (fixed three-model)
eval_clients = router.get_eval_clients()
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
import os
import re
import warnings

from src.shared.llm_interface import LLMClient
from src.shared.config import (
    STAGE1_LLM_PARAMS,
    STAGE2_LLM_PARAMS,
    STAGE2_EVAL_MODELS,
    MODEL_WEIGHT,
    ExperimentConfig,
)

# Import unified API config
from src.shared.api_config import (
    get_stage1_config,
    get_stage1_dmx_config,
    get_stage2_config,
    get_stage1_client_params,
    get_stage1_dmx_client_params,
    get_stage2_client_params,
    get_stage2_eval_models,
    get_stage2_model_weights,
    is_no_temperature_model,
    STAGE1_CONFIG,
    STAGE1_DMX_CONFIG,
    STAGE1_USE_DMX,
    STAGE2_CONFIG,
)


# ============================================================================
# Stage1 Provider Routing - Get config from api_config.py
# ============================================================================

def get_stage1_api_config() -> Tuple[str, str, Optional[str], str]:
    """
    Get Stage1 config from api_config.py.

    Returns:
        Tuple[api_type, model_name, base_url, api_key]
    """
    cfg = get_stage1_config()
    params = get_stage1_client_params()
    return (
        params["api_type"],
        params["model_name"],
        params["base_url"],
        params["api_key"],
    )


def get_stage2_api_config(model_name: str) -> Tuple[str, Optional[str], str]:
    """
    Get Stage2 config from api_config.py.

    Args:
        model_name: Evaluation model name

    Returns:
        Tuple[api_type, base_url, api_key]
    """
    params = get_stage2_client_params(model_name)
    return (
        params["api_type"],
        params["base_url"],
        params["api_key"],
    )


@dataclass
class LLMModelConfig:
    """Single LLM model configuration"""
    model_name: str
    api_type: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None  # Direct API key (priority over env var)
    base_url: Optional[str] = None
    weight: float = 1.0  # Weight for evaluation model


@dataclass
class LLMRouterConfig:
    """
    LLM Router Configuration

    Field Description:
    - generator_model: Main generation model (used by Agent 2/3/4/5)
    - helper_model: Helper model (JSON repair, compression, etc.)
    - eval_models: Evaluation model group (fixed three-model, shared by Stage 2 dual evaluation)

    Config Source:
    All configs are read from src/shared/api_config.py, users only need to modify that file.

    [2025-12 Update] Stage1 supports two call methods:
    - STAGE1_USE_DMX = False: Use original method (call official endpoint based on provider config)
    - STAGE1_USE_DMX = True: Use DMX proxy interface
    """
    # Stage 1 generation model (read from api_config.py, auto-select based on STAGE1_USE_DMX)
    generator_model: LLMModelConfig = field(
        default_factory=lambda: LLMModelConfig(
            model_name=get_stage1_client_params()["model_name"],
            api_type=get_stage1_client_params()["api_type"],
            api_key=get_stage1_client_params()["api_key"],
            base_url=get_stage1_client_params()["base_url"],
        )
    )

    # Helper model (default same as generation model)
    helper_model: Optional[LLMModelConfig] = None

    # Stage 2 evaluation model group (read from api_config.py)
    # [2025-12 Update] GPT models use OpenAI official endpoint, others use DMX interface
    eval_models: List[LLMModelConfig] = field(
        default_factory=lambda: [
            LLMModelConfig(
                model_name=model_name,
                api_type=get_stage2_client_params(model_name)["api_type"],
                api_key=get_stage2_client_params(model_name)["api_key"],
                base_url=get_stage2_client_params(model_name)["base_url"],
                weight=get_stage2_model_weights().get(model_name, 1.0 / len(get_stage2_eval_models())),
            )
            for model_name in get_stage2_eval_models()
        ]
    )

    # Stage 1 LLM params (read from api_config.py, select based on STAGE1_USE_DMX)
    # [2025-12] For models that don't support temperature, temperature may be None
    stage1_temperature: Optional[float] = field(
        default_factory=lambda: get_stage1_client_params()["temperature"]
    )
    stage1_max_tokens: int = field(
        default_factory=lambda: get_stage1_client_params()["max_tokens"]
    )

    # Stage 2 LLM params (read from api_config.py)
    stage2_temperature: float = field(default_factory=lambda: STAGE2_CONFIG.temperature)
    stage2_max_tokens: int = field(default_factory=lambda: STAGE2_CONFIG.max_tokens)

    # Enable verbose logging
    verbose: bool = False


class LLMRouter:
    """
    Unified LLM Client Router

    Responsibilities:
    1. Manage lifecycle of all LLM clients
    2. Provide unified client access interface
    3. Ensure Stage 2 dual evaluation system uses the same three-model group

    Thread Safety:
    Current implementation is a singleton variant, each Router instance manages independent client pool.
    For global singleton, initialize at program entry and pass around.
    """

    def __init__(self, config: LLMRouterConfig):
        """
        Initialize Router

        Args:
            config: Router configuration
        """
        self.config = config

        # Client cache
        self._generator_client: Optional[LLMClient] = None
        self._helper_client: Optional[LLMClient] = None
        self._eval_clients: Optional[List[LLMClient]] = None

        # Model weight mapping (for evaluation aggregation)
        self._eval_model_weights: Dict[str, float] = {}

        print(f"[LLMRouter] Initialization complete")
        print(f"  - Generation model: {config.generator_model.model_name}")
        print(f"  - Helper model: {(config.helper_model or config.generator_model).model_name}")
        print(f"  - Evaluation model group ({len(config.eval_models)} models):")
        for m in config.eval_models:
            print(f"    - {m.model_name} (weight: {m.weight:.2f})")

    @classmethod
    def from_config(cls, experiment_config: ExperimentConfig) -> "LLMRouter":
        """
        Create Router from ExperimentConfig

        [2025-12 Update] Centralized config management
        - All API configs read from src/shared/api_config.py
        - Stage1: Get model and key from STAGE1_CONFIG
        - Stage2: Get DMX config from STAGE2_CONFIG

        Args:
            experiment_config: Experiment configuration

        Returns:
            LLMRouter: Initialized Router instance
        """
        # ========== Stage1 generation model config (read from api_config.py) ==========
        s1_params = get_stage1_client_params()
        generator_model = LLMModelConfig(
            model_name=s1_params["model_name"],
            api_type=s1_params["api_type"],
            api_key=s1_params["api_key"],
            base_url=s1_params["base_url"],
        )

        # Print Stage1 routing info
        if STAGE1_USE_DMX:
            dmx_cfg = get_stage1_dmx_config()
            print(f"[LLMRouter] Stage1 config: Using DMX proxy, "
                  f"model={s1_params['model_name']}, "
                  f"base_url={s1_params['base_url']}, "
                  f"api_key={'configured' if s1_params['api_key'] else 'not configured'}")
        else:
            print(f"[LLMRouter] Stage1 config: provider={STAGE1_CONFIG.provider}, "
                  f"model={s1_params['model_name']}, "
                  f"api_type={s1_params['api_type']}, "
                  f"base_url={s1_params['base_url'] or 'default'}, "
                  f"api_key={'configured' if s1_params['api_key'] else 'not configured'}")

        # ========== Stage2 evaluation model config (read from api_config.py) ==========
        s2_config = get_stage2_config()
        s2_models = get_stage2_eval_models()
        s2_weights = get_stage2_model_weights()

        eval_models = []
        for model_name in s2_models:
            # [2025-12 Update] Use get_stage2_client_params to get config for each model
            # GPT models use OpenAI official endpoint, others use DMX interface
            s2_params = get_stage2_client_params(model_name)
            eval_models.append(LLMModelConfig(
                model_name=model_name,
                api_type=s2_params["api_type"],
                api_key=s2_params["api_key"],
                base_url=s2_params["base_url"],
                weight=s2_weights.get(model_name, 1.0 / len(s2_models)),
            ))

        # Print Stage2 routing info
        print(f"[LLMRouter] Stage2 config: Using {len(eval_models)} evaluation models")
        for m in eval_models:
            route_type = "OpenAI Official" if m.base_url is None else f"DMX ({m.base_url})"
            print(f"  - {m.model_name}: {route_type}")

        # [2025-12] Check if model doesn't support temperature
        # If so, set temperature to None to let downstream auto-skip
        s1_temp = s1_params.get("temperature")  # May be None

        router_config = LLMRouterConfig(
            generator_model=generator_model,
            helper_model=None,  # Use default (same as generation model)
            eval_models=eval_models,
            stage1_temperature=s1_temp,  # May be None (no-temperature model)
            stage1_max_tokens=s1_params["max_tokens"],
            stage2_temperature=s2_config.temperature,
            stage2_max_tokens=s2_config.max_tokens,
            verbose=experiment_config.llm.verbose,
        )

        return cls(router_config)

    # ========== Client Access Interface ==========

    def get_client(
        self,
        role: Literal["agent2", "agent3", "agent4", "agent4_solver", "helper"],
    ) -> LLMClient:
        """
        Get LLM client for specified role

        Args:
            role: Client role
                - agent2/agent3/agent4/agent4_solver: Use generation model
                - helper: Use helper model

        Returns:
            LLMClient: LLM client for the role
        """
        if role == "helper":
            return self._get_helper_client()
        else:
            return self._get_generator_client()

    def get_generator_client(self) -> LLMClient:
        """Get main generation model client (used by Agent 2/3/4)"""
        return self._get_generator_client()

    def get_eval_clients(self) -> List[LLMClient]:
        """
        Get evaluation model group (fixed three-model)

        Important:
        Stage 2 AI-centric and Pedagogical evaluation must use the same three-model group.
        This method returns a fixed client list to ensure both evaluation systems use exactly the same models.

        Returns:
            List[LLMClient]: Evaluation model client list (fixed length, usually 3)
        """
        return self._get_eval_clients()

    def get_eval_model_weights(self) -> Dict[str, float]:
        """
        Get evaluation model weight mapping

        Returns:
            Dict[str, float]: model_name -> weight
        """
        if not self._eval_model_weights:
            self._eval_model_weights = {
                m.model_name: m.weight for m in self.config.eval_models
            }
        return self._eval_model_weights

    def get_eval_model_names(self) -> List[str]:
        """Get evaluation model name list (for logging and reports)"""
        return [m.model_name for m in self.config.eval_models]

    # ========== Internal Methods ==========

    def _get_generator_client(self) -> LLMClient:
        """Get or create generation model client"""
        if self._generator_client is None:
            model_cfg = self.config.generator_model
            self._generator_client = LLMClient(
                api_type=model_cfg.api_type,
                model_name=model_cfg.model_name,
                verbose=self.config.verbose,
                api_key_env=model_cfg.api_key_env,
                api_key=model_cfg.api_key,  # Pass API key directly
                base_url=model_cfg.base_url,
                # [2025-12] Pass Stage1 max_tokens and temperature
                max_tokens=self.config.stage1_max_tokens,
                temperature=self.config.stage1_temperature,
            )
        return self._generator_client

    def _get_helper_client(self) -> LLMClient:
        """Get or create helper model client"""
        if self._helper_client is None:
            model_cfg = self.config.helper_model or self.config.generator_model
            self._helper_client = LLMClient(
                api_type=model_cfg.api_type,
                model_name=model_cfg.model_name,
                verbose=self.config.verbose,
                api_key_env=model_cfg.api_key_env,
                api_key=model_cfg.api_key,  # Pass API key directly
                base_url=model_cfg.base_url,
                # [2025-12] Pass Stage1 max_tokens and temperature
                max_tokens=self.config.stage1_max_tokens,
                temperature=self.config.stage1_temperature,
            )
        return self._helper_client

    def _get_eval_clients(self) -> List[LLMClient]:
        """Get or create evaluation model client group"""
        if self._eval_clients is None:
            self._eval_clients = []
            for model_cfg in self.config.eval_models:
                client = LLMClient(
                    api_type=model_cfg.api_type,
                    model_name=model_cfg.model_name,
                    verbose=self.config.verbose,
                    api_key_env=model_cfg.api_key_env,
                    api_key=model_cfg.api_key,  # Pass API key directly
                    base_url=model_cfg.base_url,
                    # [2025-12] Pass Stage2 max_tokens and temperature
                    max_tokens=self.config.stage2_max_tokens,
                    temperature=self.config.stage2_temperature,
                )
                self._eval_clients.append(client)
        return self._eval_clients

    # ========== Config Access ==========

    def get_stage1_params(self) -> Dict[str, Any]:
        """Get Stage 1 LLM parameters"""
        return {
            "temperature": self.config.stage1_temperature,
            "max_tokens": self.config.stage1_max_tokens,
        }

    def get_stage2_params(self) -> Dict[str, Any]:
        """Get Stage 2 LLM parameters"""
        return {
            "temperature": self.config.stage2_temperature,
            "max_tokens": self.config.stage2_max_tokens,
        }


# ============================================================================
# Global Router Instance (optional, for simplified dependency injection)
# ============================================================================

_global_router: Optional[LLMRouter] = None


def init_global_router(config: LLMRouterConfig) -> LLMRouter:
    """Initialize global Router instance"""
    global _global_router
    _global_router = LLMRouter(config)
    return _global_router


def get_global_router() -> LLMRouter:
    """Get global Router instance"""
    if _global_router is None:
        raise RuntimeError(
            "Global LLMRouter not initialized. Please call init_global_router() or "
            "LLMRouter.from_config() to create an instance first."
        )
    return _global_router


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "LLMModelConfig",
    "LLMRouterConfig",
    "LLMRouter",
    "init_global_router",
    "get_global_router",
    # Stage1 routing utilities
    "STAGE1_PROVIDER_CONFIGS",
    "infer_provider_from_model_name",
    "resolve_stage1_backend",
]
