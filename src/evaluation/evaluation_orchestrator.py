# src/evaluation/evaluation_orchestrator.py
# Stage 2 Evaluation Pipeline Orchestrator - Dual Independent Evaluation System Architecture

"""
[Module Description]
EvaluationOrchestrator orchestrates the Stage 2 evaluation system.

[Pipeline Diagram Mapping]
According to agent_pro_flowchart.png, Stage 2 contains two independent evaluation systems:
1. AI-centric Evaluation - Multi-model ensemble scoring
2. Pedagogical Evaluation - Based on ABC_evaluation_prompt.json prompt_eval

[2025-12 Refactoring]
- Supports running gk/cs/ai evaluation independently
- Supports gk+ai, cs+ai combined evaluation
- [2026-01] Removed gk+cs composite mode
- gk and cs evaluation can reuse ai evaluation results
- Evaluation results are independently tracked, no interference

[Key Design Principles]
- Two evaluation systems run completely independently, no dependencies
- Each outputs independent evaluation results
- Supports eval_mode parameter to control evaluation mode
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from src.evaluation.ai_centric_eval import AICentricEval
from src.evaluation.pedagogical_eval import PedagogicalHitBasedEval
from src.shared.config import ExperimentConfig
from src.shared.data_loader import DataLoader
from src.shared.llm_interface import LLMClient
from src.shared.prompt_logger import PromptLogger
from src.shared.schemas import (
    Stage2Record,
    Stage2CoreInput,
    PedagogicalEvalResult,
    EvaluationPipelineState,
    create_initial_evaluation_state,
)

if TYPE_CHECKING:
    from src.shared.llm_router import LLMRouter


def _normalize_weights(model_names: List[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Normalize weights to dict[str, float] format (dict is the single source).
    - weights=None/empty: uniform weights
    - Missing keys: fill with uniform default
    - Sum=0: fall back to uniform weights
    """
    n = max(len(model_names), 1)
    if not weights:
        return {name: 1.0 / n for name in model_names}

    merged: Dict[str, float] = {}
    for name in model_names:
        merged[name] = float(weights.get(name, 1.0 / n))

    s = sum(max(v, 0.0) for v in merged.values())
    if s <= 0:
        return {name: 1.0 / n for name in model_names}
    return {k: max(v, 0.0) / s for k, v in merged.items()}



def _weights_dict_to_list(model_names: List[str], weights: Dict[str, float]) -> List[float]:
    """
    Backward compatible with legacy eval modules that use List[float] for model_weights.
    """
    n = max(len(model_names), 1)
    return [float(weights.get(name, 1.0 / n)) for name in model_names]


def _extract_ai_score(res: object) -> Optional[float]:
    """
    Compatible with AI evaluation result formats:
    - Object: .overall_score
    - dict: overall_score / total_score / score
    """
    if res is None:
        return None
    if hasattr(res, "overall_score"):
        try:
            return float(getattr(res, "overall_score"))
        except Exception:
            return None
    if isinstance(res, dict):
        for k in ("overall_score", "total_score", "score"):
            if k in res:
                try:
                    return float(res[k])
                except Exception:
                    pass
    return None


def _extract_ai_call_count(res: object) -> Optional[int]:
    """
    Compatible with AI evaluation result call count formats:
    - res.call_count
    - res["audit"]["call_count"]
    """
    if res is None:
        return None
    if hasattr(res, "call_count"):
        try:
            return int(getattr(res, "call_count"))
        except Exception:
            return None
    if isinstance(res, dict):
        audit = res.get("audit", {})
        if isinstance(audit, dict) and "call_count" in audit:
            try:
                return int(audit["call_count"])
            except Exception:
                return None
    return None


def _extract_decision(res: object) -> Optional[str]:
    if res is None:
        return None
    if hasattr(res, "decision"):
        return getattr(res, "decision")
    if isinstance(res, dict):
        return res.get("decision")
    return None



class EvaluationOrchestrator:
    """
    Stage 2 Orchestrator (Streamlined version, main pipeline only)

    [2025-12 Refactoring]
    Supports independent evaluation modes:
    - eval_mode="ai": AI evaluation only
    - eval_mode="gk": GK pedagogical evaluation only
    - eval_mode="cs": CS pedagogical evaluation only
    - eval_mode="ai+gk": AI + GK pedagogical evaluation
    - eval_mode="ai+cs": AI + CS pedagogical evaluation
    [2026-01 Refactoring] Removed gk+cs and ai+gk+cs composite modes

    Constraints:
    - Single public entry point: run(...)
    - Batched mode enforced: only calls eval.run, no per-dim/legacy branches
    - Weights: maintain dict as single source, don't convert to list in orchestrator (AI aggregation relies on .get)
    - Decision: any evaluation failure => final_decision="error"
    """

    def __init__(
            self,
            experiment_config: ExperimentConfig,
            llm_router: Optional["LLMRouter"] = None,
            eval_mode: Optional[str] = None,
            incremental_ai: bool = False,
    ):
        """
        Initialize evaluation orchestrator (Stage2)

        Design Principles:
        - CLI only handles invocation; Stage2 internally assembles evaluation model groups from Router/Config
        - Prompt logs must be saved under the current experiment's output_dir for traceability
        - Dict is the single source of truth for weights; backward compatible with list format when needed

        [2026-01 Refactoring] eval_mode parameter (gk+cs mode removed):
        - None: Determined by config settings (default behavior)
        - "ai": AI evaluation only
        - "gk": GK pedagogical evaluation only
        - "cs": CS pedagogical evaluation only
        - "ai+gk": AI + GK
        - "ai+cs": AI + CS

        [2026-01 Added] Low-frequency dimension experiment support:
        - Reads low-frequency mode flag from config.pipeline.stage1_ablation.use_low_freq_random
        - When enabled, pedagogical evaluation only uses low-frequency dimensions

        [2026-01 Added] Incremental AI evaluation mode:
        - When incremental_ai=True, preserves completed units even if eval_mode includes ai
        - Completion criteria: ai_eval.success=True and overall_score>0
        """
        self.incremental_ai = incremental_ai
        from pathlib import Path

        self.config = experiment_config
        self.data_loader = DataLoader()  # PedagogicalEval may use this (dimension prompt_eval, etc.)
        self.llm_router = llm_router

        # [2025-12 Added] Parse eval_mode
        self.eval_mode = eval_mode or "default"  # Default mode determined by config
        self._parse_eval_mode()

        # [2026-01 Added] Detect low-frequency dimension experiment
        self.use_low_freq_only = getattr(
            getattr(getattr(self.config, "pipeline", None), "stage1_ablation", None),
            "use_low_freq_random", False
        )
        if self.use_low_freq_only:
            print("[EvaluationOrchestrator] Low-frequency dimension experiment mode detected, pedagogical evaluation will use low-frequency dimensions only")

        self.eval_model_names: List[str] = []
        self.eval_clients: List[LLMClient] = []
        self.eval_model_weights: Dict[str, float] = {}  # dict is the single source

        # ========== Initialize evaluation model group ==========
        if llm_router is not None:
            # [OK] Prioritize Router-provided evaluation model group (unified source)
            self.eval_clients = llm_router.get_eval_clients()
            self.eval_model_names = llm_router.get_eval_model_names()
            self.eval_model_weights = _normalize_weights(
                self.eval_model_names,
                llm_router.get_eval_model_weights()
            )
            self.default_llm_client = self.eval_clients[0] if self.eval_clients else None

            print("[EvaluationOrchestrator] Using evaluation model group from LLMRouter:")
            for name in self.eval_model_names:
                print(f"  - {name}")
        else:
            # Compatibility mode: single model
            self.default_llm_client = LLMClient(
                api_type=self.config.llm.api_type,
                model_name=self.config.llm.model_name,
                verbose=self.config.llm.verbose,
            )
            self.eval_clients = [self.default_llm_client]
            self.eval_model_names = [self.config.llm.model_name]
            self.eval_model_weights = _normalize_weights(self.eval_model_names, None)

        # ========== Prompt logger (critical fix: no longer use output_dir.parent) ==========
        prompt_log_dir = (Path(self.config.output_dir) / "logs" / "prompts").resolve()
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_logger = PromptLogger(log_dir=str(prompt_log_dir))

        # ========== AI-centric evaluation ==========
        self.ai_centric_eval: Optional[AICentricEval] = None
        if getattr(self.config.evaluation, "ai_centric", None) and self.config.evaluation.ai_centric.enabled:
            # [OK] Keep your original "batched-only" approach: inject clients/weights directly into module
            self.ai_centric_eval = AICentricEval(
                config=self.config.evaluation.ai_centric,
                llm_client=self.default_llm_client,  # Some implementations may still use single client fallback
                prompt_logger=self.prompt_logger,
            )
            self.ai_centric_eval.llm_clients = self.eval_clients
            # [OK] AI aggregation uses .get internally, so must be dict
            self.ai_centric_eval.model_weights = dict(self.eval_model_weights)

            print("[EvaluationOrchestrator] AI-centric evaluation module initialized (batched-only)")

        # ========== Pedagogical evaluation (supports independent or simultaneous gk/cs initialization) ==========
        self.pedagogical_eval_gk: Optional[PedagogicalHitBasedEval] = None
        self.pedagogical_eval_cs: Optional[PedagogicalHitBasedEval] = None
        # Backward compatibility: single pedagogical_eval instance
        self.pedagogical_eval: Optional[PedagogicalHitBasedEval] = None

        if getattr(self.config.evaluation, "pedagogical", None) and self.config.evaluation.pedagogical.enabled:
            # [2025-12 Refactoring] Decide which pedagogical evaluators to initialize based on eval_mode
            # [2026-01 Added] Pass low_freq_only parameter
            if self._should_run_gk_eval:
                self.pedagogical_eval_gk = PedagogicalHitBasedEval(
                    llm_clients=self.eval_clients,
                    data_loader=self.data_loader,
                    prompt_logger=self.prompt_logger,
                    dim_mode="gk",
                    low_freq_only=self.use_low_freq_only,  # [2026-01 Added] Low-frequency mode
                )
                print(f"[EvaluationOrchestrator] GK pedagogical evaluation module initialized (low_freq_only={self.use_low_freq_only})")

            if self._should_run_cs_eval:
                self.pedagogical_eval_cs = PedagogicalHitBasedEval(
                    llm_clients=self.eval_clients,
                    data_loader=self.data_loader,
                    prompt_logger=self.prompt_logger,
                    dim_mode="cs",
                    low_freq_only=self.use_low_freq_only,  # [2026-01 Added] Low-frequency mode
                )
                print(f"[EvaluationOrchestrator] CS pedagogical evaluation module initialized (low_freq_only={self.use_low_freq_only})")

            # Backward compatibility: set single instance (prioritize gk, then cs)
            if self.pedagogical_eval_gk:
                self.pedagogical_eval = self.pedagogical_eval_gk
            elif self.pedagogical_eval_cs:
                self.pedagogical_eval = self.pedagogical_eval_cs

    def _parse_eval_mode(self) -> None:
        """
        [2025-12 Added] Parse eval_mode string and set internal flags.
        """
        mode = (self.eval_mode or "default").lower().replace(" ", "")

        # Default mode: determined by config
        if mode == "default":
            # Based on config's enabled settings
            ai_enabled = bool(
                getattr(getattr(self.config, "evaluation", None), "ai_centric", None)
                and getattr(self.config.evaluation.ai_centric, "enabled", False)
            )
            ped_enabled = bool(
                getattr(getattr(self.config, "evaluation", None), "pedagogical", None)
                and getattr(self.config.evaluation.pedagogical, "enabled", False)
            )
            # Get dim_mode
            raw_dim_mode = getattr(self.config.pipeline.agent1, "dimension_mode", "gk_only")
            ped_dim_mode = getattr(self.config.evaluation.pedagogical, "dim_mode", None)

            self._should_run_ai_eval = ai_enabled
            if ped_enabled:
                if ped_dim_mode:
                    # Use pedagogical config's dim_mode
                    self._should_run_gk_eval = "gk" in ped_dim_mode.lower()
                    self._should_run_cs_eval = "cs" in ped_dim_mode.lower()
                else:
                    # Inherit from Stage1's dimension_mode
                    self._should_run_gk_eval = raw_dim_mode in ("gk", "gk_only")
                    self._should_run_cs_eval = raw_dim_mode in ("cs", "cs_only")
            else:
                self._should_run_gk_eval = False
                self._should_run_cs_eval = False
            return

        # Parse explicit eval_mode
        self._should_run_ai_eval = "ai" in mode
        self._should_run_gk_eval = "gk" in mode
        self._should_run_cs_eval = "cs" in mode

    # ========================
    # Single public entry point: Stage2 run
    # ========================
    def run(
            self,
            stage2_input,
            existing_eval_state: Optional[Dict] = None,
    ) -> "EvaluationPipelineState":
        """
        Unified entry point for Stage2Record / Stage2CoreInput (with dict fallback):
        - Stage2Record: Uses core_input + stage1_meta
        - Stage2CoreInput: Passed directly as question_like
        Dimension list hit_dimensions comes from core_input.dimension_ids.

        [2025-12 Architecture Refactoring]
        - ai_eval_model_names / ped_eval_model_names always populated (model list when enabled, empty list [] when disabled)
        - Disabled modules don't submit tasks, don't call safe wrapper
        - skipped_modules records skip reasons
        - run doesn't depend on any CLI context, only stage2_record and config

        [2026-01 Added] Smart append evaluation support
        - existing_eval_state: Optional, existing evaluation state dict
            - None: Fresh evaluation (default, backward compatible)
            - With value: detect existing results, only run eval_mode specified evaluation, preserve others
        - eval_mode specified modules = force run (override)
        - Modules not in eval_mode = preserve existing results
        """
        import dataclasses
        from dataclasses import asdict

        # --- Initialize state ---
        state = create_initial_evaluation_state(self.config.experiment_id)

        # Evaluation model group: based on orchestrator initialization (LLMRouter/config single source)
        state.eval_models = list(getattr(self, "eval_model_names", None) or [])

        # --- Unified extraction of core + stage1_meta (if any) ---
        core = None
        stage1_meta = None

        if isinstance(stage2_input, Stage2Record):
            core = stage2_input.core_input
            stage1_meta = getattr(stage2_input, "stage1_meta", None)
        else:
            core = stage2_input

        # dict fallback: try to convert to Stage2CoreInput
        if isinstance(core, dict):
            try:
                core = Stage2CoreInput(**core)
            except Exception:
                # If conversion fails, let the process continue (may fail later in _run_parallel and be caught)
                pass

        # --- Inject input audit info to state (dynamic fields, schemas.py allows extra fields) ---
        try:
            state.unit_id = getattr(core, "unit_id", None) if core is not None else None
        except Exception:
            state.unit_id = None

        # Save "serializable snapshot" (avoid directly storing objects to prevent dump confusion)
        try:
            if core is not None and dataclasses.is_dataclass(core):
                state.input_core = asdict(core)
            elif isinstance(core, dict):
                state.input_core = core
            else:
                state.input_core = None
        except Exception:
            state.input_core = None

        try:
            if stage1_meta is not None and dataclasses.is_dataclass(stage1_meta):
                state.input_stage1_meta = asdict(stage1_meta)
            else:
                state.input_stage1_meta = None
        except Exception:
            state.input_stage1_meta = None

        # Keep original object references (if you want to access directly in memory later)
        state.input_record = core
        if stage1_meta is not None:
            state.input_stage1_meta_obj = stage1_meta  # Not required, for debugging

        # --- hit_dimensions only comes from core.dimension_ids (fixed contract) ---
        hit_dimensions = []
        try:
            hit_dimensions = list(getattr(core, "dimension_ids", None) or [])
        except Exception:
            hit_dimensions = []

        # [2025-12 Added] Separate GK and CS dimensions, check if corresponding dimensions exist
        # [2026-01-01 Fix] Support Chinese name format dimensions (based on dimension_code_mapping.json)
        # GK dimension prefixes: core value-, subject literacy-, key ability-, essential knowledge-, four wing requirements-, context-
        GK_NAME_PREFIXES = ("核心价值-", "学科素养-", "关键能力-", "必备知识-", "四翼要求-", "情境-")
        # CS dimension prefixes: core literacy (four dimensions)-, learning task group-, Chinese language ability requirements-
        CS_NAME_PREFIXES = ("核心素养（四维）-", "学习任务群-", "语文学科能力要求-")

        def is_gk_dim(d: str) -> bool:
            return d.startswith("GK") or d.startswith("gk.") or any(d.startswith(p) for p in GK_NAME_PREFIXES)

        def is_cs_dim(d: str) -> bool:
            return d.startswith("CS") or d.startswith("cs.") or any(d.startswith(p) for p in CS_NAME_PREFIXES)

        gk_hit_dims = [d for d in hit_dimensions if is_gk_dim(d)]
        cs_hit_dims = [d for d in hit_dimensions if is_cs_dim(d)]
        has_gk_dims = len(gk_hit_dims) > 0
        has_cs_dims = len(cs_hit_dims) > 0

        # --- [2025-12 Refactoring] Determine enabled status: use _should_run_* flags ---
        # [2025-12-31 Fix] In baseline/ablation mode, allow evaluation to run even if question has no dimensions
        # Because questions may be loaded from original question bank, dimension info may be incomplete, but evaluator will handle empty dimension cases
        ai_enabled = self._should_run_ai_eval and self.ai_centric_eval is not None
        gk_enabled = self._should_run_gk_eval and self.pedagogical_eval_gk is not None
        cs_enabled = self._should_run_cs_eval and self.pedagogical_eval_cs is not None

        # If no corresponding dimensions, log warning but still run evaluation (evaluator will handle empty dimension cases)
        if gk_enabled and not has_gk_dims:
            print(f"[WARNING] GK evaluation enabled but question has no GK dimensions (unit_id: {getattr(core, 'unit_id', 'unknown')}) - still running evaluation")
        if cs_enabled and not has_cs_dims:
            print(f"[WARNING] CS evaluation enabled but question has no CS dimensions (unit_id: {getattr(core, 'unit_id', 'unknown')}) - still running evaluation")

        # Backward compatibility
        ped_enabled = gk_enabled or cs_enabled

        # ========== [2026-01 Added] Smart append evaluation logic ==========
        # Detect existing evaluation results, decide which modules need to run and which to preserve
        preserved_ai = False
        preserved_gk = False
        preserved_cs = False
        preserved_ped = False

        if existing_eval_state:
            unit_id_for_log = getattr(core, "unit_id", "unknown") if core else "unknown"

            # Check success flags for each evaluation module
            existing_ai = existing_eval_state.get("ai_eval", {})
            existing_ped = existing_eval_state.get("pedagogical_eval", {})
            existing_gk = existing_eval_state.get("gk_eval", {})
            existing_cs = existing_eval_state.get("cs_eval", {})

            ai_already_done = existing_ai.get("success", False)
            ped_already_done = existing_ped.get("success", False)
            gk_already_done = existing_gk.get("success", False)
            cs_already_done = existing_cs.get("success", False)

            # [Key Decision] Decide which evaluations to run based on eval_mode and existing state
            # eval_mode specified modules = force run (override)
            # Modules not in eval_mode = preserve existing results
            # [2026-01 Added] When incremental_ai=True, preserve completed units even if eval_mode includes ai

            # AI evaluation:
            # 1. If eval_mode doesn't specify ai, and has successful AI evaluation, preserve
            # 2. [Added] If incremental mode enabled and has successful AI evaluation (with valid score), preserve
            should_preserve_ai = False
            if ai_already_done:
                if not self._should_run_ai_eval:
                    should_preserve_ai = True
                elif self.incremental_ai:
                    # Incremental mode: check if overall_score is valid
                    ai_result = existing_ai.get("result", {})
                    ai_score = ai_result.get("overall_score") if isinstance(ai_result, dict) else None
                    if ai_score is not None and ai_score > 0:
                        should_preserve_ai = True
                        print(f"    [Incremental] Preserving completed AI evaluation (score={ai_score:.4f})")

            if should_preserve_ai:
                state.ai_eval_result = existing_ai.get("result")
                state.ai_eval_success = True
                # Backward compatibility
                state.eval2_2_result = existing_ai.get("result")
                state.eval2_2_success = True
                preserved_ai = True
                if not self.incremental_ai:
                    print(f"    [Preserved] AI evaluation result (already successful, eval_mode didn't specify ai)")

            # GK evaluation: if eval_mode doesn't specify gk, and has successful GK evaluation, preserve
            if not self._should_run_gk_eval and gk_already_done:
                state.gk_eval_result = existing_gk.get("result")
                state.gk_eval_success = True
                preserved_gk = True
                print(f"    [Preserved] GK pedagogical evaluation result (already successful, eval_mode didn't specify gk)")
                # If CS also doesn't need to run, set to pedagogical_eval_result
                if not self._should_run_cs_eval:
                    state.pedagogical_eval_result = existing_gk.get("result")
                    state.pedagogical_eval_success = True
                    state.eval2_1_result = existing_gk.get("result")
                    state.eval2_1_success = True
                    preserved_ped = True

            # CS evaluation: if eval_mode doesn't specify cs, and has successful CS evaluation, preserve
            if not self._should_run_cs_eval and cs_already_done:
                state.cs_eval_result = existing_cs.get("result")
                state.cs_eval_success = True
                preserved_cs = True
                print(f"    [Preserved] CS pedagogical evaluation result (already successful, eval_mode didn't specify cs)")
                # If GK also doesn't need to run, set to pedagogical_eval_result
                if not self._should_run_gk_eval:
                    state.pedagogical_eval_result = existing_cs.get("result")
                    state.pedagogical_eval_success = True
                    state.eval2_1_result = existing_cs.get("result")
                    state.eval2_1_success = True
                    preserved_ped = True

            # Overall pedagogical evaluation (backward compatibility): if gk and cs both not in eval_mode, and has successful ped evaluation
            if not self._should_run_gk_eval and not self._should_run_cs_eval and ped_already_done:
                if not preserved_gk and not preserved_cs:
                    state.pedagogical_eval_result = existing_ped.get("result")
                    state.pedagogical_eval_success = True
                    state.eval2_1_result = existing_ped.get("result")
                    state.eval2_1_success = True
                    preserved_ped = True
                    print(f"    [Preserved] Pedagogical evaluation result (already successful, eval_mode didn't specify gk/cs)")

            # [Important] If preserved a module's result, disable that module's execution
            if preserved_ai:
                ai_enabled = False
            if preserved_gk:
                gk_enabled = False
            if preserved_cs:
                cs_enabled = False

            # Update ped_enabled
            ped_enabled = gk_enabled or cs_enabled

        # Record preservation status to state (for _save_evaluation_state decision)
        state.ai_eval_preserved = preserved_ai
        state.gk_eval_preserved = preserved_gk
        state.cs_eval_preserved = preserved_cs
        state.ped_eval_preserved = preserved_ped
        # ========== Smart append evaluation logic end ==========

        # [2025-12 Architecture Refactoring] Fill ai_eval_model_names / ped_eval_model_names (always list)
        if ai_enabled:
            state.ai_eval_model_names = list(self.eval_model_names)
        else:
            state.ai_eval_model_names = []
            state.skipped_modules.append("ai_centric_eval: disabled or not initialized")

        if ped_enabled:
            state.ped_eval_model_names = list(self.eval_model_names)
        else:
            state.ped_eval_model_names = []
            state.skipped_modules.append("pedagogical_eval: disabled or not initialized")

        # [2025-12 Added] Record independent gk/cs evaluation status
        state.gk_eval_enabled = gk_enabled
        state.cs_eval_enabled = cs_enabled

        # --- Print execution info ---
        print("=" * 80)
        print(f"[Stage 2] Starting evaluation pipeline - Experiment ID: {self.config.experiment_id}")
        print(f"  eval_mode: {self.eval_mode}")
        if getattr(state, "unit_id", None) is not None:
            print(f"  unit_id: {state.unit_id}")
        if getattr(state, "eval_models", None):
            print(f"  Evaluation model group: {state.eval_models}")
        print(f"  AI evaluation: {ai_enabled}, GK evaluation: {gk_enabled} (dims={len(gk_hit_dims)}), CS evaluation: {cs_enabled} (dims={len(cs_hit_dims)})")
        expected_calls = len(getattr(self, "eval_clients", None) or [])
        ped_calls = expected_calls if gk_enabled else 0
        ped_calls += expected_calls if cs_enabled else 0
        print(f"  Expected call count: AI={expected_calls if ai_enabled else 0}, Ped(gk/cs)={ped_calls}")
        print(f"  hit_dimensions: {len(hit_dimensions)}")
        print("=" * 80)

        if not ai_enabled and not ped_enabled:
            state.errors.append("Stage2: Both AI and pedagogical evaluation are disabled")
            state.current_stage = "failed"
            state.final_decision = "pending"
            state.notes = "Both AI and pedagogical evaluation are disabled"
            # Still save once for audit (optional)
            try:
                self._save_evaluation_state(state)
            except Exception:
                pass
            return state

        # --- Execute evaluation (only submit tasks for enabled modules) ---
        try:
            state = self._run_parallel(
                state, core, hit_dimensions,
                ai_enabled=ai_enabled,
                gk_enabled=gk_enabled,
                cs_enabled=cs_enabled,
            )
            state.current_stage = "completed"
            state.final_decision = self._make_final_decision(
                state,
                ai_enabled=ai_enabled,
                gk_enabled=gk_enabled,
                cs_enabled=cs_enabled,
            )

            print("\n" + "=" * 80)
            print("[SUCCESS] Stage 2 evaluation pipeline completed")
            self._print_summary(state)
            print("=" * 80)

            # [OK] Unified persistence entry point (keep old field writing logic for now to avoid CLI/stats crash)
            self._save_evaluation_state(state)

        except Exception as e:
            import traceback

            print(f"\n[ERROR] Stage 2 evaluation pipeline failed: {e}")
            traceback.print_exc()
            state.errors.append(str(e))
            state.current_stage = "failed"
            state.final_decision = "error"

            # Try to save even on failure for troubleshooting
            try:
                self._save_evaluation_state(state)
            except Exception:
                pass

        return state

    # ========================
    # Internal: Parallel execution (no legacy branches)
    # ========================
    def _run_parallel(
        self,
        state: EvaluationPipelineState,
        question_like: Stage2CoreInput,
        hit_dimensions: List[str],
        *,
        ai_enabled: bool = True,
        gk_enabled: bool = False,
        cs_enabled: bool = False,
    ) -> EvaluationPipelineState:
        """
        question_like: Usually Stage2CoreInput, also compatible with future isomorphic objects.

        [2025-12 Architecture Refactoring]
        - Supports independent gk/cs evaluation
        - Only submits tasks for enabled modules
        - Disabled modules don't call safe wrapper, no errors produced
        """

        results: Dict[str, Tuple[str, object]] = {}  # {"ai":..., "gk":..., "cs":...}

        def safe_ai() -> Tuple[str, object]:
            if self.ai_centric_eval is None:
                return ("skip", None)
            try:
                res = self.ai_centric_eval.run(question_like)
                return ("ok", res)
            except Exception as e:
                return ("err", e)

        def safe_gk() -> Tuple[str, object]:
            if self.pedagogical_eval_gk is None:
                return ("skip", None)
            try:
                res = self.pedagogical_eval_gk.evaluate_single(question_like, gold_dimensions=hit_dimensions)
                return ("ok", res)
            except Exception as e:
                return ("err", e)

        def safe_cs() -> Tuple[str, object]:
            if self.pedagogical_eval_cs is None:
                return ("skip", None)
            try:
                res = self.pedagogical_eval_cs.evaluate_single(question_like, gold_dimensions=hit_dimensions)
                return ("ok", res)
            except Exception as e:
                return ("err", e)

        # [2025-12 Architecture Refactoring] Only submit tasks for enabled modules
        futures = {}
        max_workers = sum([ai_enabled, gk_enabled, cs_enabled])
        max_workers = max(max_workers, 1)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            if ai_enabled:
                futures[ex.submit(safe_ai)] = "ai"
            else:
                results["ai"] = ("skip", None)

            if gk_enabled:
                futures[ex.submit(safe_gk)] = "gk"
            else:
                results["gk"] = ("skip", None)

            if cs_enabled:
                futures[ex.submit(safe_cs)] = "cs"
            else:
                results["cs"] = ("skip", None)

            for fut in as_completed(futures):
                key = futures[fut]
                results[key] = fut.result()

        # -------- AI --------
        status_ai, payload_ai = results.get("ai", ("skip", None))
        if status_ai == "ok":
            state.ai_eval_result = payload_ai
            state.ai_eval_success = True
            state.eval2_2_result = payload_ai
            state.eval2_2_success = True

            this_run_calls = _extract_ai_call_count(payload_ai)
            if this_run_calls is not None:
                print(f"[AI-centric Evaluation] this_run_calls={this_run_calls}")

            score = _extract_ai_score(payload_ai)
            if score is not None:
                print(f"[AI-centric Evaluation] Completed - Overall score: {score:.1f}")
            else:
                print("[AI-centric Evaluation] Completed (score field not parsed, raw result saved)")

        elif status_ai == "err":
            import traceback
            err = payload_ai
            print(f"[AI-centric Evaluation] Execution failed: {err}")
            traceback.print_exc()
            state.errors.append(f"AI-centric evaluation failed: {err}")
            state.ai_eval_success = False
            state.eval2_2_success = False

        elif status_ai == "skip":
            print("[AI-centric Evaluation] Skipped (module not enabled)")

        # -------- GK Pedagogical --------
        status_gk, payload_gk = results.get("gk", ("skip", None))
        if status_gk == "ok":
            state.gk_eval_result = payload_gk
            state.gk_eval_success = True
            # Backward compatibility: if only gk, set to pedagogical_eval_result
            if not cs_enabled:
                state.pedagogical_eval_result = payload_gk
                state.pedagogical_eval_success = True
                state.eval2_1_result = payload_gk
                state.eval2_1_success = True

            gk_score = getattr(payload_gk, "overall_score", None)
            if isinstance(gk_score, (int, float)):
                print(f"[GK Pedagogical Evaluation] Completed - Overall score: {float(gk_score):.1f}")
            else:
                print("[GK Pedagogical Evaluation] Completed")

        elif status_gk == "err":
            import traceback
            err = payload_gk
            print(f"[GK Pedagogical Evaluation] Execution failed: {err}")
            traceback.print_exc()
            state.errors.append(f"GK pedagogical evaluation failed: {err}")
            state.gk_eval_success = False

        elif status_gk == "skip":
            print("[GK Pedagogical Evaluation] Skipped (module not enabled)")

        # -------- CS Pedagogical --------
        status_cs, payload_cs = results.get("cs", ("skip", None))
        if status_cs == "ok":
            state.cs_eval_result = payload_cs
            state.cs_eval_success = True
            # Backward compatibility: if only cs, set to pedagogical_eval_result
            if not gk_enabled:
                state.pedagogical_eval_result = payload_cs
                state.pedagogical_eval_success = True
                state.eval2_1_result = payload_cs
                state.eval2_1_success = True

            cs_score = getattr(payload_cs, "overall_score", None)
            if isinstance(cs_score, (int, float)):
                print(f"[CS Pedagogical Evaluation] Completed - Overall score: {float(cs_score):.1f}")
            else:
                print("[CS Pedagogical Evaluation] Completed")

        elif status_cs == "err":
            import traceback
            err = payload_cs
            print(f"[CS Pedagogical Evaluation] Execution failed: {err}")
            traceback.print_exc()
            state.errors.append(f"CS pedagogical evaluation failed: {err}")
            state.cs_eval_success = False

        elif status_cs == "skip":
            print("[CS Pedagogical Evaluation] Skipped (module not enabled)")

        # [2025-12 Added] If both gk and cs enabled, merge results to pedagogical_eval_result
        if gk_enabled and cs_enabled:
            state.pedagogical_eval_success = (
                getattr(state, "gk_eval_success", False) and
                getattr(state, "cs_eval_success", False)
            )
            # Merged results stored in respective fields, pedagogical_eval_result set to None to avoid confusion
            state.pedagogical_eval_result = None
            state.eval2_1_result = None
            state.eval2_1_success = state.pedagogical_eval_success

        return state

    # ========================
    # Internal: Final decision (simplified)
    # ========================
    def _make_final_decision(
        self,
        state: EvaluationPipelineState,
        *,
        ai_enabled: bool,
        gk_enabled: bool = False,
        cs_enabled: bool = False,
    ) -> str:
        """
        [2025-12 Refactoring] Decision rules:
        - Any enabled evaluation fails -> "error" (system error/unavailable)
        - All enabled evaluations succeed -> "pass" (no threshold judgment)

        Note: Users only care about scores, no pass/reject judgment needed
        """
        if ai_enabled and not getattr(state, "ai_eval_success", False):
            return "error"
        if gk_enabled and not getattr(state, "gk_eval_success", False):
            return "error"
        if cs_enabled and not getattr(state, "cs_eval_success", False):
            return "error"

        # [2025-12 Simplification] Evaluation success means pass, no threshold judgment
        return "pass"

    # ========================
    # Internal: Summary output (streamlined)
    # ========================
    def _print_summary(self, state: EvaluationPipelineState) -> None:
        print("\n[Evaluation Summary]")
        print(f"  Experiment ID: {state.pipeline_id}")

        if getattr(state, "eval_models", None):
            print(f"  Evaluation model group: {state.eval_models}")

        if getattr(state, "ai_eval_result", None) is not None:
            score = _extract_ai_score(state.ai_eval_result)
            print("  AI-centric Evaluation:")
            if score is not None:
                print(f"    - Overall score: {score:.1f} / 100")

        if getattr(state, "pedagogical_eval_result", None) is not None:
            ped: PedagogicalEvalResult = state.pedagogical_eval_result  # Type annotation to eliminate unused warning
            print("  Pedagogical Evaluation:")
            if hasattr(ped, "overall_score"):
                print(f"    - Overall score: {float(ped.overall_score):.1f} / 100")
            dim_results = getattr(ped, "dimension_results", None)
            if dim_results is not None:
                try:
                    print(f"    - Evaluated dimension count: {len(dim_results)}")
                except Exception:
                    pass

        # [2025-12 Removed] No longer output final decision, users judge based on scores

        if state.errors:
            print("\n  Error messages:")
            for err in state.errors:
                print(f"    - {err}")

    # ========================
    # Internal: Collect high-variance dimensions
    # ========================
    def _collect_high_variance_dims(self, state: EvaluationPipelineState, unit_id: str) -> Dict[str, Any]:
        """
        [2025-12 Added] Collect high-variance dimension info from AI and pedagogical evaluation results

        Purpose: Detect cases where different models give significantly different scores for the same dimension
        (e.g., one model gives 80 points, another gives 30 points)

        Return structure:
        {
            "has_high_variance": bool,  # Whether high-variance dimensions exist
            "total_count": int,         # Total count of high-variance dimensions
            "ai_centric": [...],        # High-variance dimensions from AI evaluation
            "pedagogical": [...],       # High-variance dimensions from pedagogical evaluation
        }
        """
        result = {
            "has_high_variance": False,
            "total_count": 0,
            "ai_centric": [],
            "pedagogical": [],
        }

        # Extract high-variance dimensions from AI evaluation result
        ai_result = getattr(state, "ai_eval_result", None)
        if ai_result is not None:
            ai_audit = None
            if hasattr(ai_result, "audit"):
                ai_audit = ai_result.audit
            elif isinstance(ai_result, dict):
                ai_audit = ai_result.get("audit", {})

            if isinstance(ai_audit, dict):
                ai_high_variance = ai_audit.get("high_variance_dims", [])
                if ai_high_variance:
                    # Add unit_id to each record
                    for item in ai_high_variance:
                        if isinstance(item, dict):
                            item["unit_id"] = unit_id
                    result["ai_centric"] = ai_high_variance

        # Extract high-variance dimensions from pedagogical evaluation result
        ped_result = getattr(state, "pedagogical_eval_result", None)
        if ped_result is not None:
            ped_audit = None
            if hasattr(ped_result, "audit"):
                ped_audit = ped_result.audit
            elif isinstance(ped_result, dict):
                ped_audit = ped_result.get("audit", {})

            if isinstance(ped_audit, dict):
                ped_high_variance = ped_audit.get("high_variance_dims", [])
                if ped_high_variance:
                    # Add unit_id to each record
                    for item in ped_high_variance:
                        if isinstance(item, dict):
                            item["unit_id"] = unit_id
                    result["pedagogical"] = ped_high_variance

        # Statistics
        total = len(result["ai_centric"]) + len(result["pedagogical"])
        result["total_count"] = total
        result["has_high_variance"] = total > 0

        # If high-variance dimensions exist, print warning
        if result["has_high_variance"]:
            print(f"\n[HIGH VARIANCE WARNING] unit_id={unit_id} found {total} high-variance dimensions:")
            for item in result["ai_centric"]:
                dim_name = item.get("dimension_name", item.get("dimension_id", "unknown"))
                print(f"  - [AI] {dim_name}: max {item.get('max_score', '?')} points ({item.get('max_model', '?')}) vs min {item.get('min_score', '?')} points ({item.get('min_model', '?')})")
            for item in result["pedagogical"]:
                dim_name = item.get("dimension_name", item.get("dimension_id", "unknown"))
                print(f"  - [Ped] {dim_name}: max {item.get('max_score', '?')} points ({item.get('max_model', '?')}) vs min {item.get('min_score', '?')} points ({item.get('min_model', '?')})")

        return result

    # ========================
    # Internal: Save evaluation state
    # ========================
    def _save_evaluation_state(self, state: EvaluationPipelineState) -> None:
        """
        Unified persistence (clean structure):
        - [OK] Save by unit_id directories to avoid full/40/60 batch conflicts
        - [OK] No longer dumps entire state to avoid legacy field pollution
        - [OK] Saves necessary audit info: input core, stage1_meta, model groups, errors, final decision
        - [OK] T4 fix: forced serialization protection to prevent JSON serialization failures
        - [OK] 2025-12 enhancement: transparent scoring details showing per-dimension and per-model scores with weights
        """
        import json
        import dataclasses
        from pathlib import Path

        def _force_serializable(obj):
            """
            Force convert object to JSON serializable form:
            - dict: recursive processing
            - list/tuple: recursive processing
            - pydantic model: model_dump(mode="json")
            - dataclass: dataclasses.asdict()
            - Others: downgrade to {"repr": str(obj)} or extract key fields
            """
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: _force_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_force_serializable(x) for x in obj]
            # pydantic model (v2)
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump(mode="json")
                except Exception:
                    pass
            # pydantic model (v1)
            if hasattr(obj, "dict"):
                try:
                    return obj.dict()
                except Exception:
                    pass
            # dataclass
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                try:
                    return dataclasses.asdict(obj)
                except Exception:
                    pass
            # Downgrade: try to extract common fields
            fallback = {}
            for key in ("overall_score", "decision", "score", "audit", "reasoning", "errors"):
                if hasattr(obj, key):
                    val = getattr(obj, key, None)
                    if val is not None:
                        fallback[key] = _force_serializable(val)
            if fallback:
                return fallback
            # Final downgrade: repr
            return {"repr": str(obj)}

        def _build_transparent_ai_eval(ai_result, model_weights: Dict[str, float]) -> Dict:
            """
            Build transparent AI evaluation result, including:
            - overall_score: final score
            - decision: pass/reject
            - dimension_details: details for each dimension (score, weight, per-model scores)
            - model_summary: overall score and weight for each model
            - score_calculation: score calculation formula explanation
            """
            if ai_result is None:
                return None

            result = {}

            # Extract basic info
            if isinstance(ai_result, dict):
                result["overall_score"] = ai_result.get("overall_score") or ai_result.get("total_score")
                result["decision"] = ai_result.get("decision")
                dimensions = ai_result.get("dimensions", {})
                model_results = ai_result.get("model_results", {})
                audit = ai_result.get("audit", {})
            else:
                result["overall_score"] = getattr(ai_result, "overall_score", None) or getattr(ai_result, "total_score", None)
                result["decision"] = getattr(ai_result, "decision", None)
                dimensions = getattr(ai_result, "dimensions", {}) or {}
                model_results = getattr(ai_result, "model_results", {}) or {}
                audit = getattr(ai_result, "audit", {}) or {}

            # Dimension details (transparent)
            dimension_details = {}
            for dim_id, dim_data in (dimensions if isinstance(dimensions, dict) else {}).items():
                if isinstance(dim_data, dict):
                    dim_detail = {
                        "aggregated_score": dim_data.get("score"),
                        "level": dim_data.get("level"),
                        "weight": dim_data.get("weight", 1.0),
                        "reason": dim_data.get("reason", ""),
                        "per_model_scores": {}
                    }
                    # Add per-model scores for this dimension
                    for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
                        if isinstance(model_dims, dict) and dim_id in model_dims:
                            model_dim = model_dims[dim_id]
                            if isinstance(model_dim, dict):
                                dim_detail["per_model_scores"][model_name] = {
                                    "score": model_dim.get("score"),
                                    "level": model_dim.get("level"),
                                    "reason": model_dim.get("reason", "")[:200],  # Truncate overly long reasons
                                    "weight": model_weights.get(model_name, 0.0)
                                }
                    dimension_details[dim_id] = dim_detail
            result["dimension_details"] = dimension_details

            # Model summary (average score and weight for each model)
            model_summary = {}
            for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
                if isinstance(model_dims, dict):
                    scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                    if scores:
                        model_summary[model_name] = {
                            "average_score": round(sum(scores) / len(scores), 2),
                            "weight": model_weights.get(model_name, 0.0),
                            "dimensions_evaluated": len(scores)
                        }
            result["model_summary"] = model_summary

            # Score calculation formula explanation
            result["score_calculation"] = {
                "method": "weighted_average",
                "description": "Weighted average of dimension scores, dimension weights from ai_eval_prompt.json, model weights from config",
                "formula": "overall = sum(dimension_score * dimension_weight) / sum(dimension_weight)"
            }

            # Audit info
            result["audit"] = _force_serializable(audit)

            return result

        def _build_transparent_ped_eval(ped_result, model_weights: Dict[str, float]) -> Dict:
            """
            Build transparent pedagogical evaluation result, including:
            - overall_score: final score
            - decision: pass/reject
            - dimension_details: details for each dimension (score, hit level, per-model scores)
            - model_summary: overall score and weight for each model
            - score_calculation: score calculation formula explanation
            """
            if ped_result is None:
                return None

            result = {}

            # Extract basic info (compatible with dataclass and dict)
            if hasattr(ped_result, "overall_score"):
                result["overall_score"] = float(ped_result.overall_score)
                result["decision"] = getattr(ped_result, "decision", None)
                result["pass_threshold"] = getattr(ped_result, "pass_threshold", 70.0)
                dimension_results = getattr(ped_result, "dimension_results", []) or []
                model_results = getattr(ped_result, "model_results", {}) or {}
                audit = getattr(ped_result, "audit", {}) or {}
            elif isinstance(ped_result, dict):
                result["overall_score"] = ped_result.get("overall_score")
                result["decision"] = ped_result.get("decision")
                result["pass_threshold"] = ped_result.get("pass_threshold", 70.0)
                dimension_results = ped_result.get("dimension_results", []) or []
                model_results = ped_result.get("model_results", {}) or {}
                audit = ped_result.get("audit", {}) or {}
            else:
                return _force_serializable(ped_result)

            # Dimension details (transparent)
            dimension_details = {}
            for dim_res in dimension_results:
                if hasattr(dim_res, "dimension_name"):
                    dim_name = dim_res.dimension_name
                    dim_detail = {
                        "dimension_id": getattr(dim_res, "dimension_id", ""),
                        "aggregated_score": float(getattr(dim_res, "score", 0)),
                        "hit_level": getattr(dim_res, "hit_level", ""),
                        "reasoning": getattr(dim_res, "reasoning", ""),
                        "per_model_scores": {}
                    }
                elif isinstance(dim_res, dict):
                    dim_name = dim_res.get("dimension_name", "")
                    dim_detail = {
                        "dimension_id": dim_res.get("dimension_id", ""),
                        "aggregated_score": dim_res.get("score", 0),
                        "hit_level": dim_res.get("hit_level", ""),
                        "reasoning": dim_res.get("reasoning", ""),
                        "per_model_scores": {}
                    }
                else:
                    continue

                # Add per-model scores for this dimension
                for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
                    if isinstance(model_dims, dict) and dim_name in model_dims:
                        model_dim = model_dims[dim_name]
                        if isinstance(model_dim, dict):
                            dim_detail["per_model_scores"][model_name] = {
                                "score": model_dim.get("score"),
                                "hit_level": model_dim.get("hit_level"),
                                "reason": model_dim.get("reason", "")[:200],
                                "weight": model_weights.get(model_name, 0.0)
                            }
                dimension_details[dim_name] = dim_detail
            result["dimension_details"] = dimension_details

            # Model summary
            model_summary = {}
            for model_name, model_dims in (model_results if isinstance(model_results, dict) else {}).items():
                if isinstance(model_dims, dict):
                    scores = [v.get("score") for v in model_dims.values() if isinstance(v, dict) and v.get("score") is not None]
                    if scores:
                        model_summary[model_name] = {
                            "average_score": round(sum(scores) / len(scores), 2),
                            "weight": model_weights.get(model_name, 0.0),
                            "dimensions_evaluated": len(scores)
                        }
            result["model_summary"] = model_summary

            # Score calculation formula explanation
            result["score_calculation"] = {
                "method": "weighted_average",
                "description": "Weighted average of dimension scores (weighted between models -> averaged between dimensions)",
                "hit_level_mapping": {
                    "high hit": 100.0,
                    "basic hit": 80.0,
                    "partially relevant": 60.0,
                    "significant deviation": 30.0
                },
                "formula": "overall = sum(dimension_score) / dimension_count"
            }

            # Audit info
            result["audit"] = _force_serializable(audit)

            return result

        try:
            unit_id = str(getattr(state, "unit_id", None) or "unknown")

            # [OK] One directory per question
            rel = Path("stage2") / f"unit_{unit_id}" / "evaluation_state.json"
            output_file = self.config.get_output_path(str(rel))

            # Get model weights
            model_weights = dict(getattr(self, "eval_model_weights", {}) or {})

            payload = {
                "experiment_id": getattr(state, "pipeline_id", None),
                "unit_id": unit_id,
                "final_decision": getattr(state, "final_decision", None),
                "current_stage": getattr(state, "current_stage", None),

                # [OK] Input audit (snapshot injected by run(); None if not available)
                "input": _force_serializable(getattr(state, "input_core", None)),
                "stage1_meta": _force_serializable(getattr(state, "input_stage1_meta", None)),

                # [OK] Two-way evaluation structure (transparency enhancement)
                # [2026-01 Added] preserved field marks whether evaluation is preserved or newly run
                "ai_eval": {
                    "enabled": bool(getattr(getattr(self.config.evaluation, "ai_centric", None), "enabled", False)) or bool(getattr(state, "ai_eval_success", False)),
                    "success": bool(getattr(state, "ai_eval_success", False)),
                    "preserved": bool(getattr(state, "ai_eval_preserved", False)),  # [2026-01 Added]
                    "result": _build_transparent_ai_eval(getattr(state, "ai_eval_result", None), model_weights),
                    "raw_result": _force_serializable(getattr(state, "ai_eval_result", None)),  # Keep raw result for audit
                },
                "pedagogical_eval": {
                    "enabled": bool(getattr(getattr(self.config.evaluation, "pedagogical", None), "enabled", False)) or bool(getattr(state, "pedagogical_eval_success", False)),
                    "success": bool(getattr(state, "pedagogical_eval_success", False)),
                    "preserved": bool(getattr(state, "ped_eval_preserved", False)),  # [2026-01 Added]
                    "result": _build_transparent_ped_eval(getattr(state, "pedagogical_eval_result", None), model_weights),
                    "raw_result": _force_serializable(getattr(state, "pedagogical_eval_result", None)),  # Keep raw result for audit
                },
                # [2026-01 Added] Independent GK/CS evaluation results
                "gk_eval": {
                    "enabled": bool(getattr(state, "gk_eval_enabled", False)) or bool(getattr(state, "gk_eval_success", False)),
                    "success": bool(getattr(state, "gk_eval_success", False)),
                    "preserved": bool(getattr(state, "gk_eval_preserved", False)),
                    "result": _build_transparent_ped_eval(getattr(state, "gk_eval_result", None), model_weights),
                    "raw_result": _force_serializable(getattr(state, "gk_eval_result", None)),
                },
                "cs_eval": {
                    "enabled": bool(getattr(state, "cs_eval_enabled", False)) or bool(getattr(state, "cs_eval_success", False)),
                    "success": bool(getattr(state, "cs_eval_success", False)),
                    "preserved": bool(getattr(state, "cs_eval_preserved", False)),
                    "result": _build_transparent_ped_eval(getattr(state, "cs_eval_result", None), model_weights),
                    "raw_result": _force_serializable(getattr(state, "cs_eval_result", None)),
                },

                "models": {
                    "eval_models": list(getattr(state, "eval_models", None) or []),
                    "ai_eval_models": list(getattr(state, "ai_eval_model_names", None) or []),
                    "ped_eval_models": list(getattr(state, "ped_eval_model_names", None) or []),
                    "model_weights": model_weights,
                    "weight_explanation": "Model weights used for aggregating multi-model scores, higher weight has greater impact on final score"
                },

                "skipped_modules": list(getattr(state, "skipped_modules", None) or []),
                "notes": getattr(state, "notes", "") or "",
                "errors": list(getattr(state, "errors", []) or []),

                # [2025-12 Added] High-variance dimension summary (for manual review)
                "high_variance_summary": self._collect_high_variance_dims(state, unit_id),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()  # Ensure data written to disk

            print(f"\n[INFO] Evaluation result saved: {output_file}")

        except Exception as e:
            import traceback
            print(f"\n[WARNING] Failed to save evaluation result: {e}")
            traceback.print_exc()
            # Try to save minimized debug info
            try:
                debug_payload = {
                    "error": str(e),
                    "unit_id": str(getattr(state, "unit_id", None) or "unknown"),
                    "experiment_id": str(getattr(state, "pipeline_id", None) or "unknown"),
                    "traceback": traceback.format_exc(),
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(debug_payload, f, ensure_ascii=False, indent=2)
                print(f"[DEBUG] Debug info saved to: {output_file}")
            except Exception as e2:
                print(f"[WARNING] Failed to save debug info as well: {e2}")



def create_evaluation_orchestrator(
    experiment_config: ExperimentConfig, llm_router: Optional["LLMRouter"] = None
) -> EvaluationOrchestrator:
    """
    Factory function (retained for external convenience)
    """
    return EvaluationOrchestrator(experiment_config, llm_router=llm_router)


__all__ = ["EvaluationOrchestrator", "create_evaluation_orchestrator"]
