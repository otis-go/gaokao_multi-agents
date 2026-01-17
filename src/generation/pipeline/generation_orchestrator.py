# src/generation/pipeline/generation_orchestrator.py
# Stage 1 Generation Pipeline Orchestrator

"""
Module Description:
GenerationOrchestrator orchestrates the four Agents in Stage 1, chaining them into a complete generation pipeline.

Position in Pipeline:
Master orchestrator for Stage 1 (Question Generation - Solving - Verification area)

Main Responsibilities:
1. Initialize all Agents (Agent 1, 2, 3, 4)
2. Call Agents in sequence, passing intermediate results
3. Handle exceptions and retry logic
4. Maintain complete PipelineState
5. Log and save intermediate results

Legacy Mapping:
Corresponds to old:
  - three_stage_orchestrator.py (Three-stage orchestrator)
  - experiment_orchestrator.py (Experiment orchestrator)

[2025-12] Refactoring Notes:
1. Question type is fully data-driven, no longer passed externally
2. Supports optional LLMRouter injection (unified LLM client management)
3. Added run_single(unit_id) method for single-question mode

[2025-12-27] Agent Renaming:
- Original Agent2 (Prompt Synthesizer) deleted, prompts extracted directly from data
- Original Agent3 (Anchor Finder) -> Agent2
- Original Agent4 (Question Generator) -> Agent3
- Original Agent5 (Verifier) -> Agent4
- Pipeline: Agent1 -> Prompt Extraction -> Agent2 -> Agent3 -> Agent4

Usage Example:
```python
from src.shared.config import create_default_config
from src.generation.pipeline.generation_orchestrator import GenerationOrchestrator
from src.shared.llm_router import LLMRouter

config = create_default_config("EXP_001")

# Method 1: Use default LLM client
orchestrator = GenerationOrchestrator(config)

# Method 2: Use LLMRouter (recommended)
router = LLMRouter.from_config(config)
orchestrator = GenerationOrchestrator(config, llm_router=router)

# Full mode
state = orchestrator.run()

# Single-question mode
state = orchestrator.run_single(unit_id="unit_001")
```
"""
import json
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from src.shared.schemas import (
    GenerationPipelineState,
    create_initial_generation_state,
    Stage2Record,
    Stage2CoreInput,
    Stage1Meta,
    # [2025-12] Output types needed for ablation experiments
    SynthesizedPrompt,
    AnchorSet,
    AnchorSpan,
    GeneratedQuestion,
    VerifiedQuestionSet,
    LightweightQualityScore,
    MaterialDimSelection,
)
from src.shared.config import ExperimentConfig
# [2025-12] Use unified adapter to build Stage2Record
from src.shared.adapters.stage1_to_stage2 import build_stage2_record as _adapter_build_stage2_record
from src.shared.llm_interface import LLMClient
from src.shared.prompt_logger import PromptLogger
from src.shared.data_loader import DataLoader

# Import implemented Agents
from src.generation.agents.agent1_material_dim_selector import Agent1MaterialDimSelector
# [2025-12-27] Original Agent2 (prompt synthesis) deleted, prompts extracted directly from data
from src.generation.agents.agent2_anchor_finder import Agent2AnchorFinder

# Agent 3 and 4 implemented (former Agent4 and Agent5)
from src.generation.agents.agent3_qg_solver import Agent3QuestionGeneratorSolver
from src.generation.agents.agent4_lightweight_verifier import Agent4LightweightVerifier

# [2026-01] Dimension check needed for low-freq ablation experiments
from src.evaluation.pedagogical_eval import get_low_freq_dims_by_mode

if TYPE_CHECKING:
    from src.shared.llm_router import LLMRouter


# [2026-01] Dimension code to label mapping (for low-freq ablation check)
# [2026-01-06] Updated to use authoritative names from ABC_prompt.json
# Chinese dimension labels - these are domain-specific Gaokao education terminology
GK_DIM_LABELS = {
    "GK01": "核心价值-爱国主义情怀",  # Core Values - Patriotism
    "GK02": "核心价值-品德修养",  # Core Values - Moral Cultivation
    "GK03": "核心价值-世界观和方法论",  # Core Values - Worldview and Methodology
    "GK04": "核心价值-法治意识",  # Core Values - Rule of Law Awareness
    "GK05": "核心价值-以人民为中心的思想",  # Core Values - People-centered Philosophy
    "GK06": "四翼要求-应用性",  # Four Wings - Applicability
    "GK07": "四翼要求-基础性",  # Four Wings - Foundational
    "GK08": "四翼要求-综合性",  # Four Wings - Comprehensive
    "GK09": "情境-生活实践情境",  # Context - Life Practice Context
    "GK10": "关键能力-思维认知能力群",  # Key Ability - Thinking Cognition Cluster
    "GK11": "学科素养-科学思维",  # Subject Literacy - Scientific Thinking
    "GK12": "学科素养-人文思维",  # Subject Literacy - Humanistic Thinking
    "GK13": "学科素养-信息获取",  # Subject Literacy - Information Acquisition
    "GK14": "学科素养-知识整合",  # Subject Literacy - Knowledge Integration
    "GK15": "学科素养-理解掌握",  # Subject Literacy - Comprehension Mastery
    "GK16": "学科素养-语言表达",  # Subject Literacy - Language Expression
    "GK17": "情境-学习探索情境",  # Context - Learning Exploration Context
}
CS_DIM_LABELS = {
    "CS01": "核心素养（四维）-语言建构与运用",  # Core Literacy (4D) - Language Construction
    "CS02": "核心素养（四维）-文化传承与理解",  # Core Literacy (4D) - Cultural Heritage
    "CS03": "核心素养（四维）-思维发展与提升",  # Core Literacy (4D) - Thinking Development
    "CS04": "学习任务群-实用性阅读与交流",  # Task Group - Practical Reading
    "CS05": "学习任务群-跨媒介阅读与交流",  # Task Group - Cross-media Reading
    "CS06": "学习任务群-语言积累、梳理与探究",  # Task Group - Language Accumulation
    "CS07": "学习任务群-中华传统文化专题研讨",  # Task Group - Chinese Traditional Culture
    "CS08": "语文学科能力要求-比较阅读与辩证判断",  # Chinese Ability - Comparative Reading
    "CS09": "语文学科能力要求-多材料整合",  # Chinese Ability - Multi-material Integration
    "CS10": "语文学科能力要求-表现手法识别与效果分析",  # Chinese Ability - Expression Technique
    "CS11": "语文学科能力要求-观点与材料关系",  # Chinese Ability - Viewpoint-Material Relation
    "CS12": "语文学科能力要求-方法与语言规律的迁移应用",  # Chinese Ability - Method Transfer
    "CS13": "语文学科能力要求-关键信息提取与概括主次",  # Chinese Ability - Key Info Extraction
    "CS14": "语文学科能力要求-立场识别与评议",  # Chinese Ability - Position Identification
    "CS15": "语文学科能力要求-论证方式识别与评估",  # Chinese Ability - Argumentation Evaluation
    "CS16": "语文学科能力要求-议题论述",  # Chinese Ability - Topic Discussion
    "CS17": "语文学科能力要求-结构化梳理与提纲生成",  # Chinese Ability - Structure Outline
    "CS18": "语文学科能力要求-概念界定与一致使用",  # Chinese Ability - Concept Definition
    "CS19": "语文学科能力要求-文化语境与古今视角解释",  # Chinese Ability - Cultural Context
    "CS20": "语文学科能力要求-词语与搭配运用的准确性",  # Chinese Ability - Word Collocation
    "CS21": "语文学科能力要求-结构与逻辑梳理及其作用",  # Chinese Ability - Structure Logic
}


class GenerationOrchestrator:
    """
    Stage 1 Generation Pipeline Orchestrator

    Main Methods:
    - run(): Execute complete generation pipeline (random material selection), returns GenerationPipelineState
    - run_single(unit_id): Single-question mode (specified material), returns GenerationPipelineState

    [2025-12] Refactoring:
    - Question type is fully data-driven, read from selected material data
    - Supports optional LLMRouter injection

    [2025-12-27] Agent Renaming:
    - Original Agent2 (Prompt Synthesis) deleted, prompts extracted directly from data
    - Agent2 = Anchor Finder (former Agent3)
    - Agent3 = Question Generator (former Agent4)
    - Agent4 = Verifier (former Agent5)
    - Pipeline: Agent1 -> Prompt Extraction -> Agent2 -> Agent3 -> Agent4

    Ablation Study Support:
    - Supports skipping Agent2/4 for ablation studies
    - Skipped Agents generate placeholder outputs, ensuring downstream Agents and Stage2 run normally
    - Configured via config.pipeline.stage1_ablation.skip_agent
    """

    def __init__(
            self,
            experiment_config: ExperimentConfig,
            llm_router: Optional["LLMRouter"] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            experiment_config: Experiment configuration
            llm_router: Optional LLM router (recommended, unified LLM client management)
        """
        self.config = experiment_config
        self.data_loader = DataLoader()
        self.llm_router = llm_router

        # [2025-12] Read ablation experiment configuration
        self.skip_agent = getattr(
            getattr(self.config.pipeline, "stage1_ablation", None),
            "skip_agent",
            "none"
        ) or "none"

        if self.skip_agent != "none":
            print(f"[GenerationOrchestrator] Ablation mode: skipping {self.skip_agent}")

        # Initialize LLM client (prefer Router)
        if llm_router is not None:
            self.llm_client = llm_router.get_generator_client()
        else:
            self.llm_client = LLMClient(
                api_type=self.config.llm.api_type,
                model_name=self.config.llm.model_name,
                verbose=self.config.llm.verbose
            )

        # Initialize Prompt logger: place in experiment output_dir (avoid mixing different experiments)
        prompt_log_dir = (Path(self.config.output_dir) / "logs" / "prompts").resolve()
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_logger = PromptLogger(log_dir=str(prompt_log_dir))

        # [2025-12-27] ABC prompt data cache (for direct extraction, replacing original Agent2)
        self._abc_prompt_cache = None

        # [2025-12-29] Iteration trigger log tracking
        self._iteration_triggers: List[Dict[str, Any]] = []

        # Initialize implemented Agents
        # [2026-01] Pass random dimension / low-freq random dimension ablation config
        use_random_dims = getattr(
            self.config.pipeline.stage1_ablation, "use_random_dims", False
        )
        use_low_freq_random = getattr(
            self.config.pipeline.stage1_ablation, "use_low_freq_random", False
        )
        # [2026-01-06] Low-freq dimension count (k=1/3/5)
        low_freq_count = getattr(
            self.config.pipeline.stage1_ablation, "low_freq_random_count", 3
        )
        self.agent1 = Agent1MaterialDimSelector(
            config=self.config.pipeline.agent1,
            data_loader=self.data_loader,
            use_random_dims=use_random_dims,
            use_low_freq_random=use_low_freq_random,
            low_freq_count=low_freq_count,
        )

        # [2025-12-27] Original Agent2 (prompt synthesis) deleted, prompts extracted directly from data

        self.agent2 = Agent2AnchorFinder(
            config=self.config.pipeline.agent2,
            llm_client=self.llm_client,
            prompt_logger=self.prompt_logger
        )

        self.agent3 = Agent3QuestionGeneratorSolver(
            config=self.config.pipeline.agent3,
            llm_client=self.llm_client,
            prompt_logger=self.prompt_logger
        )

        self.agent4 = Agent4LightweightVerifier(
            config=self.config.pipeline.agent4,
            llm_client=self.llm_client,
            prompt_logger=self.prompt_logger
        )

    # ========== Prompt Extraction Methods (replacing original Agent2) ==========

    def _is_skipped(self, agent_name: str) -> bool:
        """Check if the specified Agent is skipped."""
        return self.skip_agent == agent_name

    def _load_abc_prompt_data(self):
        """
        Load ABC prompt data (with cache).

        [2025-12-27] Replaces original Agent2's data loading.

        Returns:
            ABCPromptData: Contains dimensions, question_types, name_to_definition, etc.
        """
        if self._abc_prompt_cache is None:
            self._abc_prompt_cache = self.data_loader.load_abc_prompt()
        return self._abc_prompt_cache

    def _extract_abc_prompts_by_level(
        self,
        matched_abc_dims: list,
        question_type_label: str,
        abc_prompt_data,  # ABCPromptData object
        skip_question_type: bool = False,  # [2025-12-28] Ablation option
    ) -> dict:
        """
        Extract prompts by level from ABC prompt data.

        [2025-12-27] Replaces original Agent2's extraction method.

        [2025-12-28] Ablation option:
        - skip_question_type=True: Skip question type dimension, use only pedagogical dimensions

        Args:
            matched_abc_dims: List of matched dimension names (dimension_name)
            question_type_label: Question type label (e.g., "single-choice")
            abc_prompt_data: ABCPromptData object
            skip_question_type: Whether to skip question type dimension extraction (ablation option)

        Returns:
            Dict[str, str]: {"A": "...", "B": "...", "C": "..."}
        """
        prompts_by_level = {"A": "", "B": "", "C": ""}

        # Collect prompts for all matched dimensions
        all_level_a = []
        all_level_b = []
        all_level_c = []

        # Process dimensions - use name_to_definition mapping
        for dim_name in matched_abc_dims:
            dim_def = abc_prompt_data.name_to_definition.get(dim_name)
            if not dim_def:
                # Try to find by id
                for dim in abc_prompt_data.dimensions:
                    if dim.id == dim_name or dim.dimension_name == dim_name:
                        dim_def = dim
                        break

            if not dim_def:
                continue

            # Level A - access dataclass attributes
            level_a = dim_def.levelA if isinstance(dim_def.levelA, dict) else {}
            prompt_a = level_a.get("prompt", "")
            if prompt_a:
                all_level_a.append(f"【{dim_def.dimension_name}】{prompt_a}")

            # Level B
            level_b = dim_def.levelB if isinstance(dim_def.levelB, dict) else {}
            addon_b = level_b.get("addon", "")
            if addon_b:
                all_level_b.append(f"【{dim_def.dimension_name}】{addon_b}")

            # Level C
            level_c = dim_def.levelC if isinstance(dim_def.levelC, dict) else {}
            addon_c = level_c.get("addon", "")
            if addon_c:
                all_level_c.append(f"【{dim_def.dimension_name}】{addon_c}")

        # Process question type - find from question_types list
        # [2025-12-28] If skip_question_type=True, skip question type dimension extraction
        if question_type_label and not skip_question_type:
            for qtype in abc_prompt_data.question_types:
                if qtype.dimension_name == question_type_label or qtype.id == question_type_label:
                    level_a = qtype.levelA if isinstance(qtype.levelA, dict) else {}
                    prompt_a = level_a.get("prompt", "")
                    if prompt_a:
                        all_level_a.append(f"【QuestionType:{question_type_label}】{prompt_a}")

                    level_b = qtype.levelB if isinstance(qtype.levelB, dict) else {}
                    addon_b = level_b.get("addon", "")
                    if addon_b:
                        all_level_b.append(f"【QuestionType:{question_type_label}】{addon_b}")

                    level_c = qtype.levelC if isinstance(qtype.levelC, dict) else {}
                    addon_c = level_c.get("addon", "")
                    if addon_c:
                        all_level_c.append(f"【QuestionType:{question_type_label}】{addon_c}")
                    break
        elif skip_question_type and question_type_label:
            print(f"[SynthesizedPrompt] [Ablation] Skipping question type dimension: {question_type_label}")

        # Assemble prompts by level
        prompts_by_level["A"] = "\n\n".join(all_level_a)
        # Level B = Level A + Level B addon
        prompts_by_level["B"] = "\n\n".join(all_level_a + all_level_b) if all_level_b else prompts_by_level["A"]
        # [2026-01-05] Level C = A + C (skip B, use real exam examples directly)
        prompts_by_level["C"] = "\n\n".join(all_level_a + all_level_c) if all_level_c else prompts_by_level["A"]

        return prompts_by_level

    def _create_synthesized_prompt(
        self,
        state: GenerationPipelineState,
        iteration: int = 1,  # [2025-12-29] Current iteration round
    ) -> SynthesizedPrompt:
        """
        Create SynthesizedPrompt directly from data (replacing original Agent2).

        [2025-12-27] Refactoring:
        - Original Agent2 used LLM to fuse multiple dimension prompts
        - Now directly extract raw dimension prompts without LLM fusion
        - Prompts concatenated by level (A/B/C)

        [2025-12-29] Enhancement:
        - If previous round Agent4 returned prompt_revision_suggestion, append to prompt
        - Allows next round to address issues specifically
        - IMPORTANT: Only append revision suggestion when iteration > 1, first round has no iteration info

        Strategy:
        - synthesized_instruction: Use raw dimension prompts (no LLM fusion)
        - When iteration > 1: Append previous round's revision suggestion
        - dimension_ids: Extract from agent1_output.matched_abc_dims

        Args:
            state: Current pipeline state
            iteration: Current iteration round (1-based), round 1 has no revision info
        """
        a1 = state.agent1_output
        if a1 is None:
            raise ValueError("Cannot create SynthesizedPrompt: agent1_output is None")

        # Get dimension and question type info
        matched_abc_dims = getattr(a1, "matched_abc_dims", None) or getattr(a1, "top20_dims", None) or []
        question_type_label = getattr(a1, "question_type_label", "") or ""

        # [2026-01] Low-freq ablation: For questions with low-freq dimensions, don't provide dimension prompts
        low_freq_ablation = getattr(
            getattr(self.config.pipeline, "stage1_ablation", None),
            "low_freq_ablation",
            False
        )
        if low_freq_ablation:
            # [2026-01] Removed gk+cs mode, default to gk
            dim_mode = getattr(self.config.pipeline.agent1, "dimension_mode", "gk")
            if self._check_has_low_freq_dims(matched_abc_dims, dim_mode):
                print(f"[SynthesizedPrompt] [LowFreqAblation] Detected low-freq dimension, skipping dimension prompts")
                return SynthesizedPrompt(
                    synthesized_instruction="[LOW_FREQ_ABLATION] Ablation: No dimension prompts provided",
                    prompts_by_level={},
                    used_prompt_level="ABLATION",
                    primary_dimension_template="",
                    secondary_templates=[],
                    dimension_ids=list(matched_abc_dims),
                    prompt_revision_suggestion=None,
                )

        # Use config to determine prompt_level (read from prompt_extraction config)
        used_level = (self.config.pipeline.prompt_extraction.prompt_level or "C").upper()
        # [2025-12-28] Read ablation option: whether to skip question type dimension
        skip_qtype = getattr(self.config.pipeline.prompt_extraction, "skip_question_type_prompt", False)
        prompts_by_level = {}
        raw_prompt = ""

        if matched_abc_dims or question_type_label:
            try:
                # Load ABC prompt data
                abc_prompt_data = self._load_abc_prompt_data()
                # Extract raw prompts by level
                prompts_by_level = self._extract_abc_prompts_by_level(
                    matched_abc_dims=matched_abc_dims,
                    question_type_label=question_type_label,
                    abc_prompt_data=abc_prompt_data,
                    skip_question_type=skip_qtype,  # Ablation option: pass skip question type flag
                )
                # Select raw prompt for specified level
                raw_prompt = prompts_by_level.get(used_level, "").strip()

                # If specified level is empty, try fallback
                if not raw_prompt:
                    for fallback_level in ["C", "B", "A"]:
                        cand = prompts_by_level.get(fallback_level, "").strip()
                        if cand:
                            raw_prompt = cand
                            used_level = fallback_level
                            print(f"[SynthesizedPrompt] Prompt fallback to level {used_level}")
                            break

                print(f"[SynthesizedPrompt] Extracted raw prompt (level={used_level}, length={len(raw_prompt)})")
            except Exception as e:
                print(f"[SynthesizedPrompt] [Warning] Failed to extract raw prompt: {e}, using empty prompt")
                raw_prompt = ""

        # If extraction failed or empty, use marker text
        if not raw_prompt:
            raw_prompt = "[NO_PROMPT] Cannot extract dimension prompts"
            used_level = "NONE"
            print("[SynthesizedPrompt] [Warning] Raw prompt is empty, using placeholder text")

        # [2025-12-29] Only append previous Agent4's revision suggestion when iteration > 1
        revision_suggestion = None
        if iteration > 1 and state.agent5_output is not None:
            revision_suggestion = getattr(state.agent5_output, "prompt_revision_suggestion", None)

        if revision_suggestion:
            print(f"[SynthesizedPrompt] Round {iteration} iteration: Detected previous QC feedback, appending to prompt (length={len(revision_suggestion)})")
            raw_prompt = raw_prompt + "\n\n" + revision_suggestion
        elif iteration == 1:
            print(f"[SynthesizedPrompt] Round 1: No iteration revision info (initial generation)")

        # Extract dimension_ids
        dim_ids = list(matched_abc_dims) if matched_abc_dims else []
        if not dim_ids:
            # Fallback: extract from old abc_dims
            abc_dims = getattr(a1, "abc_dims", None) or []
            for d in abc_dims:
                dn = (getattr(d, "dimension_name", None) or "").strip()
                if dn:
                    dim_ids.append(dn)
                else:
                    did = (getattr(d, "id", None) or "").strip()
                    if did:
                        dim_ids.append(did)

        # Create output
        result = SynthesizedPrompt(
            synthesized_instruction=raw_prompt,
            prompts_by_level=prompts_by_level,
            used_prompt_level=used_level,
            primary_dimension_template="",
            secondary_templates=[],
            dimension_ids=dim_ids,
            prompt_revision_suggestion=revision_suggestion,  # Save revision suggestion for tracking
        )

        print(f"[SynthesizedPrompt] Created: dims={len(dim_ids)}, level={used_level}, has_revision={revision_suggestion is not None}")
        return result

    # ========== Ablation Study Helper Methods ==========

    def _make_dummy_agent2_output(self, state: GenerationPipelineState) -> AnchorSet:
        """
        Create placeholder Agent2 output (used when Agent2 is skipped).

        Strategy:
        - anchors: Empty list (no anchors)
        - anchor_discovery_reasoning: Mark as ablation study
        """
        dummy = AnchorSet(
            anchors=[],
            anchor_discovery_reasoning="[ABLATION_SKIP_AGENT2] Ablation study: Agent2 skipped, no anchor discovery",
        )

        print("[ABLATION] Created placeholder Agent2 output: anchors=[]")
        return dummy

    def _make_dummy_agent4_output(self, state: GenerationPipelineState) -> VerifiedQuestionSet:
        """
        Create placeholder Agent4 output (used when Agent4 is skipped).

        Strategy:
        - original_question: Wrap agent3_output
        - quality_score: Set overall_score=1.0 (auto pass)
        - need_revision=False, is_reject=False (skip QC)
        """
        a3 = state.agent4_output
        if a3 is None:
            raise ValueError("Cannot create dummy Agent4 output: agent4_output is None")

        # Create perfect pass quality score
        quality_score = LightweightQualityScore(
            overall_score=1.0,
            layer_scores={"ablation_skip": 1.0},
        )

        dummy = VerifiedQuestionSet(
            original_question=a3,
            quality_score=quality_score,
            issues=[],
            need_revision=False,
            is_reject=False,
            verifier_tags={"ablation_skip": True},
            verifier_notes="[ABLATION_SKIP_AGENT4] Ablation study: Agent4 skipped, auto pass QC",
            prompt_revision_suggestion=None,
        )

        print(f"[ABLATION] Created placeholder Agent4 output: overall_score=1.0, need_revision=False")
        return dummy

    def _check_has_low_freq_dims(
        self,
        matched_dims: List[str],
        dim_mode: str,
    ) -> bool:
        """
        [2026-01] Check if dimension list contains low-frequency dimensions.

        Args:
            matched_dims: List of matched dimension names (e.g., ["Core Values-Patriotism", ...])
            dim_mode: Dimension mode ("gk" or "cs")

        Returns:
            bool: Whether contains low-frequency dimensions
        """
        if not matched_dims:
            return False

        # Get low-freq dimension code set
        low_freq_codes = get_low_freq_dims_by_mode(dim_mode)
        if not low_freq_codes:
            print(f"[LowFreqAblation] No low-freq dimension definition found, dim_mode={dim_mode}")
            return False

        # Build low-freq dimension label set
        low_freq_labels = set()
        for code in low_freq_codes:
            if code.startswith("GK"):
                label = GK_DIM_LABELS.get(code)
            elif code.startswith("CS"):
                label = CS_DIM_LABELS.get(code)
            else:
                label = None
            if label:
                low_freq_labels.add(label)

        # Check if matched_dims contains any low-freq dimension label
        for dim_name in matched_dims:
            # dim_name format may be "Category-SpecificName" or just "SpecificName"
            for label in low_freq_labels:
                if label in dim_name:
                    print(f"[LowFreqAblation] Detected low-freq dimension: {dim_name} (matched label: {label})")
                    return True

        return False

    def run(self) -> GenerationPipelineState:
        """
        Execute complete Stage 1 generation pipeline (full mode, random material selection).

        Returns:
            GenerationPipelineState: Complete pipeline state

        [2025-12] Refactoring:
        - Question type is fully data-driven, read from data inside Agent1
        - Supports iteration (up to verifier_max_rounds rounds)
        """
        print("=" * 80)
        print(f"Starting Stage 1 generation pipeline - Experiment ID: {self.config.experiment_id}")
        print(f"  Max iterations: {self.config.pipeline.agent4.verifier_max_rounds}")
        print("=" * 80)

        # Create initial state
        state = create_initial_generation_state(self.config.experiment_id)

        try:
            # ========== Agent 1: Material and Dimension Selection (execute once) ==========
            if self.config.pipeline.agent1.enabled:
                print("\n" + "-" * 80)
                state = self._run_agent1(state)
                if not state.agent1_success:
                    return state

            # Agent 2-5 iteration loop
            state = self._run_agents_2_to_5(state)

        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {e}")
            traceback.print_exc()
            state.errors.append(str(e))
            state.current_agent = "failed"

        # -------- Stage 2 standardized input construction (only when Stage1 all successful) --------
        if (
                state.agent1_success
                and state.agent2_success
                and state.agent3_success
                and state.agent4_success
                and state.agent5_success
        ):
            try:
                state.stage2_record = self.build_stage2_record(state)
            except Exception as e:
                print(f"[WARN] Failed to build stage2_record: {e}")
                state.errors.append(f"Failed to build stage2_record: {e}")

        return state

    def run_single(self, unit_id: str) -> GenerationPipelineState:
        """
        Single-question mode: Run Stage 1 generation pipeline with specified unit_id.

        Args:
            unit_id: Specified material unit ID

        Returns:
            GenerationPipelineState: Complete pipeline state

        [2025-12] Added:
        - Question type is read from data, no longer passed externally
        - Uses agent1.run_single(unit_id) for execution
        """
        print("=" * 80)
        print(f"Starting Stage 1 generation pipeline (single mode) - Experiment ID: {self.config.experiment_id}")
        print(f"  Specified material unit: {unit_id}")
        print("=" * 80)

        # Create initial state
        state = create_initial_generation_state(self.config.experiment_id)

        try:
            # ========== Agent 1: Dimension selection for specified material ==========
            if self.config.pipeline.agent1.enabled:
                print("\n" + "-" * 80)
                state = self._run_agent1_single(state, unit_id)
                if not state.agent1_success:
                    return state

            # Subsequent Agent 2-5 flow same as run()
            state = self._run_agents_2_to_5(state)

        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {e}")
            traceback.print_exc()
            state.errors.append(str(e))
            state.current_agent = "failed"

        # -------- Stage 2 standardized input construction (only when Stage1 all successful) --------
        if (
            state.agent1_success
            and state.agent2_success
            and state.agent3_success
            and state.agent4_success
            and state.agent5_success
        ):
            try:
                state.stage2_record = self.build_stage2_record(state)
            except Exception as e:
                print(f"[WARN] Failed to build stage2_record: {e}")
                state.errors.append(f"Failed to build stage2_record: {e}")

        return state

    def _run_agent1(self, state: GenerationPipelineState) -> GenerationPipelineState:
        """Execute Agent 1 (full mode, random material selection).

        [2025-12] Refactoring:
        - Question type is fully data-driven, no longer passed externally
        - Agent1.run() reads question type from data internally

        [2025-12-31] Added:
        - In gk_only/cs_only mode, if question has no corresponding dimensions, skip it
        - Avoid meaningless generation without dimensions
        """
        try:
            # [Refactoring] No longer pass question_type, Agent1 reads from data internally
            agent1_output = self.agent1.run()

            # [2026-01] Check if has corresponding dimensions (based on dimension_mode, removed gk+cs mode)
            dimension_mode = getattr(self.config.pipeline.agent1, "dimension_mode", "gk")
            gk_dims = getattr(agent1_output, "gk_dims", {}) or {}
            cs_dims = getattr(agent1_output, "cs_dims", {}) or {}
            unit_id = getattr(agent1_output, "unit_id", "unknown")

            # gk/gk_only mode: Check if has any gk dimensions
            if dimension_mode in ("gk_only", "gk"):
                has_gk_dims = any(gk_dims.values()) if gk_dims else False
                if not has_gk_dims:
                    skip_msg = f"unit_id={unit_id} has no GK dimensions in gk mode, skipping"
                    print(f"[Agent 1] [SKIP] {skip_msg}")
                    state.errors.append(f"Agent 1 skipped: {skip_msg}")
                    state.agent1_output = agent1_output  # Save output for debugging
                    state.agent1_success = False
                    state.current_agent = "skipped_no_dim"
                    return state

            # cs/cs_only mode: Check if has any cs dimensions
            elif dimension_mode in ("cs_only", "cs"):
                has_cs_dims = any(cs_dims.values()) if cs_dims else False
                if not has_cs_dims:
                    skip_msg = f"unit_id={unit_id} has no CS dimensions in cs mode, skipping"
                    print(f"[Agent 1] [SKIP] {skip_msg}")
                    state.errors.append(f"Agent 1 skipped: {skip_msg}")
                    state.agent1_output = agent1_output  # Save output for debugging
                    state.agent1_success = False
                    state.current_agent = "skipped_no_dim"
                    return state

            state.agent1_output = agent1_output
            state.agent1_success = True
            state.current_agent = "agent2"

        except Exception as e:
            print(f"[Agent 1] Execution failed: {e}")
            state.errors.append(f"Agent 1 failed: {e}")
            state.agent1_success = False
            state.current_agent = "failed"

        return state

    def _run_agent1_single(
        self, state: GenerationPipelineState, unit_id: str
    ) -> GenerationPipelineState:
        """Execute Agent 1 (single mode, specified material).

        [2025-12] Added:
        - Use agent1.run_single(unit_id) to specify material
        - Question type read from data

        [2025-12-31] Added:
        - In gk_only/cs_only mode, if question has no corresponding dimensions, skip it
        - Avoid meaningless generation without dimensions
        """
        try:
            agent1_output = self.agent1.run_single(unit_id)

            # [2026-01] Check if has corresponding dimensions (removed gk+cs mode)
            dimension_mode = getattr(self.config.pipeline.agent1, "dimension_mode", "gk")
            gk_dims = getattr(agent1_output, "gk_dims", {}) or {}
            cs_dims = getattr(agent1_output, "cs_dims", {}) or {}

            # gk/gk_only mode: Check if has any gk dimensions
            if dimension_mode in ("gk_only", "gk"):
                has_gk_dims = any(gk_dims.values()) if gk_dims else False
                if not has_gk_dims:
                    skip_msg = f"unit_id={unit_id} has no GK dimensions in gk mode, skipping"
                    print(f"[Agent 1] [SKIP] {skip_msg}")
                    state.errors.append(f"Agent 1 skipped: {skip_msg}")
                    state.agent1_output = agent1_output  # Save output for debugging
                    state.agent1_success = False
                    state.current_agent = "skipped_no_dim"
                    return state

            # cs/cs_only mode: Check if has any cs dimensions
            elif dimension_mode in ("cs_only", "cs"):
                has_cs_dims = any(cs_dims.values()) if cs_dims else False
                if not has_cs_dims:
                    skip_msg = f"unit_id={unit_id} has no CS dimensions in cs mode, skipping"
                    print(f"[Agent 1] [SKIP] {skip_msg}")
                    state.errors.append(f"Agent 1 skipped: {skip_msg}")
                    state.agent1_output = agent1_output  # Save output for debugging
                    state.agent1_success = False
                    state.current_agent = "skipped_no_dim"
                    return state

            state.agent1_output = agent1_output
            state.agent1_success = True
            state.current_agent = "agent2"

        except Exception as e:
            print(f"[Agent 1] Execution failed: {e}")
            state.errors.append(f"Agent 1 failed: {e}")
            state.agent1_success = False
            state.current_agent = "failed"

        return state

    def _run_agents_2_to_5(self, state: GenerationPipelineState) -> GenerationPipelineState:
        """
        Execute Agent 2-4 iteration loop (original Agent2 prompt synthesis deleted, replaced with direct extraction).

        Note:
        Extracted from run() method, shared by run() and run_single(), avoiding code duplication.

        [2025-12-27] Refactoring:
        - Original Agent2 (prompt synthesis) deleted, prompts extracted directly from data
        - Agent2 = Anchor Finder (former Agent3)
        - Agent3 = Question Generator (former Agent4)
        - Agent4 = Verifier (former Agent5)
        - Pipeline: Prompt Extraction -> Agent2 -> Agent3 -> Agent4

        Ablation Study Support:
        - Skip agent2/agent4 uses placeholder output
        - Skip agent4 forces max_iterations=1 (cannot iterate)
        """
        max_iterations = self.config.pipeline.agent4.verifier_max_rounds

        # [Ablation] Force single round when Agent4 skipped (cannot iterate)
        if self._is_skipped("agent4"):
            max_iterations = 1
            print("[ABLATION] Agent4 skipped, forcing max_iterations=1 (no iteration)")

        for iteration in range(1, max_iterations + 1):
            print("\n" + "=" * 80)
            print(f"[Iteration Round {iteration}/{max_iterations}] Starting Agent 2-4 execution")
            if self.skip_agent != "none":
                print(f"  [Ablation mode] Skipping: {self.skip_agent}")
            print("=" * 80)

            # ========== Prompt Extraction (replacing original Agent2 prompt synthesis) ==========
            print("\n" + "-" * 80)
            print("[Prompt Extraction] Extracting dimension prompts directly from ABC_prompt.json...")
            try:
                # [2025-12-29] Pass iteration, only append revision suggestion when iteration > 1
                state.agent2_output = self._create_synthesized_prompt(state, iteration=iteration)
                state.agent2_success = True
                state.current_agent = "agent3"
            except Exception as e:
                print(f"[Prompt Extraction] Execution failed: {e}")
                state.errors.append(f"Prompt extraction failed: {e}")
                state.agent2_success = False
                state.current_agent = "failed"
                return state

            # ========== Agent 2: Anchor Discovery (former Agent3) ==========
            if self.config.pipeline.agent2.enabled:
                print("\n" + "-" * 80)
                if self._is_skipped("agent2"):
                    # [Ablation] Use placeholder output
                    print("[Agent 2] [ABLATION] Skipping execution, using placeholder output")
                    state.agent3_output = self._make_dummy_agent2_output(state)
                    state.agent3_success = True
                    state.current_agent = "agent4"
                else:
                    state = self._run_agent2(state)
                    if not state.agent3_success:
                        return state

            # ========== Agent 3: Question Generation + Solving (former Agent4) ==========
            if self.config.pipeline.agent3.enabled:
                print("\n" + "-" * 80)
                # Agent3 cannot be skipped (blocked at CLI layer)
                state = self._run_agent3(state)
                if not state.agent4_success:
                    return state

            # ========== Agent 4: Lightweight QC (former Agent5) ==========
            if self.config.pipeline.agent4.enabled:
                print("\n" + "-" * 80)
                if self._is_skipped("agent4"):
                    # [Ablation] Use placeholder output (auto pass)
                    print("[Agent 4] [ABLATION] Skipping execution, using placeholder output (auto pass)")
                    state.agent5_output = self._make_dummy_agent4_output(state)
                    state.agent5_success = True
                    state.current_agent = "agent4"
                else:
                    state = self._run_agent4(state)
                    if not state.agent5_success:
                        return state

            # ---------- Iteration decision logic ----------
            if state.agent5_output:
                need_revision = state.agent5_output.need_revision
                is_reject = getattr(state.agent5_output, "is_reject", False)

                print(f"\n[Agent 4] need_revision = {need_revision}, is_reject = {is_reject}")

                # [2025-12-29] Record iteration trigger info
                if need_revision or is_reject:
                    trigger_info = self._record_iteration_trigger(state, iteration, need_revision, is_reject)
                    if trigger_info:
                        print(f"  [Iteration trigger] Recorded: {trigger_info.get('trigger_type', 'unknown')}")

                # [Ablation] When Agent4 skipped, need_revision is always False, exit directly
                if self._is_skipped("agent4"):
                    print(f"\n[Iteration] [ABLATION] Agent4 skipped, auto pass, exiting iteration loop")
                    break

                # 1) Hard reject: Treat as Stage1 failure, don't enter Stage2
                if is_reject:
                    msg = (
                        f"Agent 4 determined hard reject in round {iteration} "
                        f"(overall_score={state.agent5_output.quality_score.overall_score:.2f}), "
                        f"terminating pipeline, not sending to Stage 2."
                    )
                    print("[Iteration] " + msg)
                    state.errors.append(msg)
                    state.agent5_success = False
                    state.current_agent = "failed"
                    # Can still save current state for later analysis
                    if self.config.save_intermediate:
                        self._save_intermediate_state(state, suffix=f"_iter{iteration}_reject")
                    return state

                # 2) No revision needed: QC passed, end iteration
                if not need_revision:
                    print(f"\n[Iteration] Round {iteration} QC passed (need_revision=False), exiting iteration loop early")
                    break

                # 3) Need revision but not at max rounds: Go to next round
                if iteration < max_iterations:
                    print(f"\n[Iteration] Round {iteration} QC not passed, preparing round {iteration + 1}")
                else:
                    # 4) Reached max rounds still need_revision: Treat as failure, don't enter Stage2
                    msg = (
                        f"Agent 4 still marked need_revision=True after {max_iterations} iteration rounds, "
                        f"terminating pipeline, not sending to Stage 2."
                    )
                    print("[Iteration] " + msg)
                    state.errors.append(msg)
                    state.agent5_success = False
                    state.current_agent = "failed"
                    # [2026-01] Mark iteration exceeded failure
                    state.iteration_count = iteration
                    state.max_iteration_exceeded = True
                    state.iteration_fail_reason = f"need_revision=True after {max_iterations} iterations"
                    if self.config.save_intermediate:
                        self._save_intermediate_state(state, suffix=f"_iter{iteration}_need_revision")
                    return state

        # [2026-01] Record final iteration count
        state.iteration_count = iteration
        state.current_agent = "completed"
        print("\n" + "=" * 80)
        print("[SUCCESS] Stage 1 generation pipeline completed (Agent 1, 2, 3, 4)")
        if self.skip_agent != "none":
            print(f"  [Ablation mode] Skipped: {self.skip_agent}")
        print("=" * 80)
        return state

    def _run_agent2(self, state: GenerationPipelineState) -> GenerationPipelineState:
        """Execute Agent 2 (Anchor Discovery, former Agent3)"""
        if not state.agent1_output or not state.agent2_output:
            state.errors.append("Agent 2: Missing prerequisite Agent output")
            return state

        try:
            agent2_output = self.agent2.run(
                state.agent1_output,
                state.agent2_output
            )

            state.agent3_output = agent2_output
            state.agent3_success = True
            state.current_agent = "agent4"

        except Exception as e:
            print(f"[Agent 2] Execution failed: {e}")
            state.errors.append(f"Agent 2 failed: {e}")
            state.agent3_success = False
            state.current_agent = "failed"

        return state

    def _run_agent3(self, state: GenerationPipelineState) -> GenerationPipelineState:
        """Execute Agent 3 (Question Generation, former Agent4)"""
        if not state.agent1_output or not state.agent2_output or not state.agent3_output:
            state.errors.append("Agent 3: Missing prerequisite Agent output")
            return state

        try:
            agent3_output = self.agent3.run(
                state.agent1_output,
                state.agent3_output,
                state.agent2_output
            )

            state.agent4_output = agent3_output
            state.agent4_success = True
            state.current_agent = "agent5"

        except Exception as e:
            print(f"[Agent 3] Execution failed: {e}")
            state.errors.append(f"Agent 3 failed: {e}")
            state.agent4_success = False
            state.current_agent = "failed"

        return state

    def _run_agent4(self, state: GenerationPipelineState) -> GenerationPipelineState:
        """Execute Agent 4 (QC, former Agent5).

        [Phase 6+ Update] Pass agent1 output and material_text, supports LLM self-consistency check
        """
        if not state.agent4_output:
            state.errors.append("Agent 4: Missing Agent 3 output")
            return state

        try:
            # [Phase 6+] Pass complete parameters to Agent 4, supports LLM self-consistency check
            agent4_output = self.agent4.run(
                agent3_output=state.agent4_output,
                agent1_output=state.agent1_output,
                material_text=state.agent1_output.material_text if state.agent1_output else None
            )

            state.agent5_output = agent4_output
            state.agent5_success = True
            state.current_agent = "agent4"

        except Exception as e:
            print(f"[Agent 4] Execution failed: {e}")
            state.errors.append(f"Agent 4 failed: {e}")
            state.agent5_success = False
            state.current_agent = "failed"

        return state

    def build_stage2_record(self, state: GenerationPipelineState) -> Optional[Stage2Record]:
        """
        Extract minimal self-consistent input needed for Stage 2 from current state.

        [2025-12] Architecture refactoring:
        - Delegate to src/shared/adapters/stage1_to_stage2.py's build_stage2_record
        - Maintain backward compatibility: Return None means cannot build

        Constraints:
        - Must have agent1_output / agent5_output, and question body must be available (prefer from agent5)
        - If Agent5 determined is_reject or need_revision, don't build (return None)
        """
        # [2025-12] Use unified adapter
        result = _adapter_build_stage2_record(state, ablation_skip_agent=self.skip_agent)

        if result.success and result.record is not None:
            state.stage2_record = result.record
            # Record any warnings
            if result.warnings:
                for w in result.warnings:
                    print(f"[build_stage2_record] Warning: {w}")
            return result.record
        else:
            # Record skip reasons
            if result.skip_reasons:
                for reason in result.skip_reasons:
                    print(f"[build_stage2_record] Skip: {reason}")
                    state.errors.append(f"stage2_record skip: {reason}")
            return None

    # ========== [2025-12-29] Iteration Trigger Log ==========

    def _record_iteration_trigger(
        self,
        state: GenerationPipelineState,
        iteration: int,
        need_revision: bool,
        is_reject: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Record single iteration trigger info.

        [2025-12-29] Added to track which questions triggered iteration in which round.
        [2025-12-30] Enhanced with complete case info: material, question, explanation, revision reason for paper examples.

        Returns:
            Dict with trigger info, or None if no trigger
        """
        if not (need_revision or is_reject):
            return None

        # Get basic info
        unit_id = "unknown"
        question_type = "unknown"
        material_text = ""
        a1 = getattr(state, "agent1_output", None)
        if a1 is not None:
            unit_id = str(getattr(a1, "unit_id", None) or getattr(a1, "material_id", None) or "unknown")
            question_type = str(getattr(a1, "question_type", None) or "unknown")
            material_text = getattr(a1, "material_text", "") or ""

        # Get trigger reasons
        trigger_reasons = []
        issues = []
        issues_full = []  # Complete issue description for paper examples
        prompt_revision_suggestion = ""
        a5 = getattr(state, "agent5_output", None)
        if a5 is not None:
            # Layer1 format issues
            for issue in getattr(a5, "issues", []):
                issue_dict = {
                    "layer": getattr(issue, "layer", "unknown"),
                    "issue_type": getattr(issue, "issue_type", "unknown"),
                    "description": getattr(issue, "description", "")[:200]
                }
                issue_dict_full = {
                    "layer": getattr(issue, "layer", "unknown"),
                    "issue_type": getattr(issue, "issue_type", "unknown"),
                    "description": getattr(issue, "description", "")  # Complete description
                }
                issues.append(issue_dict)
                issues_full.append(issue_dict_full)
                if issue_dict["layer"] == "layer1_format":
                    trigger_reasons.append(f"Layer1 format issue: {issue_dict['issue_type']}")
                elif issue_dict["layer"] == "layer2_consistency":
                    trigger_reasons.append(f"Layer2 consistency issue: {issue_dict['issue_type']}")

            # Get revision suggestion
            prompt_revision_suggestion = getattr(a5, "prompt_revision_suggestion", "") or ""

        # Determine trigger type
        if is_reject:
            trigger_type = "hard_reject"
        elif any("layer1" in r.lower() for r in trigger_reasons):
            trigger_type = "layer1_format"
        elif any("layer2" in r.lower() for r in trigger_reasons):
            trigger_type = "layer2_consistency"
        else:
            trigger_type = "unknown"

        # [2025-12-30] Get complete question info
        question_content = {}
        explanation = ""
        if a5 is not None:
            original_question = getattr(a5, "original_question", None)
            if original_question is not None:
                question_content["stem"] = getattr(original_question, "stem", "") or ""
                explanation = getattr(original_question, "explanation", "") or ""

                # Multiple choice
                if question_type == "single-choice":
                    options = getattr(original_question, "options", None)
                    if options:
                        question_content["options"] = {}
                        for opt in options:
                            label = getattr(opt, "label", "")
                            content = getattr(opt, "content", "")
                            is_correct = getattr(opt, "is_correct", False)
                            reasoning = getattr(opt, "reasoning", "")
                            question_content["options"][label] = {
                                "content": content,
                                "is_correct": is_correct,
                                "reasoning": reasoning
                            }
                    question_content["correct_answer"] = getattr(original_question, "correct_answer", "") or ""

                # Essay
                elif question_type == "essay":
                    answer_points = getattr(original_question, "answer_points", None)
                    if answer_points:
                        question_content["answer_points"] = []
                        for ap in answer_points:
                            question_content["answer_points"].append({
                                "point": getattr(ap, "point", ""),
                                "score": getattr(ap, "score", 0),
                                "evidence_reference": getattr(ap, "evidence_reference", [])
                            })
                    question_content["total_score"] = getattr(original_question, "total_score", 0)

        trigger_info = {
            "unit_id": unit_id,
            "question_type": question_type,
            "iteration": iteration,
            "trigger_type": trigger_type,
            "trigger_reasons": trigger_reasons,
            "is_reject": is_reject,
            "need_revision": need_revision,
            "issues": issues,
            "timestamp": self._get_timestamp(),
            # [2025-12-30] Complete case info for paper statistics and examples
            "case_details": {
                "material": material_text,
                "question": question_content,
                "explanation": explanation,
                "issues_full": issues_full,
                "revision_suggestion": prompt_revision_suggestion
            }
        }

        self._iteration_triggers.append(trigger_info)
        return trigger_info

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_iteration_trigger_summary(self) -> Optional[str]:
        """
        Save iteration trigger log summary.

        [2025-12-29] Called at experiment end, output iteration trigger statistics and details.
        [2025-12-30] Enhanced to output complete case info (material, question, explanation, revision reason) for paper examples.

        Returns:
            Saved log file path, or None if no trigger records
        """
        if not self._iteration_triggers:
            print("[Iteration Log] No iteration trigger records in this experiment")
            return None

        # Statistics
        total_triggers = len(self._iteration_triggers)
        by_type = Counter(t["trigger_type"] for t in self._iteration_triggers)
        by_question_type = Counter(t["question_type"] for t in self._iteration_triggers)
        by_iteration = Counter(t["iteration"] for t in self._iteration_triggers)

        # Build summary
        summary_lines = [
            "=" * 80,
            f"[Iteration Trigger Log Summary] Experiment ID: {self.config.experiment_id}",
            f"Generated: {self._get_timestamp()}",
            "=" * 80,
            "",
            f"## Total triggers: {total_triggers}",
            "",
            "## By trigger type:",
        ]
        for trigger_type, count in by_type.most_common():
            summary_lines.append(f"  - {trigger_type}: {count}")

        summary_lines.extend([
            "",
            "## By question type:",
        ])
        for qtype, count in by_question_type.most_common():
            summary_lines.append(f"  - {qtype}: {count}")

        summary_lines.extend([
            "",
            "## By iteration round:",
        ])
        for iter_num, count in sorted(by_iteration.items()):
            summary_lines.append(f"  - Round {iter_num}: {count}")

        summary_lines.extend([
            "",
            "=" * 80,
            "## Detailed trigger records:",
            "=" * 80,
        ])

        for i, trigger in enumerate(self._iteration_triggers, 1):
            summary_lines.extend([
                "",
                f"### [{i}] Unit {trigger['unit_id']} (Round {trigger['iteration']})",
                f"  - Question type: {trigger['question_type']}",
                f"  - Trigger type: {trigger['trigger_type']}",
                f"  - Trigger reasons: {', '.join(trigger['trigger_reasons']) if trigger['trigger_reasons'] else 'Unknown'}",
                f"  - is_reject: {trigger['is_reject']}, need_revision: {trigger['need_revision']}",
            ])
            if trigger['issues']:
                summary_lines.append("  - Issue details:")
                for issue in trigger['issues'][:3]:  # Show max 3
                    summary_lines.append(f"    * [{issue['layer']}] {issue['issue_type']}: {issue['description'][:100]}...")

        summary_text = "\n".join(summary_lines)

        # Save to file
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"iteration_triggers_{self.config.experiment_id}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # Also save JSON format (with complete case info)
        json_file = log_dir / f"iteration_triggers_{self.config.experiment_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": self.config.experiment_id,
                "total_triggers": total_triggers,
                "summary": {
                    "by_type": dict(by_type),
                    "by_question_type": dict(by_question_type),
                    "by_iteration": dict(by_iteration),
                },
                "triggers": self._iteration_triggers
            }, f, ensure_ascii=False, indent=2)

        # [2025-12-30] Save complete case file for paper examples
        cases_file = log_dir / f"iteration_cases_{self.config.experiment_id}.md"
        self._save_iteration_cases_for_paper(cases_file)

        # Print summary to console
        print("\n" + "=" * 80)
        print(f"[Iteration Trigger Log] Total {total_triggers} triggers")
        for trigger_type, count in by_type.most_common():
            print(f"  - {trigger_type}: {count}")
        print(f"Log saved: {log_file}")
        print(f"Complete cases saved: {cases_file}")
        print("=" * 80)

        return str(log_file)

    def _save_iteration_cases_for_paper(self, output_path) -> None:
        """
        [2025-12-30] Save complete iteration trigger cases for paper examples.

        Output format is Markdown, includes:
        - Full material text
        - Question (stem, options/answer points)
        - Explanation
        - Trigger situation
        - Revision reason

        Args:
            output_path: Output file path
        """
        if not self._iteration_triggers:
            return

        lines = [
            "# Iteration Trigger Case Summary (For Paper Examples)",
            "",
            f"Experiment ID: {self.config.experiment_id}",
            f"Generated: {self._get_timestamp()}",
            f"Total cases: {len(self._iteration_triggers)}",
            "",
            "---",
            "",
        ]

        for i, trigger in enumerate(self._iteration_triggers, 1):
            case_details = trigger.get("case_details", {})
            question_content = case_details.get("question", {})

            lines.extend([
                f"## Case {i}: Unit {trigger['unit_id']}",
                "",
                f"**Basic Info**",
                f"- Question type: {trigger['question_type']}",
                f"- Iteration round: {trigger['iteration']}",
                f"- Trigger type: {trigger['trigger_type']}",
                f"- Timestamp: {trigger['timestamp']}",
                "",
            ])

            # Trigger reasons
            lines.append("### 1. Trigger Reasons")
            if trigger.get("trigger_reasons"):
                for reason in trigger["trigger_reasons"]:
                    lines.append(f"- {reason}")
            else:
                lines.append("- Unknown reason")
            lines.append("")

            # Material
            lines.append("### 2. Full Material")
            material = case_details.get("material", "")
            if material:
                lines.append("```")
                lines.append(material)
                lines.append("```")
            else:
                lines.append("(No material info)")
            lines.append("")

            # Question
            lines.append("### 3. Generated Question")
            stem = question_content.get("stem", "")
            if stem:
                lines.append(f"**Stem**: {stem}")
                lines.append("")

                if trigger['question_type'] == "single-choice":
                    options = question_content.get("options", {})
                    correct_answer = question_content.get("correct_answer", "")
                    lines.append("**Options**:")
                    for label in ["A", "B", "C", "D"]:
                        if label in options:
                            opt = options[label]
                            mark = "PASS" if opt.get("is_correct") or label == correct_answer else ""
                            lines.append(f"- {label}. {opt.get('content', '')} {mark}")
                    if correct_answer:
                        lines.append(f"\n**Correct Answer**: {correct_answer}")

                elif trigger['question_type'] == "essay":
                    answer_points = question_content.get("answer_points", [])
                    total_score = question_content.get("total_score", 0)
                    lines.append("**Answer Points**:")
                    for j, ap in enumerate(answer_points, 1):
                        lines.append(f"{j}. {ap.get('point', '')} ({ap.get('score', 0)} pts)")
                    if total_score:
                        lines.append(f"\n**Total Score**: {total_score} pts")
            else:
                lines.append("(No question info)")
            lines.append("")

            # Explanation
            lines.append("### 4. Explanation")
            explanation = case_details.get("explanation", "")
            if explanation:
                lines.append(explanation)
            else:
                lines.append("(No explanation info)")
            lines.append("")

            # Issue details
            lines.append("### 5. Detected Issues")
            issues_full = case_details.get("issues_full", [])
            if issues_full:
                for issue in issues_full:
                    lines.append(f"- **[{issue.get('layer', '')}] {issue.get('issue_type', '')}**")
                    lines.append(f"  {issue.get('description', '')}")
            else:
                lines.append("(No detailed issue info)")
            lines.append("")

            # Revision suggestion
            lines.append("### 6. Revision Suggestion")
            revision_suggestion = case_details.get("revision_suggestion", "")
            if revision_suggestion:
                lines.append(revision_suggestion)
            else:
                lines.append("(No revision suggestion)")

            lines.extend(["", "---", ""])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def get_iteration_triggers(self) -> List[Dict[str, Any]]:
        """Get iteration trigger record list"""
        return self._iteration_triggers.copy()

    def clear_iteration_triggers(self) -> None:
        """Clear iteration trigger records (for multiple runs)"""
        self._iteration_triggers.clear()

    def _save_intermediate_state(self, state: GenerationPipelineState, suffix: str = ""):
        """Save intermediate result.

        Key fix:
        - Save by unit_id directory to avoid overwriting in full/40/60 batch mode
        - Retain suffix, support intermediate state for each iteration/stage
        """
        # Get unit_id from agent1_output (fallback material_id)
        unit_id = "unknown"
        a1 = getattr(state, "agent1_output", None)
        if a1 is not None:
            unit_id = str(getattr(a1, "unit_id", None) or getattr(a1, "material_id", None) or "unknown")

        # Optional: Use current_agent as part of filename for clarity
        agent_tag = str(getattr(state, "current_agent", "") or "state")
        agent_tag = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in agent_tag)

        # Each question in its own directory to completely avoid overwriting
        rel = Path("stage1") / f"unit_{unit_id}" / f"pipeline_state_{state.pipeline_id}_{agent_tag}{suffix}.json"
        output_file = self.config.get_output_path(str(rel))

        state_dict = state.model_dump(mode="json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Intermediate state saved: {output_file}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "GenerationOrchestrator",
]
