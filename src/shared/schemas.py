from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional, Dict, Any, Literal



# ======================
# Common Exceptions
# ======================


class NoMoreUnitsError(Exception):
    """
    Raised when data is exhausted (no more unit_id / materials available for selection).
    """
    pass


# ======================
# Basic Raw Data Structures
# ======================


@dataclass
class RawMaterial:
    """
    Raw material record, corresponding to one entry in raw_material.json.
    """
    material_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    json: Dict[str, Any] = field(default_factory=dict)

MaterialRecord = RawMaterial



@dataclass
class DimensionDefinition:
    """
    A dimension definition entry from ABC_evaluation_prompt.json.

    Data Source:
    Stage 0 static data area -> ABC_evaluation_prompt.json

    Field Description:
    - id: Dimension identifier (e.g., "gk.value")
    - dimension_name: Dimension name (e.g., "Core Value - Patriotism")
    - levelA/B/C: Different level prompts for Stage 1 Agent 2
    - prompt_eval: Pedagogical evaluation prompt for Stage 2

    Position in Pipeline:
    - Stage 1: Agent 2 uses levelA/B/C prompts for dimension prompt synthesis
    - Stage 2: Pedagogical Evaluation uses prompt_eval for dimension assessment
    """
    id: str
    dimension_name: str
    levelA: Dict[str, Any] = field(default_factory=dict)
    levelB: Dict[str, Any] = field(default_factory=dict)
    levelC: Dict[str, Any] = field(default_factory=dict)
    # Evaluation prompt used by Stage 2 pedagogical evaluation
    prompt_eval: str = ""



@dataclass
class QuestionDimensionMapping:
    """
    Dimension mapping info for an authentic question from merged_kaocha_jk_cs.json.
    Fields currently used by Agent1.
    """
    unit_id: str
    material: str
    question_type: str
    type: str = ""  # Genre type: expository/argumentative/mixed

    # exam_skill two-level competency hierarchy
    exam_skill_level1: List[str] = field(default_factory=list)
    exam_skill_level2: List[str] = field(default_factory=list)

    # gk dimensions (Gaokao dimensions)
    gk_value: List[str] = field(default_factory=list)
    gk_subject_literacy: List[str] = field(default_factory=list)
    gk_key_ability: List[str] = field(default_factory=list)
    gk_essential_knowledge: List[str] = field(default_factory=list)
    gk_context: List[str] = field(default_factory=list)
    gk_wings: List[str] = field(default_factory=list)

    # cs dimensions (Curriculum Standard dimensions)
    cs_core_literacy: List[str] = field(default_factory=list)
    cs_task_group: List[str] = field(default_factory=list)
    cs_ability: List[str] = field(default_factory=list)


# ======================
# Agent1 Output: Material + Dimension Selection
# ======================

@dataclass
class MaterialDimSelection:
    """
    Agent1 output: packaged "material + dimension" result for a single pipeline round.

    [2025-12] Refactoring:
    - matched_abc_dims: Matched pedagogical dimensions from ABC_evaluation_prompt.json (list of dimension_name)
    - question_type_label: Question type label (not a pedagogical dimension, but used for question generation)
    - abc_dims: Retained for Agent2 prompt synthesis
    """
    material_id: str
    material_text: str
    question_type: str

    # Complete dimension info (raw data)
    exam_skill: Dict[str, List[str]] = field(default_factory=dict)
    gk_dims: Dict[str, List[str]] = field(default_factory=dict)
    cs_dims: Dict[str, List[str]] = field(default_factory=dict)
    unit_id: str = ""

    # [2025-12] Matched pedagogical dimensions from ABC_evaluation_prompt (dimension_name list)
    matched_abc_dims: List[str] = field(default_factory=list)
    # Backward compatibility alias
    top20_dims: List[str] = field(default_factory=list)

    # [2025-12] Question type label (not a dimension)
    question_type_label: str = ""

    # Matched ABC dimension entries (for Agent2 prompt synthesis)
    abc_dims: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Legacy fields for backward compatibility
    primary_dimension: str = ""
    secondary_dimensions: List[str] = field(default_factory=list)

    # Selection reasoning
    selection_reasoning: str = ""



# ======================
# Agent2 Output: Dimension Prompt Synthesis
# ======================

@dataclass
class SynthesizedPrompt:
    """
    Agent2 output: synthesized question design prompt.
    """
    synthesized_instruction: str
    # [2025-12-28] Deprecated: ability_point no longer used, kept for backward compatibility
    ability_point: str = ""
    # Retained for analysis / ablation study, keeping original A/B/C text
    prompts_by_level: Dict[str, str] = field(default_factory=dict)
    # Actually used prompt level
    used_prompt_level: str = "C"
    # Legacy fields for backward compatibility
    primary_dimension_template: str = ""
    secondary_templates: List[str] = field(default_factory=list)
    dimension_ids: List[str] = field(default_factory=list)
    # [2025-12-29] Revision suggestion from previous Agent4 round (for iteration feedback)
    prompt_revision_suggestion: Optional[str] = None



# ======================
# Agent2 Output: Anchor Set
# ======================

@dataclass
class AnchorSpan:
    """
    A single evidence anchor's position and semantic info in the material.

    [2025-12] Enhancement:
    Added reason_for_anchor and loc fields to store Agent2's anchor semantic analysis,
    for Agent3 reference during question design.
    """

    paragraph_idx: int
    start_char: int
    end_char: int
    snippet: str = ""
    # [2025-12] Anchor semantic info
    reason_for_anchor: str = ""  # Why this anchor suits question design (Agent2 analysis)
    loc: str = ""  # Anchor location description in source text


@dataclass
class AnchorSet:
    """
    Agent2 output: collection of anchors.
    """
    anchors: List[AnchorSpan] = field(default_factory=list)
    # Reasoning for anchor discovery
    anchor_discovery_reasoning: str = ""


# ======================
# Agent3/4 Question and Answer Structures
# ======================

@dataclass
class OptionItem:
    """
    A single multiple-choice option.
    """
    label: str                 # "A" / "B" / "C" / "D"
    content: str               # Option content
    is_correct: bool           # Whether this is the correct option
    error_type: Optional[str] = None  # Error type for distractor
    reasoning: str = ""               # Reasoning for option design (optional)


@dataclass
class AnswerPoint:
    """
    A single answer point for essay questions.
    """
    point: str
    score: int
    # Corresponding evidence anchor indices, e.g., ["A1", "A3"] or [0, 2]
    evidence_reference: Optional[List[Any]] = None



@dataclass
class DistractorBlueprint:
    """
    Distractor design blueprint (placeholder for future extension).
    """
    label: Optional[str] = None
    error_type: Optional[str] = None
    description: str = ""


@dataclass
class GeneratedQuestion:
    """
    Agent3 output: generated question object (single-choice or essay).

    For single-choice:
        - question_type = "single-choice"
        - options is not None, correct_answer is the correct option label

    For essay:
        - question_type = "essay"
        - answer_points / total_score are valid
    """
    stem: str
    question_type: str  # "single-choice" / "essay"

    # Single-choice specific fields
    options: Optional[List[OptionItem]] = None
    correct_answer: Optional[str] = None
    distractor_blueprints: Optional[List[DistractorBlueprint]] = None

    # Essay specific fields
    answer_points: Optional[List[AnswerPoint]] = None
    total_score: Optional[int] = None

    # Common fields
    material_text: str = ""

    explanation: str = ""
    # Record main anchor indices the question relies on (indices in AnchorSet.anchors)
    evidence_anchors: List[int] = field(default_factory=list)
    # Chain-of-thought during question generation (optional)
    generation_reasoning: Optional[str] = None


# ======================
# Agent4 Output: Lightweight Quality Check Results
# ======================

@dataclass
class LightweightQualityScore:
    """
    Agent4's multi-layer quality scores.
    """
    overall_score: float

    layer_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityIssue:
    """
    A single quality issue record.
    """
    layer: str           # "layer1_format" / "layer2_consistency" / "layer3_logic" etc.
    issue_type: str      # Specific issue type, e.g., "empty_stem" / "multi_correct_answers"
    description: str     # Issue description


@dataclass
class VerifiedQuestionSet:
    """
    Agent4's overall quality verification output.
    """
    original_question: GeneratedQuestion

    quality_score: LightweightQualityScore
    issues: List[QualityIssue] = field(default_factory=list)
    need_revision: bool = False
    is_reject: bool = False
    verifier_tags: Dict[str, Any] = field(default_factory=dict)
    verifier_notes: str = ""

    # Attributes that can be appended by later stages (e.g., revision suggestions for Agent2/4)
    prompt_revision_suggestion: Optional[str] = None

    @property
    def question(self) -> GeneratedQuestion:
        """Provides question property for backward compatibility with legacy code."""

        return self.original_question

    @property
    def decision(self) -> str:
        """
        Abstracts need_revision / is_reject into a simple decision label:
            - "reject" / "revise" / "pass"
        """
        if self.is_reject:
            return "reject"
        if self.need_revision:
            return "revise"
        return "pass"

@dataclass
class Stage2CoreInput:
    """
    Stage 2 core input data structure.
    """
    experiment_id: str
    unit_id: str

    # Agent1 info
    material_text: str
    # For stricter typing, can change to Literal["single-choice", "essay"]
    question_type: str

    # Question body (required fields without default values)
    stem: str

    # ------------ Fields with default values below ------------

    explanation: str = ""

    # Agent1's structured dimension info
    gk_dims: Dict[str, List[str]] = field(default_factory=dict)
    cs_dims: Dict[str, List[str]] = field(default_factory=dict)
    exam_skill: Dict[str, List[str]] = field(default_factory=dict)

    # Hit dimension IDs (from Agent2, for Stage 2 pedagogical eval to fetch prompt_eval from ABC table)
    dimension_ids: List[str] = field(default_factory=list)

    # Single-choice / essay additional fields
    options: Optional[List[Dict[str, str]]] = None   # [{"label": "A", "content": "..."}]
    correct_answer: Optional[str] = None
    answer_points: Optional[List[Dict[str, Any]]] = None  # [{"point": "...", "score": 3}, ...]
    total_score: Optional[int] = None

    # [2026-01] Agent2 (anchor finder) output
    # Only experiments using Stage1 without agent2 ablation will have anchor info
    anchors: Optional[List[Dict[str, Any]]] = None  # [{"snippet": "...", "reason_for_anchor": "...", "loc": "..."}]
    anchor_count: int = 0  # Anchor count for statistics



@dataclass
class Stage1Meta:
    """
    Stage 1 metadata, wrapping Agent4 (verifier) results for Stage 2 reference.

    [2025-12-27] Note:
    - Field names keep agent5_* prefix for backward compatibility with serialized data
    - These fields actually store Agent4 (verifier, formerly Agent5) output

    [2025-12] Ablation study support:
    - ablation_skip_agent: Skipped agent name ("none"/"agent2"/"agent4")
    """
    agent5_overall_score: Optional[float] = None
    agent5_layer_scores: Dict[str, float] = field(default_factory=dict)
    agent5_need_revision: Optional[bool] = None
    agent5_is_reject: Optional[bool] = None
    agent5_issue_types: List[str] = field(default_factory=list)
    # [2025-12] Ablation study marker
    ablation_skip_agent: str = "none"


@dataclass
class Stage2Record:
    """
    Packaged record when a question enters Stage 2:
    - core_input: Minimal self-contained input for Stage 2
    - stage1_meta: Stage 1 gatekeeper result metadata
    """
    core_input: Stage2CoreInput
    stage1_meta: Stage1Meta


# ======================
# Pipeline State Objects (for Orchestrator use)
# ======================

class GenerationPipelineState:
    """
    Stage 1 generation pipeline runtime state.

    Design Goals:
      - Fully aligned with GenerationOrchestrator code
      - Supports state.errors.append(...)
      - Supports state.pipeline_id for JSON output
      - Provides model_dump(mode="json") for serialization
    """

    def __init__(self, **kwargs: Any) -> None:
        # Experiment / pipeline ID (for naming output files)
        self.pipeline_id: str = kwargs.pop("pipeline_id", "")

        # Current agent name: "init" / "agent1" / ... / "completed" / "failed"
        self.current_agent: str = kwargs.pop("current_agent", "init")

        # Each agent's output
        self.agent1_output: Optional[MaterialDimSelection] = kwargs.pop("agent1_output", None)
        self.agent2_output: Optional[SynthesizedPrompt] = kwargs.pop("agent2_output", None)
        self.agent3_output: Optional[AnchorSet] = kwargs.pop("agent3_output", None)
        self.agent4_output: Optional[GeneratedQuestion] = kwargs.pop("agent4_output", None)
        self.agent5_output: Optional[VerifiedQuestionSet] = kwargs.pop("agent5_output", None)

        # Success flags for each agent (written by GenerationOrchestrator)
        self.agent1_success: bool = kwargs.pop("agent1_success", False)
        self.agent2_success: bool = kwargs.pop("agent2_success", False)
        self.agent3_success: bool = kwargs.pop("agent3_success", False)
        self.agent4_success: bool = kwargs.pop("agent4_success", False)
        self.agent5_success: bool = kwargs.pop("agent5_success", False)

        # Stage 2 standardized input record
        self.stage2_record: Optional[Stage2Record] = kwargs.pop("stage2_record", None)

        # Error list: GenerationOrchestrator uses state.errors.append(...)
        self.errors: List[str] = kwargs.pop("errors", [])

        # [2026-01] Iteration related fields
        self.iteration_count: int = kwargs.pop("iteration_count", 0)  # Actual iteration count
        self.max_iteration_exceeded: bool = kwargs.pop("max_iteration_exceeded", False)  # Max iteration exceeded
        self.iteration_fail_reason: str = kwargs.pop("iteration_fail_reason", "")  # Iteration failure reason

        # Allow custom fields (for backward compatibility)
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Called by GenerationOrchestrator._save_intermediate_state
    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        """
        Simple serialization method, mimicking pydantic's model_dump interface.

        Currently only supports mode="json".
        """

        if mode != "json":
            raise ValueError("GenerationPipelineState.model_dump currently only supports mode='json'")

        def _to_serializable(obj: Any) -> Any:
            # dataclass -> dict (recursive expansion)
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            # Process list/dict recursively
            if isinstance(obj, list):
                return [_to_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            # Return other primitive types directly
            return obj

        return {k: _to_serializable(v) for k, v in self.__dict__.items()}



class EvaluationPipelineState:
    """
    Stage 2 evaluation pipeline runtime state.

    Pipeline Mapping:
    According to agent_pro_flowchart.png, Stage 2 contains two independent evaluation systems:
    1. AI-centric Evaluation - Multi-model ensemble scoring
    2. Pedagogical Evaluation - Dimension-based prompt_eval assessment

    Field Description:
    - ai_eval_result: AI-centric evaluation result
    - pedagogical_eval_result: Pedagogical evaluation result
    - ai_eval_success / pedagogical_eval_success: Success flags for each evaluation system
    - ai_eval_model_names / ped_eval_model_names: Model lists used by each system
    - final_decision: Final decision combining both systems

    [2025-12] Architecture Refactoring:
    - Always list (actual models when enabled, empty list when disabled)
    - CLI stats only depend on overall_score from ai_eval_result / pedagogical_eval_result
    """

    def __init__(self, **kwargs: Any) -> None:
        self.pipeline_id: str = kwargs.pop("pipeline_id", "")
        self.current_stage: str = kwargs.pop("current_stage", "init")

        # Stage 2 dual independent evaluation system outputs (refactored per flowchart)
        # AI-centric evaluation result (multi-model ensemble scoring)
        self.ai_eval_result: Any = kwargs.pop("ai_eval_result", None)
        self.ai_eval_success: bool = kwargs.pop("ai_eval_success", False)

        # Pedagogical evaluation result (based on dimension prompt_eval)
        self.pedagogical_eval_result: Any = kwargs.pop("pedagogical_eval_result", None)
        self.pedagogical_eval_success: bool = kwargs.pop("pedagogical_eval_success", False)

        # [2026-01] Independent GK/CS pedagogical evaluation results
        # Removed gk+cs compound mode, gk and cs evaluated independently
        self.gk_eval_result: Any = kwargs.pop("gk_eval_result", None)
        self.gk_eval_success: bool = kwargs.pop("gk_eval_success", False)
        self.cs_eval_result: Any = kwargs.pop("cs_eval_result", None)
        self.cs_eval_success: bool = kwargs.pop("cs_eval_success", False)

        # Legacy fields (eval1/eval2_1/eval2_2), to be deprecated
        self.eval1_result: Any = kwargs.pop("eval1_result", None)
        self.eval1_success: bool = kwargs.pop("eval1_success", False)
        self.eval2_1_result: Any = kwargs.pop("eval2_1_result", None)
        self.eval2_1_success: bool = kwargs.pop("eval2_1_success", False)
        self.eval2_2_result: Any = kwargs.pop("eval2_2_result", None)
        self.eval2_2_success: bool = kwargs.pop("eval2_2_success", False)

        # Also use list for errors, convenient for append
        self.errors: List[str] = kwargs.pop("errors", [])

        # Final decision label (e.g., "pass" / "reject")
        self.final_decision: Optional[str] = kwargs.pop("final_decision", None)

        # [2025-12] Evaluation model triplet (for tracing)
        self.eval_models: List[str] = kwargs.pop("eval_models", [])

        # [2025-12] Model lists for each evaluation module (always list, empty [] when disabled)
        self.ai_eval_model_names: List[str] = kwargs.pop("ai_eval_model_names", [])
        self.ped_eval_model_names: List[str] = kwargs.pop("ped_eval_model_names", [])

        # [2025-12] Skipped modules record
        self.skipped_modules: List[str] = kwargs.pop("skipped_modules", [])
        self.notes: str = kwargs.pop("notes", "")

        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, mode: str = "json") -> Dict[str, Any]:
        if mode != "json":
            raise ValueError("EvaluationPipelineState.model_dump currently only supports mode='json'")

        def _to_serializable(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, list):
                return [_to_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            return obj

        return {k: _to_serializable(v) for k, v in self.__dict__.items()}

# ======================
# Helper Factory Functions
# ======================

def create_initial_generation_state(experiment_id: str) -> GenerationPipelineState:
    """
    Factory function to create initial state for GenerationOrchestrator.

    Args:
        experiment_id: Experiment ID, usually equals config.experiment_id

    Returns:
        GenerationPipelineState: Initialized pipeline state object
    """
    return GenerationPipelineState(
        pipeline_id=experiment_id,
        current_agent="init",
        agent1_output=None,
        agent2_output=None,
        agent3_output=None,
        agent4_output=None,
        agent5_output=None,
        errors=[],
        agent1_success=False,
        agent2_success=False,
        agent3_success=False,
        agent4_success=False,
        agent5_success=False,
    )


def create_initial_evaluation_state(experiment_id: str) -> EvaluationPipelineState:
    """
    Factory function to create initial state for EvaluationOrchestrator.

    Pipeline Mapping:
    Stage 2 evaluation initialization, preparing to run AI-centric and Pedagogical evaluation systems.

    Args:
        experiment_id: Experiment ID

    Returns:
        EvaluationPipelineState: Initialized evaluation pipeline state object
    """
    return EvaluationPipelineState(
        pipeline_id=experiment_id,
        current_stage="init",
        ai_eval_result=None,
        ai_eval_success=False,
        pedagogical_eval_result=None,
        pedagogical_eval_success=False,
        errors=[],
        final_decision=None,
    )


# ======================
# Stage 2 Evaluation Result Data Structures
# ======================

@dataclass
class ModelScorePair:
    """
    Single model's scoring result.

    Pipeline Mapping:
    Stage 2 -> AI-centric Evaluation -> Single model output in multi-model scoring

    Field Description:
    - model_name: Model name (e.g., "gpt-4", "claude-3")
    - score: Score given by this model
    - reasoning: Scoring reasoning
    """
    model_name: str
    score: float
    reasoning: str = ""


@dataclass
class AICentricEvalResult:
    """
    AI-centric evaluation result.

    Pipeline Mapping:
    Stage 2 -> AI-centric Evaluation

    Design Note:
    - Supports multi-model ensemble scoring: calls multiple LLMs to evaluate the same question
    - Uses configurable weights to aggregate model results for final AI dimension score
    - Full score: 100

    Field Description:
    - question_id: Question identifier
    - model_scores: List of model scoring results
    - dimension_scores: Scores for each AI evaluation dimension
    - overall_score: Overall score (0-100)
    - pass_threshold: Pass threshold
    - decision: Evaluation decision ("pass" / "reject" / "skip")
    - ai_notes: AI evaluation notes
    """
    question_id: str
    prompt_level: str = "ai_eval"
    model_scores: List[ModelScorePair] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    pass_threshold: float = 70.0
    decision: str = "pending"
    ai_notes: str = ""


@dataclass
class PedagogicalScoreBreakdown:
    """
    Pedagogical evaluation dimension score breakdown.

    Pipeline Mapping:
    Stage 2 -> Pedagogical Evaluation -> Single dimension evaluation result

    Field Description:
    - dimension_name: Dimension name (from ABC_evaluation_prompt.json)
    - hit_level: Hit level ("High hit"/"Basic hit"/"Partial"/"Deviated")
    - score: Dimension score (0-100)
    - reasoning: Evaluation reasoning
    """
    dimension_name: str = ""
    hit_level: str = ""  # High hit / Basic hit / Partial / Deviated
    score: float = 0.0
    reasoning: str = ""
    # Legacy fields for backward compatibility
    bloom_level: str = ""
    bloom_score: float = 0.0
    reliability_score: float = 0.0
    validity_score: float = 0.0
    discrimination_score: float = 0.0
    notes: str = ""


@dataclass
class PedagogicalDimensionEvalResult:
    """
    Single pedagogical dimension evaluation result.

    Pipeline Mapping:
    Stage 2 -> Pedagogical Evaluation -> Based on prompt_eval from ABC_evaluation_prompt.json

    Design Note:
    Based on dimensions hit in Stage 1 (linked via unit_id),
    reads corresponding prompt_eval from ABC_evaluation_prompt.json,
    and calls LLM for pedagogical dimension evaluation.

    Field Description:
    - dimension_id: Dimension ID (e.g., "gk.value")
    - dimension_name: Dimension name (e.g., "Core Value - Patriotism")
    - prompt_eval_used: Evaluation prompt used
    - hit_level: Hit level
    - score: Dimension score (0-100)
    - reasoning: Evaluation reasoning from LLM
    """
    dimension_id: str
    dimension_name: str
    prompt_eval_used: str = ""
    hit_level: str = ""
    score: float = 0.0
    reasoning: str = ""


@dataclass
class PedagogicalEvalResult:
    """
    Pedagogical evaluation result.

    Pipeline Mapping:
    Stage 2 -> Pedagogical Evaluation

    Design Note:
    - Reads prompt_eval from ABC_evaluation_prompt.json based on dimensions hit in Stage 1
    - Calls LLM evaluation for each hit dimension
    - Aggregates dimension scores for final pedagogical score
    - Full score: 100

    Field Description:
    - question_id: Question identifier
    - dimension_results: Dimension evaluation results (aggregated)
    - overall_score: Overall score (0-100)
    - pass_threshold: Pass threshold
    - decision: Evaluation decision ("pass" / "reject" / "skip")
    - pedagogical_notes: Pedagogical evaluation notes
    - model_results: Independent scoring per model per dimension
    - audit: Evaluation audit info (call counts, model status, etc.)

    Legacy Fields:
    Retained for backward compatibility
    """
    question_id: str
    dimension_results: List[PedagogicalDimensionEvalResult] = field(default_factory=list)
    overall_score: float = 0.0
    pass_threshold: float = 70.0
    decision: str = "pending"
    pedagogical_notes: str = ""
    # [2025-12] Independent scoring per model per dimension
    # Format: {model_name: {dimension_name: {"hit_level": str, "reason": str, "score": float}}}
    model_results: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # [2025-12] Evaluation audit info
    audit: Dict[str, Any] = field(default_factory=dict)
    # Legacy fields for backward compatibility
    bloom_level: str = ""
    bloom_score: float = 0.0
    reliability_score: float = 0.0
    validity_score: float = 0.0
    discrimination_score: float = 0.0
    difficulty_match_score: float = 0.0
    # [2025-12] New hit-based evaluation result
    hit_based_result: Optional["PedagogicalHitBasedResult"] = None


@dataclass
class PedagogicalHitBasedResult:
    """
    [2025-12] New pedagogical evaluation result - hit/not-hit based evaluation

    Uses STAGE2_GK_DIMENSIONS (17, GK01-GK17) or STAGE2_CS_DIMENSIONS (21, CS01-CS21) for evaluation,
    each dimension returns hit: true/false + reason, then calculates Precision/Recall/F1 scores.

    Field Description:
    - question_id: Question identifier
    - gold_dimensions: Gold standard dimensions (Stage1 hit dimension names)
    - predicted_dimensions: Predicted hit dimensions (LLM judged hit=true)
    - missing_dimensions: Missing dimensions (gold - predicted, False Negative)
    - extra_dimensions: Extra dimensions (predicted - gold, False Positive)
    - dimension_results: Dimension evaluation details {dim_name: {"hit": bool, "reason": str}}
    - tp/fp/fn: True Positive / False Positive / False Negative counts
    - precision/recall/f1: Precision/Recall/F1 scores
    - model_results: Independent evaluation results per model
    """
    question_id: str = ""
    gold_dimensions: List[str] = field(default_factory=list)
    predicted_dimensions: List[str] = field(default_factory=list)
    # [2025-12] Explicitly save missing and extra dimensions for analysis
    missing_dimensions: List[str] = field(default_factory=list)  # FN: gold - predicted
    extra_dimensions: List[str] = field(default_factory=list)    # FP: predicted - gold
    dimension_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # TP/FP/FN statistics
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # Evaluation metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # [2026-01] Off-target (false positive rate)
    off_target: float = 0.0  # 1 - precision = FP / (TP + FP)
    # Independent results per model
    model_results: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # Audit info
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PedagogicalRoundAggregation:
    """
    [2025-12] Aggregated results for one evaluation round

    Field Description:
    - round_id: Round identifier
    - question_results: List of question evaluation results
    - micro_precision/recall/f1: Micro average (global TP/FP/FN)
    - macro_precision/recall/f1: Macro average (mean of per-question metrics)
    - total_tp/fp/fn: Global TP/FP/FN statistics
    """
    round_id: str = ""
    question_results: List[PedagogicalHitBasedResult] = field(default_factory=list)
    # Micro average
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    # Macro average
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    # Global statistics
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    # Hit statistics per dimension
    dimension_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # [2025-12-31] Skipped questions without dimensions (empty gold dimensions)
    skipped_no_dims: int = 0
    # [2026-01] metrics_rare_only: metrics for rare dimensions only (excluding high-frequency)
    metrics_rare_only: Optional[Dict[str, Any]] = None  # {"excluded_dims": [...], "micro/macro_precision": ..., ...}
    # [2026-01] Off-target and Success@k metrics
    off_target: float = 0.0  # 1 - precision, False positive rate in predictions
    success_at_k: float = 0.0  # Proportion of samples with Recall >= k (default k=0.8)
    success_threshold: float = 0.8  # [2026-01] Threshold changed from 0.6 to 0.8
    # [2026-01] Per-model evaluation bias analysis
    # Format: {"model_name": {"avg_precision": 0.78, "avg_recall": 0.65, "avg_f1": 0.71, "bias": "Precision-biased (conservative)"}, ...}
    model_bias_analysis: Optional[Dict[str, Dict[str, Any]]] = None
