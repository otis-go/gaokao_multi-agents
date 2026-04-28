# src/evaluation/__init__.py
# Stage 2: Evaluation Module Entry Point

"""
Stage 2 Evaluation System Entry Point.

Architecture Description:
According to agent_pro flowchart, Stage 2 contains two independent evaluation systems:
1. AI-centric Evaluation (AICentricEval) - Multi-model ensemble scoring;
2. Pedagogical Evaluation (PedagogicalHitBasedEval) - Hit/not-hit evaluation based on ABC_evaluation_prompt.json.

2025-12 Refactoring:
- PedagogicalEval replaced with PedagogicalHitBasedEval
- Uses pedagogical dimensions from ABC_evaluation_prompt.json for evaluation
- LLM returns hit: true/false + reason
- Calculates Precision/Recall/F1 scores

Exported Classes:
- EvaluationOrchestrator: Evaluation orchestrator, coordinates both evaluation systems;
- AICentricEval: AI-centric evaluation module;
- PedagogicalHitBasedEval: Pedagogical evaluation module (new version);
- Stage2Record / Stage2CoreInput: Stage 2 standardized input schema;
- EvaluationPipelineState: Stage 2 evaluation pipeline state object.
"""

# Main orchestrator
from src.evaluation.evaluation_orchestrator import (
    EvaluationOrchestrator,
    create_evaluation_orchestrator,
)

# Two independent evaluation systems
from src.evaluation.ai_centric_eval import AICentricEval
from src.evaluation.pedagogical_eval import PedagogicalHitBasedEval

# Stage 2 input & state schema (for convenient upper-level usage)
from src.shared.schemas import Stage2Record, Stage2CoreInput, EvaluationPipelineState


__all__ = [
    # Main orchestrator
    "EvaluationOrchestrator",
    "create_evaluation_orchestrator",
    # Independent evaluation systems
    "AICentricEval",
    "PedagogicalHitBasedEval",
    # Stage 2 schema / state
    "Stage2Record",
    "Stage2CoreInput",
    "EvaluationPipelineState",
]
