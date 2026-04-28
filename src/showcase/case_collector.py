# -*- coding: utf-8 -*-
"""
[2026-01 New] Case Data Collection Module

Collects complete Case data from logs and state files for showcase display.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentInteraction:
    """Single Agent-LLM interaction"""
    agent_name: str
    prompt: str
    response: str
    model: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Complete record of a single iteration"""
    iteration: int
    agent2_interaction: Optional[AgentInteraction] = None  # Anchor discovery
    agent3_interaction: Optional[AgentInteraction] = None  # Question generation
    agent3_repair_interaction: Optional[AgentInteraction] = None  # Repair (if any)
    agent4_interaction: Optional[AgentInteraction] = None  # Quality check
    agent4_decision: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2EvalRecord:
    """Stage2 evaluation record"""
    eval_prompts: List[Dict[str, str]] = field(default_factory=list)  # Prompts for each model
    eval_responses: List[Dict[str, str]] = field(default_factory=list)  # Responses from each model
    dimension_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Results per dimension
    model_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Independent results per model
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FullCaseRecord:
    """Complete record for a single case"""
    unit_id: str
    case_type: str  # "good" / "bad"
    fail_reason: Optional[str] = None

    # Stage1 info
    material_text: str = ""
    question_type: str = ""
    target_dimensions: List[str] = field(default_factory=list)  # Gold dimensions
    predicted_dimensions: List[str] = field(default_factory=list)  # Predicted dimensions

    # Iteration records
    iterations: List[IterationRecord] = field(default_factory=list)
    total_iterations: int = 0
    max_iteration_exceeded: bool = False

    # Final generated result
    final_question: Optional[Dict[str, Any]] = None

    # Stage2 evaluation
    stage2_eval: Optional[Stage2EvalRecord] = None

    # ABC prompts
    abc_prompts_used: Dict[str, Dict[str, str]] = field(default_factory=dict)
    prompt_level: str = "C"

    # Metrics
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


class CaseCollector:
    """Collects complete Case data from logs and state files"""

    def __init__(self, output_dir: Path, experiment_id: str):
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.prompt_logs_dir = self.output_dir / "logs" / "prompts"
        self.stage2_dir = self.output_dir / "stage2"

        # Cache ABC prompt data
        self._abc_prompt_cache: Optional[List[Dict]] = None

    def collect_full_case(
        self,
        unit_id: str,
        case_type: str,
        basic_info: Optional[Dict[str, Any]] = None,
    ) -> FullCaseRecord:
        """
        Collect complete data for a single unit

        Args:
            unit_id: Question ID
            case_type: "good" or "bad"
            basic_info: Basic info (from collect_good_bad_cases)
        """
        basic_info = basic_info or {}

        # 1. Load all Agent logs for this unit
        agent_logs = self._load_agent_logs_for_unit(unit_id)

        # 2. Load generation_state.json
        gen_state = self._load_generation_state(unit_id)

        # 3. Load evaluation_state.json
        eval_state = self._load_evaluation_state(unit_id)

        # 4. Extract basic info
        material_text = ""
        question_type = ""
        target_dimensions = []
        predicted_dimensions = []
        prompt_level = "C"

        if eval_state:
            input_data = eval_state.get("input") or eval_state.get("input_core") or {}
            material_text = input_data.get("material_text", "")
            question_type = input_data.get("question_type", "")
            target_dimensions = input_data.get("dimension_ids", [])

            # Extract from pedagogical evaluation results
            ped_eval = eval_state.get("pedagogical_eval", {})
            if ped_eval.get("result"):
                ped_result = ped_eval["result"]
                target_dimensions = ped_result.get("gold_dimensions", target_dimensions)
                predicted_dimensions = ped_result.get("predicted_dimensions", [])

        if gen_state:
            prompt_level = gen_state.get("prompt_level", "C")

        # 5. Extract ABC prompts
        abc_prompts = self._extract_abc_prompts(target_dimensions)

        # 6. Build iteration records
        iterations = self._build_iteration_records(agent_logs, gen_state)

        # 7. Extract final question
        final_question = self._extract_final_question(gen_state, eval_state)

        # 8. Extract Stage2 evaluation record
        stage2_eval = self._extract_stage2_eval(eval_state, agent_logs)

        # 9. Extract metrics
        metrics = basic_info.get("ped_metrics") or basic_info.get("gk_metrics") or {}
        f1 = metrics.get("f1", basic_info.get("f1", 0.0))
        precision = metrics.get("precision", basic_info.get("precision", 0.0))
        recall = metrics.get("recall", basic_info.get("recall", 0.0))

        return FullCaseRecord(
            unit_id=unit_id,
            case_type=case_type,
            fail_reason=basic_info.get("fail_reason") or basic_info.get("iteration_fail_reason"),
            material_text=material_text,
            question_type=question_type,
            target_dimensions=target_dimensions,
            predicted_dimensions=predicted_dimensions,
            iterations=iterations,
            total_iterations=len(iterations),
            max_iteration_exceeded=basic_info.get("max_iteration_exceeded", False),
            final_question=final_question,
            stage2_eval=stage2_eval,
            abc_prompts_used=abc_prompts,
            prompt_level=prompt_level,
            f1=f1 or 0.0,
            precision=precision or 0.0,
            recall=recall or 0.0,
        )

    def _load_agent_logs_for_unit(self, unit_id: str) -> Dict[str, List[Dict]]:
        """Load all Agent interactions for specified unit from JSONL logs"""
        result: Dict[str, List[Dict]] = {}

        if not self.prompt_logs_dir.exists():
            return result

        # Iterate all JSONL files
        for jsonl_file in self.prompt_logs_dir.glob("*.jsonl"):
            agent_name = jsonl_file.stem  # Agent2, Agent3_SingleChoice, etc.
            logs_for_unit = []

            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            metadata = record.get("metadata", {})
                            # Check if belongs to this unit
                            record_unit_id = (
                                metadata.get("unit_id") or
                                metadata.get("material_id") or
                                ""
                            )
                            # Compatible with both "unit_42" and "42" formats
                            record_unit_id = str(record_unit_id).replace("unit_", "")
                            if record_unit_id == str(unit_id):
                                logs_for_unit.append(record)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

            if logs_for_unit:
                result[agent_name] = logs_for_unit

        return result

    def _load_generation_state(self, unit_id: str) -> Optional[Dict]:
        """Load generation_state.json"""
        # Try multiple paths
        possible_paths = [
            self.stage2_dir / f"unit_{unit_id}" / "generation_state.json",
            self.output_dir / "stage1" / f"unit_{unit_id}" / "generation_state.json",
        ]

        # Also try to find pipeline_state_*.json
        unit_dir = self.stage2_dir / f"unit_{unit_id}"
        if unit_dir.exists():
            for f in unit_dir.glob("pipeline_state*.json"):
                possible_paths.insert(0, f)

        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    continue

        return None

    def _load_evaluation_state(self, unit_id: str) -> Optional[Dict]:
        """Load evaluation_state.json"""
        path = self.stage2_dir / f"unit_{unit_id}" / "evaluation_state.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _extract_abc_prompts(self, dimension_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Extract A/B/C level prompts for corresponding dimensions from ABC_prompt.json"""
        if not dimension_ids:
            return {}

        # Load ABC prompt data (with cache)
        if self._abc_prompt_cache is None:
            self._abc_prompt_cache = self._load_abc_prompt_data()

        if not self._abc_prompt_cache:
            return {}

        result = {}
        for dim_entry in self._abc_prompt_cache:
            dim_name = dim_entry.get("dimension_name", "")
            dim_code = dim_entry.get("code", "")
            dim_id = dim_entry.get("id", "")

            # Check if matches
            if dim_name in dimension_ids or dim_code in dimension_ids or dim_id in dimension_ids:
                result[dim_name] = {
                    "code": dim_code,
                    "levelA": dim_entry.get("levelA", {}).get("prompt", ""),
                    "levelB": dim_entry.get("levelB", {}).get("addon", ""),
                    "levelC": dim_entry.get("levelC", {}).get("addon", ""),
                }

        return result

    def _load_abc_prompt_data(self) -> List[Dict]:
        """Load ABC_prompt.json"""
        # Try multiple paths
        possible_paths = [
            Path("data/ABC_prompt.json"),
            self.output_dir.parent.parent / "data" / "ABC_prompt.json",
            Path(__file__).parent.parent.parent / "data" / "ABC_prompt.json",
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    continue

        return []

    def _build_iteration_records(
        self,
        agent_logs: Dict[str, List[Dict]],
        gen_state: Optional[Dict],
    ) -> List[IterationRecord]:
        """Build iteration records"""
        if not agent_logs:
            return []

        # Collect all logs and sort by timestamp
        all_logs = []
        for agent_name, logs in agent_logs.items():
            for log in logs:
                log["_agent_name"] = agent_name
                all_logs.append(log)

        all_logs.sort(key=lambda x: x.get("timestamp", ""))

        # Split iterations by Agent4 occurrence
        iterations: List[IterationRecord] = []
        current_iter = IterationRecord(iteration=1)

        for log in all_logs:
            agent_name = log["_agent_name"]
            interaction = AgentInteraction(
                agent_name=agent_name,
                prompt=log.get("prompt", ""),
                response=log.get("response", ""),
                model=log.get("model", ""),
                timestamp=log.get("timestamp", ""),
                metadata=log.get("metadata", {}),
            )

            if agent_name == "Agent2":
                current_iter.agent2_interaction = interaction
            elif agent_name.startswith("Agent3") and "Repair" not in agent_name:
                current_iter.agent3_interaction = interaction
            elif "Agent3" in agent_name and "Repair" in agent_name:
                current_iter.agent3_repair_interaction = interaction
            elif agent_name == "Agent4_Solver":
                current_iter.agent4_interaction = interaction
                # Try to parse Agent4's decision
                try:
                    resp_data = json.loads(log.get("response", "{}"))
                    current_iter.agent4_decision = {
                        "verdict": resp_data.get("verdict"),
                        "need_revision": resp_data.get("need_revision", False),
                        "confidence": resp_data.get("confidence"),
                        "revision_suggestion": resp_data.get("revision_suggestion", ""),
                    }
                except json.JSONDecodeError:
                    pass

                # Agent4 completion marks end of iteration
                iterations.append(current_iter)
                current_iter = IterationRecord(iteration=len(iterations) + 1)

        # Add last iteration even without Agent4 (may be failure case)
        if current_iter.agent2_interaction or current_iter.agent3_interaction:
            iterations.append(current_iter)

        return iterations

    def _extract_final_question(
        self,
        gen_state: Optional[Dict],
        eval_state: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        """Extract final generated question"""
        # Prefer getting from evaluation_state
        if eval_state:
            input_data = eval_state.get("input") or eval_state.get("input_core") or {}
            if input_data.get("stem"):
                return {
                    "stem": input_data.get("stem", ""),
                    "options": input_data.get("options", []),
                    "correct_answer": input_data.get("correct_answer", ""),
                    "answer_points": input_data.get("answer_points", []),
                    "explanation": input_data.get("explanation", ""),
                }

        # Get from generation_state
        if gen_state:
            agent4_output = gen_state.get("agent4_output") or gen_state.get("agent3_output")
            if agent4_output:
                if isinstance(agent4_output, dict):
                    return agent4_output.get("generated_question") or agent4_output

        return None

    def _extract_stage2_eval(
        self,
        eval_state: Optional[Dict],
        agent_logs: Dict[str, List[Dict]],
    ) -> Optional[Stage2EvalRecord]:
        """Extract Stage2 evaluation record"""
        if not eval_state:
            return None

        record = Stage2EvalRecord()

        # Extract results from evaluation_state
        ped_eval = eval_state.get("pedagogical_eval", {})
        if ped_eval.get("result"):
            ped_result = ped_eval["result"]
            record.dimension_results = ped_result.get("dimension_results", {})
            record.model_results = ped_result.get("model_results", {})
            record.metrics = {
                "f1": ped_result.get("f1", 0),
                "precision": ped_result.get("precision", 0),
                "recall": ped_result.get("recall", 0),
                "tp": ped_result.get("tp", 0),
                "fp": ped_result.get("fp", 0),
                "fn": ped_result.get("fn", 0),
            }

        # Extract evaluation prompt/response from Agent logs
        eval_logs = agent_logs.get("PedagogicalHitBasedEval", [])
        for log in eval_logs:
            record.eval_prompts.append({
                "model": log.get("model", ""),
                "prompt": log.get("prompt", ""),
            })
            record.eval_responses.append({
                "model": log.get("model", ""),
                "response": log.get("response", ""),
            })

        return record
