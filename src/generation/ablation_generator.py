# -*- coding: utf-8 -*-
"""
Stage1 Ablation Generator - Baseline Generation without Dimension Prompts

Ablation Study Purpose:
Validate the value of Stage1 multi-Agent pipeline (dimension matching, prompt synthesis).
This module skips all dimension-related processing, directly using raw materials to generate questions.

Pipeline:
1. Read raw materials from raw_material.json
2. Get question type info from merged_kaocha_jk_cs.json (single-choice/essay)
3. Use basic prompts (no dimension guidance) for LLM generation
4. Output GeneratedQuestion object, compatible with Stage2 evaluation

Differences from Normal Stage1:
- No Agent1 dimension matching
- No Agent2 anchor discovery
- No dimension prompts from ABC_prompt.json
- Directly uses generic question generation prompts
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

from src.shared.schemas import (
    GeneratedQuestion,
    OptionItem,
    AnswerPoint,
)
from src.shared.llm_interface import LLMClient
from src.shared.llm_json import extract_json_candidate, repair_common_json
from src.shared.api_config import STAGE1_TEMPERATURE

logger = logging.getLogger(__name__)


# ============================================================================
# Ablation Prompt Templates (No Dimension Guidance)
# ============================================================================

# Chinese LLM prompt - kept for domain-specific Gaokao question generation
ABLATION_SINGLE_CHOICE_PROMPT = """你是一位高考语文命题专家。请根据以下材料，命制一道高考语文阅读理解选择题。

【材料】
{material}

【要求】
1. 根据材料设计一个题干和四个选项（A、B、C、D）
2. 只有一个答案，其余三个为干扰项
3. 干扰项应具有一定迷惑性，但必须能从材料中判断出错误
4. 提供详细的解析说明为什么答案是对的，以及每个干扰项错在哪里

【输出格式】
请以 JSON 格式输出，结构如下：
```json
{{
    "stem": "题干内容",
    "options": [
        {{"label": "A", "content": "选项A内容", "is_correct": false}},
        {{"label": "B", "content": "选项B内容", "is_correct": true}},
        {{"label": "C", "content": "选项C内容", "is_correct": false}},
        {{"label": "D", "content": "选项D内容", "is_correct": false}}
    ],
    "correct_answer": "B",
    "explanation": "解析内容，说明答案的依据和干扰项的错误原因"
}}
```

请直接输出 JSON，不要有其他内容。"""

# Chinese LLM prompt - kept for domain-specific Gaokao question generation
ABLATION_ESSAY_PROMPT = """你是一位高考语文命题专家。请根据以下材料，命制一道高考语文阅读理解简答题。

【材料】
{material}

【要求】
1. 根据材料设计一个明确的问题，符合简答题的特征
2. 提供标准答案要点和评分标准
3. 总分设置为 6 分，分为 2-3 个答案要点

【输出格式】
请以 JSON 格式输出，结构如下：
```json
{{
    "stem": "题干内容（问题）",
    "answer_points": [
        {{"point": "答案要点1", "score": 2}},
        {{"point": "答案要点2", "score": 2}},
        {{"point": "答案要点3", "score": 2}}
    ],
    "total_score": 6,
    "explanation": "解析内容，说明答案要点的依据和评分标准"
}}
```

请直接输出 JSON，不要有其他内容。"""


# ============================================================================
# Ablation Generator Class
# ============================================================================

@dataclass
class AblationGeneratorConfig:
    """
    Ablation generator configuration
    """
    temperature: float = STAGE1_TEMPERATURE
    max_tokens: int = 4096
    max_retries: int = 3


class AblationGenerator:
    """
    Stage1 Ablation Generator

    Directly generates questions from raw materials without any dimension prompts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[AblationGeneratorConfig] = None,
    ):
        self.llm_client = llm_client
        self.config = config or AblationGeneratorConfig()

    def generate(
        self,
        material: str,
        question_type: str,
        unit_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedQuestion:
        """
        Generate a question

        Args:
            material: Raw material text
            question_type: Question type ("选择题" or "简答题")
            unit_id: Unit ID
            metadata: Additional metadata

        Returns:
            GeneratedQuestion object
        """
        # Determine internal question type
        internal_qtype = "single-choice" if question_type == "选择题" else "essay"

        # Select corresponding prompt template
        if internal_qtype == "single-choice":
            prompt = ABLATION_SINGLE_CHOICE_PROMPT.format(material=material)
        else:
            prompt = ABLATION_ESSAY_PROMPT.format(material=material)

        print(f"[AblationGenerator] 开始生成 unit_id={unit_id}, 题型={question_type}")

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        question = self._parse_response(response, internal_qtype, material)

        print(f"[AblationGenerator] 生成完成: stem 长度={len(question.stem)}")

        return question

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate response."""
        messages = [{"role": "user", "content": prompt}]

        gen_kwargs = {
            "max_tokens": self.config.max_tokens,
            "metadata": {"agent": "AblationGenerator"},
        }

        # Add temperature parameter
        gen_kwargs["temperature"] = self.config.temperature

        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.generate(messages, **gen_kwargs)
                return response
            except Exception as e:
                logger.warning(f"[AblationGenerator] LLM 调用失败 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise

        return ""

    def _parse_response(
        self,
        response: str,
        question_type: str,
        material: str,
    ) -> GeneratedQuestion:
        """Parse LLM response into GeneratedQuestion object."""

        # Extract JSON
        json_str = extract_json_candidate(response)
        if not json_str:
            json_str = response

        # Try to fix common JSON issues
        json_str = repair_common_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[AblationGenerator] JSON 解析失败: {e}")
            # Return an empty question object
            return self._create_fallback_question(question_type, material, str(e))

        # Build GeneratedQuestion
        if question_type == "single-choice":
            return self._build_single_choice(data, material)
        else:
            return self._build_essay(data, material)

    def _build_single_choice(self, data: Dict, material: str) -> GeneratedQuestion:
        """Build single-choice question."""
        options = []
        for opt in data.get("options", []):
            options.append(OptionItem(
                label=opt.get("label", ""),
                content=opt.get("content", ""),
                is_correct=opt.get("is_correct", False),
            ))

        return GeneratedQuestion(
            stem=data.get("stem", ""),
            question_type="single-choice",
            options=options,
            correct_answer=data.get("correct_answer", ""),
            explanation=data.get("explanation", ""),
            material_text=material,
        )

    def _build_essay(self, data: Dict, material: str) -> GeneratedQuestion:
        """Build essay question."""
        answer_points = []
        for pt in data.get("answer_points", []):
            answer_points.append(AnswerPoint(
                point=pt.get("point", ""),
                score=pt.get("score", 0),
            ))

        return GeneratedQuestion(
            stem=data.get("stem", ""),
            question_type="essay",
            answer_points=answer_points,
            total_score=data.get("total_score", 6),
            explanation=data.get("explanation", ""),
            material_text=material,
        )

    def _create_fallback_question(
        self,
        question_type: str,
        material: str,
        error_msg: str,
    ) -> GeneratedQuestion:
        """Create fallback question object on failure."""
        if question_type == "single-choice":
            return GeneratedQuestion(
                stem=f"[生成失败] {error_msg}",
                question_type="single-choice",
                options=[
                    OptionItem(label="A", content="选项A", is_correct=False),
                    OptionItem(label="B", content="选项B", is_correct=True),
                    OptionItem(label="C", content="选项C", is_correct=False),
                    OptionItem(label="D", content="选项D", is_correct=False),
                ],
                correct_answer="B",
                explanation=f"生成失败: {error_msg}",
                material_text=material,
            )
        else:
            return GeneratedQuestion(
                stem=f"[生成失败] {error_msg}",
                question_type="essay",
                answer_points=[AnswerPoint(point="答案要点", score=6)],
                total_score=6,
                explanation=f"生成失败: {error_msg}",
                material_text=material,
            )


# ============================================================================
# Ablation Pipeline Orchestration
# ============================================================================

class AblationOrchestrator:
    """
    Ablation Experiment Orchestrator

    Manages ablation generation pipeline, including data loading, generation, and result saving.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: Path,
        config: Optional[AblationGeneratorConfig] = None,
    ):
        self.generator = AblationGenerator(llm_client, config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.raw_materials = self._load_raw_materials()
        self.question_types = self._load_question_types()

    def _load_raw_materials(self) -> Dict[str, Dict]:
        """Load raw materials."""
        path = Path("data/raw_material.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to unit_id -> material data mapping
        return {str(item["unit_id"]): item for item in data}

    def _load_question_types(self) -> Dict[str, str]:
        """Load question type information."""
        path = Path("data/merged_kaocha_jk_cs.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to unit_id -> question_type mapping
        return {str(item["unit_id"]): item.get("question_type", "选择题") for item in data}

    def run_single(self, unit_id: str) -> Dict[str, Any]:
        """
        Run ablation generation for a single unit.

        Returns:
            Dictionary containing generation results
        """
        unit_id = str(unit_id)

        # Get raw material
        raw = self.raw_materials.get(unit_id)
        if not raw:
            raise ValueError(f"未找到 unit_id={unit_id} 的原始材料")

        material = raw.get("material", "")
        question_type = self.question_types.get(unit_id, "选择题")

        # Generate question
        question = self.generator.generate(
            material=material,
            question_type=question_type,
            unit_id=unit_id,
            metadata={"source": raw.get("source", "")},
        )

        # Build result
        result = {
            "unit_id": unit_id,
            "source": raw.get("source", ""),
            "original_stem": raw.get("stem", ""),
            "original_answer": raw.get("answer", ""),
            "question_type": question_type,
            "generated_question": asdict(question),
            "material": material,
        }

        # Save result
        self._save_result(unit_id, result)

        return result

    def run_batch(self, unit_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Batch run ablation generation.

        Args:
            unit_ids: List of unit_ids to process

        Returns:
            List of all generation results
        """
        results = []
        total = len(unit_ids)

        for i, unit_id in enumerate(unit_ids, 1):
            print(f"\n>>> [{i}/{total}] 正在处理 unit_id={unit_id}...")
            try:
                result = self.run_single(unit_id)
                results.append(result)
                print(f"    完成: 题型={result['question_type']}")
            except Exception as e:
                logger.error(f"[AblationOrchestrator] unit_id={unit_id} 处理失败: {e}")
                results.append({
                    "unit_id": unit_id,
                    "error": str(e),
                })

        return results

    def _save_result(self, unit_id: str, result: Dict):
        """Save single result to file. | 保存单个结果"""
        stage1_dir = self.output_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        output_path = stage1_dir / f"unit_{unit_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
