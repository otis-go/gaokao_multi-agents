# src/evaluation/baseline_evaluator.py
# Real Exam Baseline Evaluator - Skip Stage1, directly evaluate with real exams in Stage2

"""
[Module Description]
BaselineEvaluator directly uses real exam data (raw_material.json) to construct Stage2CoreInput,
skipping Stage1 generation, directly feeding into Stage2 for evaluation to get baseline scores.

Purpose:
1. Establish real exam scoring baseline
2. Compare quality difference between generated and real questions
3. Quick validation of evaluation module

Data Sources:
- raw_material.json: Real exam data (stem, material, answer, analysis, etc.)
- merged_kaocha_jk_cs.json: Dimension annotation data (gk.*, cs.*, exam_skill, etc.)
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.shared.schemas import (
    Stage2CoreInput,
    Stage2Record,
    Stage1Meta,
)


@dataclass
class BaselineQuestion:
    """Real exam data structure."""
    unit_id: str
    question_type: str  # "single-choice" | "essay"
    material_type: str  # "说明文"(expository) | "议论文"(argumentative)
    stem: str
    material_text: str
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    options: Optional[List[Dict[str, str]]] = None
    answer_points: Optional[List[Dict[str, Any]]] = None
    total_score: Optional[int] = None

    # Dimension information
    gk_dims: Dict[str, List[str]] = field(default_factory=dict)
    cs_dims: Dict[str, List[str]] = field(default_factory=dict)
    exam_skill: Dict[str, List[str]] = field(default_factory=dict)
    dimension_ids: List[str] = field(default_factory=list)

    # Raw data
    source: str = ""


# =============================================================================
# [2025-12 Added] National/Local Exam Detection Logic
# =============================================================================

# National exam keywords (containing these keywords means national exam)
NATIONAL_EXAM_KEYWORDS = [
    "新课标",     # New Curriculum Standard I/II/III
    "全国",       # National A/B/I/II
    "新高考",     # New Gaokao I/II
]

# Local exam keywords (containing these keywords means local exam)
LOCAL_EXAM_KEYWORDS = [
    "北京", "天津", "上海", "重庆",           # Municipalities
    "山东", "江苏", "浙江", "广东", "福建",   # Eastern coastal provinces
    "湖南", "湖北", "河南", "河北",           # Central provinces
    "四川", "云南", "贵州", "陕西", "甘肃",   # Western provinces
    "辽宁", "吉林", "黑龙江",                 # Northeastern provinces
    "安徽", "江西", "山西", "海南",           # Other provinces
]


def is_national_exam(source: str) -> bool:
    """
    Determine if the source is a national exam.

    Rules:
    1. If contains local keywords (Beijing, Tianjin, Shandong, etc.), then it's local exam
    2. If contains national exam keywords (New Curriculum Standard, National, New Gaokao), then it's national exam
    3. Default to national exam for other cases

    Args:
        source: Source string, e.g., "2024·New Gaokao I", "2025·Beijing"

    Returns:
        True for national exam, False for local exam
    """
    if not source:
        return True  # Default to national exam when no source info

    # Check local keywords first (more specific)
    for kw in LOCAL_EXAM_KEYWORDS:
        if kw in source:
            return False

    # Then check national exam keywords
    for kw in NATIONAL_EXAM_KEYWORDS:
        if kw in source:
            return True

    # Default to national exam
    return True


def get_exam_type_label(source: str) -> str:
    """
    Get exam type label.

    Args:
        source: Source string

    Returns:
        "全国卷" or "地方卷"
    """
    return "全国卷" if is_national_exam(source) else "地方卷"


def _normalize_question_type(qt: str) -> str:
    """Normalize question type to single-choice or essay"""
    qt_lower = (qt or "").lower()
    if any(kw in qt_lower for kw in ["选择", "单选", "多选", "choice"]):
        return "single-choice"
    return "essay"


def _parse_stem_options(stem: str, question_type: str) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    """
    Parse pure stem and option list from stem.
    Real exam's stem may include options, like:
    "Which of the following is correct? (  )
    A. ...
    B. ...
    C. ...
    D. ..."
    """
    if question_type != "single-choice":
        return stem, None

    # [2026-01-07] re already imported at module top

    # Try to match options
    lines = stem.strip().split('\n')
    options = []
    stem_lines = []
    in_options = False

    for line in lines:
        line = line.strip()
        # Match A. / A、/ A． formats
        match = re.match(r'^([A-D])[.、．\s]\s*(.+)$', line)
        if match:
            in_options = True
            label = match.group(1)
            content = match.group(2).strip()
            options.append({"label": label, "content": content})
        elif not in_options:
            stem_lines.append(line)
        elif line:  # Option continuation line
            if options:
                options[-1]["content"] += " " + line

    if len(options) >= 2:
        return '\n'.join(stem_lines).strip(), options
    return stem, None


def _parse_essay_answer(answer_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[int]]:
    """
    [2025-12 Added] Parse answer points from essay question's answer text.

    Real exam answers usually have these formats:
    1. Example 1/Example 2 + question + answer
    2. Pure text answer
    3. Bullet point answers (1. xxx 2. xxx)

    Returns: (answer_points, total_score)
    """
    # [2026-01-07] re already imported at module top

    if not answer_text or not answer_text.strip():
        return None, None

    answer_text = answer_text.strip()
    answer_points = []

    # Try to parse "Example 1/2" format
    # Pattern: Example 1/2/一/二 followed by question and answer
    example_pattern = r'示例[一二三四五六七八九十\d]+[：:]\s*'
    if re.search(example_pattern, answer_text):
        # Split by examples
        parts = re.split(example_pattern, answer_text)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Extract question and answer
            lines = part.split('\n')
            point_text = part[:500] if len(part) > 500 else part
            answer_points.append({
                "point": point_text,
                "score": None,  # Real exams usually don't provide scores
                "evidence_reference": "",
            })
        if answer_points:
            return answer_points, None

    # Try to parse bullet point format (1. xxx or ① xxx)
    point_pattern = r'(?:^|\n)\s*(?:[\d①②③④⑤⑥⑦⑧⑨⑩]+[.、．）\)]\s*|[（\(][\d一二三四五六七八九十]+[）\)]\s*)'
    if re.search(point_pattern, answer_text):
        parts = re.split(point_pattern, answer_text)
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Filter too short fragments
                answer_points.append({
                    "point": part[:500] if len(part) > 500 else part,
                    "score": None,
                    "evidence_reference": "",
                })
        if answer_points:
            return answer_points, None

    # If cannot parse, treat entire answer as one point
    answer_points.append({
        "point": answer_text[:1000] if len(answer_text) > 1000 else answer_text,
        "score": None,
        "evidence_reference": "",
    })
    return answer_points, None


class BaselineEvaluator:
    """
    Real Exam Baseline Evaluator

    Functions:
    1. Load real exam data and dimension data
    2. Construct Stage2Record (skip Stage1)
    3. Call EvaluationOrchestrator for evaluation
    4. Collect scoring results and statistics

    [2026-01 Refactoring] Baseline mode definition:
    - Input dimensions (for model): Random dimensions from merged_mix_dimension_jk_cs.json
    - Gold dimensions (for metrics): Real exam dimensions from merged_kaocha_jk_cs.json
    """

    def __init__(
        self,
        raw_material_path: Optional[Path] = None,
        dimension_mapping_path: Optional[Path] = None,
        gold_dimension_path: Optional[Path] = None,
        abc_prompt_path: Optional[Path] = None,
        use_random_dims: bool = False,
        baseline_mode: bool = False,
    ):
        """
        Initialize Real Exam Baseline Evaluator.

        Args:
            raw_material_path: Real exam data path
            dimension_mapping_path: Input dimension mapping data path (dimensions for model)
            gold_dimension_path: Gold dimension data path (for pedagogical metrics)
            abc_prompt_path: ABC prompt data path
            use_random_dims: Whether to use random dimension file (only effective when dimension_mapping_path not specified)
                - False: Use original merged_kaocha_jk_cs.json (default)
                - True: Use random dimensions merged_mix_dimension_jk_cs.json
            baseline_mode: [2026-01 Added] Baseline mode
                - True: Input uses random dimensions, gold uses real exam dimensions
                - False: Input and gold use same dimension source (default)
        """
        project_root = Path(__file__).resolve().parents[2]
        self.raw_material_path = raw_material_path or (project_root / "data" / "raw_material.json")

        # [2026-01 Refactoring] Baseline mode: input uses random dimensions, gold uses real exam dimensions
        self.baseline_mode = baseline_mode
        self.use_random_dims = use_random_dims

        if baseline_mode:
            # Baseline mode: input dimensions=random, gold dimensions=real exam
            self.dimension_mapping_path = dimension_mapping_path or (project_root / "data" / "merged_mix_dimension_jk_cs.json")
            self.gold_dimension_path = gold_dimension_path or (project_root / "data" / "merged_kaocha_jk_cs.json")
        elif dimension_mapping_path:
            # Explicitly specified input dimension path
            self.dimension_mapping_path = dimension_mapping_path
            self.gold_dimension_path = gold_dimension_path or dimension_mapping_path
        elif use_random_dims:
            # Use random dimensions (input and gold are same)
            self.dimension_mapping_path = project_root / "data" / "merged_mix_dimension_jk_cs.json"
            self.gold_dimension_path = gold_dimension_path or self.dimension_mapping_path
        else:
            # Default: use real exam dimensions (input and gold are same)
            self.dimension_mapping_path = project_root / "data" / "merged_kaocha_jk_cs.json"
            self.gold_dimension_path = gold_dimension_path or self.dimension_mapping_path

        self.abc_prompt_path = abc_prompt_path or (project_root / "data" / "ABC_prompt.json")

        self._raw_materials: Dict[str, Dict] = {}
        self._dimension_mappings: Dict[str, Dict] = {}  # Input dimensions
        self._gold_dimension_mappings: Dict[str, Dict] = {}  # Gold dimensions
        self._abc_prompts: List[Dict] = []

    def _load_data(self) -> None:
        """Load all necessary data files"""
        # Load real exam data
        with open(self.raw_material_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            self._raw_materials = {str(item["unit_id"]): item for item in raw_list}

        # Load input dimension mapping data (dimensions for model)
        with open(self.dimension_mapping_path, "r", encoding="utf-8") as f:
            dim_list = json.load(f)
            self._dimension_mappings = {str(item["unit_id"]): item for item in dim_list}

        # [2026-01 Added] Load Gold dimension mapping data (for pedagogical metrics)
        if self.gold_dimension_path != self.dimension_mapping_path:
            with open(self.gold_dimension_path, "r", encoding="utf-8") as f:
                gold_list = json.load(f)
                self._gold_dimension_mappings = {str(item["unit_id"]): item for item in gold_list}
        else:
            self._gold_dimension_mappings = self._dimension_mappings

        # Load ABC prompt data (to get dimension_ids)
        if self.abc_prompt_path.exists():
            with open(self.abc_prompt_path, "r", encoding="utf-8") as f:
                self._abc_prompts = json.load(f)

    def _build_baseline_question(self, unit_id: str) -> Optional[BaselineQuestion]:
        """Build BaselineQuestion from real exam data and dimension data"""
        unit_id = str(unit_id)

        raw = self._raw_materials.get(unit_id)
        dim = self._dimension_mappings.get(unit_id)

        if not raw:
            print(f"[BaselineEvaluator] Real exam data not found for unit_id={unit_id}")
            return None
        if not dim:
            print(f"[BaselineEvaluator] Dimension data not found for unit_id={unit_id}")
            return None

        # Determine question type
        qt_raw = dim.get("question_type") or raw.get("question_type", "")
        question_type = _normalize_question_type(qt_raw)

        # Determine material type
        material_type = dim.get("type", raw.get("type", "未知"))

        # Parse stem and options
        stem_raw = raw.get("stem", "")
        stem, options = _parse_stem_options(stem_raw, question_type)

        # [2025-12 Added] Parse answer based on question type
        answer_points = None
        total_score = None
        correct_answer = None

        if question_type == "essay":
            # Essay question: parse answer points
            answer_text = raw.get("answer", "")
            answer_points, total_score = _parse_essay_answer(answer_text)
        else:
            # Choice question: directly use answer as correct_answer
            correct_answer = raw.get("answer")

        # Build dimension information
        gk_dims = {}
        cs_dims = {}
        exam_skill = {}

        for key in ["gk.value", "gk.subject_literacy", "gk.key_ability",
                    "gk.essential_knowledge", "gk.wings", "gk.context"]:
            if key in dim and dim[key]:
                gk_dims[key] = dim[key] if isinstance(dim[key], list) else [dim[key]]

        for key in ["cs.core_literacy", "cs.task_group", "cs.ability"]:
            if key in dim and dim[key]:
                cs_dims[key] = dim[key] if isinstance(dim[key], list) else [dim[key]]

        if dim.get("exam_skill_level1"):
            exam_skill["level1"] = dim["exam_skill_level1"] if isinstance(dim["exam_skill_level1"], list) else [dim["exam_skill_level1"]]
        if dim.get("exam_skill_level2"):
            exam_skill["level2"] = dim["exam_skill_level2"] if isinstance(dim["exam_skill_level2"], list) else [dim["exam_skill_level2"]]

        # Match dimension_ids (from ABC prompt)
        dimension_ids = self._match_dimension_ids(gk_dims, cs_dims, exam_skill)

        return BaselineQuestion(
            unit_id=unit_id,
            question_type=question_type,
            material_type=material_type,
            stem=stem,
            material_text=raw.get("material", ""),
            correct_answer=correct_answer,
            explanation=raw.get("analysis", ""),
            options=options,
            answer_points=answer_points,
            total_score=total_score,
            gk_dims=gk_dims,
            cs_dims=cs_dims,
            exam_skill=exam_skill,
            dimension_ids=dimension_ids,
            source=raw.get("source", ""),
        )

    def _match_dimension_ids(
        self,
        gk_dims: Dict[str, List[str]],
        cs_dims: Dict[str, List[str]],
        exam_skill: Dict[str, List[str]],
    ) -> List[str]:
        """
        Match hit dimension_ids

        Return format: ["dimension_name", ...] for Stage2 evaluation
        Example: ["基础要求-基础性", "学科素养-理解能力"]

        PedagogicalEval uses dimension_name as unique identifier.
        This is because the same id (like gk.wings) in ABC_evaluation_prompt.json may correspond to multiple
        dimension_names (like 基础要求-应用性, 基础要求-基础性, 基础要求-综合性).
        Need to match to specific dimension_name based on specific labels in real exam data.
        """
        if not self._abc_prompts:
            return []

        matched_names = []

        # Build mapping from (id, label keyword) to dimension_name
        # Real exam data: gk.wings: ['基础性'] -> need to match to "基础要求-基础性"
        def match_label_to_dim_name(dim_id: str, label: str) -> Optional[str]:
            """Find complete dimension_name based on id and label keyword"""
            for dim_def in self._abc_prompts:
                if dim_def.get("id") == dim_id:
                    dim_name = dim_def.get("dimension_name", "")
                    # Label may be part of dimension_name
                    # E.g., "基础性" matches "基础要求-基础性"
                    if label in dim_name:
                        return dim_name
            return None

        # Process gk dimensions
        for dim_key, labels in gk_dims.items():
            for label in labels:
                dim_name = match_label_to_dim_name(dim_key, label)
                if dim_name and dim_name not in matched_names:
                    matched_names.append(dim_name)

        # Process cs dimensions
        for dim_key, labels in cs_dims.items():
            for label in labels:
                dim_name = match_label_to_dim_name(dim_key, label)
                if dim_name and dim_name not in matched_names:
                    matched_names.append(dim_name)

        # Process exam_skill (choice/essay type dimensions)
        # exam_skill format may be {"level1": [...], "level2": [...]}
        # [2025-12 Fix] Only use level2 labels for precise matching
        # level1 is category info (like "信息筛选类主观题"), should not be used for matching fine-grained dimensions
        # level2 is specific dimension labels (like "具体信息筛选题"), should match precisely to dimension_name
        level2_labels = exam_skill.get("level2", []) or exam_skill.get("exam_skill_level2", [])
        for label in level2_labels:
            # exam_skill labels directly search all ABC prompts
            for dim_def in self._abc_prompts:
                dim_name = dim_def.get("dimension_name", "")
                # Match: label is contained in dimension_name
                if label in dim_name:
                    if dim_name not in matched_names:
                        matched_names.append(dim_name)
                    break

        return matched_names

    def build_stage2_record(
        self,
        unit_id: str,
        experiment_id: str,
        dim_mode: str = "gk",  # [2026-01 Refactoring] Removed gk+cs mode, default gk
    ) -> Optional[Stage2Record]:
        """
        Build Stage2Record for specified unit_id (skip Stage1)

        Args:
            unit_id: Question ID
            experiment_id: Experiment ID
            dim_mode: Dimension mode - "gk_only" / "cs_only" / "gk" / "cs"

        Returns:
            Stage2Record, returns None (skip) if the question has no corresponding type of dimensions
        """
        if not self._raw_materials:
            self._load_data()

        bq = self._build_baseline_question(unit_id)
        if not bq:
            return None

        # Filter dimensions based on dim_mode
        gk_dims = bq.gk_dims.copy() if bq.gk_dims else {}
        cs_dims = bq.cs_dims.copy() if bq.cs_dims else {}

        if dim_mode == "gk_only" or dim_mode == "gk":
            cs_dims = {}  # Clear cs dimensions
            # [2025-12-31 Fix] If no gk dimensions in gk mode, still return (allow evaluator to handle empty dimensions)
            # Previously would skip, causing ablation-nodim mode unable to run evaluation
            if not any(gk_dims.values()):
                print(f"[BaselineEvaluator] unit_id={unit_id} has no GK dimensions, but still return empty dimension record for evaluation")
        elif dim_mode == "cs_only" or dim_mode == "cs":
            gk_dims = {}  # Clear gk dimensions
            # [2025-12-31 Fix] If no cs dimensions in cs mode, still return (allow evaluator to handle empty dimensions)
            if not any(cs_dims.values()):
                print(f"[BaselineEvaluator] unit_id={unit_id} has no CS dimensions, but still return empty dimension record for evaluation")
        # [2026-01 Refactoring] Removed gk+cs mode, no longer have "keep all dimensions" branch

        # [2025-12 Fix] Recalculate dimension_ids based on filtered dimensions
        # Need to re-match after filtering by dim_mode
        dimension_ids = self._match_dimension_ids(gk_dims, cs_dims, bq.exam_skill)

        # Build Stage2CoreInput
        core_input = Stage2CoreInput(
            experiment_id=experiment_id,
            unit_id=bq.unit_id,
            material_text=bq.material_text,
            question_type=bq.question_type,
            stem=bq.stem,
            explanation=bq.explanation or "",
            gk_dims=gk_dims,
            cs_dims=cs_dims,
            exam_skill=bq.exam_skill,
            dimension_ids=dimension_ids,
            options=bq.options,
            correct_answer=bq.correct_answer,
            answer_points=bq.answer_points,
            total_score=bq.total_score,
        )

        # Build Stage1Meta (mark as baseline evaluation)
        # Stage1Meta field explanation:
        # - agent5_overall_score: Agent5 overall score
        # - agent5_need_revision: Whether revision is needed
        # - agent5_is_reject: Whether rejected
        # - ablation_skip_agent: Agent skipped in ablation experiment
        stage1_meta = Stage1Meta(
            agent5_overall_score=1.0,  # Real exam default full score
            agent5_need_revision=False,
            agent5_is_reject=False,
            ablation_skip_agent="none",  # Don't skip any Agent (although here actually skipped all Stage1)
        )

        return Stage2Record(
            core_input=core_input,
            stage1_meta=stage1_meta,
        )

    def get_all_unit_ids(self) -> List[str]:
        """Get all available unit_id list"""
        if not self._raw_materials:
            self._load_data()
        return list(self._raw_materials.keys())

    def get_recent_years_unit_ids(self, start_year: int = 2020, end_year: int = 2025) -> List[str]:
        """
        [2025-12 Added] Get unit_id list for recent years' real exams

        Filter questions within specified year range based on source field.
        Source field format examples:
        - "2025·National I"
        - "2020·Shandong"
        - "2023年New Gaokao II"

        Args:
            start_year: Start year (inclusive), default 2020
            end_year: End year (inclusive), default 2025

        Returns:
            unit_id list matching year range
        """
        # [2026-01-07] re already imported at module top

        if not self._raw_materials:
            self._load_data()

        recent_unit_ids = []
        year_distribution = {}  # For counting questions per year

        for unit_id, raw_data in self._raw_materials.items():
            source = raw_data.get("source", "")
            # Extract year from source (assume year is at the front, format like "2020·xxx" or "2020年xxx")
            year_match = re.match(r'^(\d{4})', source)
            if year_match:
                year = int(year_match.group(1))
                if start_year <= year <= end_year:
                    recent_unit_ids.append(unit_id)
                    # Count year distribution
                    year_distribution[year] = year_distribution.get(year, 0) + 1

        # Print year distribution info
        if year_distribution:
            print(f"\n[BaselineEvaluator] Recent {end_year - start_year + 1} years real exam filter results ({start_year}-{end_year}):")
            for year in sorted(year_distribution.keys()):
                print(f"  Year {year}: {year_distribution[year]} questions")
            print(f"  Total: {len(recent_unit_ids)} questions\n")

        return recent_unit_ids

    def get_unit_ids_by_year_range(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> List[str]:
        """
        [2025-12 Added] Get unit_id list for real exams within specified year range

        Supports three modes:
        1. Specify both start and end year: start_year=2021, end_year=2025
        2. Only specify end year (before certain year): end_year=2020 means 2020 and before
        3. Only specify start year (after certain year): start_year=2021 means 2021 and after

        Args:
            start_year: Start year (inclusive), None means no limit
            end_year: End year (inclusive), None means no limit

        Returns:
            unit_id list matching year range
        """
        # [2026-01-07] re already imported at module top

        if not self._raw_materials:
            self._load_data()

        matched_unit_ids = []
        year_distribution = {}
        unknown_year_count = 0

        for unit_id, raw_data in self._raw_materials.items():
            source = raw_data.get("source", "")
            year_match = re.match(r'^(\d{4})', source)

            if year_match:
                year = int(year_match.group(1))
                # Check if within range
                in_range = True
                if start_year is not None and year < start_year:
                    in_range = False
                if end_year is not None and year > end_year:
                    in_range = False

                if in_range:
                    matched_unit_ids.append(unit_id)
                    year_distribution[year] = year_distribution.get(year, 0) + 1
            else:
                unknown_year_count += 1

        # Print year distribution info
        if year_distribution or unknown_year_count > 0:
            range_str = ""
            if start_year is not None and end_year is not None:
                range_str = f"{start_year}-{end_year}"
            elif start_year is not None:
                range_str = f"Year {start_year} and after"
            elif end_year is not None:
                range_str = f"Year {end_year} and before"
            else:
                range_str = "All years"

            print(f"\n[BaselineEvaluator] Year range filter results ({range_str}):")
            for year in sorted(year_distribution.keys()):
                print(f"  Year {year}: {year_distribution[year]} questions")
            if unknown_year_count > 0:
                print(f"  (Unknown year: {unknown_year_count} questions, excluded)")
            print(f"  Total: {len(matched_unit_ids)} questions\n")

        return matched_unit_ids

    def get_unit_ids_by_exam_type(
        self,
        exam_type: str = "all",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[str]:
        """
        [2025-12 Added] Filter real exam unit_id list by exam type

        Args:
            exam_type: Exam type
                - "national": National exams only (New Curriculum Standard, National A/B/I/II, New Gaokao)
                - "local": Local exams only (Beijing, Tianjin, Shandong, Jiangsu, Zhejiang, etc.)
                - "all": All questions (default)
            start_year: Start year (inclusive), None means no limit
            end_year: End year (inclusive), None means no limit

        Returns:
            unit_id list matching conditions
        """
        # [2026-01-07] re already imported at module top

        if not self._raw_materials:
            self._load_data()

        matched_unit_ids = []
        national_count = 0
        local_count = 0
        source_distribution: Dict[str, int] = {}  # Source distribution statistics

        for unit_id, raw_data in self._raw_materials.items():
            source = raw_data.get("source", "")

            # Year filter
            if start_year is not None or end_year is not None:
                year_match = re.match(r'^(\d{4})', source)
                if year_match:
                    year = int(year_match.group(1))
                    if start_year is not None and year < start_year:
                        continue
                    if end_year is not None and year > end_year:
                        continue
                else:
                    continue  # Skip questions with unparseable year

            # Exam type filter
            is_national = is_national_exam(source)

            if exam_type == "national" and not is_national:
                continue
            if exam_type == "local" and is_national:
                continue

            matched_unit_ids.append(unit_id)

            # Statistics
            if is_national:
                national_count += 1
            else:
                local_count += 1
            source_distribution[source] = source_distribution.get(source, 0) + 1

        # Print filter results
        exam_type_label = {
            "national": "全国卷",
            "local": "地方卷",
            "all": "全部"
        }.get(exam_type, exam_type)

        year_range_str = ""
        if start_year is not None and end_year is not None:
            year_range_str = f", Year range: {start_year}-{end_year}"
        elif start_year is not None:
            year_range_str = f", Year {start_year} and after"
        elif end_year is not None:
            year_range_str = f", Year {end_year} and before"

        print(f"\n[BaselineEvaluator] Exam type filter results ({exam_type_label}{year_range_str}):")
        if exam_type == "all":
            print(f"  National exams: {national_count} questions")
            print(f"  Local exams: {local_count} questions")
        print(f"  Total: {len(matched_unit_ids)} questions")

        # Print source distribution (sorted by count, only show top 10)
        sorted_sources = sorted(source_distribution.items(), key=lambda x: -x[1])
        if len(sorted_sources) > 0:
            print("  Source distribution (top 10):")
            for src, cnt in sorted_sources[:10]:
                exam_label = "全国卷" if is_national_exam(src) else "地方卷"
                print(f"    - {src} ({exam_label}): {cnt} questions")
            if len(sorted_sources) > 10:
                print(f"    ... and {len(sorted_sources) - 10} other sources")
        print()

        return matched_unit_ids

    def get_baseline_question(self, unit_id: str) -> Optional[BaselineQuestion]:
        """Get real exam information for specified unit_id"""
        if not self._raw_materials:
            self._load_data()
        return self._build_baseline_question(unit_id)

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get statistical summary of real exam data"""
        if not self._raw_materials:
            self._load_data()

        total = len(self._raw_materials)

        # Question type statistics
        question_type_counts = {"single-choice": 0, "essay": 0}
        material_type_counts = {"说明文": 0, "议论文": 0, "其他": 0}

        for unit_id in self._raw_materials:
            bq = self._build_baseline_question(unit_id)
            if bq:
                qt = bq.question_type
                if qt in question_type_counts:
                    question_type_counts[qt] += 1

                mt = bq.material_type
                if mt in material_type_counts:
                    material_type_counts[mt] += 1
                else:
                    material_type_counts["其他"] += 1

        return {
            "total_questions": total,
            "question_type_distribution": question_type_counts,
            "material_type_distribution": material_type_counts,
        }

    def get_gold_dimensions(self, unit_id: str, dim_mode: str = "gk") -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        [2026-01 Added] Get Gold dimension information for specified unit_id

        For baseline mode: input dimensions use random ones, but use real exam gold dimensions for pedagogical metrics.

        Args:
            unit_id: Question ID
            dim_mode: Dimension mode - "gk" / "cs"

        Returns:
            (dimension_ids, gk_dims, cs_dims) tuple
        """
        if not self._gold_dimension_mappings:
            self._load_data()

        unit_id = str(unit_id)
        gold_dim = self._gold_dimension_mappings.get(unit_id, {})

        if not gold_dim:
            return [], {}, {}

        # Build dimension information
        gk_dims = {}
        cs_dims = {}
        exam_skill = {}

        for key in ["gk.value", "gk.subject_literacy", "gk.key_ability",
                    "gk.essential_knowledge", "gk.wings", "gk.context"]:
            if key in gold_dim and gold_dim[key]:
                gk_dims[key] = gold_dim[key] if isinstance(gold_dim[key], list) else [gold_dim[key]]

        for key in ["cs.core_literacy", "cs.task_group", "cs.ability"]:
            if key in gold_dim and gold_dim[key]:
                cs_dims[key] = gold_dim[key] if isinstance(gold_dim[key], list) else [gold_dim[key]]

        if gold_dim.get("exam_skill_level1"):
            exam_skill["level1"] = gold_dim["exam_skill_level1"] if isinstance(gold_dim["exam_skill_level1"], list) else [gold_dim["exam_skill_level1"]]
        if gold_dim.get("exam_skill_level2"):
            exam_skill["level2"] = gold_dim["exam_skill_level2"] if isinstance(gold_dim["exam_skill_level2"], list) else [gold_dim["exam_skill_level2"]]

        # Filter based on dim_mode
        if dim_mode == "gk_only" or dim_mode == "gk":
            cs_dims = {}
        elif dim_mode == "cs_only" or dim_mode == "cs":
            gk_dims = {}

        # Match dimension_ids
        dimension_ids = self._match_dimension_ids(gk_dims, cs_dims, exam_skill)

        return dimension_ids, gk_dims, cs_dims

    def is_baseline_mode(self) -> bool:
        """[2026-01 Added] Check if in baseline mode (input random dimensions + gold real exam dimensions)"""
        return self.baseline_mode


__all__ = ["BaselineEvaluator", "BaselineQuestion"]
