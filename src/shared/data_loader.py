from __future__ import annotations

"""
src/shared/data_loader.py

Responsible for loading from data/ directory:
- raw_material.json            -> RawMaterial list
- merged_kaocha_jk_cs.json     -> QuestionDimensionMapping list
- ABC_evaluation_prompt.json   -> DimensionDefinition list

Agent1MaterialDimSelector accesses data only through this class, without direct file operations.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from src.shared.schemas import (
    RawMaterial,
    QuestionDimensionMapping,
    DimensionDefinition,
)


# ABC Prompt dimension definitions (for Stage1 question design and pedagogical evaluation)
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ABCDimensionDefinition:
    """
    Dimension definition from ABC_evaluation_prompt.json.

    [2025-12] Used for Stage1 question design prompts and pedagogical evaluation
    - id: Dimension category ID (e.g., "gk.value", "gk.wings", "cs.ability")
    - dimension_name: Dimension name (e.g., "Core Value - Patriotism")
    - levelA/B/C: Prompt levels for question design
    - prompt_eval: Evaluation criteria for pedagogical assessment (deprecated)
    """
    id: str
    dimension_name: str
    levelA: Dict[str, Any] = field(default_factory=dict)
    levelB: Dict[str, Any] = field(default_factory=dict)
    levelC: Dict[str, Any] = field(default_factory=dict)
    prompt_eval: str = ""


@dataclass
class QuestionTypeDefinition:
    """
    Question type definition from ABC_evaluation_prompt.json.

    [2025-12] Question types are not pedagogical dimensions, but used for Stage1 question design
    - id: Question type category ID (e.g., "Multiple-choice types", "Essay types")
    - dimension_name: Type name (e.g., "Distractor setting type - temporal")
    - levelA/B/C: Prompt levels for question design
    - prompt_eval: Evaluation criteria (deprecated)
    """
    id: str
    dimension_name: str
    levelA: Dict[str, Any] = field(default_factory=dict)
    levelB: Dict[str, Any] = field(default_factory=dict)
    levelC: Dict[str, Any] = field(default_factory=dict)
    prompt_eval: str = ""


@dataclass
class ABCPromptData:
    """
    Complete data structure for ABC_evaluation_prompt.json.

    - dimensions: Pedagogical dimensions (GK 17 + CS 21, total 38)
    - question_types: Question type definitions (multiple-choice/essay related)
    - dimension_id_to_names: Mapping from id to dimension names
    - name_to_definition: Mapping from dimension name to definition object
    """
    dimensions: List[ABCDimensionDefinition] = field(default_factory=list)
    question_types: List[QuestionTypeDefinition] = field(default_factory=list)
    dimension_id_to_names: Dict[str, List[str]] = field(default_factory=dict)
    name_to_definition: Dict[str, ABCDimensionDefinition] = field(default_factory=dict)


# Legacy compatibility aliases (to be removed in future versions)
Top20DimensionDefinition = ABCDimensionDefinition
Top20Data = ABCPromptData


class DataLoader:
    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        """
        Args:
            data_dir: Data directory; if None, defaults to <project_root>/data.
        """
        if data_dir is None:
            # Current file is at src/shared/data_loader.py
            # parents[0] -> shared
            # parents[1] -> src
            # parents[2] -> project_root
            project_root = Path(__file__).resolve().parents[2]
            data_dir = project_root / "data"

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            print(f"[DataLoader] [Warning] Data directory not found: {self.data_dir}")

        self._materials: Optional[List[RawMaterial]] = None
        self._dimension_mappings: Optional[List[QuestionDimensionMapping]] = None
        self._random_dimension_mappings: Optional[List[QuestionDimensionMapping]] = None
        self._hardmix_dimension_mappings: Optional[List[QuestionDimensionMapping]] = None  # [2026-01-14] Hard negative control cache
        self._low_freq_random_mappings: Optional[List[QuestionDimensionMapping]] = None  # [2026-01-05] Added
        self._low_freq_count: int = 3  # [2026-01-06] Low-freq dimension count (default k=3)
        self._cached_low_freq_count: Optional[int] = None  # Cached k value
        self._dimensions: Optional[List[DimensionDefinition]] = None
        self._abc_prompt: Optional[ABCPromptData] = None

    def set_low_freq_count(self, count: int) -> None:
        """
        [2026-01-06] Set low-frequency dimension count (k=1/3/5)

        Args:
            count: Number of low-freq dimensions per question, recommended: 1, 3, 5
        """
        if count != self._low_freq_count:
            self._low_freq_count = count
            # Clear cache, use new k value on next load
            self._low_freq_random_mappings = None
            self._cached_low_freq_count = None
            print(f"[DataLoader] Low-freq dimension count updated to k={count}")

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------
    def load_materials(self) -> List[RawMaterial]:
        """
        Read raw_material.json and convert to RawMaterial list.
        """
        if self._materials is not None:
            return self._materials

        path = self.data_dir / "raw_material.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        materials: List[RawMaterial] = []
        for item in data:
            # unit_id is int in JSON, convert to str for alignment with mapping files
            material_id = str(item.get("unit_id") or item.get("material_id") or "")
            content = item.get("material") or item.get("content") or ""
            metadata: Dict[str, Any] = dict(item)
            materials.append(
                RawMaterial(
                    material_id=material_id,
                    content=content,
                    metadata=metadata,
                    json=item,
                )
            )

        self._materials = materials
        print(f"[DataLoader] Loaded {len(materials)} materials.")
        return materials

    def load_question_dimension_mappings(
        self,
        use_random_dims: bool = False,
        use_hardmix_dims: bool = False,
        use_low_freq_random: bool = False,
    ) -> List[QuestionDimensionMapping]:
        """
        Read dimension mapping file and convert to QuestionDimensionMapping list.

        Args:
            use_random_dims: Whether to use random dimension file
                - False: Use original merged_kaocha_jk_cs.json (default)
                - True: Use random dimension merged_mix_dimension_jk_cs.json
            use_hardmix_dims: Whether to use hardmix dimension file (2026-01-14 added)
                - False: Don't use (default)
                - True: Use merged_hardmix_dimension_jk_cs.json (global permutation)
            use_low_freq_random: Whether to use low-freq random dimension file (2026-01-05 added)
                - False: Don't use (default)
                - True: Use merged_low_freq_k{count}_jk_cs.json (based on low_freq_count)

        Returns:
            QuestionDimensionMapping list
        """
        # Select cache and file
        # [2026-01-05] Low-freq random has highest priority
        # [2026-01-06] Support different k values (k=1/3/5)
        if use_low_freq_random:
            # Select file based on k value
            low_freq_count = getattr(self, '_low_freq_count', 3)
            if self._low_freq_random_mappings is not None and getattr(self, '_cached_low_freq_count', None) == low_freq_count:
                return self._low_freq_random_mappings
            filename = f"merged_low_freq_k{low_freq_count}_jk_cs.json"
            # If specific k-value file doesn't exist, try default file
            if not (self.data_dir / filename).exists():
                filename = "merged_low_freq_random_jk_cs.json"
            tag = f"low-freq random dimension mapping (k={low_freq_count})"
        elif use_random_dims:
            if self._random_dimension_mappings is not None:
                return self._random_dimension_mappings
            filename = "merged_mix_dimension_jk_cs.json"
            tag = "random dimension mapping"
        # [2026-01-14] Hardmix dimension file
        elif use_hardmix_dims:
            if self._hardmix_dimension_mappings is not None:
                return self._hardmix_dimension_mappings
            filename = "merged_hardmix_dimension_jk_cs.json"
            tag = "hardmix dimension mapping (global permutation)"
        else:
            if self._dimension_mappings is not None:
                return self._dimension_mappings
            filename = "merged_kaocha_jk_cs.json"
            tag = "dimension mapping"

        path = self.data_dir / filename

        # Check if file exists
        if not path.exists():
            if use_low_freq_random:
                print(f"[DataLoader] [Warning] Low-freq random dimension file not found: {path}")
                print(f"[DataLoader] Please run: python scripts/generate_low_freq_random_file.py")
                raise FileNotFoundError(f"Low-freq random dimension file not found: {path}")
            elif use_random_dims:
                print(f"[DataLoader] [Warning] Random dimension file not found: {path}")
                print(f"[DataLoader] Please run: python scripts/generate_random_dimension_file.py")
                raise FileNotFoundError(f"Random dimension file not found: {path}")
            elif use_hardmix_dims:
                print(f"[DataLoader] [Warning] Hardmix dimension file not found: {path}")
                print(f"[DataLoader] Please run: python scripts/generate_hardmix_dimension.py")
                raise FileNotFoundError(f"Hardmix dimension file not found: {path}")
            else:
                raise FileNotFoundError(f"Dimension mapping file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        mappings: List[QuestionDimensionMapping] = []
        for item in data:
            def _get_list(*keys: str) -> List[str]:
                for k in keys:
                    v = item.get(k)
                    if v:
                        return v if isinstance(v, list) else [v]
                return []

            q = QuestionDimensionMapping(
                unit_id=str(item.get("unit_id") or ""),
                material=item.get("material") or "",
                question_type=item.get("question_type") or "",
                type=item.get("type") or "",  # Text genre type
                exam_skill_level1=_get_list("exam_skill_level1"),
                exam_skill_level2=_get_list("exam_skill_level2"),
                gk_value=_get_list("gk.value", "gk_value"),
                gk_subject_literacy=_get_list("gk.subject_literacy", "gk_subject_literacy"),
                gk_key_ability=_get_list("gk.key_ability", "gk_key_ability"),
                gk_essential_knowledge=_get_list("gk.essential_knowledge", "gk_essential_knowledge"),
                gk_context=_get_list("gk.context", "gk_context"),
                gk_wings=_get_list("gk.wings", "gk_wings"),
                cs_core_literacy=_get_list("cs.core_literacy", "cs_core_literacy"),
                cs_task_group=_get_list("cs.task_group", "cs_task_group"),
                cs_ability=_get_list("cs.ability", "cs_ability"),
            )
            mappings.append(q)

        # Cache results
        if use_low_freq_random:
            self._low_freq_random_mappings = mappings
            self._cached_low_freq_count = self._low_freq_count  # Record current cached k value
        elif use_random_dims:
            self._random_dimension_mappings = mappings
        elif use_hardmix_dims:
            self._hardmix_dimension_mappings = mappings
        else:
            self._dimension_mappings = mappings

        print(f"[DataLoader] Loaded {tag}: {len(mappings)} entries (file: {filename})")
        return mappings

    def load_dimensions(self) -> List[DimensionDefinition]:
        """
        Read ABC_evaluation_prompt.json and convert to DimensionDefinition list.

        Expected JSON structure (per dimension) example:

        {
          "id": "gk.value",
          "dimension_name": "Core Value - Patriotism",
          "levelA": {
            "prompt": "..."              # Level A basic constraint prompt
          },
          "levelB": {
            "addon": "..."               # Level B addon to A
          },
          "levelC": {
            "addon": "...",              # Level C addon to A/B
            "definition": "..."          # Level C complete definition
          },
          "prompt_eval": "..."           # Dimension description for pedagogical eval
        }
        """
        if self._dimensions is not None:
            return self._dimensions

        path = self.data_dir / "ABC_prompt.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dimensions: List[DimensionDefinition] = []
        for item in data:
            dim = DimensionDefinition(
                id=item.get("id"),
                dimension_name=item.get("dimension_name"),
                levelA=item.get("levelA") or {},
                levelB=item.get("levelB") or {},
                levelC=item.get("levelC") or {},
                prompt_eval=item.get("prompt_eval") or "",
            )
            dimensions.append(dim)

        self._dimensions = dimensions
        print(f"[DataLoader] Loaded {len(dimensions)} dimension definitions.")

        # Sanity check: validate key fields across levels
        self._validate_dimension_prompts(dimensions)

        return dimensions


    def _validate_dimension_prompts(self, dimensions: List[DimensionDefinition]) -> None:
        """
        Validate key fields in levelA/B/C of loaded dimension definitions.

        Current Prompt Format Convention (2025-12 updated):
        - levelA: Only "prompt" field (dimension name)
        - levelB: Mainly uses "addon" field (constraints + value connotations + definition)
        - levelC: Mainly uses "addon" field (real exam examples)

        Note: levelC.definition field is deprecated, no longer checked

        Check Items:
        1. Randomly sample up to 3 dimensions, print field lengths
        2. Warn for:
           - levelA.prompt is empty
           - levelC.addon is empty or too short
        """
        import random

        if not dimensions:
            return

        sample_size = min(3, len(dimensions))
        sample_dims = random.sample(dimensions, sample_size)

        print("[DataLoader] === Dimension Config Field Length Validation (Random Sample) ===")

        warnings: list[str] = []

        for dim in sample_dims:
            dim_id = dim.id
            dim_name = dim.dimension_name

            levelA = dim.levelA if isinstance(dim.levelA, dict) else {}
            levelB = dim.levelB if isinstance(dim.levelB, dict) else {}
            levelC = dim.levelC if isinstance(dim.levelC, dict) else {}

            len_A_prompt = len(levelA.get("prompt", "") or "")
            len_B_addon = len(levelB.get("addon", "") or "")
            len_C_addon = len(levelC.get("addon", "") or "")

            print(f"  [{dim_id}] {dim_name}")
            print(f"    levelA.prompt      : {len_A_prompt} chars")
            print(f"    levelB.addon       : {len_B_addon} chars")
            print(f"    levelC.addon       : {len_C_addon} chars")

            # Rule-based warnings
            if len_A_prompt == 0:
                warnings.append(
                    f"[Warning] Dimension {dim_id}({dim_name}): levelA.prompt is empty, "
                    f"may result in vague Level A prompts."
                )

            if len_C_addon == 0:
                warnings.append(
                    f"[Warning] Dimension {dim_id}({dim_name}): levelC.addon is empty, "
                    f"consider adding real exam examples for Level C."
                )
            elif len_C_addon < 100:
                warnings.append(
                    f"[Warning] Dimension {dim_id}({dim_name}): levelC.addon is too short ({len_C_addon} chars), "
                    f"may lack sufficient real exam examples."
                )

        if warnings:
            print("[DataLoader] === Dimension Config Sanity Check Warnings ===")
            for w in warnings:
                print("  " + w)


    # ----------------------------
    # Convenience Methods
    # ----------------------------
    def get_material_by_id(
        self,
        material_id: str,
        materials: Optional[List[RawMaterial]] = None,
    ) -> Optional[RawMaterial]:
        """
        Get RawMaterial by material_id / unit_id from material list.
        If materials not provided, auto-loads from cache/file.
        """
        if materials is None:
            materials = self.load_materials()

        for m in materials:
            if str(m.material_id) == str(material_id):
                return m
        return None

    def load_abc_prompt(self) -> ABCPromptData:
        """
        Read ABC_evaluation_prompt.json and parse into ABCPromptData object.

        ABC_evaluation_prompt.json contains:
        - Pedagogical dimensions (id starts with gk.* or cs.*)
        - Question type definitions (id is "choice type list" or "essay type list" or "essay distractors")

        Returned ABCPromptData contains:
        - dimensions: All dimension definitions
        - question_types: Question type definitions
        - dimension_id_to_names: id to names mapping
        - name_to_definition: name to definition mapping
        """
        if self._abc_prompt is not None:
            return self._abc_prompt

        path = self.data_dir / "ABC_prompt.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dimensions: List[ABCDimensionDefinition] = []
        question_types: List[QuestionTypeDefinition] = []
        dimension_id_to_names: Dict[str, List[str]] = {}
        name_to_definition: Dict[str, ABCDimensionDefinition] = {}

        # Question type category IDs (including MC and essay distractor types)
        QUESTION_TYPE_IDS = {"选择题题型梳理", "主观题题型梳理", "选项设置干扰项", "简答题干扰项"}

        for item in data:
            item_id = item.get("id", "")
            dimension_name = item.get("dimension_name", "")

            # Determine if it's a question type or dimension definition
            if item_id in QUESTION_TYPE_IDS:
                # Question type definition
                qtype = QuestionTypeDefinition(
                    id=item_id,
                    dimension_name=dimension_name,
                    levelA=item.get("levelA") or {},
                    levelB=item.get("levelB") or {},
                    levelC=item.get("levelC") or {},
                    prompt_eval=item.get("prompt_eval") or "",
                )
                question_types.append(qtype)
            else:
                # Pedagogical dimension definition
                dim = ABCDimensionDefinition(
                    id=item_id,
                    dimension_name=dimension_name,
                    levelA=item.get("levelA") or {},
                    levelB=item.get("levelB") or {},
                    levelC=item.get("levelC") or {},
                    prompt_eval=item.get("prompt_eval") or "",
                )
                dimensions.append(dim)

                # Build id -> names mapping (same id may have multiple dimension_names)
                if item_id not in dimension_id_to_names:
                    dimension_id_to_names[item_id] = []
                dimension_id_to_names[item_id].append(dimension_name)

                # Build name -> definition mapping
                name_to_definition[dimension_name] = dim

        self._abc_prompt = ABCPromptData(
            dimensions=dimensions,
            question_types=question_types,
            dimension_id_to_names=dimension_id_to_names,
            name_to_definition=name_to_definition,
        )

        print(f"[DataLoader] Loaded ABC_prompt.json:")
        print(f"  - Pedagogical dimensions: {len(dimensions)}")
        print(f"  - Question type definitions: {len(question_types)}")
        print(f"  - Dimension ID categories: {len(dimension_id_to_names)}")

        return self._abc_prompt

    # Legacy compatibility alias
    def load_top20(self) -> ABCPromptData:
        """
        Legacy compatibility: equivalent to load_abc_prompt()
        """
        return self.load_abc_prompt()

    def get_abc_dimension_by_name(
        self,
        dimension_name: str,
        abc_data: Optional[ABCPromptData] = None,
    ) -> Optional[ABCDimensionDefinition]:
        """
        Get ABC dimension definition by dimension_name.
        """
        if abc_data is None:
            abc_data = self.load_abc_prompt()
        return abc_data.name_to_definition.get(dimension_name)

    # Legacy compatibility alias
    def get_top20_dimension_by_name(
        self,
        dimension_name: str,
        top20_data: Optional[ABCPromptData] = None,
    ) -> Optional[ABCDimensionDefinition]:
        """
        Legacy compatibility: equivalent to get_abc_dimension_by_name()
        """
        return self.get_abc_dimension_by_name(dimension_name, top20_data)

    def get_abc_all_dimension_names(
        self,
        abc_data: Optional[ABCPromptData] = None,
    ) -> List[str]:
        """
        Get all dimension names from ABC_evaluation_prompt (for matching).
        """
        if abc_data is None:
            abc_data = self.load_abc_prompt()
        return list(abc_data.name_to_definition.keys())

    # Legacy compatibility alias
    def get_top20_all_dimension_names(
        self,
        top20_data: Optional[ABCPromptData] = None,
    ) -> List[str]:
        """
        Legacy compatibility: equivalent to get_abc_all_dimension_names()
        """
        return self.get_abc_all_dimension_names(top20_data)

    # ========== [2025-12-31] Year Information Loading ==========

    def load_unit_year_mapping(self) -> Dict[str, int]:
        """
        Load unit_id -> year mapping from raw_material.json.

        Extract year by parsing source field (e.g., "2025 National Paper I" -> 2025).

        Returns:
            Mapping dict, e.g., {"1": 2025, "2": 2024, ...}
        """
        import re

        raw_path = self.data_dir / "raw_material.json"
        if not raw_path.exists():
            return {}

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        result: Dict[str, int] = {}
        for item in raw_data:
            unit_id = str(item.get("unit_id", ""))
            source = item.get("source", "")

            # Extract year from source, e.g., "2025 National Paper I" -> 2025
            year_match = re.search(r"(\d{4})", source)
            if year_match:
                result[unit_id] = int(year_match.group(1))

        return result

    def load_unit_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load unit_id metadata from raw_material.json.

        Returns:
            {unit_id: {"year": int, "source": str, "question_type": str}}
        """
        import re

        raw_path = self.data_dir / "raw_material.json"
        if not raw_path.exists():
            return {}

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        result: Dict[str, Dict[str, Any]] = {}
        for item in raw_data:
            unit_id = str(item.get("unit_id", ""))
            source = item.get("source", "")

            # Extract year from source
            year = None
            year_match = re.search(r"(\d{4})", source)
            if year_match:
                year = int(year_match.group(1))

            result[unit_id] = {
                "year": year,
                "source": source,
            }

        return result
