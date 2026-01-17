# src/generation/agents/agent1_material_dim_selector.py
# Agent 1: Material & Dimension Selector (data-driven question type + external stratified sampling)

"""
Agent 1: Material & Dimension Selector (Stage1 Entry Point)

Core Design (sampling-decoupled version):
- run(): No longer has built-in sequential/random/manual material selection logic; only responsible for:
  1) Calling external subset_sampler to build subset_unit_ids once (stratified sampling by default)
  2) Yielding the next question from subset within one experiment (no duplicate unit_id)
  3) Packaging selected unit_id with "material + dimensions + question_type (data-driven) + ABC dims"

- run_single(unit_id): Retained for CLI single-question debugging, bypasses sampling.

Notes:
- unit_id is always normalized to str, eliminating deduplication failures from int/str mixing.
- No longer uses dangerous fallback like "use mappings[0] if not found": raises error directly to avoid silent mismatch.

Features:
- Select a material from the annotated material library;
- Extract exam_skill, gk_dims, cs_dims, unit_id from question-dimension mappings;
- Read question_type from data (data-driven);
- Extract matched ABC dimension entries (for Agent 2);
- Ensure no duplicate unit_id within one experiment.

Input:
- DataLoader: Loads material list and question-dimension mappings;
- Agent1Config: Contains material selection strategy, dimension selection strategy, dimension mode, etc.

Output:
- MaterialDimSelection: Packaged result of "material + dimensions" for one pipeline round.
"""

from typing import List, Optional, Dict, Set, Tuple, Any

from src.shared.schemas import (
    MaterialDimSelection,
    RawMaterial,
    DimensionDefinition,
    QuestionDimensionMapping,
    NoMoreUnitsError,
)
from src.shared.config import Agent1Config
from src.shared.data_loader import DataLoader, ABCPromptData

# Stratified subset sampling (Stage1-internal capability)
from src.generation.utils.subset_sampler import build_subset_unit_ids

# Question type alias mapping: unify various formats to internal labels
# Chinese labels are mapped to internal English keys
QUESTION_TYPE_ALIASES: Dict[str, str] = {
    "single-choice": "single-choice",
    "single_choice": "single-choice",
    "单选题": "single-choice",  # single-choice question
    "选择题": "single-choice",  # multiple-choice question
    "essay": "essay",
    "简答题": "essay",  # short-answer question
    "主观题": "essay",  # subjective question
}


def _normalize_question_type(qt: Optional[str]) -> str:
    if qt is None:
        return ""
    qt = str(qt).strip()
    return QUESTION_TYPE_ALIASES.get(qt, qt)


def _uid(x: Any) -> str:
    return str(x).strip()


class Agent1MaterialDimSelector:
    """
    Agent 1: Material & Dimension Selector (external sampling version)

    - run(): Calls subset_sampler to build subset, then sequentially yields MaterialDimSelection for next question
    - run_single(unit_id): Packages specified unit_id directly, bypasses sampling
    """

    def __init__(
        self,
        config: Agent1Config,
        data_loader: DataLoader,
        use_random_dims: bool = False,
        use_low_freq_random: bool = False,
        low_freq_count: int = 3,  # [2026-01-06] Low-freq dimension count (k=1/3/5)
    ) -> None:
        self.config = config
        self.data_loader = data_loader

        # [2026-01] Random dimension ablation experiment flag
        self.use_random_dims = use_random_dims
        # [2026-01] Low-freq random dimension experiment flag
        self.use_low_freq_random = use_low_freq_random
        # [2026-01-06] Low-freq dimension count
        self.low_freq_count = low_freq_count

        # Set DataLoader's low-freq dimension count
        if use_low_freq_random and hasattr(data_loader, 'set_low_freq_count'):
            data_loader.set_low_freq_count(low_freq_count)

        # Deduplication for this experiment (store as str)
        self._used_unit_ids: Set[str] = set()

        # Subset sampling state (for run())
        self._subset_unit_ids: Optional[List[str]] = None
        self._subset_cursor: int = 0

    # -------------------------
    # Public APIs
    # -------------------------

    def run(self) -> MaterialDimSelection:
        """
        Full/Batch mode: Returns selection for next question in subset on each call.

        Subset construction is done by external subset_sampler (default: stratified coverage-first).
        """
        print("[Agent 1] Starting material & dimension selection (subset-driven)...")
        print(f"[Agent 1] Dimension selection strategy: {getattr(self.config, 'dimension_selection_strategy', 'N/A')}")
        print(f"[Agent 1] Dimension mode: {getattr(self.config, 'dimension_mode', 'gk')}")

        materials, mappings = self._load_data()

        # Build subset on first run() call (only once)
        if self._subset_unit_ids is None:
            self._subset_unit_ids = self._build_subset_unit_ids(materials, mappings)
            self._subset_cursor = 0
            print(f"[Agent 1] [OK] Subset built: {len(self._subset_unit_ids)} questions")

        # Get next unused question from subset
        while self._subset_cursor < len(self._subset_unit_ids):
            unit_id = _uid(self._subset_unit_ids[self._subset_cursor])
            self._subset_cursor += 1

            if unit_id in self._used_unit_ids:
                continue

            selection = self._build_selection_by_unit_id(unit_id, materials, mappings)
            self._used_unit_ids.add(unit_id)

            self._print_selection_summary(selection, prefix="[Agent 1] [OK] Selection complete")
            return selection

        raise NoMoreUnitsError(
            f"[Agent 1] Subset exhausted: subset_size={len(self._subset_unit_ids or [])}, used={len(self._used_unit_ids)}"
        )

    def run_single(self, unit_id: str) -> MaterialDimSelection:
        """
        Single-question mode: Run with specified unit_id (bypasses sampling).
        """
        unit_id = _uid(unit_id)
        print(f"[Agent 1] Single-question mode: loading unit_id = {unit_id}")
        print(f"[Agent 1] Dimension mode: {getattr(self.config, 'dimension_mode', 'gk')}")

        materials, mappings = self._load_data()
        selection = self._build_selection_by_unit_id(unit_id, materials, mappings)

        # Also record single question as used (prevent duplicates when mixing run/run_single)
        self._used_unit_ids.add(unit_id)

        self._print_selection_summary(selection, prefix="[Agent 1] [OK] Single-question selection complete")
        return selection

    def reset_state(self) -> None:
        """
        Reset experiment-level state (call when starting new experiment)
        """
        self._used_unit_ids.clear()
        self._subset_unit_ids = None
        self._subset_cursor = 0
        print("[Agent 1] State reset")

    def get_used_unit_ids(self) -> Set[str]:
        return set(self._used_unit_ids)

    # -------------------------
    # Internals
    # -------------------------

    def _load_data(self) -> Tuple[List[RawMaterial], List[QuestionDimensionMapping]]:
        materials = self.data_loader.load_materials()
        if not materials:
            raise ValueError("Material list is empty, please check data/raw_material.json")

        # [2026-01] Support random dimension ablation / low-freq random dimension experiments
        mappings = self.data_loader.load_question_dimension_mappings(
            use_random_dims=self.use_random_dims,
            use_low_freq_random=self.use_low_freq_random,
        )
        if not mappings:
            if self.use_low_freq_random:
                filename = "merged_low_freq_random_jk_cs.json"
            elif self.use_random_dims:
                filename = "merged_mix_dimension_jk_cs.json"
            else:
                filename = "merged_kaocha_jk_cs.json"
            raise ValueError(f"Dimension mapping is empty, please check data/{filename}")

        return materials, mappings

    def _build_subset_unit_ids(
        self,
        materials: List[RawMaterial],
        mappings: List[QuestionDimensionMapping],
    ) -> List[str]:
        """
        Build unit_id subset using external subset_sampler.

        Default strategy (based on requirements):
        - strategy: stratified (coverage-first)
        - use_year_bin: True (source year is clear)
        - strata_keys: ("question_type","type") + year_bin
        """
        subset_size = int(getattr(self.config, "subset_size", 40))
        seed = int(getattr(self.config, "subset_seed", getattr(self.config, "seed", 42)))
        strategy = str(getattr(self.config, "subset_strategy", "stratified"))

        # Allow custom config; otherwise use "question_type x genre x year_bin" strategy
        strata_keys = getattr(self.config, "subset_strata_keys", ("question_type", "type"))
        use_year_bin = bool(getattr(self.config, "subset_use_year_bin", True))

        print("[Agent 1] Calling subset_sampler to build subset...")
        print(f"  - subset_size={subset_size}, seed={seed}, strategy={strategy}")
        print(f"  - strata_keys={tuple(strata_keys)}, use_year_bin={use_year_bin}")

        res = build_subset_unit_ids(
            materials=materials,
            mappings=mappings,
            subset_size=subset_size,
            seed=seed,
            strategy=strategy,
            strata_keys=tuple(strata_keys),
            use_year_bin=use_year_bin,
        )

        # Normalize to str
        return [_uid(x) for x in res.unit_ids]

    def _build_selection_by_unit_id(
        self,
        unit_id: str,
        materials: List[RawMaterial],
        mappings: List[QuestionDimensionMapping],
    ) -> MaterialDimSelection:
        """Build MaterialDimSelection for specified unit_id"""
        unit_id = _uid(unit_id)

        material_index: Dict[str, RawMaterial] = {_uid(m.material_id): m for m in materials}
        mapping_index: Dict[str, QuestionDimensionMapping] = {_uid(mp.unit_id): mp for mp in mappings}

        matched_mapping = mapping_index.get(unit_id)
        if matched_mapping is None:
            raise ValueError(f"[Agent 1] Cannot find dimension mapping for unit_id={unit_id}")

        selected_material = material_index.get(unit_id)
        if selected_material is None:
            # Fallback: content prefix matching (only when material_id doesn't match)
            prefix = (matched_mapping.material or "")[:100].strip()
            if prefix:
                for mat in materials:
                    if (mat.content or "")[:100].strip() == prefix:
                        selected_material = mat
                        break
        if selected_material is None:
            raise ValueError(f"[Agent 1] Cannot find material for unit_id={unit_id}")

        # Data-driven question type
        raw_qt = matched_mapping.question_type
        normalized_qt = _normalize_question_type(raw_qt)
        print(f"[Agent 1] [Data-driven] Read question type from data: {raw_qt} -> {normalized_qt}")

        # Dimension extraction
        exam_skill, gk_dims, cs_dims, extracted_uid = self._extract_complete_dimensions(matched_mapping)
        extracted_uid = _uid(extracted_uid)

        if extracted_uid != unit_id:
            # Should not happen; indicates data or index issue
            raise ValueError(f"[Agent 1] unit_id mismatch: requested={unit_id}, in_mapping={extracted_uid}")

        # [2025-12] Match ABC_evaluation_prompt pedagogical dimensions
        abc_prompt_data = self.data_loader.load_abc_prompt()
        matched_abc_dims, question_type_label = self._extract_matched_abc_dims(gk_dims, cs_dims, abc_prompt_data)

        # Build question type label (extracted from exam_skill, not a dimension)
        # Format: L1-L2 (e.g., "distractor_type-range")
        skill_l1 = exam_skill.get("level1", []) or []
        skill_l2 = exam_skill.get("level2", []) or []
        if skill_l1 and skill_l2:
            question_type_label = f"{skill_l1[0]}-{skill_l2[0]}"
        elif skill_l2:
            question_type_label = skill_l2[0]
        elif skill_l1:
            question_type_label = skill_l1[0]
        print(f"[Agent 1] Question type label: {question_type_label}")

        # ABC dims (for Agent2) - [2025-12] Question type dimension no longer used
        abc_dims = self._extract_abc_dims(gk_dims, cs_dims)

        return MaterialDimSelection(
            material_id=selected_material.material_id,
            material_text=selected_material.content,
            question_type=normalized_qt,
            exam_skill=exam_skill,
            gk_dims=gk_dims,
            cs_dims=cs_dims,
            unit_id=unit_id,
            # [2025-12] Matched ABC dimensions
            matched_abc_dims=matched_abc_dims,
            top20_dims=matched_abc_dims,  # Backward compatibility
            question_type_label=question_type_label,
            abc_dims=abc_dims,
            # Legacy field compatibility (minimize downstream impact)
            primary_dimension=gk_dims.get("gk.value", [""])[0] if gk_dims.get("gk.value") else "",
            secondary_dimensions=[],
            selection_reasoning=f"subset-driven/run_single; data-driven question_type={normalized_qt}; mode={getattr(self.config, 'dimension_mode', 'gk')}; matched_abc_dims={len(matched_abc_dims)}",
        )

    def _extract_complete_dimensions(
        self, mapping: QuestionDimensionMapping
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], str]:
        """
        Extract complete dimensions from question-dimension mapping.
        """
        exam_skill: Dict[str, List[str]] = {
            "level1": mapping.exam_skill_level1 if mapping.exam_skill_level1 else [],
            "level2": mapping.exam_skill_level2 if mapping.exam_skill_level2 else [],
        }

        # Gaokao dimensions
        gk_dims: Dict[str, List[str]] = {
            "gk.value": mapping.gk_value if mapping.gk_value else [],
            "gk.subject_literacy": mapping.gk_subject_literacy if mapping.gk_subject_literacy else [],
            "gk.key_ability": mapping.gk_key_ability if mapping.gk_key_ability else [],
            "gk.essential_knowledge": mapping.gk_essential_knowledge if mapping.gk_essential_knowledge else [],
            "gk.wings": mapping.gk_wings if mapping.gk_wings else [],
            "gk.context": mapping.gk_context if mapping.gk_context else [],
        }

        # Curriculum standard dimensions
        cs_dims: Dict[str, List[str]] = {
            "cs.core_literacy": mapping.cs_core_literacy if mapping.cs_core_literacy else [],
            "cs.task_group": mapping.cs_task_group if mapping.cs_task_group else [],
            "cs.ability": mapping.cs_ability if mapping.cs_ability else [],
        }

        return exam_skill, gk_dims, cs_dims, _uid(mapping.unit_id)

    def _extract_abc_dims(
            self,
            gk_dims: Dict[str, List[str]],
            cs_dims: Dict[str, List[str]],
    ) -> List[DimensionDefinition]:
        """
        [2025-12] Extract ABC dimension definitions from gk_dims/cs_dims.

        Question type dimensions (exam_skill) are no longer used.
        """
        from collections import defaultdict

        abc_dims: List[DimensionDefinition] = []
        seen: Set[str] = set()

        all_dimensions = self.data_loader.load_dimensions()
        if not all_dimensions:
            print("[Agent 1] [Warning] ABC_evaluation_prompt.json is empty, cannot extract abc_dims")
            return []

        by_id = defaultdict(list)
        for d in all_dimensions:
            by_id[getattr(d, "id", "")].append(d)

        def _norm(s: str) -> str:
            """Normalize string: remove spaces"""
            return (s or "").strip().replace(" ", "").replace("\u3000", "")

        def _tail(name: str) -> str:
            """Extract tail part after separator"""
            name = (name or "").strip()
            if not name:
                return ""
            for sep in ("-", "—", "–", "－"):
                if sep in name:
                    return name.split(sep)[-1].strip()
            return name

        def _pick(dim_id: str, label: str):
            """Pick best matching dimension"""
            cands = by_id.get(dim_id, [])
            if not cands:
                return None
            nl = _norm(label)
            ranked = []
            for d in cands:
                dn = getattr(d, "dimension_name", "") or ""
                nd = _norm(dn)
                nt = _norm(_tail(dn))
                score = 0
                if nl and nt == nl:
                    score = 3  # Exact tail match
                elif nl and nd.endswith(nl):
                    score = 2  # Ends with label
                elif nl and nl in nd:
                    score = 1  # Contains label
                if score > 0:
                    ranked.append((score, len(dn), d))
            if not ranked:
                return None
            ranked.sort(key=lambda x: (-x[0], x[1]))
            top_score = ranked[0][0]
            if sum(1 for s, _, _ in ranked if s == top_score) > 1:
                names = [getattr(d, "dimension_name", "") for s, _, d in ranked if s == top_score][:5]
                print(f"[Agent 1] [Warning] ABC dimension match ambiguity dim_id={dim_id}, label={label}, candidates={names}")
            return ranked[0][2]

        def _add(d):
            """Add dimension to list if not seen"""
            if d is None:
                return
            key = f"{getattr(d, 'id', '')}::{getattr(d, 'dimension_name', '')}"
            if key in seen:
                return
            abc_dims.append(d)
            seen.add(key)

        # [2026-01] Remove gk+cs mode, default to gk
        mode = getattr(self.config, "dimension_mode", "gk")
        process_gk = mode in ("gk_only", "gk")
        process_cs = mode in ("cs_only", "cs")

        # ---- 1) GK/CS dimensions ----
        if process_gk:
            for dim_category, labels in (gk_dims or {}).items():
                if not labels:
                    continue
                for lb in labels:
                    d = _pick(dim_category, lb)
                    if d is None:
                        print(f"[Agent 1] [Warning] ABC dimension not matched dim_id={dim_category}, label={lb}")
                    _add(d)

        if process_cs:
            for dim_category, labels in (cs_dims or {}).items():
                if not labels:
                    continue
                for lb in labels:
                    d = _pick(dim_category, lb)
                    if d is None:
                        print(f"[Agent 1] [Warning] ABC dimension not matched dim_id={dim_category}, label={lb}")
                    _add(d)

        # [2025-12] Question type dimensions no longer used, exam_skill not matched

        print(f"[Agent 1] Matched ABC dimension entries: {len(abc_dims)} (mode: {mode})")
        if abc_dims:
            preview = [f"{getattr(d, 'id', '')}|{getattr(d, 'dimension_name', '')}" for d in abc_dims[:12]]
            print(f"[Agent 1] ABC match preview (first 12): {preview}")
        return abc_dims

    def _extract_matched_abc_dims(
        self,
        gk_dims: Dict[str, List[str]],
        cs_dims: Dict[str, List[str]],
        abc_prompt_data: "ABCPromptData",
    ) -> Tuple[List[str], str]:
        """
        [2025-12] Match dimension labels from merged_kaocha_jk_cs.json to
        pedagogical dimensions in ABC_evaluation_prompt.json.

        Matching Logic:
        1. Iterate all labels in gk_dims/cs_dims
        2. For each label, find dimension_name containing the label in ABC_evaluation_prompt.json
        3. Question type labels (exam_skill) are not dimensions, handled separately

        Args:
            gk_dims: Gaokao dimensions (gk.value, gk.wings, ...)
            cs_dims: Curriculum standard dimensions (cs.core_literacy, ...)
            abc_prompt_data: ABC_evaluation_prompt.json data

        Returns:
            (matched_dims, question_type_label):
            - matched_dims: List of matched ABC dimension_names
            - question_type_label: Question type label (if any)
        """
        matched_dims: List[str] = []
        seen: Set[str] = set()

        # Get all dimension names from ABC_evaluation_prompt
        all_abc_names = list(abc_prompt_data.name_to_definition.keys())

        def _norm(s: str) -> str:
            """Normalize string: remove spaces"""
            return (s or "").strip().replace(" ", "").replace("\u3000", "")

        def _match_label_to_abc(label: str, dim_id: str) -> Optional[str]:
            """
            Try to match label to dimension_name in ABC_evaluation_prompt.

            Matching strategies (by priority):
            1. Exact tail match: label == tail of dimension_name
            2. Contains match: label is substring of dimension_name
            3. Prefix/ID match: filter candidates by dim_id prefix
            """
            nl = _norm(label)
            if not nl:
                return None

            # Strategy 1: Find exact tail match from same ID dimensions
            if dim_id in abc_prompt_data.dimension_id_to_names:
                candidates = abc_prompt_data.dimension_id_to_names[dim_id]
                for name in candidates:
                    # Tail match: dimension_name format is usually "Category-SpecificName"
                    if "-" in name:
                        tail = name.split("-", 1)[-1]
                        if _norm(tail) == nl:
                            return name
                    if _norm(name) == nl:
                        return name

            # Strategy 2: Global contains match
            for name in all_abc_names:
                nn = _norm(name)
                if nl in nn or nn.endswith(nl):
                    return name

            return None

        # Process dimension mode
        # [2026-01] Remove gk+cs mode, default to gk
        mode = getattr(self.config, "dimension_mode", "gk")
        process_gk = mode in ("gk_only", "gk")
        process_cs = mode in ("cs_only", "cs")

        # Match gk dimensions
        if process_gk:
            for dim_id, labels in (gk_dims or {}).items():
                for label in labels:
                    matched = _match_label_to_abc(label, dim_id)
                    if matched and matched not in seen:
                        matched_dims.append(matched)
                        seen.add(matched)
                        print(f"[Agent 1] [ABC matched] {dim_id}:{label} -> {matched}")
                    elif not matched:
                        print(f"[Agent 1] [ABC not matched] {dim_id}:{label}")

        # Match cs dimensions
        if process_cs:
            for dim_id, labels in (cs_dims or {}).items():
                for label in labels:
                    matched = _match_label_to_abc(label, dim_id)
                    if matched and matched not in seen:
                        matched_dims.append(matched)
                        seen.add(matched)
                        print(f"[Agent 1] [ABC matched] {dim_id}:{label} -> {matched}")
                    elif not matched:
                        print(f"[Agent 1] [ABC not matched] {dim_id}:{label}")

        # Question type label is empty for now (extracted separately from exam_skill, not a dimension)
        question_type_label = ""

        print(f"[Agent 1] Matched ABC pedagogical dimensions: {len(matched_dims)}")
        if matched_dims:
            print(f"[Agent 1] ABC match list: {matched_dims[:10]}...")

        return matched_dims, question_type_label

    def _print_selection_summary(self, selection: MaterialDimSelection, prefix: str = "") -> None:
        """Print selection summary"""
        unit_id = _uid(selection.unit_id)
        gk_cnt = sum(len(v) for v in (selection.gk_dims or {}).values())
        cs_cnt = sum(len(v) for v in (selection.cs_dims or {}).values())
        abc_cnt = len(selection.abc_dims or [])
        matched_abc_cnt = len(selection.matched_abc_dims or [])

        if prefix:
            print(prefix)
        print(f"  - material_id: {selection.material_id}")
        print(f"  - unit_id: {unit_id}")
        print(f"  - question_type: {selection.question_type}")
        print(f"  - question_type_label: {selection.question_type_label}")
        print(f"  - gk_dim_count: {gk_cnt}")
        print(f"  - cs_dim_count: {cs_cnt}")
        print(f"  - matched_abc_dim_count: {matched_abc_cnt}")
        print(f"  - abc_dim_count: {abc_cnt}")
        if selection.matched_abc_dims:
            print(f"  - matched_abc_dims: {selection.matched_abc_dims[:5]}...")


__all__ = [
    "Agent1MaterialDimSelector",
    "QUESTION_TYPE_ALIASES",
    "_normalize_question_type",
]
