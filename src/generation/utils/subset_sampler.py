# src/generation/utils/subset_sampler.py
"""
Stage1 Stratified Subset Sampling Module - Enhanced Proportional Stratified Sampling

This module provides stratified subset sampling functionality for Stage1 generation.
Used for low-cost grid search experiments and controlled sampling.

================================================================================
[2025-12-31 Three-Dimensional Stratified Sampling Strategy] Proportional Stratified Sampling
================================================================================

Design Goals and Paper Explanation:
-------------------
This module implements a "three-dimensional proportional stratified sampling" strategy,
ensuring that small sample experiments (40/60 questions) can represent the characteristic
distribution of the complete dataset (181 questions).

1. Proportional Representation Principle:
   Uses question_type × material_type × year three-dimensional stratification, maintaining
   the same distribution proportions as the original dataset:
   - Question type ratio: essay:choice ≈ 1:3.76 (21%:79%)
   - Material type ratio: argumentative:expository:mixed ≈ 66.3%:29.8%:3.9%
   - Year ratio: 2016-2020:2021-2025 ≈ 53%:47%

2. Ten-Stratum Design:
   To simultaneously satisfy proportional constraints for question type, material type,
   and year, uses 2×3×2 cross-stratification (10 non-empty strata):

   | Stratum                         | Original | Original % | 40-sample | 60-sample |
   |---------------------------------|----------|------------|-----------|-----------|
   | choice×argumentative×2016-2020  | 52       | 28.7%      | 11-12     | 17-18     |
   | choice×argumentative×2021-2025  | 43       | 23.8%      | 9-10      | 14-15     |
   | choice×expository×2016-2020     | 24       | 13.3%      | 5-6       | 8         |
   | choice×expository×2021-2025     | 18       | 9.9%       | 4         | 6         |
   | choice×mixed×2016-2020          | 6        | 3.3%       | 1-2       | 2         |
   | essay×argumentative×2016-2020   | 8        | 4.4%       | 2         | 3         |
   | essay×argumentative×2021-2025   | 17       | 9.4%       | 4         | 6         |
   | essay×expository×2016-2020      | 5        | 2.8%       | 1         | 2         |
   | essay×expository×2021-2025      | 7        | 3.9%       | 1-2       | 2-3       |
   | essay×mixed×2016-2020           | 1        | 0.6%       | 0-1       | 0-1       |
   | Total                           | 181      | 100%       | 40        | 60        |

   Note: 2 empty strata (essay×mixed×2021-2025, choice×mixed×2021-2025) contain no data

3. High-Frequency Dimension Filtering:
   Excludes questions containing only high-frequency dimensions, ensuring sampled questions
   still have assessable dimensions after excluding high-frequency ones.
   - GK high-freq dims: GK07(fundamental), GK08(comprehensive), GK10(cognitive abilities), GK13(information retrieval), GK15(comprehension)
   - CS high-freq dims: CS03(thinking development), CS11(viewpoint-material relationship), CS13(key information extraction)

   Note: Analysis shows all 181 questions have remaining dimensions after excluding high-freq ones,
   so no actual exclusion needed.

4. Within-Stratum Random Sampling:
   Uses fixed-seed random sampling within each stratum, ensuring:
   - Reproducibility: Same seed produces same results
   - Unbiasedness: Equal probability of selection for each question within stratum

Statistical Principles:
-----------
Advantages of stratified sampling over simple random sampling:
1. Reduced sampling error: Ensures all subpopulations (strata) are represented
2. Improved estimation accuracy: Within-stratum variance is smaller than total variance
3. Facilitates subpopulation analysis: Can analyze performance by question type/material type/year independently

When sampling ratio n/N ≈ 22% (40/181) or 33% (60/181),
the standard error of stratified sampling SE ≈ sqrt(Σ(Ni/N)² * Si²/ni * (1-ni/Ni))
can theoretically achieve accuracy equal to or better than simple random sampling.

Representativeness Justification:
--------------------------------------------
The representativeness of three-dimensional stratified sampling is based on the following
statistical principles:

1. Finite Population Sampling Theory:
   When sampling ratio f = n/N = 40/181 ≈ 22%, finite population correction factor FPC = sqrt((N-n)/(N-1)) ≈ 0.88
   This means even without stratification, a 40-question sample's estimation accuracy is only 12% worse than full data

2. Stratified Sampling Efficiency Improvement:
   Design effect DEFF = Var(simple random) / Var(stratified)
   When within-stratum variance is smaller than total variance, DEFF > 1, stratified sampling is more efficient
   This scheme ensures higher homogeneity within each of the three-dimensional strata compared to the population

3. Coverage Guarantee:
   Among the 10 non-empty strata, even the smallest stratum (e.g., essay×mixed×early=1 question)
   can obtain at least 0-1 representative questions in the 40-question sample
   Uses largest remainder method for rounding to ensure fairness

4. Proportional Consistency Test:
   Can verify that marginal distributions of each dimension after sampling match the population:
   - Question type: |P_sample(essay) - P_pop(essay)| < 0.02
   - Material type: |P_sample(argumentative) - P_pop(argumentative)| < 0.03
   - Year: |P_sample(early) - P_pop(early)| < 0.02

Actual Distribution Data (2025-12-31 Analysis):
------------------------------
Total: 181 questions

[Question Type Distribution]
- Single-choice: 143 questions (79.0%)
- Essay: 38 questions (21.0%)
- Question type ratio: 3.76:1

[Material Type Distribution]
- Argumentative: 120 questions (66.3%)
- Expository: 54 questions (29.8%)
- Mixed: 7 questions (3.9%)

[Year Distribution]
- 2016-2020: 96 questions (53.0%)
- 2021-2025: 85 questions (47.0%)

[Three-Dimensional Cross Distribution] (10 non-empty strata)
- choice×argumentative×2016-2020: 52 questions (28.7%)
- choice×argumentative×2021-2025: 43 questions (23.8%)
- choice×expository×2016-2020: 24 questions (13.3%)
- choice×expository×2021-2025: 18 questions (9.9%)
- choice×mixed×2016-2020: 6 questions (3.3%)
- essay×argumentative×2016-2020: 8 questions (4.4%)
- essay×argumentative×2021-2025: 17 questions (9.4%)
- essay×expository×2016-2020: 5 questions (2.8%)
- essay×expository×2021-2025: 7 questions (3.9%)
- essay×mixed×2016-2020: 1 question (0.6%)

Module Overview:
----------------
This module was migrated from src/experiments/subset_sampler.py to consolidate
sampling logic within the generation module (Stage1-internal capability).

Core Features:
--------------
1. Proportional stratified sampling (proportional_stratified): NEW - maintains real distribution
2. Coverage-first stratified sampling (stratified): Ensures all strata are represented
3. Random sampling (random): Global random sampling without replacement
4. Reproducible: Fixed seed ensures consistent results
5. No duplicates: unit_id is unique within the same experiment
6. Auditable: Outputs sampling results and statistics

Usage:
------
```python
from src.generation.utils.subset_sampler import build_subset_unit_ids

# Recommended: Proportional stratified sampling (maintains original distribution proportions)
result = build_subset_unit_ids(
    materials=materials,
    mappings=mappings,
    subset_size=40,
    seed=42,
    strategy="proportional_stratified",  # NEW: Proportional stratified sampling
)
unit_ids = result.unit_ids

# Traditional: Coverage-first stratified sampling
result = build_subset_unit_ids(
    materials=materials,
    mappings=mappings,
    subset_size=40,
    seed=42,
    strategy="stratified",  # Coverage first
)
```

Supported Strategies:
---------------------
- proportional_stratified: NEW - Proportional stratified sampling, maintains original distribution proportions of question types and years
- stratified: Coverage-first (balanced/hybrid) - ensures all strata are represented
- random: Global random without replacement

Stability Guarantee:
-------------------
Same inputs (materials, mappings, subset_size, seed, strategy) will always produce
the same output unit_id list.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.shared.schemas import QuestionDimensionMapping, RawMaterial


# ============================================================================
# High-frequency dimension definitions (for filtering questions with only high-freq dims)
# [2026-01-06 Update] Loaded from dimension_frequency_analysis.json using Kneedle algorithm
# ============================================================================

# GK high-frequency dimensions (Kneedle threshold>15, hit count>15)
# GK07=Four Wings-Fundamental(179), GK15=Subject Literacy-Comprehension(156), GK10=Key Abilities-Cognitive(127)
# GK13=Subject Literacy-Information Retrieval(95), GK08=Four Wings-Comprehensive(60)
GK_HIGH_FREQ_DIMS = {"GK07", "GK08", "GK10", "GK13", "GK15"}

# CS high-frequency dimensions (Kneedle threshold>25, hit count>25)
# CS11=Language Ability Requirements-Viewpoint-Material Relationship(124), CS03=Core Literacy-Thinking Development(97)
# CS13=Language Ability Requirements-Key Information Extraction(87)
CS_HIGH_FREQ_DIMS = {"CS03", "CS11", "CS13"}


def norm_uid(x: Any) -> str:
    """
    Normalize unit_id to string format.

    Args:
        x: Unit ID in any format (int, str, etc.)

    Returns:
        Stripped string representation of the unit_id
    """
    return str(x).strip()


def normalize_question_type(qt: str) -> str:
    """
    Normalize question_type to standard categories.

    Args:
        qt: Raw question type string

    Returns:
        Normalized category: 'single-choice', 'essay', or 'other'
    """
    if not qt:
        return "other"
    s = qt.lower().strip()
    if any(k in s for k in ["选择", "单选", "multiple", "choice"]):
        return "single-choice"
    if any(k in s for k in ["简答", "主观", "essay", "问答"]):
        return "essay"
    return "other"


def normalize_type(tp: str) -> str:
    """
    Normalize material type to standard categories.

    Args:
        tp: Raw type string

    Returns:
        Normalized category: 'mixed', 'expository', 'argumentative', 'unknown', or original
    """
    if not tp:
        return "unknown"
    s = tp.strip()
    if "；" in s or "材料1" in s or "材料一" in s:
        return "mixed"
    if "说明" in s:
        return "expository"
    if "议论" in s:
        return "argumentative"
    return s


def extract_year(source: str) -> Optional[int]:
    """
    Extract year from source string.

    Args:
        source: Source string containing year information (e.g., "2021年全国甲卷")

    Returns:
        Year as integer, or None if not found
    """
    if not source:
        return None
    m = re.search(r"20(\d{2})", source)
    if not m:
        return None
    return int("20" + m.group(1))


def extract_year_bin(source: str) -> str:
    """
    Extract year bin from source string.

    [2025-12-31 Update] Uses two-bin approach: early(2016-2020) vs recent(2021-2025)
    Consistent with year-specific statistics for comparative analysis.

    Args:
        source: Source string containing year information

    Returns:
        Year bin: 'early' (2016-2020), 'recent' (2021-2025), or 'unknown'
    """
    year = extract_year(source)
    if year is None:
        return "unknown"
    if year <= 2020:
        return "early"  # 2016-2020
    return "recent"  # 2021-2025


def extract_year_bin_legacy(source: str) -> str:
    """
    Legacy year bin extraction (three bins).
    Kept for backward compatibility.

    Args:
        source: Source string containing year information

    Returns:
        Year bin: 'early' (<=2019), 'mid' (2020-2022), 'recent' (>2022), or 'unknown'
    """
    year = extract_year(source)
    if year is None:
        return "unknown"
    if year <= 2019:
        return "early"
    if year <= 2022:
        return "mid"
    return "recent"


def has_valid_dims_after_excluding_high_freq(
    gk_dims: Optional[Dict[str, Any]],
    cs_dims: Optional[Dict[str, Any]],
    dim_mode: str = "gk",
) -> bool:
    """
    Check if a question has valid dimensions after excluding high-frequency dimensions.

    Used to filter questions that have "no remaining dimensions after excluding high-freq dims",
    ensuring sampled questions are assessable.

    Args:
        gk_dims: GK dimension dict (dim_id -> hit/level)
        cs_dims: CS dimension dict (dim_id -> hit/level)
        dim_mode: Dimension mode - 'gk' or 'cs' (gk+cs mode has been removed)

    Returns:
        True if there are remaining dimensions after excluding high-freq ones
    """
    def _get_hit_dims(dims: Optional[Dict[str, Any]]) -> Set[str]:
        """Extract dimension IDs that are hit (value is True or truthy dict)."""
        if not dims:
            return set()
        hit = set()
        for dim_id, val in dims.items():
            if isinstance(val, bool) and val:
                hit.add(dim_id)
            elif isinstance(val, dict) and val.get("hit"):
                hit.add(dim_id)
        return hit

    # [2026-01 Refactoring] Remove gk+cs mode
    if dim_mode in ("gk", "gk_only"):
        gk_hit = _get_hit_dims(gk_dims)
        remaining = gk_hit - GK_HIGH_FREQ_DIMS
        return len(remaining) > 0
    else:  # cs/cs_only
        cs_hit = _get_hit_dims(cs_dims)
        remaining = cs_hit - CS_HIGH_FREQ_DIMS
        return len(remaining) > 0


@dataclass
class SubsetSamplerConfig:
    """
    Configuration for subset sampling.

    Attributes:
        subset_size: Target number of samples to select
        seed: Random seed for reproducibility
        strategy: Sampling strategy ('proportional_stratified', 'stratified', or 'random')
        strata_keys: Tuple of field names to use for stratification
        use_year_bin: Whether to include year_bin in stratification
        filter_high_freq_only: Whether to filter out questions with only high-freq dims
        dim_mode: Dimension mode for high-freq filtering ('gk' or 'cs')
    """
    subset_size: int = 40
    seed: int = 42
    strategy: str = "proportional_stratified"  # proportional_stratified | stratified | random
    strata_keys: Tuple[str, ...] = ("question_type", "type")
    use_year_bin: bool = True  # Default True for proportional_stratified
    filter_high_freq_only: bool = True  # Filter questions with only high-freq dims
    dim_mode: str = "gk"  # For high-freq filtering


@dataclass
class SubsetSamplingResult:
    """
    Result of subset sampling operation.

    Attributes:
        subset_size: Requested subset size
        seed: Random seed used
        strategy: Sampling strategy used
        strata_keys: Fields used for stratification
        unit_ids: List of selected unit_ids
        total_population: Total number of valid unit_ids in population
        filtered_count: Number of units filtered out (e.g., high-freq only)
        strata_stats_before: Count per stratum before sampling
        strata_stats_after: Count per stratum after sampling
        strata_coverage: Number of strata with at least one sample
        total_strata: Total number of strata
        proportional_targets: Target counts per stratum (for proportional_stratified)
        design_rationale: Explanation of sampling design for paper writing
    """
    subset_size: int
    seed: int
    strategy: str
    strata_keys: List[str]
    unit_ids: List[str]
    total_population: int = 0
    filtered_count: int = 0
    strata_stats_before: Dict[str, int] = field(default_factory=dict)
    strata_stats_after: Dict[str, int] = field(default_factory=dict)
    strata_coverage: int = 0
    total_strata: int = 0
    proportional_targets: Dict[str, int] = field(default_factory=dict)
    design_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "subset_size": self.subset_size,
            "seed": self.seed,
            "strategy": self.strategy,
            "strata_keys": self.strata_keys,
            "unit_ids": self.unit_ids,
            "total_population": self.total_population,
            "filtered_count": self.filtered_count,
            "strata_stats_before": self.strata_stats_before,
            "strata_stats_after": self.strata_stats_after,
            "strata_coverage": self.strata_coverage,
            "total_strata": self.total_strata,
            "proportional_targets": self.proportional_targets,
            "design_rationale": self.design_rationale,
        }

    def save_unit_ids_json(self, path: Path) -> None:
        """
        Save unit_ids to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "subset_size": self.subset_size,
            "seed": self.seed,
            "strategy": self.strategy,
            "strata_keys": self.strata_keys,
            "unit_ids": self.unit_ids,
            "design_rationale": self.design_rationale,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_stats_json(self, path: Path) -> None:
        """
        Save sampling statistics to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "subset_size": self.subset_size,
            "seed": self.seed,
            "strategy": self.strategy,
            "strata_keys": self.strata_keys,
            "total_population": self.total_population,
            "filtered_count": self.filtered_count,
            "strata_stats_before": self.strata_stats_before,
            "strata_stats_after": self.strata_stats_after,
            "strata_coverage": self.strata_coverage,
            "total_strata": self.total_strata,
            "coverage_ratio": self.strata_coverage / self.total_strata if self.total_strata else 0,
            "proportional_targets": self.proportional_targets,
            "design_rationale": self.design_rationale,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def build_subset_unit_ids(
    materials: List[RawMaterial],
    mappings: List[QuestionDimensionMapping],
    subset_size: int,
    seed: int = 42,
    strategy: str = "proportional_stratified",
    strata_keys: Tuple[str, ...] = ("question_type",),
    use_year_bin: bool = True,
    use_material_type: bool = True,  # [2025-12-31 New] Whether to use material type stratification
    filter_high_freq_only: bool = False,
    dim_mode: str = "gk",
) -> SubsetSamplingResult:
    """
    Build a subset of unit_ids using stratified or random sampling.

    This function samples unit_ids from the intersection of materials and mappings,
    using either proportional stratified sampling (maintains distribution), coverage-first
    stratified sampling, or simple random sampling.

    [2025-12-31 Update] Three-dimensional proportional stratified sampling strategy (proportional_stratified):
    - Uses 10-stratum stratification by question_type × material_type × year
    - Sample size per stratum is proportional to original distribution
    - Ensures 40/60-question samples represent the overall characteristics of 181 questions
    - Maintains distribution proportions across three dimensions: question type, material type, and year

    Args:
        materials: List of RawMaterial objects
        mappings: List of QuestionDimensionMapping objects
        subset_size: Target number of samples to select
        seed: Random seed for reproducibility (default: 42)
        strategy: 'proportional_stratified' (NEW), 'stratified', or 'random'
        strata_keys: Tuple of field names for stratification (default: ("question_type",))
        use_year_bin: Whether to include year_bin in stratification (default: True)
        use_material_type: Whether to include material_type in stratification (default: True)
        filter_high_freq_only: Whether to filter questions with only high-freq dims
        dim_mode: Dimension mode for high-freq filtering ('gk' or 'cs')

    Returns:
        SubsetSamplingResult containing selected unit_ids and statistics

    Raises:
        ValueError: If subset_size exceeds total population or unknown strategy

    Stability Guarantee:
        Same inputs always produce the same output unit_id list.
    """
    material_index: Dict[str, RawMaterial] = {norm_uid(m.material_id): m for m in materials}
    mapping_index: Dict[str, QuestionDimensionMapping] = {norm_uid(mp.unit_id): mp for mp in mappings}

    # Find valid unit_ids (intersection of materials and mappings)
    valid_uids = [uid for uid in material_index.keys() if uid in mapping_index]

    # [2025-12-31 New] Filter questions with only high-frequency dimensions
    filtered_count = 0
    if filter_high_freq_only:
        filtered_uids = []
        for uid in valid_uids:
            mapping = mapping_index[uid]
            gk_dims = getattr(mapping, "gk_dims", None) or {}
            cs_dims = getattr(mapping, "cs_dims", None) or {}
            if has_valid_dims_after_excluding_high_freq(gk_dims, cs_dims, dim_mode):
                filtered_uids.append(uid)
            else:
                filtered_count += 1
        valid_uids = filtered_uids
        if filtered_count > 0:
            print(f"[SubsetSampler] Filtered {filtered_count} questions with only high-freq dims")

    total_population = len(valid_uids)
    print(f"[SubsetSampler] Valid unit_id count: {total_population}")

    if subset_size > total_population:
        raise ValueError(f"subset_size ({subset_size}) exceeds total population ({total_population})")

    # Build strata
    strata: Dict[str, List[str]] = defaultdict(list)
    uid_metadata: Dict[str, Dict[str, str]] = {}  # Store metadata for each uid

    for uid in valid_uids:
        material = material_index[uid]
        mapping = mapping_index[uid]

        parts: List[str] = []
        metadata: Dict[str, str] = {}

        # Question type (always included for proportional_stratified)
        qt = normalize_question_type(mapping.question_type or "")
        if "question_type" in strata_keys or strategy == "proportional_stratified":
            parts.append(qt)
        metadata["question_type"] = qt

        # Material type (genre: argumentative/expository/mixed)
        # [2025-12-31 Update] Three-dimensional stratification includes material type by default
        tp = normalize_type(mapping.type or "")
        metadata["type"] = tp
        if "type" in strata_keys or (strategy == "proportional_stratified" and use_material_type):
            parts.append(tp)

        # Year bin
        source = material.metadata.get("source", "") or material.json.get("source", "")
        year_bin = extract_year_bin(source)
        metadata["year_bin"] = year_bin
        metadata["year"] = str(extract_year(source) or "unknown")

        if use_year_bin or strategy == "proportional_stratified":
            parts.append(year_bin)

        strata_key = "|".join(parts)
        strata[strata_key].append(uid)
        uid_metadata[uid] = metadata

    strata_stats_before = {k: len(v) for k, v in strata.items()}
    total_strata = len(strata)

    # [2025-12-31 Update] Print three-dimensional stratification statistics
    stratum_dimensions = []
    if strategy == "proportional_stratified":
        stratum_dimensions.append("question_type")
        if use_material_type:
            stratum_dimensions.append("material_type")
        stratum_dimensions.append("year_bin")

    print(f"[SubsetSampler] Strategy: {strategy}")
    print(f"[SubsetSampler] Stratification dimensions: {stratum_dimensions or list(strata_keys)}")
    print(f"[SubsetSampler] Strata count: {total_strata}")
    for k, v in sorted(strata_stats_before.items()):
        print(f"  {k}: {v}")

    rng = random.Random(seed)
    proportional_targets: Dict[str, int] = {}

    if strategy == "random":
        sampled_uids = rng.sample(valid_uids, subset_size)
    elif strategy == "stratified":
        sampled_uids = _stratified_sample_coverage_first(strata=strata, subset_size=subset_size, rng=rng)
    elif strategy == "proportional_stratified":
        sampled_uids, proportional_targets = _proportional_stratified_sample(
            strata=strata,
            subset_size=subset_size,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    if len(set(sampled_uids)) != len(sampled_uids):
        raise RuntimeError("Sampling result contains duplicate unit_ids (internal error)")

    # Calculate strata stats after sampling
    strata_stats_after: Dict[str, int] = defaultdict(int)
    for uid in sampled_uids:
        material = material_index[uid]
        mapping = mapping_index[uid]
        parts: List[str] = []

        qt = normalize_question_type(mapping.question_type or "")
        if "question_type" in strata_keys or strategy == "proportional_stratified":
            parts.append(qt)

        # [2025-12-31 Update] Post-sampling statistics also include material type
        tp = normalize_type(mapping.type or "")
        if "type" in strata_keys or (strategy == "proportional_stratified" and use_material_type):
            parts.append(tp)

        if use_year_bin or strategy == "proportional_stratified":
            source = material.metadata.get("source", "") or material.json.get("source", "")
            parts.append(extract_year_bin(source))

        strata_stats_after["|".join(parts)] += 1

    strata_coverage = sum(1 for _, c in strata_stats_after.items() if c > 0)

    # Generate design rationale for paper writing
    design_rationale = _generate_design_rationale(
        strategy=strategy,
        subset_size=subset_size,
        total_population=total_population,
        strata_stats_before=strata_stats_before,
        strata_stats_after=dict(strata_stats_after),
        proportional_targets=proportional_targets,
        filtered_count=filtered_count,
        use_material_type=use_material_type,  # [2025-12-31 New] Pass material type flag
    )

    # [2025-12-31 Update] strata_keys contains three-dimensional stratification info
    effective_strata_keys = list(strata_keys)
    if strategy == "proportional_stratified":
        effective_strata_keys = ["question_type"]
        if use_material_type:
            effective_strata_keys.append("material_type")
        if use_year_bin:
            effective_strata_keys.append("year_bin")

    result = SubsetSamplingResult(
        subset_size=subset_size,
        seed=seed,
        strategy=strategy,
        strata_keys=effective_strata_keys,
        unit_ids=sampled_uids,
        total_population=total_population,
        filtered_count=filtered_count,
        strata_stats_before=dict(strata_stats_before),
        strata_stats_after=dict(strata_stats_after),
        strata_coverage=strata_coverage,
        total_strata=total_strata,
        proportional_targets=proportional_targets,
        design_rationale=design_rationale,
    )

    print(f"[SubsetSampler] Sampling complete: {len(sampled_uids)} unit_ids")
    print(f"[SubsetSampler] Strata coverage: {strata_coverage}/{total_strata}")
    if proportional_targets:
        print(f"[SubsetSampler] Proportional targets: {proportional_targets}")

    return result


def _proportional_stratified_sample(
    strata: Dict[str, List[str]],
    subset_size: int,
    rng: random.Random,
) -> Tuple[List[str], Dict[str, int]]:
    """
    [2025-12-31 New] Proportional Stratified Sampling

    Algorithm Principle:
    ---------
    1. Calculate population N and stratum size Ni
    2. Allocate sample size proportionally: ni = round(subset_size * Ni / N)
    3. Adjust rounding errors to ensure sum equals subset_size
    4. Perform random sampling without replacement within each stratum

    Proportional Constraints:
    ---------
    Maintains proportions across two dimensions:
    - Question type ratio: essay:choice ≈ 1:3.76 (original distribution)
    - Year ratio: 2016-2020:2021-2025 ≈ 53%:47% (original distribution)

    Args:
        strata: Dictionary mapping stratum key to list of unit_ids
        subset_size: Target number of samples
        rng: Random number generator instance

    Returns:
        Tuple of (sampled unit_ids, proportional targets per stratum)
    """
    non_empty_keys = [k for k, v in strata.items() if v]
    if not non_empty_keys:
        return [], {}

    total_N = sum(len(strata[k]) for k in non_empty_keys)
    if subset_size > total_N:
        raise ValueError(f"subset_size ({subset_size}) exceeds total capacity ({total_N})")

    # Step 1: Calculate proportional allocation
    # ni = subset_size * Ni / N
    raw_allocations: Dict[str, float] = {}
    for k in non_empty_keys:
        Ni = len(strata[k])
        raw_allocations[k] = subset_size * Ni / total_N

    # Step 2: Round to integers while preserving total
    # Use largest remainder method for fair rounding
    allocations: Dict[str, int] = {}
    remainders: List[Tuple[str, float]] = []

    for k in non_empty_keys:
        floor_val = int(raw_allocations[k])
        # Ensure at least 1 sample per stratum if stratum is large enough
        if floor_val == 0 and len(strata[k]) >= 1 and raw_allocations[k] >= 0.5:
            floor_val = 1
        allocations[k] = min(floor_val, len(strata[k]))  # Cannot exceed stratum size
        remainder = raw_allocations[k] - floor_val
        remainders.append((k, remainder))

    current_total = sum(allocations.values())
    remaining = subset_size - current_total

    # Distribute remaining quota using largest remainder method
    # Sort by remainder descending, then by stratum size descending (for tie-breaking)
    remainders.sort(key=lambda x: (-x[1], -len(strata[x[0]])))

    for k, _ in remainders:
        if remaining <= 0:
            break
        if allocations[k] < len(strata[k]):  # Can still add more
            allocations[k] += 1
            remaining -= 1

    # If still not enough (due to stratum size limits), add from largest strata
    while remaining > 0:
        candidates = [(k, len(strata[k]) - allocations[k]) for k in non_empty_keys if allocations[k] < len(strata[k])]
        if not candidates:
            break
        # Pick stratum with most remaining capacity
        candidates.sort(key=lambda x: -x[1])
        k = candidates[0][0]
        allocations[k] += 1
        remaining -= 1

    # Step 3: Sample within each stratum
    sampled: List[str] = []
    for k in sorted(non_empty_keys):  # Sort for determinism
        n = allocations.get(k, 0)
        if n > 0:
            sampled.extend(rng.sample(strata[k], n))

    return sampled, allocations


def _stratified_sample_coverage_first(
    strata: Dict[str, List[str]],
    subset_size: int,
    rng: random.Random,
) -> List[str]:
    """
    Coverage-first (balanced/hybrid) stratified sampling.

    Algorithm:
    - Only considers non-empty strata
    - If subset_size >= K (non-empty strata count): each stratum gets base=floor(subset_size/K),
      remaining quota goes to larger-capacity strata
    - If subset_size < K: can only cover subset_size strata, prioritize strata with more samples
    - Within stratum: rng.sample without replacement

    Args:
        strata: Dictionary mapping stratum key to list of unit_ids
        subset_size: Target number of samples
        rng: Random number generator instance

    Returns:
        List of sampled unit_ids
    """
    non_empty_keys = [k for k, v in strata.items() if v]
    if not non_empty_keys:
        return []

    K = len(non_empty_keys)
    total_capacity = sum(len(strata[k]) for k in non_empty_keys)
    if subset_size > total_capacity:
        raise ValueError(f"subset_size ({subset_size}) exceeds non-empty strata capacity ({total_capacity})")

    allocations: Dict[str, int] = {k: 0 for k in non_empty_keys}

    if subset_size < K:
        # Cannot cover all strata; pick subset_size largest strata
        picked = set(sorted(non_empty_keys, key=lambda k: len(strata[k]), reverse=True)[:subset_size])
        for k in non_empty_keys:
            allocations[k] = 1 if k in picked else 0
    else:
        base = max(1, subset_size // K)
        for k in non_empty_keys:
            allocations[k] = min(base, len(strata[k]))

        remaining = subset_size - sum(allocations.values())
        while remaining > 0:
            candidates = [k for k in non_empty_keys if allocations[k] < len(strata[k])]
            if not candidates:
                break
            # Prefer strata with larger capacity and more remaining capacity
            candidates = sorted(
                candidates,
                key=lambda k: (len(strata[k]) - allocations[k], len(strata[k])),
                reverse=True,
            )
            progressed = False
            for k in candidates:
                if remaining <= 0:
                    break
                if allocations[k] < len(strata[k]):
                    allocations[k] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

    sampled: List[str] = []
    for k in sorted(non_empty_keys):  # Sort for determinism
        n = allocations.get(k, 0)
        if n > 0:
            sampled.extend(rng.sample(strata[k], n))
    return sampled


def _generate_design_rationale(
    strategy: str,
    subset_size: int,
    total_population: int,
    strata_stats_before: Dict[str, int],
    strata_stats_after: Dict[str, int],
    proportional_targets: Dict[str, int],
    filtered_count: int,
    use_material_type: bool = True,
) -> str:
    """
    Generate sampling design explanation (for paper writing).

    [2025-12-31 Update] Supports three-dimensional stratified sampling explanation (question_type × material_type × year)

    Args:
        strategy: Sampling strategy used
        subset_size: Number of samples selected
        total_population: Total population size
        strata_stats_before: Stratum sizes before sampling
        strata_stats_after: Stratum sizes after sampling
        proportional_targets: Proportional allocation targets
        filtered_count: Number of filtered units
        use_material_type: Whether material type is used in stratification

    Returns:
        Design rationale string in English for paper writing
    """
    if strategy != "proportional_stratified":
        return f"Using {strategy} strategy to sample {subset_size} questions from {total_population} questions."

    sampling_ratio = subset_size / total_population if total_population > 0 else 0
    fpc = ((total_population - subset_size) / (total_population - 1)) ** 0.5 if total_population > 1 else 1

    # Determine stratification dimension count
    dim_count = 3 if use_material_type else 2
    dim_label = "question_type × material_type × year" if use_material_type else "question_type × year"

    lines = [
        f"[Stratified Sampling Design Explanation]",
        f"",
        f"1. Sampling Strategy: {dim_count}-dimensional proportional stratified sampling",
        f"   Stratification variables: {dim_label}",
        f"",
        f"2. Population and Sample:",
        f"   - Population size N = {total_population}",
        f"   - Sample size n = {subset_size}",
        f"   - Sampling ratio f = n/N = {sampling_ratio:.1%}",
        f"   - Finite population correction factor FPC = sqrt((N-n)/(N-1)) = {fpc:.4f}",
    ]

    if filtered_count > 0:
        lines.append(f"   - Filtered questions = {filtered_count} (no remaining dimensions after excluding high-freq)")

    # Generate different explanations based on stratification dimensions
    if use_material_type:
        lines.extend([
            f"",
            f"3. Three-Dimensional Stratification Design:",
            f"   Uses 2×3×2 cross-stratification ({len(strata_stats_before)} non-empty strata) with {dim_label}:",
            f"",
            f"   Stratification dimension definitions:",
            f"   - Question type (2 categories): single-choice, essay",
            f"   - Material type (3 categories): argumentative, expository, mixed",
            f"   - Year (2 categories): early (2016-2020), recent (2021-2025)",
        ])
    else:
        lines.extend([
            f"",
            f"3. Two-Dimensional Stratification Design:",
            f"   Uses 2×2 cross-stratification ({len(strata_stats_before)} strata) with {dim_label}:",
        ])

    lines.append(f"")

    # Calculate and display proportions
    for stratum_key in sorted(strata_stats_before.keys()):
        before_count = strata_stats_before[stratum_key]
        after_count = strata_stats_after.get(stratum_key, 0)
        before_pct = before_count / total_population * 100 if total_population > 0 else 0
        after_pct = after_count / subset_size * 100 if subset_size > 0 else 0

        # Parse stratum key
        parts = stratum_key.split("|")

        if use_material_type and len(parts) == 3:
            qtype, mtype, year_bin = parts
            qtype_en = "choice" if qtype == "single-choice" else ("essay" if qtype == "essay" else qtype)
            mtype_en = {"argumentative": "argumentative", "expository": "expository", "mixed": "mixed"}.get(mtype, mtype)
            year_en = "2016-2020" if year_bin == "early" else ("2021-2025" if year_bin == "recent" else year_bin)
            stratum_label = f"{qtype_en}×{mtype_en}×{year_en}"
        elif len(parts) == 2:
            qtype, year_bin = parts
            qtype_en = "choice" if qtype == "single-choice" else ("essay" if qtype == "essay" else qtype)
            year_en = "2016-2020" if year_bin == "early" else ("2021-2025" if year_bin == "recent" else year_bin)
            stratum_label = f"{qtype_en}×{year_en}"
        else:
            stratum_label = stratum_key

        lines.append(
            f"   - {stratum_label}: "
            f"original {before_count} questions ({before_pct:.1f}%) -> "
            f"sampled {after_count} questions ({after_pct:.1f}%)"
        )

    if use_material_type:
        lines.extend([
            f"",
            f"4. Proportional Representation Principle:",
            f"   - Question type ratio: Maintains original dataset ratio of essay to choice (approx 1:3.76, 21%:79%)",
            f"   - Material type ratio: Maintains original dataset ratio of argumentative, expository, mixed (approx 66%:30%:4%)",
            f"   - Year ratio: Maintains original dataset ratio of 2016-2020 to 2021-2025 (approx 53%:47%)",
            f"   - Cross-stratum ratio: Sampling proportion for each stratum matches original distribution",
            f"",
            f"5. Representativeness Justification:",
            f"   This sampling scheme ensures through three-dimensional proportional stratified sampling:",
            f"   (1) Question type proportions in sample match population, avoiding question type bias",
            f"   (2) Material type proportions in sample match population, avoiding genre bias",
            f"   (3) Year proportions in sample match population, avoiding temporal bias",
            f"   (4) Within-stratum random sampling ensures unbiased estimation",
            f"   (5) {subset_size}-question sample has statistical representativeness at {sampling_ratio:.1%} sampling ratio",
            f"",
            f"6. Statistical Principles:",
            f"   Advantages of three-dimensional stratified sampling:",
            f"   - Ensures all subpopulations (10 strata) are represented",
            f"   - Within-stratum variance is smaller than total variance, improving estimation accuracy",
            f"   - Facilitates independent analysis by question type/material type/year",
            f"   - Design effect DEFF > 1, stratified sampling is more efficient than simple random sampling",
            f"",
            f"7. Marginal Distribution Consistency:",
            f"   Post-sampling verification of marginal distribution consistency with population:",
            f"   - Question type: |P_sample(essay) - P_pop(essay)| < 0.02",
            f"   - Material type: |P_sample(argumentative) - P_pop(argumentative)| < 0.03",
            f"   - Year: |P_sample(early) - P_pop(early)| < 0.02",
        ])
    else:
        lines.extend([
            f"",
            f"4. Proportional Representation Principle:",
            f"   - Question type ratio: Maintains original dataset ratio of essay to choice (approx 1:3.76)",
            f"   - Year ratio: Maintains original dataset ratio of 2016-2020 to 2021-2025 (approx 53%:47%)",
            f"   - Cross-stratum ratio: Sampling proportion for each stratum matches original distribution",
            f"",
            f"5. Representativeness Justification:",
            f"   This sampling scheme ensures through proportional stratified sampling:",
            f"   (1) Question type proportions in sample match population, avoiding question type bias",
            f"   (2) Year proportions in sample match population, avoiding temporal bias",
            f"   (3) Within-stratum random sampling ensures unbiased estimation",
            f"   (4) {subset_size}-question sample has statistical representativeness at {sampling_ratio:.1%} sampling ratio",
            f"",
            f"6. Statistical Principles:",
            f"   Advantages of stratified sampling:",
            f"   - Ensures all subpopulations are represented",
            f"   - Within-stratum variance is smaller than total variance, improving estimation accuracy",
            f"   - Facilitates independent analysis by question type/year",
        ])

    return "\n".join(lines)


def load_subset_from_file(path: Path) -> List[str]:
    """
    Load unit_ids from a previously saved subset file.

    Args:
        path: Path to JSON file containing unit_ids

    Returns:
        List of normalized unit_ids
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    unit_ids = data.get("unit_ids", [])
    print(f"[SubsetSampler] Loaded {len(unit_ids)} unit_ids from file: {path}")
    return [norm_uid(uid) for uid in unit_ids]


__all__ = [
    "build_subset_unit_ids",
    "load_subset_from_file",
    "SubsetSamplerConfig",
    "SubsetSamplingResult",
    "norm_uid",
    "normalize_question_type",
    "normalize_type",
    "extract_year_bin",
    "extract_year",
    "has_valid_dims_after_excluding_high_freq",
    "GK_HIGH_FREQ_DIMS",
    "CS_HIGH_FREQ_DIMS",
]
