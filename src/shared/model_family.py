from __future__ import annotations

"""
Model-family utilities for Stage1/Stage2 evaluator decoupling.

The evaluation stage must not reuse a model from the same provider family as
the model that generated the question. Unknown model names are only considered
the same family when their normalized names are identical.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


_KNOWN_FAMILY_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("openai", ("gpt", "chatgpt", "o1", "o3", "o4", "o5", "openai")),
    ("deepseek", ("deepseek",)),
    ("doubao", ("doubao", "volcengine", "volc", "ark")),
    ("gemini", ("gemini", "google")),
    ("qwen", ("qwen", "qwq", "tongyi", "dashscope")),
    ("claude", ("claude", "anthropic")),
    ("glm", ("glm", "zhipu")),
    ("hunyuan", ("hunyuan", "tencent")),
    ("ernie", ("ernie", "wenxin", "baidu")),
    ("llama", ("llama", "meta-llama")),
    ("mistral", ("mistral", "mixtral")),
)


def normalize_model_name(model_name: Optional[str]) -> str:
    """Normalize a model name for robust comparisons."""
    text = (model_name or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def infer_model_family(model_name: Optional[str]) -> Optional[str]:
    """
    Return a canonical model family.

    Known providers map to stable families. Unknown non-empty names map to an
    exact-name family so unrelated custom model names are not grouped together.
    """
    normalized = normalize_model_name(model_name)
    if not normalized:
        return None

    tokens = tuple(part for part in normalized.split("-") if part)
    for family, patterns in _KNOWN_FAMILY_PATTERNS:
        for pattern in patterns:
            if normalized == pattern or normalized.startswith(f"{pattern}-") or pattern in tokens:
                return family

    return f"model:{normalized}"


def is_same_model_family(model_a: Optional[str], model_b: Optional[str]) -> bool:
    """Return True when two model names belong to the same inferred family."""
    family_a = infer_model_family(model_a)
    family_b = infer_model_family(model_b)
    return bool(family_a and family_b and family_a == family_b)


@dataclass
class ModelFamilyFilterResult:
    """Result metadata for evaluator-family filtering."""

    generator_model: Optional[str]
    generator_family: Optional[str]
    active_models: List[str]
    excluded_models: List[str]
    original_models: List[str]

    @property
    def enabled(self) -> bool:
        return bool(self.generator_family)

    @property
    def exclusion_applied(self) -> bool:
        return bool(self.excluded_models)

    @property
    def aggregation_policy(self) -> str:
        if len(self.active_models) == 2 and self.exclusion_applied:
            return "two_model_consensus_for_pedagogical"
        return "strict_majority_for_pedagogical"

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "generator_model": self.generator_model,
            "generator_family": self.generator_family,
            "original_eval_models": list(self.original_models),
            "excluded_eval_models": list(self.excluded_models),
            "active_eval_models": list(self.active_models),
            "aggregation_policy": self.aggregation_policy,
        }


def filter_models_by_generator_family(
    models: Sequence[T],
    *,
    model_name_getter,
    generator_model: Optional[str],
    enable_filter: bool = True,
) -> Tuple[List[T], ModelFamilyFilterResult]:
    """
    Remove evaluator models whose family matches the generation model family.

    Args:
        models: Client/config objects or plain model-name values.
        model_name_getter: Callable returning the model name for each item.
        generator_model: Stage1 generation model name.
        enable_filter: Set False for baseline/original-question evaluation.
    """
    original_models = [str(model_name_getter(item) or "") for item in models]
    generator_family = infer_model_family(generator_model) if enable_filter else None

    if not enable_filter or not generator_family:
        return list(models), ModelFamilyFilterResult(
            generator_model=generator_model if enable_filter else None,
            generator_family=generator_family,
            active_models=list(original_models),
            excluded_models=[],
            original_models=list(original_models),
        )

    active: List[T] = []
    excluded: List[str] = []
    for item in models:
        name = str(model_name_getter(item) or "")
        if is_same_model_family(generator_model, name):
            excluded.append(name)
        else:
            active.append(item)

    active_names = [str(model_name_getter(item) or "") for item in active]
    return active, ModelFamilyFilterResult(
        generator_model=generator_model,
        generator_family=generator_family,
        active_models=active_names,
        excluded_models=excluded,
        original_models=list(original_models),
    )


def normalize_weights_for_models(
    model_names: Iterable[str],
    weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Normalize a weight mapping after evaluator filtering."""
    names = list(model_names)
    n = max(len(names), 1)
    if not names:
        return {}
    if not weights:
        return {name: 1.0 / n for name in names}

    selected = {name: float(weights.get(name, 1.0 / n)) for name in names}
    total = sum(max(value, 0.0) for value in selected.values())
    if total <= 0:
        return {name: 1.0 / n for name in names}
    return {name: max(value, 0.0) / total for name, value in selected.items()}
