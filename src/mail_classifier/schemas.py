from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    random_state: int = 42
    max_features: int = 50000
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    min_df: int = 1
    classifier: str = "logistic_regression"
    class_weight: str | None = "balanced"
    max_iter: int = 1200


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Any
    labels: list[str]
    config: TrainingConfig
    metrics: dict[str, Any] = field(default_factory=dict)

