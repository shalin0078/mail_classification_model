from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from mail_classifier.data import load_dataset
from mail_classifier.features import EmailRiskFeatureExtractor
from mail_classifier.preprocessing import normalize_email_text
from mail_classifier.schemas import ModelArtifact, TrainingConfig


def load_config(path: str | Path | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return TrainingConfig(**raw)


def build_pipeline(config: TrainingConfig) -> Pipeline:
    if config.classifier != "logistic_regression":
        raise ValueError(f"Unsupported classifier: {config.classifier}")

    text_features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_email_text,
                    analyzer="word",
                    ngram_range=(1, config.word_ngram_max),
                    min_df=config.min_df,
                    max_features=config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_email_text,
                    analyzer="char_wb",
                    ngram_range=(config.char_ngram_min, config.char_ngram_max),
                    min_df=config.min_df,
                    max_features=max(config.max_features // 2, 1000),
                    sublinear_tf=True,
                ),
            ),
            (
                "risk_features",
                Pipeline(
                    [
                        ("extract", EmailRiskFeatureExtractor()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        n_jobs=None,
        solver="liblinear",
    )
    return Pipeline([("features", text_features), ("classifier", classifier)])


def train_model(data_path: str | Path, config_path: str | Path | None = None) -> ModelArtifact:
    config = load_config(config_path)
    x, y = load_dataset(data_path)

    class_counts = y.value_counts()
    estimated_test_rows = int(round(len(y) * config.test_size))
    can_stratify = class_counts.min() >= 2 and estimated_test_rows >= len(class_counts)
    stratify = y if can_stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)
    metrics = evaluate_pipeline(pipeline, x_test, y_test)
    labels = sorted(y.unique().tolist())
    return ModelArtifact(pipeline=pipeline, labels=labels, config=config, metrics=metrics)


def evaluate_pipeline(pipeline: Pipeline, x, y) -> dict[str, Any]:
    predictions = pipeline.predict(x)
    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "weighted_f1": f1_score(y, predictions, average="weighted"),
        "classification_report": classification_report(y, predictions, zero_division=0),
    }


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> ModelArtifact:
    return joblib.load(path)


def predict_one(artifact: ModelArtifact, text: str) -> dict[str, Any]:
    pipeline = artifact.pipeline
    label = str(pipeline.predict([text])[0])

    probabilities: dict[str, float] = {}
    confidence = 1.0
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        classes = [str(item) for item in pipeline.named_steps["classifier"].classes_]
        scores = pipeline.predict_proba([text])[0]
        probabilities = {class_name: float(score) for class_name, score in zip(classes, scores)}
        confidence = float(max(scores))

    return {"label": label, "confidence": confidence, "probabilities": probabilities}


def batch_predict(artifact: ModelArtifact, data_path: str | Path):
    import pandas as pd

    frame = pd.read_csv(data_path)
    if "text" not in frame.columns:
        raise ValueError("Batch prediction CSV must contain a text column.")
    predictions = [predict_one(artifact, text=str(text)) for text in frame["text"]]
    frame["predicted_label"] = [item["label"] for item in predictions]
    frame["confidence"] = [item["confidence"] for item in predictions]
    return frame
