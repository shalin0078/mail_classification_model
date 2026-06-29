from __future__ import annotations

import argparse
import json
from pathlib import Path

from mail_classifier.data import load_dataset
from mail_classifier.model import (
    batch_predict,
    evaluate_pipeline,
    load_artifact,
    predict_one,
    save_artifact,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run an advanced mail classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a model from labeled CSV data.")
    train.add_argument("--data", required=True, help="Path to labeled CSV.")
    train.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    train.add_argument("--output", default="models/mail_classifier.joblib", help="Model output path.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on labeled data.")
    evaluate.add_argument("--model", required=True, help="Path to saved joblib model.")
    evaluate.add_argument("--data", required=True, help="Path to labeled CSV.")

    predict = subparsers.add_parser("predict", help="Predict one email.")
    predict.add_argument("--model", required=True, help="Path to saved joblib model.")
    input_group = predict.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Email text to classify.")
    input_group.add_argument("--input", help="Path to a text file containing an email.")

    batch = subparsers.add_parser("batch-predict", help="Predict labels for an unlabeled CSV.")
    batch.add_argument("--model", required=True, help="Path to saved joblib model.")
    batch.add_argument("--data", required=True, help="CSV with a text column.")
    batch.add_argument("--output", required=True, help="CSV output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        artifact = train_model(args.data, args.config)
        save_artifact(artifact, args.output)
        print(json.dumps({k: v for k, v in artifact.metrics.items() if k != "classification_report"}, indent=2))
        print(artifact.metrics["classification_report"])
        print(f"Saved model to {args.output}")
        return 0

    if args.command == "evaluate":
        artifact = load_artifact(args.model)
        x, y = load_dataset(args.data)
        metrics = evaluate_pipeline(artifact.pipeline, x, y)
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
        print(metrics["classification_report"])
        return 0

    if args.command == "predict":
        text = args.text if args.text is not None else Path(args.input).read_text(encoding="utf-8")
        artifact = load_artifact(args.model)
        print(json.dumps(predict_one(artifact, text), indent=2))
        return 0

    if args.command == "batch-predict":
        artifact = load_artifact(args.model)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = batch_predict(artifact, args.data)
        frame.to_csv(output, index=False)
        print(f"Saved predictions to {output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

