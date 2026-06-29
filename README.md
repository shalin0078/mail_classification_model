# Advanced Mail Classification Model

Production-ready baseline for classifying emails into categories such as `primary`, `promotion`, `social`, `updates`, `spam`, and `phishing`.

This repository contains a complete Python ML workflow:

- Email-aware cleaning and feature extraction
- TF-IDF word and character n-gram features
- Metadata/risk features for URLs, money terms, urgency, attachments, and suspicious language
- Train/evaluate/predict command line tools
- Reproducible config-driven training
- Unit tests for parsing and feature behavior

## 1. Project Structure

```text
.
├── config.yaml                     # Training configuration
├── data/sample_emails.csv           # Small demo dataset
├── examples/example_email.txt       # Example raw email for prediction
├── requirements.txt                 # Runtime dependencies
├── src/mail_classifier/
│   ├── cli.py                       # CLI commands
│   ├── data.py                      # Dataset loading and validation
│   ├── features.py                  # Email-specific numeric features
│   ├── model.py                     # Pipeline creation, train, eval, save/load
│   ├── preprocessing.py             # Email parsing and normalization
│   └── schemas.py                   # Shared dataclasses
└── tests/                           # Dependency-light unit tests
```

## 2. Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest
```

## 3. Dataset Format

Training data must be a CSV file with these columns:

| column | required | description |
| --- | --- | --- |
| `text` | yes | Raw email body or full RFC822-like message |
| `label` | yes | Target class, for example `spam` or `primary` |
| `subject` | no | Subject line if stored separately |
| `sender` | no | Sender address/domain |

Minimum example:

```csv
text,label
"Your package was delivered today.",updates
"URGENT: verify your bank password now http://bad.example",phishing
```

Use your real labeled emails for serious training. The included sample file is intentionally tiny and exists only to prove the workflow.

## 4. Train

Train with the default config:

```bash
python -m mail_classifier.cli train \
  --data data/sample_emails.csv \
  --config config.yaml \
  --output models/mail_classifier.joblib
```

The trainer will:

1. Validate input columns.
2. Combine subject, sender, and body into a single model input.
3. Split data into train/test sets.
4. Train an email-aware scikit-learn pipeline.
5. Print accuracy, macro F1, weighted F1, and a classification report.
6. Save the trained artifact with labels and metadata.

## 5. Predict

Predict from a text file:

```bash
python -m mail_classifier.cli predict \
  --model models/mail_classifier.joblib \
  --input examples/example_email.txt
```

Predict inline text:

```bash
python -m mail_classifier.cli predict \
  --model models/mail_classifier.joblib \
  --text "Please review the attached quarterly invoice."
```

The output is JSON:

```json
{
  "label": "updates",
  "confidence": 0.82,
  "probabilities": {
    "updates": 0.82,
    "primary": 0.11,
    "spam": 0.07
  }
}
```

## 6. Evaluate

Evaluate an already trained model on a labeled CSV:

```bash
python -m mail_classifier.cli evaluate \
  --model models/mail_classifier.joblib \
  --data data/sample_emails.csv
```

## 7. Configuration

`config.yaml` controls vectorizer and classifier behavior:

```yaml
test_size: 0.25
random_state: 42
max_features: 50000
word_ngram_max: 2
char_ngram_min: 3
char_ngram_max: 5
min_df: 1
classifier: logistic_regression
class_weight: balanced
```

Recommended production changes:

- Increase `min_df` to `2` or `5` for larger datasets.
- Keep `class_weight: balanced` if categories are imbalanced.
- Track macro F1, not only accuracy, because rare classes such as phishing matter.
- Store a held-out test set that is never used while tuning.

## 8. Model Design

The pipeline combines three feature families:

1. **Word TF-IDF**: captures topic and phrase-level patterns.
2. **Character TF-IDF**: improves robustness for misspellings, obfuscation, odd domains, and phishing variants.
3. **Numeric email risk features**: counts URLs, currency symbols, exclamation marks, uppercase ratio, urgency terms, attachment terms, credential terms, and sender hints.

This gives a strong classical ML baseline that trains quickly, is explainable enough to debug, and does not require a GPU.

## 9. Deployment Notes

For a web/API service:

1. Train and version `models/mail_classifier.joblib`.
2. Load the model once at process startup with `load_artifact`.
3. Call `predict_one` for every request.
4. Log predicted label, confidence, model version, and latency.
5. Keep raw email content out of logs unless you have explicit privacy approval.

For batch inference:

```bash
python -m mail_classifier.cli batch-predict \
  --model models/mail_classifier.joblib \
  --data data/unlabeled_emails.csv \
  --output reports/predictions.csv
```

## 10. Quality Checklist

Before trusting the model:

- Use at least several hundred labeled emails per class.
- De-duplicate near-identical marketing or spam messages.
- Split data by time or sender when possible to avoid leakage.
- Review the confusion matrix for high-risk mistakes.
- Calibrate thresholds for phishing/spam if false negatives are costly.
- Re-train regularly as email patterns drift.

## 11. GitHub Push

If the remote is not configured:

```bash
git remote add origin https://github.com/shalin0078/mail_classification_model.git
```

Then push:

```bash
git push -u origin main
```

