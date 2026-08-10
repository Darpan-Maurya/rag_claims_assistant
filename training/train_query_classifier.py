import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from core.config import settings


TRAINING_DATA_PATH = Path("training/query_classifier_samples.jsonl")
MODEL_VERSION = "intent-safety-tfidf-logreg-v1"


def load_examples(path: Path = TRAINING_DATA_PATH) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip()
            if not text or not label:
                raise ValueError(f"Invalid training example at line {line_number}")
            texts.append(text)
            labels.append(label)
    if len(set(labels)) < 2:
        raise ValueError("Training data must contain at least two labels")
    return texts, labels


def train(output_path: Path | None = None) -> dict[str, object]:
    texts, labels = load_examples()
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    destination = output_path or settings.query_classifier_model_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": len(texts),
        "labels": sorted(set(labels)),
    }
    joblib.dump(payload, destination)
    return {
        "model_path": str(destination),
        "model_version": MODEL_VERSION,
        "training_examples": len(texts),
        "labels": payload["labels"],
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
