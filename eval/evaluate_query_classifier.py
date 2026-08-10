import json
from collections import Counter, defaultdict
from pathlib import Path

from core.query_classifier import QueryClassifier


GOLDEN_PATH = Path("eval/query_classifier_golden.jsonl")


def main() -> None:
    classifier = QueryClassifier()
    rows = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()]
    correct = 0
    by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    prediction_counts: Counter[str] = Counter()

    for row in rows:
        prediction = classifier.classify(row["text"])
        expected = row["expected_label"]
        prediction_counts[prediction.label] += 1
        by_label[expected]["total"] += 1
        if prediction.label == expected:
            correct += 1
            by_label[expected]["correct"] += 1

    print(
        json.dumps(
            {
                "queries": len(rows),
                "accuracy": round(correct / len(rows), 3) if rows else 0.0,
                "per_label_accuracy": {
                    label: round(values["correct"] / values["total"], 3)
                    for label, values in sorted(by_label.items())
                },
                "prediction_counts": dict(sorted(prediction_counts.items())),
                "model": classifier.readiness(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
