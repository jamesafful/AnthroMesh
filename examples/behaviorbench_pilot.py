"""Exploratory cross-wallet prediction on BehaviorBench Belief validation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HISTORY = re.compile(r"^\d+\. Question: (.*?) \| Event: (.*?) \| Choice: (yes|no) \| Confidence:", re.M)
WORD = re.compile(r"[a-z]{3,}")


def load(path):
    features, labels, wallets, groups = [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("This exploratory protocol uses publisher validation only.")
            history = [(question, int(side == "yes"))
                       for question, event, side in HISTORY.findall(row["input_text"])
                       if event != row["event_slug"] and question != row["question"]]
            if len(history) < 5 or row["label_choice"] not in {"yes", "no"}:
                continue
            target = set(WORD.findall(row["question"].lower()))
            similar = [side for question, side in history
                       if len(target & set(WORD.findall(question.lower()))) >= 4]
            past = [side for _, side in history]
            recent = past[-5:]
            features.append([1, float("draw" in target), (sum(past) + 1) / (len(past) + 2),
                             (sum(recent) + 1) / 7, past[-1],
                             (sum(similar) + 1) / (len(similar) + 2)])
            labels.append(int(row["label_choice"] == "yes"))
            wallet = row["wallet"]
            wallets.append(wallet)
            groups.append(int(hashlib.sha256(wallet.encode()).hexdigest()[:8], 16) % 10)
    return np.asarray(features), np.asarray(labels), np.asarray(wallets), np.asarray(groups)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def fit(x, y):
    weights = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])
    for _ in range(15):
        p = sigmoid(x @ weights)
        curvature = x.T @ ((p * (1 - p))[:, None] * x) + penalty
        weights += np.linalg.solve(curvature, x.T @ (y - p) - penalty @ weights)
    return weights


def metrics(p, y, cuts):
    confidence = abs(p - .5)
    correct = (p >= .5) == y
    curve = []
    for nominal, threshold in cuts:
        keep = confidence >= threshold
        curve.append({"validation_target": nominal, "coverage": float(keep.mean()),
                      "error": float((~correct[keep]).mean()) if keep.any() else None,
                      "escalation": float((~keep).mean())})
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": float(correct.mean()),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
            "risk_coverage": curve}


def evaluate(path):
    x, y, wallets, group = load(path)
    train, calibration, test = group < 6, (group >= 6) & (group < 8), group >= 8
    ones = np.full(len(y), y[train].mean())
    models = {
        "population": ones,
        "last_choice": .25 + .5 * x[:, 4],
        "recent_frequency": x[:, 3],
        "personal_history_frequency": x[:, 2],
        "target_only": sigmoid(x[:, :2] @ fit(x[train, :2], y[train])),
        "pooled_history_model": sigmoid(x @ fit(x[train], y[train])),
    }
    result = {}
    for name, p in models.items():
        cuts = [(coverage, np.quantile(abs(p[calibration] - .5), 1 - coverage))
                for coverage in (.10, .25, .50)]
        cuts.append((1.0, 0.0))
        result[name] = metrics(p[test], y[test], cuts)
    return {"design": "exploratory internal wallet-disjoint split of publisher validation",
            "archive_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "splits": {name: {"cases": int(mask.sum()), "wallets": len(set(wallets[mask]))}
                       for name, mask in (("train", train), ("calibration", calibration), ("test", test))},
            "models": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.validation), indent=2))
