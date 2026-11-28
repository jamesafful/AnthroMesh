"""Exploratory cross-wallet prediction on BehaviorBench Belief validation."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HISTORY = re.compile(r"^\d+\. Question: (.*?) \| Event: (.*?) \| Choice: (yes|no) \| Confidence: ([0-9.eE+-]+)[ \t]*$", re.M)
WORD = re.compile(r"[a-z]{3,}")


def load(path):
    features, labels, wallets, groups, markets, start, end = [], [], [], [], [], [], []
    rich, sequences, mixed = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("This exploratory protocol uses publisher validation only.")
            history = [(question, int(side == "yes"), float(confidence))
                       for question, event, side, confidence in HISTORY.findall(row["input_text"])
                       if event != row["event_slug"] and question != row["question"]]
            if len(history) < 5 or row["label_choice"] not in {"yes", "no"}:
                continue
            target = set(WORD.findall(row["question"].lower()))
            similar = [side for question, side, _ in history
                       if len(target & set(WORD.findall(question.lower()))) >= 4]
            past = [side for _, side, _ in history]
            recent = past[-5:]
            features.append([1, float("draw" in target), (sum(past) + 1) / (len(past) + 2),
                             (sum(recent) + 1) / 7, past[-1],
                             (sum(similar) + 1) / (len(similar) + 2)])
            weights = [confidence for _, _, confidence in history]
            weighted_yes = sum(side * weight for side, weight in zip(past, weights))
            switches = sum(a != b for a, b in zip(past, past[1:]))
            rich.append([(weighted_yes + 1) / (sum(weights) + 2),
                         sum(weights) / len(weights), weights[-1],
                         switches / (len(past) - 1), np.log1p(len(past))])
            sequences.append(4 * past[-3] + 2 * past[-2] + past[-1])
            mixed.append(min(sum(past), len(past) - sum(past)) / len(past) >= .25)
            labels.append(int(row["label_choice"] == "yes"))
            wallet = row["wallet"]
            wallets.append(wallet)
            markets.append(row["market_id"])
            start.append(row["stance_first_block"])
            end.append(row["stance_last_block"])
            groups.append(int(hashlib.sha256(wallet.encode()).hexdigest()[:8], 16) % 10)
    return tuple(np.asarray(column) for column in (features, labels, wallets, groups, markets, start, end, rich, sequences, mixed))


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


def market_prior(labels, markets, start, end, train):
    observations = defaultdict(list)
    for market, block, label in zip(markets[train], end[train], labels[train]):
        observations[market].append((block, label))
    lookup = {}
    for market, records in observations.items():
        ordered = sorted(records)
        lookup[market] = ([block for block, _ in ordered],
                          np.r_[0, np.cumsum([label for _, label in ordered])])
    result = np.full(len(labels), labels[train].mean())
    for i, (market, block) in enumerate(zip(markets, start)):
        if market in lookup:
            blocks, sums = lookup[market]
            n = bisect_left(blocks, block)
            result[i] = (sums[n] + 10 * result[i]) / (n + 10)
    return result


def loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired_interval(delta, wallets):
    _, membership = np.unique(wallets, return_inverse=True)
    sums = np.bincount(membership, weights=delta)
    sizes = np.bincount(membership)
    rng = np.random.default_rng(20261128)
    draws = rng.integers(len(sizes), size=(2000, len(sizes)))
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {"mean": float(delta.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def matched_risk_interval(correct, other, first, second, wallets):
    _, membership = np.unique(wallets, return_inverse=True)
    n = len(np.unique(membership))
    a_count = np.bincount(membership[first], minlength=n)
    b_count = np.bincount(membership[second], minlength=n)
    a_errors = np.bincount(membership[first], weights=~correct[first], minlength=n)
    b_errors = np.bincount(membership[second], weights=~other[second], minlength=n)
    draws = np.random.default_rng(20261128).integers(n, size=(2000, n))
    a_total, b_total = a_count[draws].sum(axis=1), b_count[draws].sum(axis=1)
    valid = (a_total > 0) & (b_total > 0)
    reduction = (b_errors[draws].sum(axis=1)[valid] / b_total[valid]
                 - a_errors[draws].sum(axis=1)[valid] / a_total[valid])
    return np.quantile(reduction, [.025, .975]).tolist()

def compare(models, x, y, wallets, train, calibration, test):
    rng = np.random.default_rng(20261128)
    xt, yt, wt = x[test], y[test], wallets[test]
    ids = rng.permutation(np.unique(wt))
    donor = dict(zip(ids, np.roll(ids, 1)))
    donor_rows = {wallet: np.flatnonzero(wt == wallet) for wallet in ids}
    shuffled = xt.copy()
    for i, owner in enumerate(wt):
        shuffled[i, 2:] = xt[rng.choice(donor_rows[donor[owner]]), 2:]
    predictions = {name: p[test] for name, p in models.items()}
    predictions["shuffled_history"] = sigmoid(shuffled @ fit(x[train], y[train]))
    p = predictions["pooled_history_model"]
    correct = (p >= .5) == yt
    comparisons, matched = {}, {}
    ties = rng.random(len(yt))
    for name, q in predictions.items():
        other = (q >= .5) == yt
        comparisons[name] = {
            "accuracy": float(other.mean()),
            "accuracy_gain": paired_interval(correct.astype(float) - other, wt),
            "log_loss_reduction": paired_interval(loss(q, yt) - loss(p, yt), wt),
        }
        a_order = np.lexsort((ties, -abs(p - .5)))
        b_order = np.lexsort((ties, -abs(q - .5)))
        matched[name] = []
        for coverage in (.10, .25, .50, 1.):
            n = round(len(yt) * coverage)
            a, b = a_order[:n], b_order[:n]
            matched[name].append({"coverage": n / len(yt), "cases": n,
                                  "pooled_error": float((~correct[a]).mean()),
                                  "comparison_error": float((~other[b]).mean()),
                                  "risk_reduction_ci95": matched_risk_interval(correct, other, a, b, wt)})
    # Consistency uses visible-history choice polarization and calibration
    # boundaries, rather than evaluation labels or a bot classification.
    cuts = np.quantile(abs(x[calibration, 2] - .5), [1/3, 2/3])
    wallet_consistency = {w: np.mean(abs(xt[wt == w, 2] - .5)) for w in ids}
    bands = np.digitize([wallet_consistency[w] for w in wt], cuts)
    base_correct = (predictions["personal_history_frequency"] >= .5) == yt
    subgroup = []
    for i in range(3):
        mask = bands == i
        if mask.any():
            subgroup.append({"band": i + 1, "cases": int(mask.sum()),
                             "wallets": len(set(wt[mask])),
                             "pooled_accuracy": float(correct[mask].mean()),
                             "frequency_accuracy": float(base_correct[mask].mean())})
    return {"paired_comparisons": comparisons, "matched_coverage": matched,
            "history_consistency_bands": subgroup,
            "market_prior_cases_with_earlier_observations": int(
                np.count_nonzero(predictions["market_prior"] != models["population"][test]))}

def markov_prior(sequences, labels, train):
    """Estimate a pooled third-order transition with second-order backoff."""
    previous = sequences % 4
    n2 = np.bincount(previous[train], minlength=4)
    y2 = np.bincount(previous[train], weights=labels[train], minlength=4)
    backoff = (y2 + 10 * labels[train].mean()) / (n2 + 10)
    n3 = np.bincount(sequences[train], minlength=8)
    y3 = np.bincount(sequences[train], weights=labels[train], minlength=8)
    return (y3[sequences] + 10 * backoff[previous]) / (n3[sequences] + 10)


def mixed_curves(models, y, wallets, mask):
    selected = ("personal_history_frequency", "recent_frequency", "markov_three_choice",
                "pooled_history_model", "enriched_pooled_model")
    names = {name: models[name][mask] for name in selected}
    yt, wt = y[mask], wallets[mask]
    ties = np.random.default_rng(20261128).random(len(yt))
    correct = {name: (p >= .5) == yt for name, p in names.items()}
    rank = {name: np.lexsort((ties, -abs(p - .5))) for name, p in names.items()}
    curves = {}
    reference = "enriched_pooled_model"
    for name in names:
        curve = []
        for coverage in (.10, .15, .20, .25, .50, 1.):
            count = round(len(yt) * coverage)
            a, b = rank[reference][:count], rank[name][:count]
            curve.append({"coverage": count / len(yt), "error": float((~correct[name][b]).mean()),
                          "enriched_risk_reduction_ci95": matched_risk_interval(
                              correct[reference], correct[name], a, b, wt)})
        curves[name] = curve
    return {"definition": "At least 25% of visible prior choices are on the minority side",
            "cases": len(yt), "wallets": len(set(wt)), "curves": curves}

def evaluate(path):
    x, y, wallets, group, markets, start, end, rich, sequences, mixed = load(path)
    train, calibration, test = group < 6, (group >= 6) & (group < 8), group >= 8
    ones = np.full(len(y), y[train].mean())
    enriched = np.column_stack((x, rich))
    models = {
        "population": ones,
        "markov_three_choice": markov_prior(sequences, y, train),
        "enriched_pooled_model": sigmoid(enriched @ fit(enriched[train], y[train])),
        "market_prior": market_prior(y, markets, start, end, train),
        "last_choice": .25 + .5 * x[:, 4],
        "recent_frequency": x[:, 3],
        "personal_history_frequency": x[:, 2],
        "target_only": sigmoid(x[:, :2] @ fit(x[train, :2], y[train])),
        "pooled_history_model": sigmoid(x @ fit(x[train], y[train])),
        "without_similar_questions": sigmoid(x[:, :-1] @ fit(x[train, :-1], y[train])),
    }
    result = {}
    for name, p in models.items():
        cuts = [(coverage, np.quantile(abs(p[calibration] - .5), 1 - coverage))
                for coverage in (.10, .25, .50)]
        cuts.append((1.0, 0.0))
        result[name] = metrics(p[test], y[test], cuts)
    controls = compare(models, x, y, wallets, train, calibration, test)
    mixed_result = mixed_curves(models, y, wallets, test & mixed)
    return {"mixed_history": mixed_result, "controls": controls, "design": "exploratory internal wallet-disjoint split of publisher validation",
            "archive_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "splits": {name: {"cases": int(mask.sum()), "wallets": len(set(wallets[mask]))}
                       for name, mask in (("train", train), ("calibration", calibration), ("test", test))},
            "models": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.validation), indent=2))
