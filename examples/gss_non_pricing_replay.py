
"""Exploratory two-answer memory and personal intercept replay on GSS 2010."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

ARCHIVE = "ffe09fa4edaae23a169b0737a982df28fb6acc0ec19e15762a0019f5de3abbd7"
ITEMS = ("nataid", "natarms", "natchld", "natcity", "natcrime", "natdrug",
         "nateduc", "natenvir", "natfare", "natheal", "natmass", "natpark",
         "natroad", "natsci", "natsoc", "natspac")
TARGET = "natenvir"
SEED = 20261126


def compare(predictions, truth, majority, rng):
    hits = predictions == truth
    base = majority == truth
    change = hits.astype(float) - base
    # One target response per participant in both comparisons.
    draws = rng.integers(len(change), size=(3000, len(change)))
    return {"n": len(change), "accuracy": float(hits.mean()),
            "majority_accuracy": float(base.mean()), "accuracy_gain": float(change.mean()),
            "gain_ci95": np.quantile(change[draws].mean(axis=1), [.025, .975]).tolist()}


def evaluate(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE:
        raise ValueError("Unexpected GSS panel revision.")
    columns = [f"{item}_{wave}" for item in ITEMS for wave in (1, 2, 3)]
    with TemporaryDirectory() as tmp, ZipFile(archive) as z:
        member, = [n for n in z.namelist() if n.lower().endswith(".dta")]
        data = pd.read_stata(z.extract(member, tmp), columns=columns, convert_categoricals=False)
    # Panel codes: 1 too little, 2 about right, 3 too much. Exclude other codes.
    raw = data[columns].to_numpy().reshape(-1, len(ITEMS), 3)
    eligible = np.isin(raw, [1, 2, 3])
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(len(raw))
    fit, test = np.array_split(ids, 2)
    target_index = ITEMS.index(TARGET)
    choose = eligible[test, target_index].all(axis=1)
    matched = test[choose]
    if not len(matched):
        raise ValueError("No participants answered the target in all three waves.")
    y = raw[matched, target_index, 2]
    equal = raw[matched, target_index, 0] == raw[matched, target_index, 1]
    train = raw[fit, target_index, :2]
    count = np.bincount(train[np.isin(train, [1, 2, 3])].astype(int), minlength=4)
    majority = int(count.argmax())
    exact = compare(raw[matched[equal], target_index, 1], y[equal], majority, rng)
    exact["coverage"] = float(equal.mean())
    exact["qualified_pairs"] = len(matched)

    history = [i for i in range(len(ITEMS)) if i != target_index]
    train = raw[fit, :, :2]
    p = ((train == 1).sum(axis=(0, 2)) + .5) / (np.isin(train, [1, 2, 3]).sum(axis=(0, 2)) + 1)
    offsets = np.log(p / (1 - p))
    history_rows = raw[test][:, history, :2]
    observed = np.isin(history_rows, [1, 2, 3])
    available = eligible[test, target_index, 2] & (observed.sum(axis=(1, 2)) >= 8)
    history_rows, observed = history_rows[available], observed[available]
    if len(history_rows) < 2:
        raise ValueError("Insufficient participants with eight usable non-target responses.")
    y_binary = raw[test[available], target_index, 2] == 1
    bias = np.zeros(len(y_binary))
    for _ in range(20):
        eta = offsets[history][None, :, None] + bias[:, None, None]
        fit_p = 1 / (1 + np.exp(-eta))
        gradient = (((history_rows == 1) - fit_p) * observed).sum(axis=(1, 2)) - 2 * bias
        curvature = (fit_p * (1 - fit_p) * observed).sum(axis=(1, 2)) + 2
        bias += gradient / curvature
    baseline = offsets[target_index]
    logits = {"population": np.full(len(bias), baseline),
              "personal_intercept": baseline + bias,
              "shuffled_identity": baseline + np.roll(bias, 1)}
    transfer = {name: {"accuracy": float(((z >= 0) == y_binary).mean()),
                       "log_loss": float((np.logaddexp(0, z) - y_binary * z).mean())}
                for name, z in logits.items()}
    transfer["paired_accuracy_gain"] = compare(
        logits["personal_intercept"] >= 0, y_binary, baseline >= 0, rng)
    return {"dataset": "GSS 2010 panel, release 6", "archive_sha256": ARCHIVE,
            "scope": "exploratory unweighted panel and survey response comparison",
            "runtime_validation": False, "seed": SEED, "target": TARGET,
            "history_items": [ITEMS[i] for i in history], "fit_people": len(fit),
            "exact_repeat_two_answers": exact, "transfer": transfer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(evaluate(parser.parse_args().archive), indent=2))

