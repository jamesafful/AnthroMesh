
"""Score the frozen independent GSS studies only after verifying public registration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_registration(record_id, manifest):
    if not record_id.isdecimal():
        raise ValueError("Provide the published version-specific Zenodo record ID.")
    with urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=30) as response:
        record = json.load(response)
    if record["doi"] != f"10.5281/zenodo.{record_id}":
        raise ValueError("Registration DOI mismatch.")
    files = {file["key"] for file in record["files"]}
    for path in (manifest, Path(__file__)):
        if path.name not in files:
            raise ValueError(f"Registered artifact missing: {path.name}")
        url = f"https://zenodo.org/api/records/{record_id}/files/{path.name}/content"
        with urlopen(url, timeout=30) as response:
            if response.read() != path.read_bytes():
                raise ValueError(f"Registered artifact differs: {path.name}")
    return record["doi"]


def frame(archive, columns):
    with TemporaryDirectory() as temp, ZipFile(archive) as container:
        member, = [name for name in container.namelist() if name.lower().endswith(".dta")]
        path = container.extract(member, temp)
        with pd.io.stata.StataReader(path) as reader:
            available = reader.variable_labels()
        present = [column for column in columns if column in available]
        data = pd.read_stata(path, columns=present, convert_categoricals=False)
    return data.reindex(columns=columns).to_numpy()


def accuracy(prediction, truth, prior, rng):
    wins = prediction == truth
    baseline = truth == prior
    gain = wins.astype(float) - baseline
    draws = rng.integers(len(gain), size=(5000, len(gain)))
    return {"n": len(gain), "accuracy": float(wins.mean()),
            "majority_accuracy": float(baseline.mean()), "gain": float(gain.mean()),
            "gain_ci95": np.quantile(gain[draws].mean(axis=1), [.025, .975]).tolist()}


def personal_offsets(x, baseline, codes, minimum):
    observed = np.isin(x, codes)
    theta = np.zeros(len(x))
    for _ in range(20):
        logit = baseline[None, :, None] + theta[:, None, None]
        p = 1 / (1 + np.exp(-logit))
        gradient = (((x == 1) - p) * observed).sum(axis=(1, 2)) - 2 * theta
        curvature = (p * (1 - p) * observed).sum(axis=(1, 2)) + 2
        theta += gradient / curvature
    return theta, observed.sum(axis=(1, 2)) >= minimum


def run(folder, plan):
    for name, expected in plan["archive_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Unrecognized panel revision: {name}")
    items = plan["spending_items"] + plan["unrelated_items"]
    codes = [1, 2, 3]
    dev = frame(folder / "panel2006.zip", [f"{item}_{j}" for item in items for j in (1, 2)])
    dev = dev.reshape(-1, len(items), 2)
    test = frame(folder / "panel2008.zip", [f"{item}_{j}" for item in items for j in (1, 2, 3)])
    test = test.reshape(-1, len(items), 3)
    rng = np.random.default_rng(plan["seed"])
    report, exclusions, deltas = {}, {}, []
    for i, item in enumerate(items):
        earlier = dev[:, i].ravel()
        population = earlier[np.isin(earlier, codes)].astype(int)
        history = test[:, i, :2]
        y = test[:, i, 2]
        eligible = np.isin(history, codes).all(axis=1) & np.isin(y, codes)
        approve = eligible & (history[:, 0] == history[:, 1])
        if min(len(population), int(eligible.sum())) < plan["min_complete"]:
            exclusions[item] = "insufficient eligible cases"
            continue
        prior = np.bincount(population, minlength=4).argmax()
        rows = {"complete": int(eligible.sum()), "coverage": float(approve.sum() / eligible.sum()),
                "baseline_majority_share": float(np.mean(population == prior))}
        rows["baseline_share_bin"] = "under_70" if rows["baseline_majority_share"] < .7 else (
            "70_to_under_90" if rows["baseline_majority_share"] < .9 else "90_or_higher")
        if approve.any():
            rows.update(accuracy(history[approve, 1], y[approve], prior, rng))
            delta = np.full(len(test), np.nan)
            delta[approve] = (history[approve, 1] == y[approve]).astype(float) - (y[approve] == prior)
            deltas.append(delta)
        report[item] = rows
    if not deltas:
        raise ValueError("No qualifying multi-item observations.")
    delta = np.stack(deltas, axis=1)
    with_approval = (~np.isnan(delta)).any(axis=1)
    average = np.nanmean(delta[with_approval], axis=1)
    draws = rng.integers(len(average), size=(5000, len(average)))
    bounds = np.quantile(average[draws].mean(axis=1), [.025, .975]).tolist()

    p = ((dev == 1).sum(axis=(0, 2)) + .5) / (np.isin(dev, codes).sum(axis=(0, 2)) + 1)
    baseline = np.log(p / (1 - p))
    target = items.index(plan["tendency_target"])
    spending = [i for i in range(len(plan["spending_items"])) if i != target]
    unrelated = list(range(len(plan["spending_items"]), len(items)))
    within, within_ok = personal_offsets(test[:, spending, :2], baseline[spending], codes, 8)
    outside, outside_ok = personal_offsets(test[:, unrelated, :2], baseline[unrelated], codes, 4)
    valid = within_ok & outside_ok & np.isin(test[:, target, 2], codes)
    truth = test[valid, target, 2] == 1
    if len(truth) < plan["min_complete"]:
        raise ValueError("Insufficient shared tendency evaluation cases.")
    offsets = {"population": np.zeros(len(truth)), "within": within[valid],
               "unrelated_response_style": outside[valid],
               "shuffled": np.roll(within[valid], 1)}
    tendency = {}
    for arm, offset in offsets.items():
        logit = baseline[target] + offset
        tendency[arm] = accuracy(logit >= 0, truth, baseline[target] >= 0, rng)
        tendency[arm]["log_loss"] = float((np.logaddexp(0, logit) - truth * logit).mean())
    comparisons = {}
    within_logit = baseline[target] + offsets["within"]
    for arm in ("unrelated_response_style", "shuffled"):
        comparisons[arm] = accuracy(within_logit >= 0, truth, baseline[target] + offsets[arm] >= 0, rng)
    primary = tendency["within"]
    return {"memory_items": report, "excluded": exclusions,
            "memory_participant_mean_gain": float(average.mean()), "memory_gain_ci95": bounds,
            "memory_passes": len(deltas) >= 5 and average.mean() >= .05 and bounds[0] > 0,
            "tendency": tendency, "within_vs_controls": comparisons,
            "tendency_passes": primary["gain"] >= .03 and primary["gain_ci95"][0] > 0
            and primary["log_loss"] < tendency["population"]["log_loss"]
            and all(row["gain_ci95"][0] > 0 for row in comparisons.values()),
            "runtime_temporal_scope": "Survey waves are years apart. The 90-day fast path cannot be validated by the two-answer comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_folder", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--zenodo-record", required=True)
    args = parser.parse_args()
    doi = verify_registration(args.zenodo_record, args.manifest)
    with args.result.with_suffix(".started").open("x", encoding="utf-8") as marker:
        marker.write(doi + "\n")
    findings = run(args.dataset_folder, json.loads(args.manifest.read_text(encoding="utf-8")))
    with args.result.open("x", encoding="utf-8") as output:
        json.dump({"registration": doi, "manifest_sha256": digest(args.manifest), **findings},
                  output, indent=2, allow_nan=False)
        output.write("\n")

