"""Freeze and score the official BehaviorBench mixed-history comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from behaviorbench_pilot import fit, load, sigmoid

HASHES = {
    "train": "4269b0b843bcea344fea412a410de59e63bb961493ef057911ccb5bdee5a9e34",
    "validation": "9cf5d6a4af3763edc69e0d517625535abcefed620636578b74bae91124da0a7e",
    "test": "27b541d3da3267fb999ffd73591ffec28672e3c5df1ee62fa95a3a22afd119f2",
}
SEED = 20261128
COVERAGES = (.10, .15, .20, .25, .50, 1.0)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def code_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("behaviorbench_official.py", "behaviorbench_pilot.py")}


def data(folder, split):
    path = folder / f"{split}.jsonl.gz"
    if digest(path) != HASHES[split]:
        raise ValueError(f"Archive hash mismatch for {split}.")
    x, y, wallet, _, _, _, _, _, seq, mixed = load(path, split)
    return x, y, wallet, seq, mixed


def predict(x, sequence, model):
    return {"pooled": sigmoid(x @ np.array(model["weights"])),
            "recent_frequency": x[:, 3], "all_history_frequency": x[:, 2],
            "markov": np.asarray(model["markov"])[sequence]}


def freeze(folder):
    x, y, _, seq, _ = data(folder, "train")
    n2 = np.bincount(seq % 4, minlength=4)
    y2 = np.bincount(seq % 4, weights=y, minlength=4)
    backoff = (y2 + 10 * y.mean()) / (n2 + 10)
    n3 = np.bincount(seq, minlength=8)
    y3 = np.bincount(seq, weights=y, minlength=8)
    markov = (y3 + 10 * backoff[np.arange(8) % 4]) / (n3 + 10)
    model = {"weights": fit(x, y).tolist(), "markov": markov.tolist()}
    vx, vy, vw, vs, mixed = data(folder, "validation")
    probs = predict(vx, vs, model)["pooled"]
    confidence = abs(probs[mixed] - .5)
    cuts = {str(c): float(np.quantile(confidence, 1 - c)) for c in (.10, .25)}
    observed = {}
    for target, threshold in cuts.items():
        keep = mixed & (abs(probs - .5) >= threshold)
        observed[target] = {"cases": int(keep.sum()), "coverage": float(keep.sum() / mixed.sum()),
                            "error": float(np.mean((probs[keep] >= .5) != vy[keep]))}
    return {"archives": HASHES, "code": code_hashes(), "model": model, "seed": SEED,
            "numpy": np.__version__, "cutoffs": cuts, "validation": observed,
            "train_cases": len(y), "validation_mixed_wallets": len(set(vw[mixed])),
            "population": "mixed history with at least 25% minority-side choices",
            "gates": {"risk_caps": {"0.1": .15, "0.25": .20},
                      "minimum_gain_at_25": .01, "noninferiority_margin_at_10": .02,
                      "primary_comparator": "recent_frequency", "bootstrap_draws": 4000,
                      "paired_interval_level": .975},
            "llm": "unrun: no capable matched backend in this evaluation"}


def interval(errors, selected, baseline_errors, baseline_selected, wallet, quantiles):
    _, ids = np.unique(wallet, return_inverse=True)
    n = len(set(ids))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(n, size=(4000, n))
    def boot(values, mask):
        size = np.bincount(ids[mask], minlength=n)[draws].sum(axis=1)
        total = np.bincount(ids[mask], weights=values[mask], minlength=n)[draws].sum(axis=1)
        return np.divide(total, size, out=np.full(len(size), np.nan), where=size > 0)
    a = boot(errors, selected)
    values = a if baseline_errors is None else boot(baseline_errors, baseline_selected) - a
    return np.nanquantile(values, quantiles).tolist()


def cohort(probs, y, wallet):
    ties = np.random.default_rng(SEED).random(len(y))
    orders = {key: np.lexsort((ties, -abs(p - .5))) for key, p in probs.items()}
    errors = {key: (p >= .5) != y for key, p in probs.items()}
    rows = {}
    for c in COVERAGES:
        masks = {}
        for key, order in orders.items():
            mask = np.zeros(len(y), dtype=bool)
            mask[order[:round(c * len(y))]] = True
            masks[key] = mask
        error = {key: float(errors[key][mask].mean()) for key, mask in masks.items()}
        row = {"cases": int(masks["pooled"].sum()), "errors": error}
        if c in (.10, .25):
            row["pooled_reduction_vs_recent"] = error["recent_frequency"] - error["pooled"]
            row["paired_ci97_5"] = interval(errors["pooled"], masks["pooled"],
                errors["recent_frequency"], masks["recent_frequency"], wallet, [.0125, .9875])
        rows[str(c)] = row
    return {"cases": len(y), "wallets": len(set(wallet)), "matched": rows}


def score(folder, frozen):
    if frozen["code"] != code_hashes() or frozen["archives"] != HASHES or frozen["numpy"] != np.__version__:
        raise ValueError("Frozen inputs do not match this scorer.")
    x, y, wallet, seq, mixed = data(folder, "test")
    p = predict(x, seq, frozen["model"])
    primary_p = {name: value[mixed] for name, value in p.items()}
    primary = cohort(primary_p, y[mixed], wallet[mixed])
    secondary = cohort(p, y, wallet)
    error = (primary_p["pooled"] >= .5) != y[mixed]
    deployed = {}
    for target, cutoff in frozen["cutoffs"].items():
        keep = abs(primary_p["pooled"] - .5) >= cutoff
        deployed[target] = {"coverage": float(keep.mean()), "answered": int(keep.sum()),
                            "abstentions": int((~keep).sum()),
                            "error": float(error[keep].mean()) if keep.any() else None,
                            "upper95": interval(error, keep, None, None, wallet[mixed], [.95])[0]
                            if keep.any() else None}
    at10, at25 = (primary["matched"][key] for key in ("0.1", "0.25"))
    gates = frozen["gates"]
    criteria = {
        "25pct_gain": at25["pooled_reduction_vs_recent"] >= gates["minimum_gain_at_25"]
                      and at25["paired_ci97_5"][0] > 0,
        "10pct_noninferiority": at10["paired_ci97_5"][0] > -gates["noninferiority_margin_at_10"],
    }
    for key, stats in deployed.items():
        criteria[f"risk_cap_{key}"] = stats["error"] is not None and stats["error"] <= gates["risk_caps"][key]
    return {"primary": primary, "secondary": secondary, "validation_selected_cutoffs": deployed,
            "criteria": criteria, "passed": all(criteria.values()), "llm": frozen["llm"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.folder)
        with args.manifest.open("x", encoding="utf-8") as destination:
            json.dump(result, destination, indent=2)
            destination.write("\n")
    else:
        if not args.output:
            parser.error("score requires a new output file")
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        with args.output.open("x", encoding="utf-8") as destination:
            result = score(args.folder, frozen)
            json.dump(result, destination, indent=2)
            destination.write("\n")
