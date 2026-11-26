"""Survey replay on a previously used public response pool. No reserved rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from math import log, sqrt
from statistics import NormalDist, fmean

import numpy as np

from anthromesh.benchmarks.public_twin import _items, _rank, _reference_rows
from anthromesh.cohort.categorical_anchor import _cross_estimate, _estimate

REPEATS = 300
METHODS = ("human_only", "human_equal_label_budget", "fixed", "power_tuned", "cross_ppi", "bounded_human", "bounded_fixed")
SEED = 20261124


def read(path, wanted, items):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["pid"]: {key: row[key] for key in items}
                for row in csv.DictReader(stream) if row["pid"] in wanted}


def evaluate(project, source, *, published_llm=False):
    ledger_path = project / "research/participant-use-ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        used = set(ledger["groups"]["cross_question_closed_holdout"]["public_dataset_pids"])
        reserve = set(ledger["groups"]["memory_replication_reserve"]["public_dataset_pids"])
    else:
        with (source / "wave1_3_response.csv").open(encoding="utf-8-sig", newline="") as stream:
            ids = [row["pid"] for row in csv.DictReader(stream)]
        if len(ids) != 2058 or len(set(ids)) != 2058:
            raise ValueError("Unexpected early-wave ID inventory.")
        original = sorted(ids, key=lambda pid: _rank("anthromesh-public-v1", "participant", pid))
        followup = sorted(original[92:], key=lambda pid: _rank("anthromesh-public-validation-v2", pid))
        memory = sorted(followup[400:], key=lambda pid: _rank("anthromesh-repeated-memory-v1", pid))
        public_ledger = json.loads((project / "research/participant-accounting.json").read_text(encoding="utf-8"))
        if (_rank(*followup[200:400]) != public_ledger["groups"]["cross_question_closed_holdout"]["ordered_sha256"]
                or _rank(*memory[400:]) != public_ledger["groups"]["memory_replication_reserve"]["ordered_sha256"]):
            raise ValueError("Reproduced memberships do not match the public hashes.")
        used, reserve = set(followup[200:400]), set(memory[400:])
    if len(used) != 200 or used & reserve:
        raise ValueError("Unexpected participant ledger.")
    metadata = json.loads((source / "question_catalog.json").read_text(encoding="utf-8-sig"))
    catalog = _items(metadata)
    report = json.loads((project / "examples/public-twin-2k-500-report.json").read_text(encoding="utf-8"))
    keys = report["evaluation_question_columns"]
    early = read(source / "wave1_3_response.csv", used, keys)
    late = read(source / "wave4_response.csv", used, keys)
    forecast_hashes = {}
    if published_llm:
        names = ("wave4_formatted_to_catalog_mapping.json", "responses_wave4_formatted.csv",
                 "responses_llm_imputed_formatted.csv")
        mapping_path, human_path, model_path = [source / name for name in names]
        forecast_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
        mapping = {row["catalog_csv_column"]: row["formatted_column"]
                   for row in json.loads(mapping_path.read_text(encoding="utf-8-sig"))}
        published_human = _reference_rows(human_path)
        published_model = _reference_rows(model_path)
    rng = np.random.default_rng(SEED)
    results = []
    subgroup_diagnostics = []
    for key in keys:
        item = catalog[key]
        categories = [option.option_id for option in item.options]
        matched = []
        for pid in sorted(used):
            earlier, later = item.answer(early[pid]), item.answer(late[pid])
            if earlier is None or later is None:
                continue
            choice = earlier
            if published_llm:
                if key not in mapping or pid not in published_human or pid not in published_model:
                    continue
                column = mapping[key]
                try:
                    if item.answer({key: published_human[pid].get(column, "")}) != later:
                        continue
                    choice = item.answer({key: published_model[pid].get(column, "")})
                except ValueError:
                    continue
                if choice is None:
                    continue
            matched.append((earlier, later, choice))
        n, k = len(matched), len(categories)
        if n < 100:
            continue
        early_codes = np.array([categories.index(a) for a, _, _ in matched])
        observed = np.eye(k)[[categories.index(b) for _, b, _ in matched]]
        forecast_codes = [categories.index(c) for _, _, c in matched]
        proxy = np.eye(k)[forecast_codes]
        if not published_llm:
            proxy = proxy * .8 + .2 / k
        truth = observed.mean(axis=0)
        cells = [early_codes == c for c in range(k)]
        errors = [float(np.abs(proxy[mask] - observed[mask]).mean()) for mask in cells if int(mask.sum()) >= 10]
        if len(errors) >= 2:
            subgroup_diagnostics.append({"column": key, "groups": len(errors), "error_gap": max(errors) - min(errors)})
        z = NormalDist().inv_cdf(1 - .05 / (2 * k))
        anchor_size = round(.15 * n)
        tune_size = round(.15 * n)
        for scenario in ("random", "selected_anchors", "prediction_drift"):
            metrics = {name: [] for name in METHODS}
            for _ in range(REPEATS):
                permutation = rng.permutation(n)
                tuning, rest = permutation[:tune_size], permutation[tune_size:]
                if scenario == "selected_anchors":
                    rest = rest[np.argsort(early_codes[rest] != 0, kind="stable")]
                anchors, forecast = rest[:anchor_size], rest[anchor_size:]
                corrected_proxy = np.roll(proxy, 1, axis=1) if scenario == "prediction_drift" else proxy
                arms = {}
                for method in ("power_tuned", "fixed"):
                    arms[method] = [
                        _estimate(observed[anchors, c].tolist(), proxy[anchors, c].tolist(),
                                  corrected_proxy[forecast, c].tolist(), observed[tuning, c].tolist(),
                                  proxy[tuning, c].tolist(), method, z)
                        for c in range(k)
                    ]
                bounds = {"human_only": [row["human_only_interval"] for row in arms["fixed"]]}
                bounds.update({method: [row["interval"] for row in rows] for method, rows in arms.items()})
                full_labels = np.concatenate((anchors, tuning))
                folds = [list(range(i, len(full_labels), 5)) for i in range(5)]
                bounds["cross_ppi"] = [
                    _cross_estimate(observed[full_labels, c].tolist(), proxy[full_labels, c].tolist(),
                                    corrected_proxy[forecast, c].tolist(), folds, z)["interval"]
                    for c in range(k)
                ]
                bounds["human_equal_label_budget"] = [
                    _estimate(observed[full_labels, c].tolist(), proxy[full_labels, c].tolist(),
                              corrected_proxy[forecast, c].tolist(), [], [], "fixed", z)["human_only_interval"]
                    for c in range(k)
                ]
                factor = log(6 * k / .05) / 2
                anchor_radius = sqrt(factor / anchor_size)
                fixed_radius = 2 * anchor_radius + sqrt(factor / len(forecast))
                def bounded(center, radius):
                    return (max(0.0, min(1.0, center - radius)),
                            max(0.0, min(1.0, center + radius)))
                bounds["bounded_human"] = [bounded(row["human_only"], anchor_radius) for row in arms["fixed"]]
                bounds["bounded_fixed"] = [
                    bounded(row["estimate_unclipped"], fixed_radius) for row in arms["fixed"]
                ]
                for method, intervals in bounds.items():
                    covers = all(lo <= target <= hi for (lo, hi), target in zip(intervals, truth))
                    width = fmean(hi - lo for lo, hi in intervals)
                    metrics[method].append((covers, width))
            results.append({"column": key, "outcomes": k, "n": n, "scenario": scenario,
                            "anchor_count": anchor_size, "tuning_count": tune_size,
                            "metrics": {name: {"joint_coverage": fmean(x[0] for x in rows),
                                               "mean_marginal_width": fmean(x[1] for x in rows)}
                                        for name, rows in metrics.items()}})
    if not results:
        raise ValueError("No item had at least 100 aligned responses.")
    summary = {}
    for scenario in ("random", "selected_anchors", "prediction_drift"):
        rows = [r for r in results if r["scenario"] == scenario]
        summary[scenario] = {method: {
            field: fmean(r["metrics"][method][field] for r in rows)
            for field in ("joint_coverage", "mean_marginal_width")
        } for method in METHODS}
    predictor = ("publisher GPT-4.1 mini choices, encoded as one-hot vectors" if published_llm
                 else "fixed earlier-answer proxy, 80% one-hot plus 20% uniform; no LLM")
    return {"study": "finite respondent-pool resampling on previously used Twin-2K-500 participants",
            "predictor": predictor, "publisher_file_sha256": forecast_hashes,
            "seed": SEED, "resamples_per_item_scenario": REPEATS, "items": len(results) // 3,
            "nominal_joint_coverage_per_item": .95,
            "sampling_note": "Disjoint draws without replacement; human-only comparison uses anchor labels only, and tuning uses extra labels. Cross-PPI and the equal-budget human-only arm use the same pooled labels.",
            "summary": summary, "item_results": results, "subgroup_diagnostics": subgroup_diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--published-llm", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.project, args.dataset, published_llm=args.published_llm), indent=2))
