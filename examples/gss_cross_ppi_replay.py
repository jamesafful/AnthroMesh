
"""Exploratory real-LLM Cross-PPI replay on the public GSS panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from anthromesh.cohort.categorical_anchor import CategoricalAnchorStudy, analyze_categories

ARCHIVE = "d1ae45c793a4fc3f8bbf4af65b11762c8d814106cecd11f244a2831d2272dc6d"
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
SEED = 20261126
REPEATS = 300
OPTIONS = {1: "can trust most people", 2: "cannot be too careful", 3: "depends"}


def predict(endpoint, earlier):
    prompt = (
        f"A US survey respondent previously answered: {OPTIONS[earlier]}. "
        "Predict how they will respond in a later interview. "
        "Which best describes their answer about trust in most people? "
        "A: can trust most people. B: cannot be too careful. "
        "Return a JSON string containing only A or B."
    )
    body = {"model": MODEL, "temperature": 0, "seed": SEED, "max_tokens": 8,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "schema": {"type": "string", "enum": ["A", "B"]}}}
    request = Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        choice = json.loads(json.load(response)["choices"][0]["message"]["content"])
    if choice not in ("A", "B"):
        raise ValueError("Invalid model response.")
    return choice


def evaluate(archive, endpoint):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE:
        raise ValueError("Unexpected GSS archive revision.")
    columns = ["trust_1a", "trust_1b", "trust_2"]
    with TemporaryDirectory() as tmp, ZipFile(archive) as z:
        path = Path(z.extract("gss2020panel_r1a.dta", tmp))
        data = pd.read_stata(path, columns=columns, convert_categoricals=False)
    a, b, target = (data[col].to_numpy() for col in columns)
    history = np.where(np.isin(b, [1, 2, 3]), b, a)
    valid = np.isin(history, [1, 2, 3]) & np.isin(target, [1, 2])
    outcomes = np.where(target[valid] == 1, "A", "B")
    earlier = history[valid].astype(int)
    choices = {value: predict(endpoint, value) for value in sorted(set(earlier.tolist()))}
    forecasts = [choices[value] for value in earlier]
    n = len(outcomes)
    budget = max(10, round(.15 * n))
    if n < 100:
        raise ValueError("Insufficient eligible public panel responses.")
    rows = [dict(unit=str(i), group="panel", probabilities={"A": float(c == "A"), "B": float(c == "B")})
            for i, c in enumerate(forecasts)]
    truth = {category: float(np.mean(outcomes == category)) for category in ("A", "B")}
    draws = np.random.default_rng(SEED)
    metrics = {"cross_ppi": [], "human_only": []}
    for _ in range(REPEATS):
        indices = draws.permutation(n)
        anchors = [dict(rows[i], outcome=str(outcomes[i])) for i in indices[:budget]]
        study = CategoricalAnchorStudy(
            outcome_id="GSS trust_2", model_revision=MODEL, method="cross_ppi",
            sampling_design="exploratory finite-pool sampling without replacement",
            categories=["A", "B"], anchors=anchors, minimum_subgroup_size=n + 1,
            predictions=[rows[i] for i in indices[budget:]],
        )
        # Only the population comparison is in this benchmark family.
        marginals = analyze_categories(study)["population"]["marginals"]
        for arm, field in (("cross_ppi", "interval"), ("human_only", "human_only_interval")):
            intervals = {k: row[field] for k, row in marginals.items()}
            metrics[arm].append((all(lo <= truth[k] <= hi for k, (lo, hi) in intervals.items()),
                                 fmean(hi - lo for lo, hi in intervals.values())))
    summary = {arm: {"joint_coverage": fmean(x[0] for x in values),
                     "mean_marginal_width": fmean(x[1] for x in values)}
               for arm, values in metrics.items()}
    return {
        "dataset": "NORC GSS 2016-2020 Panel Release 1a", "archive_sha256": ARCHIVE,
        "outcome": "trust_2 coded responses 1 and 2", "nominal_joint_coverage": .95,
        "predictor": MODEL, "seed": SEED, "repeats": REPEATS, "eligible": n,
        "labels_per_draw": budget, "forecast_by_earlier_response_code": choices,
        "two_record_eligible": int((np.isin(a, [1, 2, 3]) & np.isin(b, [1, 2, 3]) & valid).sum()),
        "scope": "complete-case finite respondent pool; no survey population weighting",
        "passes": summary["cross_ppi"]["joint_coverage"] >= .95
        and summary["cross_ppi"]["mean_marginal_width"] < summary["human_only"]["mean_marginal_width"],
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18192/v1/chat/completions")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.archive, args.endpoint), indent=2))

