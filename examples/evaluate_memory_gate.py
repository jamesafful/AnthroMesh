"""Prespecified validation of memory on an identical repeated question."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from anthromesh.benchmarks.public_twin import _rank
from anthromesh.benchmarks.public_validation import _choices, _means, load_public_validation
from anthromesh.benchmarks.study_report import _interval

SEED = "anthromesh-repeated-memory-v1"


def evaluate(folder: Path, original: Path) -> dict:
    data = load_public_validation(folder, original)
    available = sorted(data.reference, key=lambda pid: _rank(SEED, pid))
    development, holdout = available[:200], available[200:400]
    if len(holdout) != 200:
        raise ValueError("The registered split requires 400 unused Wave 4 outcomes.")
    dev_early = _choices(data, development, data.questions, data.earlier)
    dev_late = _choices(data, development, data.questions, data.later)
    paired = (dev_early >= 0) & (dev_late >= 0)
    count = paired.sum(axis=0)
    stability = (1 + ((dev_early == dev_late) & paired).sum(axis=0)) / (count + 2)
    approved_columns = (count >= 30) & (stability >= .80)
    priors = [
        np.bincount(column[column >= 0]).argmax()
        for column in dev_early.T
    ]
    early = _choices(data, holdout, data.questions, data.earlier)
    late = _choices(data, holdout, data.questions, data.later)
    eligible = (early >= 0) & (late >= 0)
    approved = eligible & approved_columns
    correct = early == late
    majority = late == priors
    differences = _means(correct.astype(int) - majority.astype(int), approved)
    interval = _interval(differences, draws=5000, seed=int(_rank(SEED, "bootstrap")[:8], 16), tail=.025)
    total = int(eligible.sum())
    accepted = int(approved.sum())
    coverage = accepted / total
    gain = fmean(differences) if differences else None
    person_total = eligible.sum(axis=1)
    person_approved = approved.sum(axis=1)
    person_correct = (correct & eligible).sum(axis=1)
    represented = person_approved > 0
    rng = np.random.default_rng(int(_rank(SEED, "random-control")[:8], 16))
    random_counts = rng.hypergeometric(
        person_correct[represented],
        (person_total - person_correct)[represented],
        person_approved[represented],
        size=(5000, int(represented.sum())),
    )
    random_scores = (random_counts / person_approved[represented]).mean(axis=1)
    escalated = eligible & ~approved
    item_results = []
    for j, key in enumerate(data.questions):
        scored = eligible[:, j]
        item_results.append({
            "column": key, "qualified": bool(approved_columns[j]),
            "development_pairs": int(count[j]), "development_stability": float(stability[j]),
            "evaluation_pairs": int(scored.sum()),
            "copy_accuracy": float(correct[scored, j].mean()) if scored.any() else None,
            "majority_accuracy": float(majority[scored, j].mean()) if scored.any() else None,
        })
    return {
        "protocol": SEED,
        "plan_sha256": hashlib.sha256(Path(__file__).with_name("public-memory-gate-plan.md").read_bytes()).hexdigest(),
        "available_fresh_wave4_outcomes_before_run": len(available),
        "unused_wave4_outcomes_after_run": len(available) - 400,
        "development_participants": len(development),
        "evaluation_participants": len(holdout),
        "paired_answers": total,
        "approved_answers": accepted,
        "escalated_answers": total - accepted,
        "coverage": coverage,
        "escalation_share": 1 - coverage,
        "covered_participants": len(differences),
        "approved_question_columns": [key for key, valid in zip(data.questions, approved_columns) if valid],
        "participant_mean_memory_accuracy": fmean(_means(correct, approved)) if accepted else None,
        "participant_mean_matched_majority_accuracy": fmean(_means(majority, approved)) if accepted else None,
        "participant_mean_gain": gain,
        "paired_participant_bootstrap_95pct": interval,
        "ungated_memory_accuracy": fmean(_means(correct, eligible)),
        "full_coverage_majority_accuracy": fmean(_means(majority, eligible)),
        "post_hoc_controls": {
            "participant_mean_random_matched_accuracy": float(random_scores.mean()),
            "random_matched_accuracy_central_95pct": np.quantile(random_scores, [.025, .975]).tolist(),
            "gate_minus_random_matched_accuracy": fmean(_means(correct, approved)) - float(random_scores.mean()),
            "participant_mean_copy_accuracy_escalated": fmean(_means(correct, escalated)) if escalated.any() else None,
            "participant_mean_majority_accuracy_escalated": fmean(_means(majority, escalated)) if escalated.any() else None,
            "paired_escalated_copy_minus_majority_95pct": _interval(
                _means(correct.astype(int) - majority.astype(int), escalated),
                draws=5000, seed=int(_rank(SEED, "escalated-control")[:8], 16), tail=.025,
            ),
            "qualifying_item_share": float(approved_columns.mean()),
            "items": item_results,
        },
        "passes": bool(gain is not None and gain >= .05 and coverage >= .20 and interval and interval[0] > 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("original_report", type=Path)
    options = parser.parse_args()
    print(json.dumps(evaluate(options.folder, options.original_report), indent=2))
