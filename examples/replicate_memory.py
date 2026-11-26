"""Run the frozen survey memory policy after checking a published Zenodo record."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean
from urllib.request import urlopen

import numpy as np

from anthromesh.benchmarks.public_twin import FILE_HASHES, _items, _rank
from anthromesh.benchmarks.public_validation import _means
from anthromesh.benchmarks.study_report import _interval


def digest(body):
    return hashlib.sha256(body).hexdigest()


def verify_record(record_id, freeze_bytes, runner_bytes):
    if not re.fullmatch(r"[0-9]+", record_id):
        raise ValueError("Use the numeric version-specific production Zenodo record ID.")
    endpoint = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(endpoint, timeout=20) as response:
        record = json.load(response)
    if record.get("doi") != f"10.5281/zenodo.{record_id}" or not record.get("created"):
        raise ValueError("Expected a published version-specific DOI.")
    files = {entry["key"] for entry in record["files"]}
    artifacts = {"memory-replication-freeze.json": freeze_bytes,
                 "replicate_memory.py": runner_bytes}
    if not artifacts.keys() <= files:
        raise ValueError("The public record must include the exact manifest and scoring script.")
    for name, content in artifacts.items():
        with urlopen(f"{endpoint}/files/{name}/content", timeout=20) as response:
            if response.read(len(content) + 1) != content:
                raise ValueError(f"The deposited {name} does not match the local bytes.")
    return {"doi": record["doi"], "published_record_created": record["created"],
            "record_url": f"https://zenodo.org/records/{record_id}"}


def scores(columns, earlier, later, majority, approved, draws):
    valid = (earlier >= 0) & (later >= 0) & columns
    kept = valid & approved
    matches, prior = earlier == later, later == majority
    delta = _means(matches.astype(int) - prior.astype(int), kept)
    estimate = lambda values, mask: fmean(_means(values, mask)) if mask.any() else None
    interval = _interval(delta, draws=draws, seed=int(_rank("anthromesh-repeated-memory-v1", "bootstrap")[:8], 16), tail=.025)
    return {"paired": int(valid.sum()), "answered": int(kept.sum()),
            "participants_answered": int(np.count_nonzero(kept.any(axis=1))),
            "coverage": float(kept.sum() / valid.sum()) if valid.any() else None,
            "copy_accuracy": estimate(matches, kept),
            "copy_accuracy_ci95": _interval(_means(matches, kept), draws=draws, seed=20261126, tail=.025),
            "matched_majority_accuracy": estimate(prior, kept),
            "gain": fmean(delta) if delta else None, "gain_ci95": interval,
            "copy_all_accuracy": estimate(matches, valid),
            "random_matched_copy_expected_accuracy": estimate(matches, valid & kept.any(axis=1)[:, None]),
            "escalated_copy_accuracy": estimate(matches, valid & ~kept)}


def evaluate(project, folder, freeze):
    for key, expected in freeze["source_artifact_sha256"].items():
        if digest((project / key).read_bytes()) != expected:
            raise ValueError(f"Frozen source mismatch: {key}")
    ledger = json.loads((project / "research/participant-use-ledger.json").read_text(encoding="utf-8"))
    development = ledger["groups"]["memory_development"]["public_dataset_pids"]
    reserve = ledger["groups"]["memory_replication_reserve"]["public_dataset_pids"]
    if (len(set(reserve)) != freeze["reserved_count"] or set(development) & set(reserve)
            or _rank(*reserve) != freeze["reserve_ordered_participant_ids_sha256"]
            or _rank(*development) != freeze["role_hashes"]["memory_development"]):
        raise ValueError("Frozen participant membership mismatch.")
    original = json.loads((project / "examples/public-twin-2k-500-report.json").read_text(encoding="utf-8"))
    columns = original["evaluation_question_columns"]
    if digest(json.dumps(columns, ensure_ascii=True, separators=(",", ":")).encode()) != freeze["evaluation_columns_sha256"]:
        raise ValueError("Frozen outcome inventory mismatch.")
    for name in ("question_catalog.json", "wave1_3_response.csv", "wave4_response.csv"):
        if digest((folder / name).read_bytes()) != FILE_HASHES[name]:
            raise ValueError(f"Input digest mismatch: {name}")
    items = _items(json.loads((folder / "question_catalog.json").read_text(encoding="utf-8-sig")))
    def read(name, people):
        index = {pid: i for i, pid in enumerate(people)}
        values = np.full((len(people), len(columns)), -1)
        seen = set()
        with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                pid = row["pid"]
                if pid not in index:
                    continue
                if pid in seen:
                    raise ValueError("Duplicate participant.")
                seen.add(pid)
                for j, column in enumerate(columns):
                    answer = items[column].answer(row)
                    if answer is not None:
                        values[index[pid], j] = [o.option_id for o in items[column].options].index(answer)
        if seen != set(people):
            raise ValueError("Missing participant.")
        return values
    dev = read("wave1_3_response.csv", development)
    earlier = read("wave1_3_response.csv", reserve)
    later = read("wave4_response.csv", reserve)
    prior = [np.bincount(col[col >= 0]).argmax() for col in dev.T]
    rules = freeze["replication_rules"]
    approved = np.isin(columns, rules["items"])
    pricing = np.array([column.startswith("QID9_") for column in columns])
    result = {name: scores(mask, earlier, later, prior, approved, rules["bootstrap_draws"])
              for name, mask in (("overall", np.ones(len(columns), dtype=bool)),
                                 ("pricing", pricing), ("non_pricing", ~pricing))}
    overall = result["overall"]
    passed = (overall["gain_ci95"] is not None and overall["gain_ci95"][0] > 0
              and overall["coverage"] >= rules["minimum_coverage"]
              and overall["gain"] >= rules["minimum_paired_gain"])
    return {"participants": len(reserve), "policy": "frozen previous-answer survey policy",
            "runtime_fast_tier_validation": False, "passes": bool(passed), "results": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--zenodo-record", required=True)
    args = parser.parse_args()
    raw = (args.project / "research/memory-replication-freeze.json").read_bytes()
    registration = verify_record(args.zenodo_record, raw, Path(__file__).read_bytes())
    destination = args.project / "research/memory-replication-result.json"
    marker = destination.with_suffix(".started")
    if destination.exists():
        raise FileExistsError(destination)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(registration, stream)
    result = evaluate(args.project, args.dataset, json.loads(raw))
    result["registration"] = registration
    result["manifest_sha256"] = digest(raw)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(destination)
