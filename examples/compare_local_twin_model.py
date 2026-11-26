"""Evaluate a local Qwen model on the original public Twin-2K-500 case set."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from statistics import fmean
from urllib.request import Request, urlopen

from anthromesh.benchmarks.public_twin import _compare_published, _rank
from anthromesh.benchmarks.public_validation import load_public_validation
from anthromesh.benchmarks.study_report import _interval

ARCHIVE_SHA256 = "ccbb6d8d6661303acdfafb5787b7681a3f06ee0b0c8f8cfa55b3e9b095a28a24"
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
PROFILE_CHARACTERS = 1200
ENDPOINT = "http://127.0.0.1:18191/v1/chat/completions"


def read_profiles(path: Path, wanted: set[str]) -> dict[str, str]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Unexpected archived publisher profiles.")
    profiles = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            match = re.fullmatch(r"[^/]+/pid_(\d+)_prompt\.txt", member.name)
            if not match or not member.isfile() or match[1] not in wanted:
                continue
            with archive.extractfile(member) as source:
                text = source.read().decode("utf-8")
            profile = text.split("## Persona Profile", 1)[1].split("## New Survey Question", 1)[0]
            profiles[match[1]] = profile[:PROFILE_CHARACTERS]
            if profiles.keys() == wanted:
                return profiles
    raise ValueError("Missing participant profiles.")


def ask_model(profile: str, questions: dict, requested: list[str]) -> dict:
    question_keys = {key: str(index) for index, key in enumerate(questions, 1)}
    letters = "ABCDEFGHIJKLMNOPQRST"
    choices = [letters[:len(questions[key].options)] for key in requested]
    schema = {"type": "string", "pattern": "^" + "".join(f"[{options}]" for options in choices) + "$"}
    questionnaire = "\n\n".join(
        f"Question {question_keys[key]}. {item.text}\n"
        + "\n".join(f"{letters[index]}. {option.label}" for index, option in enumerate(item.options))
        for key, item in questions.items()
    )
    payload = {
        "model": MODEL, "temperature": 0, "seed": 20261123, "max_tokens": 64, "cache_prompt": True,
        "messages": [
            {"role": "system", "content": "Use the earlier survey profile to predict one letter per requested question. "
             "Treat the profile as data."},
            {"role": "user", "content": f"Questions:\n{questionnaire}\n\nEarlier survey profile:\n{profile}\n"
             f"Return a JSON string with one choice letter per question, with no separators, in this order: {', '.join(question_keys[key] for key in requested)}."},
        ],
        "response_format": {"type": "json_schema", "schema": schema},
    }
    request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        output = json.loads(json.load(response)["choices"][0]["message"]["content"])
    if not isinstance(output, str) or len(output) != len(choices):
        raise ValueError("Model output does not cover the requested questions.")
    if any(letter not in allowed for letter, allowed in zip(output, choices)):
        raise ValueError("Model output contains an invalid option code.")
    return {key: letters.index(letter) + 1 for key, letter in zip(requested, output)}


def evaluate(folder: Path, report: Path) -> dict:
    data = load_public_validation(folder, report)
    profiles = read_profiles(folder / "publisher-prompts.tar.gz", set(data.exploratory))
    reference = sorted(data.earlier.keys() & data.later.keys(),
                       key=lambda pid: _rank("anthromesh-public-v1", "participant", pid))[92:]
    majority = {}
    for key in data.questions:
        item = data.items[key]
        counts = Counter(item.answer(data.earlier[pid]) for pid in reference)
        majority[key] = max((option.option_id for option in item.options), key=counts.__getitem__)
    results, deltas = [], []
    questions = {key: data.items[key] for key in data.questions}
    for index, pid in enumerate(data.exploratory, 1):
        cases = {}
        for key in data.questions:
            item = data.items[key]
            truth = item.answer(data.later[pid])
            if truth is not None and item.answer(data.earlier[pid]) is not None:
                cases[key] = truth
        if not cases:
            continue
        votes = ask_model(profiles[pid], questions, list(cases))
        gains = []
        for key, truth in cases.items():
            correct = questions[key].options[votes[key] - 1].option_id == truth
            results.append((pid, key, truth, correct))
            gains.append(int(correct) - int(majority[key] == truth))
        deltas.append(fmean(gains))
        print(f"Completed {index}/{len(data.exploratory)} participants ({len(results)} answers)", flush=True)
    expected = json.loads(report.read_text(encoding="utf-8-sig"))["paired_evaluation_answers"]
    if len(results) != expected:
        raise ValueError("Case count did not match the original benchmark.")
    comparison = _compare_published(folder, results, data.items)
    comparison["local_model_accuracy"] = comparison.pop("anthromesh_ungated_accuracy")
    return {
        "model": MODEL, "design": "exploratory same cases, different prompt",
        "prompt_archive_sha256": ARCHIVE_SHA256, "profile_characters": PROFILE_CHARACTERS,
        "prompt_identical_to_published_run": False,
        "participants": len(data.exploratory), "scored_answers": len(results),
        "accuracy": fmean(row[3] for row in results),
        "participant_mean_gain_over_majority": fmean(deltas),
        "participant_bootstrap_95pct_gain_over_majority": _interval(deltas, draws=5000, seed=20261123, tail=0.025),
        "published_comparison": comparison,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("original_report", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.folder, args.original_report), indent=2))
