"""Matched direct-history baseline on exploratory BehaviorBench validation cases."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from behaviorbench_pilot import HISTORY, fit, load, sigmoid


def post(base, route, payload):
    request = Request(base + route, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def eligible_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("Use publisher validation only.")
            def target(match):
                return match[1] == row["question"] or match[2] == row["event_slug"]
            if row["label_choice"] in {"yes", "no"} and sum(
                    not target(match) for match in HISTORY.finditer(row["input_text"])) >= 5:
                prompt = HISTORY.sub(lambda match: "" if target(match) else match[0],
                                     row["input_text"])
                yield row, prompt


def evaluate(path, base, limit):
    x, labels, wallets, groups, *extra = load(path)
    train = groups < 6
    pooled = sigmoid(x @ fit(x[train], labels[train]))
    rows = list(eligible_rows(path))
    if len(rows) != len(labels):
        raise ValueError("Case alignment failed.")
    seed = "anthromesh-direct-history-v1/"
    order = sorted((i for i in range(len(rows)) if groups[i] >= 8 and extra[-1][i]),
                   key=lambda i: hashlib.sha256((seed + rows[i][0]["sample_id"]).encode()).digest())
    seen, counts, scores = set(), {"eligible": 0, "scored": 0, "over_context": 0,
                                   "invalid": 0}, {"llm": 0, "pooled": 0, "frequency": 0}
    lengths = []
    for i in order:
        if len(seen) >= limit:
            break
        row, visible = rows[i]
        if row["wallet"] != wallets[i] or (row["label_choice"] == "yes") != labels[i]:
            raise ValueError("Case alignment failed.")
        if wallets[i] in seen:
            continue
        seen.add(wallets[i])
        counts["eligible"] += 1
        prompt = ("<|im_start|>system\n" + row["system_prompt"] +
                  "\nReturn exactly yes or no.<|im_end|>\n<|im_start|>user\n" +
                  visible + "<|im_end|>\n<|im_start|>assistant\n")
        n = len(post(base, "/tokenize", {"content": prompt, "parse_special": True})["tokens"])
        lengths.append(n)
        if n + 16 > 8192:
            counts["over_context"] += 1
            continue
        response = post(base, "/completion", {
            "prompt": prompt, "parse_special": True, "n_predict": 8,
            "temperature": 0, "cache_prompt": False, "seed": 42,
            "grammar": 'root ::= "yes" | "no"',
        })
        answer = response.get("content", "").strip().lower()
        if response.get("truncated") or answer not in {"yes", "no"}:
            counts["invalid"] += 1
            continue
        counts["scored"] += 1
        scores["llm"] += int((answer == "yes") == labels[i])
        scores["pooled"] += int((pooled[i] >= .5) == labels[i])
        scores["frequency"] += int((x[i, 2] >= .5) == labels[i])
    return {"design": "exploratory mixed-history validation, one case per wallet",
            "model": "Qwen2.5-0.5B-Instruct Q4_K_M", "context_limit": 8192,
            "counts": counts, "correct": scores, "max_prompt_tokens": max(lengths, default=0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:18983")
    parser.add_argument("--cases", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.validation, args.base, args.cases), indent=2))
