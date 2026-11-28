"""Matched direct-history baseline on exploratory BehaviorBench validation cases."""
import argparse
import gzip
import hashlib
import json
import math
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


def tokens(base, text, special=False):
    return post(base, "/tokenize", {
        "content": text, "parse_special": special, "add_special": False,
    })["tokens"]


def binary_probability(base, prompt_tokens, ids, bias):
    # An equal additive bias preserves the odds between the two tokens.
    # Suppress other tokens numerically and verify their residual mass.
    response = post(base, "/completion", {
        "prompt": prompt_tokens, "n_predict": 1, "temperature": 1.0,
        "samplers": ["temperature"], "n_probs": 2, "post_sampling_probs": True,
        "logit_bias": [[token, bias] for token in ids.values()],
        "cache_prompt": False, "seed": 42,
    })
    steps = response.get("completion_probabilities", [])
    if response.get("truncated") or len(steps) != 1:
        raise ValueError("Incomplete binary score.")
    probabilities = {entry["id"]: entry["prob"] for entry in steps[0]["top_probs"]}
    if set(probabilities) != set(ids.values()):
        raise ValueError("Backend did not return both answer-token probabilities.")
    yes, no = (probabilities[ids[label]] for label in ("yes", "no"))
    if not all(math.isfinite(p) and 0 < p < 1 for p in (yes, no)) or abs(yes + no - 1) > 1e-5:
        raise ValueError("Binary answer distribution is invalid.")
    return yes / (yes + no)

def evaluate(path, base, limit, predictions=None):
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
    lengths, brier, log_losses = [], [], []
    pieces = {label: tokens(base, label) for label in ("yes", "no")}
    if any(len(piece) != 1 for piece in pieces.values()) or pieces["yes"] == pieces["no"]:
        raise ValueError("Each answer must have its own single token.")
    ids = {label: piece[0] for label, piece in pieces.items()}
    bias_delta = None
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
        prefix = tokens(base, prompt, special=True)
        n = len(prefix)
        lengths.append(n)
        if n + 16 > 8192:
            counts["over_context"] += 1
            continue
        for label, token in ids.items():
            if tokens(base, prompt + label, special=True) != prefix + [token]:
                raise ValueError("Answer token changes at the prompt boundary.")
        try:
            p_yes = binary_probability(base, prefix, ids, 100)
            if bias_delta is None:
                bias_delta = abs(p_yes - binary_probability(base, prefix, ids, 80))
                if bias_delta > 1e-4:
                    raise RuntimeError(f"Binary probability drift under equal token bias: {bias_delta:.8f}")
        except ValueError:
            counts["invalid"] += 1
            continue
        counts["scored"] += 1
        if predictions is not None:
            predictions.write(json.dumps({
                "sample_id": row["sample_id"],
                "p_yes": p_yes,
                "p_no": 1 - p_yes,
                "pooled_p_yes": float(pooled[i]),
                "frequency_p_yes": float(x[i, 2]),
            }) + "\n")
        truth = int(labels[i])
        brier.append((p_yes - truth) ** 2)
        log_losses.append(-math.log(p_yes if truth else 1 - p_yes))
        scores["llm"] += int((p_yes >= .5) == truth)
        scores["pooled"] += int((pooled[i] >= .5) == labels[i])
        scores["frequency"] += int((x[i, 2] >= .5) == labels[i])
    return {"design": "exploratory mixed-history validation, one case per wallet",
            "model": "Qwen2.5-0.5B-Instruct Q4_K_M", "context_limit": 8192,
            "counts": counts, "correct": scores, "max_prompt_tokens": max(lengths, default=0),
            "llm_brier": sum(brier) / len(brier) if brier else None,
            "llm_log_loss": sum(log_losses) / len(log_losses) if log_losses else None,
            "equal_bias_probability_delta": bias_delta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:18983")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--predictions", type=Path, help="Write local matched scores as JSON lines")
    args = parser.parse_args()
    if args.predictions:
        with args.predictions.open("x", encoding="utf-8") as destination:
            result = evaluate(args.validation, args.base, args.cases, destination)
    else:
        result = evaluate(args.validation, args.base, args.cases)
    print(json.dumps(result, indent=2))
