# BehaviorBench dense-history pilot

The pilot predicts the final revealed yes/no stance in a target market.
It uses the pinned publisher validation archive from revision
b89f0c73410bc94df1906077895dc2f8b7fd0096. This is exploratory. We split
wallets into fitting, threshold selection, and evaluation sets. The
publisher's official test split was unused. Histories are supplied by the
benchmark. We remove exact target questions and target event histories and
require five remaining choices. The learner is a pooled, regularized
logistic model using global and recent choice frequencies, last choice,
lexically similar histories, and a simple target-text feature.

| Model | Accuracy | Log loss |
| --- | ---: | ---: |
| Population majority | 48.43% | 0.6933 |
| Last visible choice | 70.82% | 0.6083 |
| Recent choice frequency | 73.76% | 0.5323 |
| All visible history frequency | 73.76% | 0.5154 |
| Target text feature only | 51.75% | 0.6931 |
| Pooled history model | 74.92% | 0.5008 |

Evaluation covers 3,704 cases from 372 wallets. At a confidence cutoff
set for nominal 50% coverage on calibration wallets, the pooled model
covered 48.89% of evaluation cases with 11.43% error. All-history
frequency covered 50.57% with 13.56% error. These are nearby observed
operating points. The pilot does not estimate uncertainty for their
difference, and abstentions estimate hypothetical interruptions.

Wallets are pseudonymous accounts and may be shared or automated.
Scores measure revealed market positions and provide no validation of
owner endorsement or general human judgment. A standard sequential
recommender and the publisher's direct-history LLM should be compared
on the same official cases, with shuffled histories and wallet-clustered
intervals before deciding whether to integrate a learned adapter.

Run with NumPy installed:

```text
python examples/behaviorbench_pilot.py PATH_TO_VALIDATION_JSONL_GZ
```

[Benchmark and data documentation](https://arxiv.org/html/2606.02798)
