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
frequency covered 50.57% with 13.56% error. These are nearby operating points. See the fixed-coverage analysis below.

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

## Controls on the exploratory split

The full-coverage paired wallet bootstrap gave a +1.16 percentage-point
gain over all-history frequency (95% interval +0.10 to +2.28), with a
0.0146-nat log-loss reduction (+0.0062 to +0.0231). Against recent
frequency the same point gain had an interval of -0.16 to +2.39 points.

At exactly 50% coverage (1,852 decisions), pooled error was 11.66% and
all-history frequency error was 13.39%, a 1.73-point reduction (95% wallet-bootstrap interval +0.60 to +2.95). At exactly 25%, the pooled model
had higher error (7.78% versus 6.59%). Rankings use label-independent
seeded tie breaking. The fixed-coverage results describe a retrospective
ranking. The calibration-derived thresholds above are the deployable
operating-point comparison. Abstentions estimate hypothetical interruptions.

Shuffled donor-wallet features reduced accuracy to 53.40%. This control
shuffles the history feature vector, so the similar-question feature
reflects the donor row's own question. Dropping that feature reduced
accuracy by 0.38 points (95% interval -0.26 to +1.01), with a small log-loss
cost. A temporally filtered market prior scored 47.98%. It changed the
global prior on only 620 evaluation cases, so this does not rule out
more informative market baselines. The model's incremental accuracy
gain was concentrated in the least polarized visible-history band
(58.26% versus 55.30%). Both models scored 93.95% in the most polarized
band. Polarization is a choice-history statistic and cannot identify bots.

The official train and test files were downloaded with the pinned revision.
Their published test labels remain unused. Before scoring them, add a
strong sequence model or boosted baseline and a matched direct-history LLM
arm, lock the model and prompt configurations, and publish the endpoint
and threshold rules. A candidate for the primary endpoint is at least
one point lower error at 50% matched coverage with a positive paired
wallet-bootstrap lower bound. No runtime integration is based on this
exploratory pilot.
