# AnthroMesh public validation results

## Cross-question prediction

The frozen cross-question model failed the registered fidelity threshold. On
4,936 cases from 200 fresh participants it scored 54.27%, versus 54.09% for
majority. The paired gain was 0.18 percentage points (95% participant-bootstrap
interval −0.36 to +0.72 points). Majority matched the Twin at 10%, 25%, and 50%
coverage on the same selected cases. The original heuristic was similarly weak
with linked histories (35.9%) and shuffled histories (35.4%). The holdout is
closed.

The Cohort Oracle had higher Jensen–Shannon divergence and lower minority
precision than the raw earlier-wave prior, with equal minority recall. Treat
it as a cohort evaluation tool. The Qwen run with truncated profiles is
exploratory and does not establish language-model performance with full inputs.

## Identical repeated decisions

A separate, prespecified test used 200 development participants and 200
fresh evaluation participants from the former reference population. On
4,913 paired answers, the stable-item policy covered 71.1% and escalated
28.9%. It scored 85.5% versus 56.2% for majority on those approved cases. The
participant-mean gain was 29.3 points (95% interval +27.0 to +31.5).
Ungated memory scored 77.8% across all repeated items. Seventeen of the 18
approved items were product-pricing questions. This evidence applies to exact
repeats in this evaluation. Live escalation burden remains unmeasured.
Further work has 1,166 unused Wave 4 participant outcomes available, all from
people whose earlier-wave responses previously contributed to a population prior.

- [Cross-question protocol](public-twin-validation-plan.md)
- [Cross-question results](public-twin-validation-report.json)
- [Memory-gate protocol](public-memory-gate-plan.md)
- [Memory-gate results](public-memory-gate-report.json)

Source: [Twin-2K-500 by Toubia et al.](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500),
CC BY 4.0. This report contains aggregate results only.
