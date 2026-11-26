# Mega-Study transfer test (preregistration draft)

**Question.** Does original survey history improve forecasts on genuinely new
binary outcomes? This uses Mega-Study outcomes only. The 1,166 reserved Wave 4
outcomes remain sealed for the exact-repeat replication.

**Data and split.** Pin the [Apache-2.0 Mega-Study release](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study)
and its [paper and outcome definitions](https://arxiv.org/abs/2509.19088v5).
Before examining outcomes, freeze the binary-outcome catalog, response codes
and transformations. Exclude consent, attention and administrative items,
missing responses and questions not shown. Join Twin IDs and stratify by
study and experimental condition. Among overlapping IDs in the reserve,
hash-sort with seed `anthromesh-megastudy-transfer-v1`; use the first 40% for
development and the rest for one evaluation. Require at least 200 evaluation
participants and five eligible outcomes across three studies.

**Comparisons.** Give the personalized model only original Waves 1–3 history.
Fit outcome-by-condition majority on development. Compare it with the frozen
personalized model, shuffled history, and the same LLM on the same questionnaire
with an empty profile. Lock model, context limits and prompt hashes in advance.
Reject truncation and report invalid outputs and coverage. Score participant-mean
accuracy on matched cases. Pass requires at least +3 points over both majority
and generic LLM and positive lower bounds for both paired participant bootstrap
intervals with multiplicity adjustment. Report outcome-level scores and Brier
score when probabilities are available. Use publisher results as context unless
cases and endpoints match exactly. Public data may have appeared in LLM training.

**Status.** Draft. Pin the schema and backend, verify join counts without
reading outcomes, and register the frozen plan before the first target run.
