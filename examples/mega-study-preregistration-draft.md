# Mega-Study transfer test (preregistration draft)

**Question.** Does original survey history improve forecasts on genuinely new
binary outcomes? This uses Mega-Study outcomes only. The 1,166 Wave 4 participants were used for the registered memory replication.
Their distinct Mega-Study outcome namespace remains unscored.

**Data and split.** Pin the [Apache-2.0 Mega-Study release](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study/tree/0401b715a341ac4b5f98b4424b4aecf9d29570d0)
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

## Execution requirements

Build the join from participant IDs and the questionnaire-only metadata
before reading target labels. Freeze the exact dataset revision, selected
outcomes and full prompt assembly. Measure token coverage with the chosen
backend's tokenizer, including its chat template and completion budget.
A response with server-reported truncation or an input that would exceed
the verified context window is excluded and counted. The previously used
local pilot model ran with short context and provides no well-powered
full-profile result. No Mega-Study outcomes were opened for this draft.

## Questionnaire inventory preflight

The pinned release contains 19 study folders with parallel QSF questionnaire
and response files. The metadata-only inventory retrieved 10 questionnaires
and found 154 two-option multiple-choice fields. The nine unresolved downloads
are listed explicitly. Candidate fields can include administrative checks,
repeated displays, or condition-specific variants. The 154 are raw field IDs,
so they cannot be treated as eligible study endpoints.

The inventory script reads QSF metadata only. Participant answers were not
opened. The remaining prerequisites are completing the inventory, mapping
candidate fields to paper-defined endpoints and conditions, projecting PID
alone for join counts, and freezing a full-context backend before registering
the protocol. The public checkout does not contain the reserve membership
ledger used by the earlier memory replication.
