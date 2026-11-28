# BehaviorBench official comparison protocol draft

## Population and data

Use the publisher Belief Direct split at revision
b89f0c73410bc94df1906077895dc2f8b7fd0096. The official test archive SHA-256 is
27b541d3da3267fb999ffd73591ffec28672e3c5df1ee62fa95a3a22afd119f2.
Fit on official train data. Choose confidence cutoffs on publisher validation.
The exploratory validation results are available to the investigators.

For all arms remove visible target-event rows and exact target-question rows.
Make a case eligible only when it has a binary target and at least five
remaining choices. The primary cohort has a minority-side history share
of at least 25%. Use available history alone to determine cohort membership.
Report all eligible cases as a secondary cohort. Freeze exact inclusion
code and report matched counts, invalid outputs and context failures by arm.
Cluster inference by wallet. Claims are about wallet decisions in this
benchmark, with no claim about owner endorsement or verified personhood.

## Comparators and decision rules

Compare the original pooled logistic model with the recent-five frequency
baseline, all-history frequency, third-order Markov with backoff, and a
direct-history LLM. Keep each baseline visible. Do not choose a comparator
after scoring test. The prespecified primary comparator is recent-five
frequency. At 25% matched coverage, require at least one percentage point
lower error and a paired clustered lower confidence bound above zero.
At 10% coverage, require the lower bound of the error reduction to exceed
minus two points. For these two comparisons use paired wallet-bootstrap
97.5% intervals (Bonferroni familywise 95% for the two intervals).
Seed all confidence ties independently of labels. Report exact matched
curves at 10%, 15%, 20%, 25%, 50% and 100% as retrospective ranking metrics.

Use separate validation-selected deployment thresholds. Evaluation support
requires observed selective error at most 15% at the cutoff targeted to
10% validation coverage and at most 20% at the cutoff targeted to 25%.
These caps are task-specific research decisions, not deployment guarantees.
Report the achieved coverage, one-sided upper uncertainty limits, and
abstention counts. A deployable guarantee would additionally require an
independent wallet-level risk calibration design and drift assessment.

## Freeze checklist

Before registration archive model weight and runtime identifiers, hashes
of the executable and prompts, exact tokenizer/context budget, parser and
confidence scoring, train/validation/test archive hashes, and exact seed
and scoring code. Confirm the LLM produces comparable confidence scores
for both answer tokens with no truncation. Do not substitute a hard answer
for a confidence score or omit invalid predictions silently. The local
Qwen smoke run is an exploratory check of the direct-history harness.
It does not supply a calibrated official LLM comparator.

Activity regularity is a separate analysis. Direct Belief history rows
supply ordering and stance confidence with no per-history-row timestamp
or trade count. Join only pre-query timestamped Trade records if the
publisher's released schema exposes a documented cross-layer key.
Report join coverage and keep absent trade histories explicitly missing.

The exploratory local run uses Qwen2.5-0.5B-Instruct Q4_K_M and llama.cpp
b12669 on loopback, with an 8,192-token context, greedy decoding, and a
yes/no output grammar. It checks the rendered prompt length before
inference and scores the pooled and frequency baselines on exactly the
same successfully parsed cases. The pilot weights SHA-256 is
74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db.
The Windows server binary SHA-256 is
6cdac01814e24d451692c8e5b4a6950db1ed67ff0c929b197a829382032619e9.

The matched pilot scored all 16 selected validation cases, one per wallet.
Qwen and history frequency each got 7 correct. The pooled model got 8.
There were zero invalid or over-context cases. The largest rendered
prompt was 2,104 tokens. The sample is a harness check.

The Trade Direct validation archive has 114,471 cases from 1,988 wallets.
It overlaps 221 of 2,000 Belief validation wallets. The Trade targets
contain direction and amount, and the available rows lack structured
trade timestamps or block heights. An exploratory scan found no explicit
timestamp or block markers in the prompt text. The shared wallet key
alone therefore cannot enforce a pre-query activity join.

## Binary LLM confidence implementation

The local scorer uses canonical one-token yes and no continuations at
the exact same prompt boundary. It verifies tokenizer identity for both
continuations, requests one token at unit temperature, and applies the
same additive bias to both logits. Equal bias preserves their probability
ratio. It verifies that the two returned probabilities carry essentially
all the mass, normalizes them, and checks agreement at biases of 80 and
100 on the first scored case. Invalid, truncated or over-context cases
are counted explicitly.

Confidence for ranking is the larger of the two normalized probabilities.
The values are conditional on this binary response representation.
They are not calibrated estimates of correctness. The official run still
requires validation-based cutoff selection and registration. The
optional local JSON-lines export includes both probabilities and matched
baseline scores keyed by publisher sample ID.

The invariance check failed with prompt caching enabled. The scorer now
disables caching and rejects drift above 0.0001. In the two-case live
validation smoke test, both distributions normalized, the checked bias
comparison had zero observed drift, and the local score export passed
its normalization check. Both Qwen predictions were wrong. The sample
validates score extraction and is too small to assess calibration.
