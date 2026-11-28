# BehaviorBench official test protocol

This one-shot evaluation tests revealed wallet stances. The primary
population is the eligible mixed-history subset with at least 25% of
visible prior choices on the minority side. A binary target and at least
five prior choices are required after excluding the target event and
identical question. All eligible cases are secondary. No absolute cap
is applied to that secondary group.

Fit the original pooled logistic model and Markov comparator on the
publisher train split. Freeze pooled coefficients, Markov probabilities
and confidence cutoffs on publisher validation. Report recent-five and
all-history frequencies as fixed comparators. A capable direct-history
LLM comparison is unrun. The local Qwen smoke tests are excluded from
official comparisons.

The primary comparator is recent-five frequency. The pooled model must
reduce error by at least one percentage point at exactly 25% coverage,
with the paired 97.5% wallet-bootstrap lower bound above zero. At exactly
10% coverage the lower bound must exceed minus two points. Use 4,000
wallet bootstrap draws and seeded label-independent tie breaking.
Fixed-coverage rankings are retrospective. Their intervals resample
the selected rows by wallet and condition on the ranking and model fit.

The frozen validation cutoffs targeting 10% and 25% coverage must
respect observed error caps of 15% and 20%, respectively, on the primary
population. Retain these caps even if the model fails. Report actual
coverage, abstentions and the one-sided 95% bootstrap upper bound.
Pass requires all four gates. The upper bounds are descriptive and the
protocol makes no deployment guarantee.

Archive this protocol, the manifest and scorer in a signed public tag
before running the official test. The manifest binds all three publisher
archives and both Python sources (UTF-8 with normalized line endings).
The scorer writes one aggregate result to a new path. Release that result
regardless of the decision and close this dataset study.
