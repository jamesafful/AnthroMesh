# AnthroMesh research release candidate

## Findings

Cross-question personal prediction did not clear its registered development
program threshold on the 200-person public holdout. The participant-mean
gain over majority was +0.18 percentage points, with a 95% interval from
−0.36 to +0.72. The truncated Qwen comparison is exploratory.

## Preregistered exact-memory replication

The [frozen protocol and scorer](https://doi.org/10.5281/zenodo.24560763) were published before
scoring the 1,166 reserved Wave 4 participants. The frozen previous-answer
policy passed all three gates: at least 20% coverage, at least +5 percentage
points over matched majority, and a strictly positive 95% interval lower
bound for that gain. Accuracy and gains are participant means. Coverage is
the share of eligible paired answers. Intervals bootstrap participants.

| Scope | Answered / eligible | Coverage | Copy accuracy (95% CI) | Matched majority | Gain (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 20,401 / 28,831 | 70.8% | 84.0% (83.3% to 84.7%) | 55.5% | +28.5 pp (+27.6 to +29.5) |
| Pricing | 19,822 / 20,988 | 94.4% | 83.8% (83.1% to 84.5%) | 54.4% | +29.4 pp (+28.4 to +30.3) |
| Non pricing | 579 / 7,843 | 7.4% | 93.1% (91.0% to 95.2%) | 94.5% | -1.4 pp (-3.5 to +0.7) |

Copy-all and expected random selection at matched counts both scored 76.4%
overall. Copying on escalated cases scored 57.9%. Within pricing, copy-all
scored 83.75% and the gate scored 83.77%. The non-pricing result comes from
one qualifying item and fell below majority. The same 18 of 34 items were
frozen, including 17 pricing items, so the replication extends the result
to more participants on the same items.

The survey policy copies one earlier answer. Runtime memory requires at
least two agreeing records. The [research validity envelopes](memory-replication-validity.json)
carry survey bounds. Runtime response bounds remain unset.

## Development-only pricing transfer

On held-out pricing categories in the pooled-scorer comparison, a personal
intercept reached 62.25% accuracy and 0.65898 log loss versus 51.89% and
0.69725 for the population prior. The personal slope comparison lost 0.3
points in its development experiment. The pooled scorer added 0.03 points
over the intercept and slightly increased log loss to 0.65919.

## Categorical survey-anchor replay

On 23 Twin-2K-500 items with at least 100 paired cases, we used only the
previously opened 200-participant pool. Earlier answers were represented by
a fixed 80%-one-hot/20%-uniform proxy. This is a real-response resampling
study with a fixed earlier-answer predictor. It does not use LLM forecasts.

At nominal 95%, mean joint coverage across items with random anchors was
94.1% for human-only, 98.1% for untuned PPI-style intervals, and 96.5% for
split power-tuned intervals. Average interval width was 0.378, 0.347 and
0.360 respectively. Using the same total number of human labels as tuning
plus estimation, human-only achieved 94.9% coverage with width 0.278.
Conservative bounded prediction-powered intervals achieved 100% coverage
with mean width 1.000, effectively the full probability range.

When anchors were selected by prior answer, tuned coverage fell to 87.9%.
When the forecast distribution shifted, it fell to 85.2%. These exploratory
results are specific to the observed pool and assumptions. They establish
neither uniform subgroup coverage nor lower data requirements. The earlier-
answer-defined subgroup mean absolute proxy error gap reached 0.111 for one
item. It is descriptive. Tuned estimates are marginal and approximate.

## Equal-budget cross-fitted survey replay

Across 23 Twin-2K-500 categorical items and 300 random draws per item,
Cross-PPI with the publisher's GPT-4.1 mini choices reached **98.8% mean
joint coverage** at nominal 95%. Mean marginal interval width was
**0.299 vs. 0.278** for human-only estimation using the identical labels.
That is about 7.4% wider, so this real-LLM arm missed the efficiency bar.
The model files contain discrete predictions, encoded here as one-hot
vectors. These are estimates for the previously opened finite respondent
pool with disjoint samples drawn without replacement.

On the earlier-answer proxy, Cross-PPI reached 99.3% joint coverage with
width 0.268 versus 0.278 at the same label budget. The proxy result
illustrates the role of predictiveness. With publisher LLM predictions,
joint coverage dropped to 45.6% under selected anchors and 78.3% under
forecast drift. These deliberate assumption violations establish no
deployment guarantee.

An exploratory second survey used a public GSS panel trust item with 1,221
eligible respondents and 183 labels per draw. Local Qwen2.5-0.5B returned
the same choice for all three prior-answer profiles. Cross-PPI attained
99.0% coverage with width 0.163 versus 98.3% and 0.160 for human-only.
Both real-LLM comparisons missed the equal-budget width target. Keep
human-only estimation as the reference for these settings. This GSS pool
contains zero eligible respondents with two earlier answers to the trust
item. See the [GSS replay](cross-ppi-gss-results.json).

A timestamped reanalysis on the same GSS trust item used a probabilistic
classical model conditioned on prior trust, baseline age band, and survey
cohort. It was fitted on 611 separate labeled participants and evaluated
on 610 held-out participants, with 92 evaluation labels per draw. Cross-PPI
achieved **99.0% coverage and 0.203 mean width** versus **98.3% and 0.220**
for human-only. Brier score was 0.178. The widths were about 7.7% narrower
at equal evaluation-label budget. Model training labels are an additional
cost if an existing trained predictor is unavailable. The item and data
source were previously opened, so this is a fixed reanalysis rather than
an untouched validation. See the [plan](gss-probabilistic-reanalysis-plan.md)
and [results](gss-probabilistic-reanalysis-results.json).


Reproduce from the pinned source files with
`python examples/survey_anchor_replay.py . DATASET --published-llm`.
The aggregate output is in [the LLM replay results](cross-ppi-published-llm-results.json).
See [Cross-Prediction-Powered Inference](https://doi.org/10.1073/pnas.2322083121)
and the [Twin-2K-500 data release](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).

## Independent non-pricing survey evidence

An exploratory replay on the separate 2010 GSS three-wave panel used
the environment-spending question as the target. It was an exploratory
choice from the available spending battery. No preregistered item-selection
rule supported that single-item analysis. Among 302 evaluation
participants answering all three waves, 200 had two agreeing prior
answers. Copying those answers reached **77.5% vs. 69.0%** for an
independent participant majority on the same cases, at **66.2%** coverage.
The gain was +8.5 percentage points (95% participant bootstrap interval
+1.5 to +16.0).

In a separate binary transfer comparison, the personal intercept used
prior responses about other spending categories and excluded the target
from the person's history. On 310 respondents it achieved **66.8% vs.
59.7%** for the population prior and **59.7%** for shuffled identities.
Log loss was 0.618 versus 0.674 for the population prior. The +7.1 point
gain had a 95% interval of +2.3 to +12.3. The original pricing pipeline's
eight-history minimum and intercept penalty were retained. The survey
target and category coding were specific to GSS.

These observations span years, so the runtime's recency and context
requirements were outside the survey replay. The outcome is one
unweighted panel item, and these results are exploratory. Reproduce with
the [non-pricing replay script](../examples/gss_non_pricing_replay.py)
and [NORC's public panel release](https://gss.norc.org/get-the-data/stata.html).
See the [aggregate results](gss-non-pricing-results.json).

## Preregistered independent GSS replication

The [frozen replication](https://doi.org/10.5281/zenodo.24562917) used early
2006 panel responses for population priors and the separate 2008 panel for
evaluation. All 20 prespecified spending and social-attitude items passed
the minimum-case requirement. The participant-mean gain of two-answer
memory over matched majority was **+19.5 percentage points** (95% interval
**+18.2 to +20.9**). All 20 point estimates were positive, from about +1 to
+43 points. The per-item intervals and coverage are in the [full results](gss-registered-replication-results.json).
The following are descriptive, equally weighted item gains by 2006 baseline
majority share.

| Baseline majority share | Items | Mean gain |
| --- | ---: | ---: |
| under 70 | 18 | +21.4 pp |
| 70 to under 90 | 2 | +5.2 pp |
| 90 or higher | 0 | — |

On the reserved environmental-spending target, the one-parameter personal
intercept reached **61.8%** versus **56.1%** for the population prior on 408
matched people. The +5.6 point gain had a 95% interval of +2.7 to +8.8.
The same scoring rule fed only unrelated response history reached 56.4%.
Shuffled identity reached 55.9%. The within-family model beat both controls
with positive bootstrap lower bounds and improved log loss from 0.724 to
0.669. That supports transfer within this spending battery, with a
response-style check limited to the selected unrelated items.

The GSS survey schedules determine whether the 90-day rule is feasible.
These waves are years apart and cannot yield two prior records inside
the runtime's 90-day window. The Twin-2K-500 replication supplies only one
earlier response per repeated survey item. Those datasets therefore supply
no qualifying estimates of live fast-path accuracy. Runtime envelopes leave
accuracy and interval fields unset. The earlier Twin-2K-500 non-pricing
item had a 94.5% matched majority, leaving only 5.5 points of possible gain.

## Operational scope

The memory-only route rejects autonomous approval for structured query
metadata and sends the case to the owner. The MCP client now preserves that
metadata. Security checks are internal; independent review is pending.
Usability, demand, and real-world false approvals remain unmeasured.

The aggregate accounting file documents 2,058 participant roles, allocation
seeds, and membership hashes. Raw response records and the local membership
ledger are excluded. The full-context Mega-Study transfer test remains pending.

## Sources

- [Twin-2K-500 release](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
- [Prediction-powered inference](https://arxiv.org/abs/2301.09633)
- [PPI++](https://arxiv.org/abs/2311.01453)
