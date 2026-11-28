# Public survey stability map

This descriptive crosswalk reuses registered results and separates designs by
wave pair and reuse rule. It does not score new outcomes or calibrate the
production 90-day cutoff. Survey timing comes from study design, with no
respondent-level elapsed time available for these estimates.

| Dataset and wave pair | Reuse rule and timing | Item family | Eligible cases retained | Reuse accuracy | Matched majority |
| --- | --- | --- | ---: | ---: | ---: |
| Twin-2K-500, earlier waves to Wave 4 | One previous response, survey retest | pricing | 19822 / 20988 (94.4%) | 83.8% | 54.4% |
| Twin-2K-500, earlier waves to Wave 4 | One previous response, survey retest | non pricing | 579 / 7843 (7.4%) | 93.1% | 94.5% |
| GSS 2008 panel, waves 1 and 2 to wave 3 | Two agreeing responses across multi-year waves | 20 spending and attitude items | 10646 / 16471 (64.6%) | 76.1% | 55.8% |

Counts are participant-item pairs. The GSS row weights items by retained
cases for display. Its preregistered participant-mean paired gain was
+19.5 percentage points (95% interval +18.2 to +20.9). The GSS replication
file carries the full per-item table. Different questions and reuse rules
prevent estimating a decay curve from these rows.

Sources: [Twin-2K-500 replication](memory-replication-result.json) and
[GSS replication](gss-registered-replication-results.json).

## Public PR feasibility pilot

Run `python -m examples.public_review_pilot OWNER/REPO PR_NUMBER ...` from
a checkout with AnthroMesh installed. The pilot uses the reviewed commit's
single-parent delta and GitHub check conclusions completed before submission.
This approximates part of the historical review context. It omits the PR
base diff, prior reviewer–author interaction, pending checks, and checks
whose historical records are no longer returned. Results exclude identities
and raw review text.

The anonymous Flask pilot (numbers 5650–5669) found four PRs and inspected two
reviews. Both were ineligible under the pilot filters, leaving zero scored
cases. A second, 24-PR exploratory pytest sample stopped on HTTP 403, with
GitHub reporting that the anonymous rate limit had been exceeded. These
runs provide no estimate of approval coverage or accuracy.
