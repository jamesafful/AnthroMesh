# Twin-2K-500 validation protocol

The earlier 60 evaluation and 32 calibration participants were exploratory. The
original case set is reused only for diagnostics and publisher-matched language
model comparisons. The dataset snapshot and question columns are inherited from
the original report. Evaluation remains retrospective on public data.

From the remaining participants, the SHA-256 order for
`anthromesh-public-validation-v2` assigns the first 200 to development, the next
200 to held-out evaluation, and the remainder to the reference population.
Development is split evenly between model selection and confidence calibration.
Reference choices come from Waves 1–3. Predictors receive a participant's
earlier responses only on other single-select question columns. The earlier
answer to the exact item is scored separately as a retest control. All
development and holdout outcomes come from Wave 4.

The statistical model starts with an earlier-wave population distribution for
each item. It can adjust that prior based on the earlier choices of reference
participants with similar cross-question histories. Development selects
neighbor count and strength from a fixed grid that includes zero adjustment.
Selection uses participant-mean accuracy, with ties favoring the smaller
adjustment. An independent development split calibrates confidence and freezes
the operating point targeting 25% coverage.

The predictive claim passes only if the frozen model improves participant-mean
accuracy over majority by at least three percentage points, and the lower
bound of a participant bootstrap 95% interval for the paired difference is
above zero. Report coverage and selective accuracy at 10%, 25%, and 50%, plus
the operating point frozen during development. Evaluate model behavior with
shuffled histories, the score oracle, and the copy-earlier-answer control.

For cohort forecasts, report Jensen–Shannon divergence and minority precision
and recall, including a same-wave split-half sampling reference, temporal
shift, and the publisher's persona-based predictions on aligned cases.
Public data and deterministic splits cannot rule out prior dataset access.
