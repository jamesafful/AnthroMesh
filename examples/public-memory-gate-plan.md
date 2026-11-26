# Repeated-decision memory gate

The prior 200-person holdout is closed. From participants whose Wave 4 outcomes were outside the earlier exploration, calibration, selection, and holdout sets, assign 200 to development and 200 to evaluation by SHA-256 order with seed `anthromesh-repeated-memory-v1`. Their earlier-wave answers previously contributed to population baselines. Retain the prior 34 single-select items with answers in both waves.

Forecast each person's earlier answer to the exact item. Estimate item stability on development participants with a Beta(1,1) prior. Approve memory only with at least 30 development pairs and posterior mean agreement of at least 0.80. The matched baseline is the most frequent earlier answer among development participants on those exact approved cases.

Pass criteria are at least 20% coverage, five percentage points of participant-mean gain over matched majority, and a positive lower bound of a paired participant bootstrap 95% interval (5,000 draws). Report total coverage, escalation share, covered participants, selective scores, and full-coverage memory and majority. Do not tune against the holdout. The public retrospective study cannot measure live owner burden or transfer to changed contexts.
