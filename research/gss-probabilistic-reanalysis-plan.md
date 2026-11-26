# GSS trust probabilistic reanalysis plan

This is a timestamped reanalysis of an already opened outcome. Use a seeded,
fixed half-person training and evaluation split, with 2020 target labels
used only on the training half to fit the forecast. Condition on earlier
trust, original survey cohort, and an age band split at age 50. Smooth
toward the prior trust group with ten pseudo-observations. Draw 15% of the
held-out pool as anchors 300 times, seed 20261126. Compare both methods using
exactly the same anchors. Pass requires mean width strictly below the
human-only arm and empirical joint coverage at least 94% for the nominal
95% intervals. Report Brier score and label budget. Inference concerns the
eligible complete-case pool rather than the national population. A stronger
LLM comparison requires externally scored probabilistic forecasts with a
pinned model and full context.
