# Registration and pilot handoff

## Mega-Study transfer preflight

Pin the publisher dataset at revision 0401b715a341ac4b5f98b4424b4aecf9d29570d0.
The release lists 19 study splits. Freeze a per-outcome allowlist from the
publisher's analysis code and questionnaire definitions. Join via publisher Twin
ID, tabulate overlaps and split counts using IDs only, then freeze prompts and a
context-capable model snapshot. Use original Waves 1–3 history. Give generic and
personalized arms the same model and target questionnaire, with only the profile
differing. Retain the full history without truncation.

The proposed primary endpoint is exact-match accuracy on explicitly binary,
publisher-defined outcomes. Exclude continuous and ordinal scales, free text,
missing/unshown targets and administrative fields. Compute each participant's
mean over their matched outcomes, then average across participants. Report the
omitted outcomes and arm failures separately. If either gain is below three
points, or its multiplicity-adjusted paired interval includes zero, report that
the test missed its success criterion and keep the new-question claim closed.
Public inputs may have appeared in training. Document the model cutoff when
available. No target scoring before authenticated OSF registration.

## One-shot same-item replication

Register the 1,166 reserved Wave 4 participants as one test set. Freeze the
original gate and 18-item list and the original development-derived majority.
Require 20% coverage, a five-point gain over matched majority, and positive
paired 95% interval lower bound. Also report whether the gain over random
selection shrinks by half or more relative to the exploratory 7.7-point control.
The original pricing-heavy item mix limits domain conclusions even if it passes.

## Owner pilot and stopping rule

Recruit 3–5 consenting owners after ethics review. Collect five days of low-risk
traffic without third-party email capture. Keep scripted cases out of the real
traffic denominator. Require at least three evaluable owners with ten eligible
real queries each, median memory coverage of 20% or more, median active review
time of at most five minutes per day, and willingness to continue from a majority
of evaluable owners. Require 20 completed blinded audits overall and at least
90% agreement. Report counts beside every rate, per owner and aggregate.
A consequential false match is an autonomous approval grounded in an earlier
decision when a material condition (amount, timing, recipient, permission or
required evidence) changed and the owner confirms the supported action changed.
Any such case stops autonomous use in that category and triggers owner follow-up.
Withdrawal or unexpected disclosure stops that participant's collection.

## Agent builder interviews

Ask 3–5 builders to describe a recent case requiring owner preference or
authorization, their current workaround, and the last time it failed. Ask what
a verifier would need, how they would integrate it, and who would own the
integration. Record concrete commitments separately: a test integration with a
date, a volunteer for a technical trial, or an introduction. If nobody commits
to a technical integration, keep delegation positioned as a reference
implementation. Recruit and interview only after agreeing on participants.

## External review and release

Provide an independent reviewer the scope of concurrent challenge consumption,
revocation timing, canonicalization, replay, signed action binding and tenant
isolation. Ask for written findings and retests. Internal tests are engineering
evidence. Submit the results bundle to OSF after contributor access and the
registrations are confirmed. Keep the null cross-question result alongside the
positive exact-repeat result, and label the truncated Qwen run exploratory.

## Validity envelope evidence rule

A future envelope should bind question family, evaluator and model version,
prompt hash, population, time range, evidence IDs, baseline, empirical interval,
and permitted disposition. Return an out-of-envelope referral for unknown
or expired cells. Keep empirical evaluation bounds separate from guarantees:
randomized audit weighting addresses known sampling probabilities, while
nonresponse, adaptive policy changes and population drift need separate
measurement. Existing survey outcomes do not authorize any real action.
