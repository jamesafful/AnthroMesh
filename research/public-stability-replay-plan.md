# Public stability and runtime replay protocol draft

## Survey stability

Before scoring, archive the item inventory, codebook crosswalk, dataset
hashes and participant-role ledger. Use published item labels to define
pricing, spending, social attitudes and other repeated single-select items.
Record actual measurement gaps where response timestamps establish them.
Where only fieldwork windows are available, report the interval of possible
gaps and retain a separate wave-pair cell. Keep one-prior-answer agreement
and two-agreeing-prior-answer accuracy separate.

For each family and observed gap, report eligible people, eligible cases,
matched majority, approval coverage, participant-mean accuracy, and a
participant bootstrap interval. Keep the previously opened Twin-2K and GSS
outcomes in a descriptive analysis. Use an untouched cohort for any new
confirmatory claim. A few panel intervals cannot identify a smooth decay
curve or justify replacing the current runtime cutoff.

## Public pull-request review replay

Select repositories and maintainers by event count and coverage before
scoring. Use public human APPROVED and CHANGES_REQUESTED reviews with
submission timestamps and a historical head SHA. Track COMMENTED as an
abstention rather than mapping it to approval or rejection. Exclude bots,
dismissed or ambiguous dispositions, and observations whose point-in-time
context cannot be reconstructed. Read checks only for the reviewed commit,
and include only results available by the review timestamp. Apply the same
rule to diff, test changes, and author relationship. Prefer durable public
records to mutable current PR attributes.

Freeze exact context serialization, chronological splits, and 30-, 60-,
and 90-day cutoffs before scoring. Replay the production fast gate with its
clock set to each decision timestamp. Insert owner outcomes only after
querying. Keep query fields that production treats as escalation conditions.
Report eligibility, accuracy and escalation by maintainer, and uncertainty
clustered at the maintainer level. Report changed-condition false approvals
separately and describe any handling of multiple reviews on the same PR.

Public archives support retrospective agreement measurements. Publish
aggregate results and code, with no reviewer identifiers or raw text.

### Replay input and command

Run `python examples/replay_public_reviews.py reviews.jsonl` with AnthroMesh
installed. Each JSON line must provide a unique `event_id`, a pseudonymous
`subject_id`, a timezone-aware `reviewed_at`, the serialized historical
`context`, and an `action` of `APPROVED`, `CHANGES_REQUESTED`, or
`COMMENTED`. Optional `query_metadata` is passed to the gate unmodified.
Optional `changed_condition` flags independently labeled context changes for
evaluation and stays out of the gate input.

The runner currently measures the shipping 90-day policy and reports aggregate
counts. Historical GitHub collection, alternate-cutoff sensitivity, and
maintainer-clustered confidence intervals are separate study steps. Before
building the input, define a first-review or deduplication rule for repeated
reviews on the same PR so a single PR cannot supply both supporting records.
A rate with no eligible observations is reported as null.

### Source feasibility preflight

The standardized Twin-2K-500 Waves 1–3 CSV headers contain no response time
columns. The publisher-formatted Wave 4 file exposes submission metadata.
That alone cannot establish elapsed time between item responses. Report
wave-pair cells and document the supported range of each interval.

An unauthenticated GitHub REST sample of ten closed Flask pull requests
returned one review with a submission timestamp and reviewed commit ID.
The review endpoint does not return the historical diff or checks. Before
scoring, reconstruct these separately from reviewed-commit data and
exclude cases where the original base or check timing is ambiguous.
