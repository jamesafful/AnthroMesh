# AnthroMesh

**Preference memory, owner escalation, and verifiable delegation for AI agents.**

AnthroMesh lets agents consult recorded human preferences with consent and
provenance, ask the owner to resolve uncertain cases, and present auditable
delegation for an exact proposed action. Predictive fidelity is experimental.

1. **Personal Decision Twin (`anthromesh.twin`)** — Captures structured `(context, options_considered, chosen_action, rejected_options, rationale)` decision logs in a local-first SQLite Behavioral Vault, enforces deterministic boundary/consent policies, and supports owner escalation when the recorded preference is insufficient.
2. **Research Cohort Oracle (`anthromesh.cohort`)** — Evaluates stratified forecasts against empirical distributions and population priors.
3. **Pluggable Local + Cloud Inference Adapters (`anthromesh.adapters`)** — Unified adapter interface supporting **Ollama**, **vLLM**, **OpenAI**, **Anthropic**, **Google Gemini**, and a zero-dependency **Heuristic** adapter for offline development and CI.
4. **Protocol-Native Interfaces (`anthromesh.protocols`)** — Exposes both an **MCP Server** (JSON-RPC 2.0 over stdio and Streamable HTTP, mapping low-confidence queries to `InputRequired`) and an **A2A v1.0 Server** (`/.well-known/agent-card.json`, persisted tasks, and human resolution endpoints).
5. **Reproducible Evaluation Harness (`anthromesh.benchmarks`)** — Measures Expected Calibration Error (ECE), out-of-distribution Escalation F1, Jensen-Shannon Divergence (JSD), and Normalized Entropy against zero-shot persona baselines.

---

## Quickstart

### 1. Install Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### 2. Run the Full End-to-End Demo & Benchmarks
```bash
python examples/demo_end_to_end.py
anthromesh benchmark
```

### 3. Run the Test Suite
```bash
pytest
```

### 4. Seed Vault & Launch MCP / A2A Servers
```bash
cp .env.example .env
anthromesh init
anthromesh seed --decisions examples/sample_decisions.json --cohort examples/sample_cohort.json

# Start MCP Server (Streamable HTTP on :8081)
anthromesh serve-mcp --transport http --port 8081

# Start A2A v1.0 Server (:8082)
anthromesh serve-a2a --port 8082
```

### 5. Run via Docker Compose
```bash
docker compose up --build
```


## Local implementation status

From the repository root, install the optional features with
`python -m pip install -e ".[protocols,encryption,survey,dev]"` and run
`python -m pytest`.

### Vault and retrieval

- Set `ANTHROMESH_VAULT_KEY` in the process environment to create or open a
  SQLCipher vault. Copy an existing plaintext vault to a new encrypted file
  with `anthromesh export-encrypted source.db encrypted.db`, then point
  `ANTHROMESH_DB_PATH` at the new file. The source remains on disk.
- Run `anthromesh verify-audit --expected-head HASH` with a checkpoint stored
  outside the vault to detect changes to the recorded history.
- BM25 is the default ranker. Set `ANTHROMESH_EMBEDDING_MODEL` to use an Ollama
  embedding model with dense rank fusion. Ollama receives the query and
  candidate decision text. Dense embeddings are currently recomputed in a
  batch for each query.

### Protocol entry points

- MCP discovery uses `server/discover`. Send
  `Mcp-Protocol-Version: 2026-07-28`, `Mcp-Method`, and, for tool calls,
  `Mcp-Name` with modern HTTP requests. Retry an escalated request with the
  same arguments and the returned `requestState`. Owners can inspect the
  pending ticket with `anthromesh escalations` and resolve it with
  `anthromesh resolve`.
- The optional MCP OAuth resource server requires `ANTHROMESH_OAUTH_ISSUER`,
  `ANTHROMESH_OAUTH_RESOURCE` (an HTTPS URL ending in `/mcp`), and
  `ANTHROMESH_OAUTH_JWKS` (a local copy of the issuer's trusted public keys).
  Run the MCP listener on loopback behind a TLS reverse proxy. The issuer
  supplies `sub` matching the vault subject ID, `client_id` identifying the
  caller, and scopes such as `twin:read:calendar` or
  `cohort:simulate:example-cohort`. Authorization flows and client
  registration are handled by the external authorization server.
- Discover the A2A JSON-RPC interface at `/.well-known/agent-card.json`.
  Send `SendMessage` to `/a2a` with a structured part whose data contains
  `skill` and `input`. Supported skills are `twin_decision` and
  `cohort_simulation`. Task reads, listing, cancellation, and streaming use
  the same endpoint. Run the A2A listener on a trusted local host.

To use local capability tokens, start MCP HTTP with `--allow-capability-tokens`.
Issue one with `anthromesh issue-capability --subject u1 --client agent1 --scope twin:read:calendar`
and send the returned secret as an Authorization bearer token. Use
`anthromesh revoke-capability TOKEN_ID` to revoke by the public ID.

Use `anthromesh.client.DecisionClient` for a backend MCP integration.
`langgraph_node(client)` reads the graph's `query` state and writes a
`decision` value. With the `agent-sdk` extra, build an OpenAI Agents SDK
tool with `openai_agents_tool(client, subject_id="u1", domain="calendar")`.
An `escalate` result is a pending owner decision. Keep its request state
with the original arguments if your application needs to resume it.

### Capture, calibration, and research

- `anthromesh import-decisions github-review reviews.json --subject u1 --account login`
  imports submitted GitHub review webhook payloads for that account. A payload
  can include a top-level `diff_summary`. To receive signed deliveries, set
  `ANTHROMESH_GITHUB_WEBHOOK_SECRET` and run
  `anthromesh serve-github-capture --subject u1 --account login --repository owner/repo`.
  The capture listener accepts `POST /webhooks/github` on loopback.
- `anthromesh import-decisions google-calendar events.json --subject u1` imports
  the signed-in attendee's response from a local Google Calendar events file.
  The event update timestamp may differ from the original RSVP timestamp.
- Set `ANTHROMESH_INBOX_PASSWORD` and run `anthromesh serve-inbox` to review
  and resolve tickets from the local browser inbox. The login is `owner`.
- `anthromesh import-survey survey.csv --id example --domain research --trait region --response answer`
  estimates response priors for observed trait combinations. Repeat
  `--trait` to specify more columns. Optional `--weight` takes the column
  name for survey weights. Optional `--margins` takes a JSON file with
  proportions such as `{"region":{"north":0.4,"south":0.6}}`.
- Run `anthromesh fit-calibration held-out-decisions.json` to generate profile
  JSON from held-out decisions. Each group uses only vault history before its
  earliest held-out decision. Set `ANTHROMESH_CALIBRATION_PATH` to a JSON array of profiles with
  `subject_id`, `domain`, `adapter`, `model`, `alpha`, and `scores`.
  Supply scores equal to one minus the normalized prediction mass of the
  observed human choice on held-out cases. Keep the inference and
  retrieval settings fixed. Split conformal coverage depends on
  exchangeability.
- `anthromesh score-study personal observations.jsonl` and
  `anthromesh score-study survey observations.jsonl` report metrics by
  model backend and system. Add `--baseline SYSTEM` for paired confidence
  intervals clustered by participant or survey question. The input schemas are `PersonalObservation`
  and `SurveyObservation` in `src/anthromesh/benchmarks/empirical.py`.

Install the `interop` extra to run the official Python SDK smoke tests in
`tests/test_interoperability.py`. They cover MCP discovery and tool calls
and A2A discovery, sending a message, and retrieving the resulting task.
Full protocol conformance, dense end-to-end latency, human study results, and production fleet validation remain open.


## Extended deployment options

- **PostgreSQL and vector retrieval.** Install the `postgres` extra, create a
  database and enable `vector` in its `public` schema, then set
  `ANTHROMESH_POSTGRES_DSN`. AnthroMesh stores its application tables in the
  dedicated `anthromesh` schema by default. Set `ANTHROMESH_POSTGRES_SCHEMA`
  to override it. TLS defaults to certificate and host verification. Enable
  cached vector retrieval with `ANTHROMESH_VECTOR_INDEX=1` and
  `ANTHROMESH_EMBEDDING_MODEL`. PostgreSQL uses pgvector HNSW and SQLite uses
  the `vector` extra with sqlite-vec. Indexes hold derived copies inside the
  selected vault. Use the same Ollama model and endpoint when reopening a cache.
- **Live OAuth issuer.** Set `ANTHROMESH_OAUTH_DISCOVERY=1`,
  `ANTHROMESH_OAUTH_ISSUER`, and `ANTHROMESH_OAUTH_RESOURCE` to discover and
  refresh issuer keys over HTTPS. Use `ANTHROMESH_OAUTH_TOKEN_PROFILE=keycloak`
  with a Keycloak issuer and configure its audience and scope mappers.
  An optional `ANTHROMESH_OAUTH_CA_FILE` supplies a private CA. The bounded
  clock-skew allowance is configurable with
  `ANTHROMESH_OAUTH_CLOCK_SKEW_SECONDS` and defaults to zero. OAuth authorization
  flows and CIMD registration depend on the chosen issuer and client.
- **Microsoft Graph.** `anthromesh sync-graph calendar --subject u1 --account
  person@example.org --start ISO_START --end ISO_END` imports RSVPs from a
  delegated mailbox. `--track-reschedules` records observed organizer time
  changes. `sync-graph email --subject u1 --account person@example.org
  --drafts drafts.json` captures only sent messages that carry a matching
  `X-AnthroMesh-Draft-ID`. The manifest maps those IDs to plain-text drafts.
  Supply consented delegated access with `ANTHROMESH_GRAPH_ACCESS_TOKEN`
  or `--token-file`. Polling is optional, and the token file can be rotated.
  The implementation uses the public Microsoft Graph origin. The automated tests
  use mocked Graph responses.
- **Subject records.** `anthromesh export-subject SUBJECT` prints scoped
  records and provenance. `anthromesh erase-subject SUBJECT --confirm-subject
  SUBJECT` deletes vault-managed subject records and prevents reimport.
  Repeat `--decision-id` for targeted erasure. Erasure invalidates managed
  calibration, removes affected vector and fast evidence records, and returns
  the previous and new audit checkpoints. Use `fit-calibration --store` and
  `ANTHROMESH_MANAGED_CALIBRATION=1` to keep calibration inside the vault.
  Manage disconnected exports, backups, custom model weights, and upstream
  source records under their own retention policies.
- **Low-latency tier and ROS 2.** `ANTHROMESH_FAST_TIER=1` enables a deterministic
  lookup for unanimous, recently recorded low-risk preferences with exact
  context and options. Other queries go to deliberation. Configure the
  inference budget with `ANTHROMESH_DELIBERATION_SECONDS`. Boot a Jazzy-based
  ROS node with `anthromesh serve-ros2 --subject u1 --domain interaction`
  from a ROS environment. The node accepts `std_msgs/msg/String` JSON on
  `/anthromesh/gate/request` and publishes a correlated disposition on
  `/anthromesh/gate/result`. The request includes `request_id`, `domain`,
  `context`, `risk_tier`, and `candidate_options`. Integrate the resulting
  preference with the robot controller's independent safety checks.
  To reproduce the isolated DDS test, build `docker build -f ros2/Dockerfile
  -t anthromesh-ros .` and run `docker run --rm --network none -e
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST anthromesh-ros python3 /app/ros2_smoke.py`.

### Behavioral validation and human review

The prospective fidelity study starts with `anthromesh register-study plan.json
registration.json`. The plan names the domains and exact backend identifiers
and fixes the sample minima, coverage requirement, accuracy margin, and allowable
gap from human retest agreement. Retain the generated registration and its
digest in a timestamped registry before collecting held-out outcomes. The CLI
checks the digest locally.

Use `anthromesh check-study registration.json cases.json` before running any
forecasts. The dataset has a `source` (`synthetic` or `consented`),
`participants`, and `cases`. Participants use pseudonymous IDs and carry consent
timestamps and the explicitly approved backends. Each case includes its scenario
ID, domain, context, options, independent choice, and timezone-aware
`recorded_at` timestamp. Use `retest_of` for blinded repeat decisions. The
validator partitions each participant and domain in chronological order and
checks scenario separation and consent. The full input schema and a synthetic
fixture are in `benchmarks/study.py` and `tests/test_study_pipeline.py`.

Run `anthromesh forecast-study registration.json cases.json forecasts.json`
once per registered backend with a distinct output file for each. Set
`ANTHROMESH_ADAPTER` and `ANTHROMESH_MODEL` to the adapter and model configured
for that run. The corresponding registration entry is `adapter:model`.
Confirmatory replay uses a temporary SQLCipher vault. Remote forecasting
requires both backend-specific participant consent and `--allow-remote-model`.
Run `anthromesh evaluate-study registration.json cases.json forecasts-*.json`,
passing the expanded filenames explicitly when the shell does not expand
wildcards. The report compares the Twin with persona prompting, ungated retrieval,
and a historical population majority. It reports accuracy at matched coverage,
retention, human retest agreement, calibration, and escalation performance.
Synthetic replay is a rehearsal and cannot pass the confirmatory gate.

### Public Twin-2K-500 benchmark

The retrospective benchmark uses the official [Twin-2K-500 release by Toubia
et al.](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) under CC BY
4.0. Run `anthromesh download-public-twin DATA_DIR` to fetch the pinned files
with the publisher comparison. For manual setup, download
`question_catalog.json`, `wave1_3_response.csv`, and
`wave4_response.csv` from `question_catalog_and_human_response_csv` into one
local directory. Keep the publisher revision in `revision.txt`. The reference
run uses revision `f883165a3026fde855dfd448e0cd16443ab257b6`.

Run `anthromesh evaluate-public-twin DATA_DIR --participants 60 --seed
anthromesh-public-v1 --output report.json`. The default offline backend is
heuristic. A locally hosted language model can also be selected through the
normal adapter environment variables. The command refuses cloud adapters and
writes aggregate results without participant identifiers.

The protocol draws disjoint evaluation, calibration, and population-reference
participants with a deterministic hash. It splits paired single-select
question columns between calibration and evaluation. The Twin and persona
baseline see earlier answers to other questions. A repeat-answer baseline uses
the same participant's earlier answer to the exact question. The population
majority and Cohort Oracle use earlier answers to that question from separate
participants. The empirical distribution is also scored directly as a baseline
for the Oracle. Minority recall counts options below 10% of observed responses
when the forecast assigns at least half their observed probability.
After all forecasting is complete, one escalated case per question is resolved
through the production inbox writeback path using the published answer. These
offline resolutions exercise the workflow without changing any forecast.

For a matched publisher comparison, place
`wave4_formatted_to_catalog_mapping.json`,
`responses_wave4_formatted.csv`, and `responses_llm_imputed_formatted.csv`
in the same directory. They are under `LLM_simulation_results` and its
`GPT4.1-mini-simulation-llm-vs-human` subdirectory. The report joins by
participant and mapped question and verifies the human answer in both files
before scoring. The publisher model and AnthroMesh get different input
profiles, so this comparison measures observed performance under the two
published setups. The heuristic persona baseline does not interpret persona
text and acts as an unpersonalized control.

The original release is public. A fixed split limits inadvertent leakage from
this evaluator and cannot establish that a pretrained model has never seen
the survey. Retest agreement applies to repeated questions in this subset.
The pooled prediction set is exploratory because answers are correlated
within participants and the calibration and evaluation questions differ.

The [follow-up validation protocol](examples/public-twin-validation-plan.md)
reserves a fresh 200-participant holdout and fixes the required effect in
advance. Reproduce the controls and frozen evaluation with these commands.

```bash
anthromesh audit-public-twin DATA_DIR examples/public-twin-2k-500-report.json
anthromesh validate-public-twin development DATA_DIR examples/public-twin-2k-500-report.json frozen.json
anthromesh validate-public-twin sealed DATA_DIR examples/public-twin-2k-500-report.json frozen.json
```

The [follow-up report](examples/public-twin-validation-report.json) records
54.27% participant-mean accuracy for the population-adjusted Twin and 54.09%
for majority. The difference missed the preregistered margin. The preference
memory, escalation, and delegation scope reflects this result.

The optional `examples/compare_local_twin_model.py` runner scores Qwen2.5
0.5B through llama.cpp on the original 1,472 cases. It needs the publisher
[archived text prompts](https://huggingface.co/datasets/Mike-deng-2002/Twin-2K-500-simulations/blob/main/input/text_simulation_input.tar.gz)
saved as `publisher-prompts.tar.gz` and a local llama.cpp server on
`127.0.0.1:18191`. Run it with the data directory and the original JSON report
as positional arguments. It uses the first 1,200 characters of each archived
profile, the original catalog questions and option order, and a constrained
JSON output. The unchanged archived questionnaire exceeded the tested context
window. This is a same-case comparison with different inputs.

See the [public validation results](examples/public-validation-results.md) for the negative cross-question result and the separate exact-repeat memory experiment.

### Continuous owner audits

For ongoing audits, use `anthromesh configure-judgment settings.json` to
record a subject and domain's opt-in, sampling probability, and optional
reporting group. Enable `ANTHROMESH_JUDGMENT_MONITOR=1` on the inference server.
`anthromesh serve-inbox --subject SUBJECT` shows independent review cases without
the predicted answer. `assign-elicitation` also supports scenarios, structured
tradeoffs, counterfactuals, repeat decisions, and comparisons with a past action.
Using elicitation responses as future evidence requires separate opt-in. Random
audits are excluded from training and ordinary escalations are excluded from
randomized estimates. `anthromesh trust-map --domain DOMAIN` reports randomized
audit counts, weighted response and observed agreement, and nonresponse bounds.
Cells need at least five contributing participants and 30 answered audits.

### Verified human input and delegated authority

Set `ANTHROMESH_ORGANIZATION_POLICY_PATH` to an operator-managed policy file.
Personal Twin requests must satisfy both the organization and local policy.
A denial overrides an escalation, and an escalation requires owner review.
For PostgreSQL isolation, provision a dedicated schema and database role per
tenant, with access confined to that tenant's schema.

- `panel-enroll` accepts a short-lived identity-provider attestation from a
  configured issuer. Set a question budget and per-answer amount with
  `panel-question`. A verified participant can claim and answer through
  `serve-inbox --subject SUBJECT --panel-topic TOPIC` or the panel CLI commands.
  An independent operator reviews submitted answers. Approved answers earn a
  ledger entry, and `panel-reconcile` records an external payment reference.
  The runtime has no live verifier or automatic payment processor. The signed
  attestation verifies the account subject according to the provider, and repeat
  questions and independent review supply evidence about response quality.
- Survey ingestion can cluster categorical response features and rake against
  target margins. `grant-cohort-consent` scopes a contribution to an exact
  cohort profile, domain, archetype, options, and expiration. Cohort inference
  receives bounded aggregate counts. `evaluate-cohort-panel` compares survey-only
  and consented-twin simulations with generic persona prompting on matching
  questions that received reviewed human panel answers.
- Use `delegation-key` to provision issuer and holder keys, `issue-delegation`
  to issue scoped grants for `speak` or `commit`, and `serve-delegation` to
  publish the issuer's status endpoint. A relying party pins the issuer key,
  creates a one-time challenge with `delegation-challenge`, and supplies the
  challenge and exact proposed artifact to the grant holder. The holder uses
  `prove-delegated-action`. The relying party fetches fresh signed status and
  uses `verify-delegated-action` before accepting the action. Revocation is
  checked through the issuer. Only the relying party executes the action.

Open release gates include comprehensive protocol certification, Graph
validation against a consented mailbox, performance measurements on the
chosen dense model and deployment hardware, and evaluation using consented
human outcomes. The repository tests and synthetic fixtures are engineering
checks and provide no estimate of human fidelity.

## Evidence-aware infrastructure

Set `ANTHROMESH_MEMORY_ONLY=1` on agent-facing services for exact recorded
preferences and owner escalation. A memory miss, changed context, or expired
record bypasses experimental inference. The existing fast path still requires
recent unanimous evidence and respects policy. With a configured conformal
calibrator that disables the fast path, memory-only requests go to the owner.
Without memory-only mode, deliberative inference remains available for research.

MCP and A2A responses serialize a `validity_envelope`. Recorded exact choices
include evidence IDs. Experimental forecasts, owner referrals, and denials
have distinct statuses. Accuracy and interval fields remain null until a
matched validation result is available. The envelope grants no authority.
Existing separately signed delegation controls still govern external actions.

The [preregistered survey replication](https://doi.org/10.5281/zenodo.24560763) covered 70.8% of paired
answers and reached 84.0% accuracy versus 55.5% for matched majority on
1,166 reserved Wave 4 participants. Pricing drove the gain. The [technical
report](research/TECHNICAL-REPORT.md) gives scope, controls, and intervals.

For binary population estimates, run `anthromesh calibrate-cohort study.json`.
The JSON declares `outcome_id`, `model_revision`, `sampling_design`, an `anchors`
array with `unit`, `group`, `probability`, and binary `outcome`, and a
`predictions` array with the first three fields only. Use independent samples
from the same target population and freeze the predictor before viewing human
answers. The report includes the raw prediction mean, human-only estimate,
rectified mean, simultaneous intervals, subgroup residuals, and suppressed
small subgroups. The implementation uses conservative Hoeffding bounds and
requires representative IID samples. It does not implement survey weighting,
clustered sampling, adaptive collection, or nonresponse correction.

### Categorical human-anchor calibration

Use the existing `anthromesh calibrate-cohort study.json` command. Add
`categories` and use a complete `probabilities` mapping per record. Labeled
records carry a category ID as `outcome`. Set `method` to `fixed` or
`power_tuned`, the latter with a separate labeled `tuning` sample. Unit IDs
must be distinct across tuning, anchor and forecast sets. The output reports
per-category estimates and simultaneous asymptotic intervals across reported
categories and subgroups. Marginal corrected estimates are left unclipped and
may not compose a valid probability vector. Use the binary wrapper when the
conservative bounded method is required. No finite-sample efficiency guarantee
is made for the adaptive method.

For a loopback MCP consultation, start AnthroMesh in memory-only mode and
run `python examples/minimal_mcp_agent.py`. A demo subject with no qualifying
records will request owner review. Use `ANTHROMESH_MCP_TOKEN` only for an
endpoint configured to accept that token.

