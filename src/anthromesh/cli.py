from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from anthromesh.adapters.base import post_json
from anthromesh.adapters.factory import create_adapter
from anthromesh.benchmarks.empirical import score_study
from anthromesh.benchmarks.public_twin import download_public_twin, evaluate_public_twin
from anthromesh.benchmarks.public_validation import audit_original_public_run, run_public_validation
from anthromesh.benchmarks.runner import BenchmarkRunner
from anthromesh.benchmarks.replay import forecast_study, load_dataset
from anthromesh.benchmarks.study import load_registration, prepare_study, register_study
from anthromesh.benchmarks.study_report import evaluate_study
from anthromesh.capture.graph import GraphCapture
from anthromesh.capture.github_webhook import create_github_capture_app
from anthromesh.capture.workflows import calendar_responses, github_reviews
from anthromesh.cohort.clustering import build_clustered_profile
from anthromesh.cohort.twin_priors import TwinCohortConsent, grant_cohort_consent, revoke_cohort_consent
from anthromesh.cohort.ingest import build_empirical_profile, read_survey
from anthromesh.cohort.categorical_anchor import CategoricalAnchorStudy, analyze_categories, draw_anchor_units
from anthromesh.cohort.human_anchor import AnchorStudy, analyze_anchors
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.config import AnthroMeshConfig
from anthromesh.gate.conformal import ConformalCalibration, VaultConformalCalibration
from anthromesh.gate.fit import fit_profiles
from anthromesh.judgment.elicitation import Elicitation, ElicitationCase
from anthromesh.judgment.monitor import PredictionMonitor
from anthromesh.judgment.panel import HumanPanel, PanelAnswer, PanelProfile, PanelQuestion
from anthromesh.judgment.settings import JudgmentSettings, save_judgment_settings
from anthromesh.models import CohortProfile, DecisionLog, EscalationStatus, RiskTier
from anthromesh.protocols.a2a_server import A2AService
from anthromesh.protocols.capabilities import CapabilityTokens
from anthromesh.protocols.mcp_server import MCPServer
from anthromesh.protocols import delegation
from anthromesh.protocols.oauth import OAuthResource
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.twin.inbox import create_inbox_app
from anthromesh.vault.encryption import export_encrypted_vault
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.privacy import SubjectData
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault


def _remote_model(cfg: AnthroMeshConfig, adapter_name: str) -> bool:
    from urllib.parse import urlsplit

    if adapter_name == "heuristic":
        return False
    if adapter_name not in {"ollama", "vllm"}:
        return True
    endpoint = cfg.ollama_base_url if adapter_name == "ollama" else cfg.vllm_base_url
    return urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}


def _open_vault(cfg: AnthroMeshConfig) -> BehavioralVault:
    if cfg.postgres_dsn:
        from anthromesh.vault.postgres import PostgresBehavioralVault

        if cfg.encryption_key is not None:
            raise ValueError("ANTHROMESH_VAULT_KEY applies to SQLCipher; configure PostgreSQL separately.")
        return PostgresBehavioralVault(cfg.postgres_dsn, schema=cfg.postgres_schema)
    return BehavioralVault(db_path=cfg.db_path, encryption_key=cfg.encryption_key)


def _build_runtime(cfg: AnthroMeshConfig) -> tuple[BehavioralVault, PersonalDecisionTwin, ResearchCohortOracle]:
    vault = _open_vault(cfg)
    organization = (
        PolicyEngine.from_file(cfg.organization_policy_path) if cfg.organization_policy_path else None
    )
    policy_engine = (
        PolicyEngine.from_file(cfg.policy_path, organization=organization)
        if cfg.policy_path
        else PolicyEngine(organization=organization)
    )
    adapter = create_adapter(cfg)
    retriever = None
    if cfg.embedding_model:
        def embed_batch(texts: list[str]) -> list[list[float]]:
            response = post_json(
                f"{cfg.ollama_base_url.rstrip('/')}/api/embed",
                {"model": cfg.embedding_model, "input": texts},
            )
            return response["embeddings"]

        if cfg.vector_index:
            from anthromesh.vault.dense import DenseDecisionIndex

            index = DenseDecisionIndex(
                vault, f"ollama:{cfg.ollama_base_url}:{cfg.embedding_model}", embed_batch,
            )
            retriever = HybridDecisionRetriever(dense_index=index)
        else:
            retriever = HybridDecisionRetriever(embed_batch=embed_batch)
    elif cfg.vector_index:
        raise ValueError("Configure ANTHROMESH_EMBEDDING_MODEL to use indexed retrieval.")

    conformal = None
    if cfg.calibration_path and cfg.managed_calibration:
        raise ValueError("Choose vault-managed or file calibration.")
    if cfg.managed_calibration:
        conformal = VaultConformalCalibration(vault)
    elif cfg.calibration_path:
        conformal = ConformalCalibration.from_file(Path(cfg.calibration_path))

    twin = PersonalDecisionTwin(
        vault=vault,
        policy_engine=policy_engine,
        adapter=adapter,
        confidence_threshold=cfg.confidence_threshold,
        min_evidence_count=cfg.min_evidence_count,
        retriever=retriever,
        conformal=conformal,
        fast_enabled=cfg.fast_enabled,
        memory_only=cfg.memory_only,
        deliberation_seconds=cfg.deliberation_seconds,
        monitor_enabled=cfg.monitor_enabled,
    )
    oracle = ResearchCohortOracle(vault=vault, adapter=adapter)
    return vault, twin, oracle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anthromesh",
        description="AnthroMesh preference memory, owner escalation, and delegation (MCP + A2A).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize the local SQLite Behavioral Vault.")

    export = sub.add_parser("export-encrypted", help="Copy a plaintext vault to an encrypted file.")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)

    seed_p = sub.add_parser("seed", help="Seed decisions and/or cohort profiles from JSON files.")
    seed_p.add_argument("--decisions", type=str, help="Path to JSON list of DecisionLog objects.")
    seed_p.add_argument("--cohort", type=str, help="Path to JSON CohortProfile object.")

    capture = sub.add_parser("import-decisions", help="Import local workflow exports.")
    capture.add_argument("source", choices=["github-review", "google-calendar"])
    capture.add_argument("path", type=Path)
    capture.add_argument("--subject", required=True)
    capture.add_argument("--account", help="GitHub login to import from a review export.")

    graph = sub.add_parser("sync-graph", help="Capture consented Microsoft Graph decisions.")
    graph.add_argument("stream", choices=["calendar", "email"])
    graph.add_argument("--subject", required=True)
    graph.add_argument("--account", required=True)
    graph.add_argument("--start", help="Start of the calendar window as an ISO timestamp.")
    graph.add_argument("--end", help="End of the calendar window as an ISO timestamp.")
    graph.add_argument("--drafts", type=Path, help="Draft ID to plain text JSON manifest.")
    graph.add_argument("--track-reschedules", action="store_true")
    graph.add_argument("--poll-seconds", type=int)
    graph.add_argument("--token-file", type=Path, help="Optional file containing a delegated access token.")
    graph.add_argument("--reset", action="store_true")

    webhook = sub.add_parser("serve-github-capture", help="Accept signed GitHub review deliveries.")
    webhook.add_argument("--subject", required=True)
    webhook.add_argument("--account", required=True)
    webhook.add_argument("--repository", required=True, action="append")
    webhook.add_argument("--port", type=int, default=8083)

    survey = sub.add_parser("import-survey", help="Build a cohort from CSV or Parquet microdata.")
    survey.add_argument("path", type=Path)
    survey.add_argument("--id", required=True)
    survey.add_argument("--domain", required=True)
    survey.add_argument("--trait", action="append", default=[])
    survey.add_argument("--cluster-feature", action="append", default=[])
    survey.add_argument("--cluster-count", type=int)
    survey.add_argument("--twin-prior-weight", type=float, default=0.0)
    survey.add_argument("--response", required=True)
    survey.add_argument("--weight")
    survey.add_argument("--margins", type=Path, help="JSON object of trait-to-category proportions.")

    calibrate = sub.add_parser("fit-calibration", help="Score held-out human decisions for calibration.")
    calibrate.add_argument("path", type=Path, help="JSON array of held-out DecisionLog entries.")
    calibrate.add_argument("--alpha", type=float, default=0.1)
    calibrate.add_argument("--store", action="store_true", help="Store the fitted profiles in the vault.")

    registration = sub.add_parser("register-study", help="Freeze a study plan before collecting outcomes.")
    registration.add_argument("plan", type=Path)
    registration.add_argument("destination", type=Path)
    study_check = sub.add_parser("check-study", help="Validate study consent and split eligibility.")
    study_check.add_argument("registration", type=Path)
    study_check.add_argument("dataset", type=Path)
    forecast = sub.add_parser("forecast-study", help="Run one registered model and matched baselines.")
    forecast.add_argument("registration", type=Path)
    forecast.add_argument("dataset", type=Path)
    forecast.add_argument("destination", type=Path)
    forecast.add_argument("--allow-remote-model", action="store_true")
    evaluate = sub.add_parser("evaluate-study", help="Score the complete preregistered evaluation.")
    evaluate.add_argument("registration", type=Path)
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("forecasts", type=Path, nargs="+")

    download_twin = sub.add_parser("download-public-twin", help="Download the pinned public Twin-2K-500 release.")
    download_twin.add_argument("folder", type=Path)
    sample = sub.add_parser("draw-cohort-anchors", help="Draw anchor IDs from a frozen unlabeled sampling frame.")
    sample.add_argument("frame", type=Path, help="JSON array of unique unit ID strings.")
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--seed", type=int, required=True)

    anchors = sub.add_parser("calibrate-cohort", help="Estimate a binary proportion using independent human anchors.")
    anchors.add_argument("study", type=Path)
    public_twin = sub.add_parser("evaluate-public-twin", help="Evaluate the public Twin-2K-500 wave split.")
    public_twin.add_argument("folder", type=Path, help="Directory containing the catalog and numeric response CSVs.")
    public_twin.add_argument("--participants", type=int, default=60)
    public_twin.add_argument("--seed", default="anthromesh-public-v1")
    public_twin.add_argument("--output", type=Path, help="Save the aggregate JSON report to a new file.")
    audit_public_twin = sub.add_parser("audit-public-twin", help="Audit the original published case set.")
    audit_public_twin.add_argument("folder", type=Path)
    audit_public_twin.add_argument("original_report", type=Path)
    validate_public_twin = sub.add_parser("validate-public-twin", help="Develop or score the frozen population-adjusted Twin.")
    validate_public_twin.add_argument("phase", choices=["development", "sealed"])
    validate_public_twin.add_argument("folder", type=Path)
    validate_public_twin.add_argument("original_report", type=Path)
    validate_public_twin.add_argument("frozen", type=Path)

    study = sub.add_parser("score-study", help="Score held-out personal or survey study observations.")
    study.add_argument("mode", choices=["personal", "survey"])
    study.add_argument("path", type=Path)
    study.add_argument("--baseline", help="Compare against a named system with paired observations.")
    sub.add_parser("benchmark", help="Run calibration and long-tail variance benchmarks.")
    sub.add_parser("escalations", help="List pending human-in-the-loop escalation tickets.")
    verify = sub.add_parser("verify-audit", help="Verify the vault audit hash chain.")
    verify.add_argument("--expected-head", help="Audit head retained externally.")

    judgment = sub.add_parser("configure-judgment", help="Record opt-in judgment and audit settings.")
    judgment.add_argument("settings", type=Path)
    assign = sub.add_parser("assign-elicitation", help="Assign a structured scenario to an opted-in subject.")
    assign.add_argument("case", type=Path)
    pending = sub.add_parser("list-elicitation", help="List blinded scenarios and prediction checks.")
    pending.add_argument("subject")
    respond = sub.add_parser("respond-elicitation", help="Record an owner's independent response.")
    respond.add_argument("subject")
    respond.add_argument("case_id")
    respond.add_argument("--action", required=True)
    respond.add_argument("--rationale", default="")
    respond.add_argument("--switch-condition", default="")
    trust = sub.add_parser("trust-map", help="Report aggregate randomized prediction audit results.")
    trust.add_argument("--domain")

    make_key = sub.add_parser("delegation-key", help="Create an Ed25519 issuer or holder key.")
    make_key.add_argument("path", type=Path)
    issue_grant = sub.add_parser("issue-delegation", help="Issue a signed grant of authority.")
    issue_grant.add_argument("request", type=Path)
    issue_grant.add_argument("destination", type=Path)
    revoke_grant = sub.add_parser("revoke-delegation", help="Revoke a signed delegation.")
    revoke_grant.add_argument("grant_id")
    status_server = sub.add_parser("serve-delegation", help="Publish signing keys and fresh revocation status.")
    status_server.add_argument("--port", type=int, default=8085)
    for command in (issue_grant, revoke_grant, status_server):
        command.add_argument("--issuer", required=True)
        command.add_argument("--issuer-key", required=True, type=Path)
    challenge = sub.add_parser("delegation-challenge", help="Create a one-time verifier challenge.")
    challenge.add_argument("--audience", required=True)
    prove = sub.add_parser("prove-delegated-action", help="Sign the exact challenged action with a holder key.")
    prove.add_argument("grant", type=Path)
    prove.add_argument("destination", type=Path)
    prove.add_argument("--issuer-public-key", required=True)
    prove.add_argument("--holder-key", required=True, type=Path)
    prove.add_argument("--challenge", required=True)
    verify_action = sub.add_parser("verify-delegated-action", help="Verify authority and consume its challenge.")
    verify_action.add_argument("grant", type=Path)
    verify_action.add_argument("status", type=Path, help="JSON returned by the trusted issuer status endpoint.")
    verify_action.add_argument("proof", type=Path)
    verify_action.add_argument("--issuer", required=True)
    verify_action.add_argument("--issuer-public-key", required=True)
    verify_action.add_argument("--audience", required=True)
    for command in (challenge, prove, verify_action):
        command.add_argument("--intent", required=True, type=Path)
        command.add_argument("--artifact", required=True, type=Path)

    enrol = sub.add_parser("panel-enroll", help="Enroll a consented participant with an external verification credential.")
    enrol.add_argument("profile", type=Path)
    enrol.add_argument("attestation", type=Path)
    enrol.add_argument("--identity-issuer", required=True)
    enrol.add_argument("--identity-audience", required=True)
    enrol.add_argument("--identity-public-key", required=True)
    withdraw = sub.add_parser("panel-withdraw", help="Stop routing tasks to a participant.")
    withdraw.add_argument("subject")
    panel_create = sub.add_parser("panel-question", help="Add a panel question with a compensation ceiling.")
    panel_create.add_argument("question", type=Path)
    panel_claim = sub.add_parser("panel-claim", help="Claim a question for a verified participant.")
    panel_claim.add_argument("subject")
    panel_claim.add_argument("topic")
    panel_answer = sub.add_parser("panel-answer", help="Submit an independently authored response.")
    panel_answer.add_argument("subject")
    panel_answer.add_argument("assignment_id")
    panel_answer.add_argument("answer", type=Path)
    panel_queue = sub.add_parser("panel-review-queue", help="List responses for independent review.")
    panel_queue.add_argument("topic")
    panel_review = sub.add_parser("panel-review", help="Review one response and record earned compensation.")
    panel_review.add_argument("assignment_id")
    panel_review.add_argument("--reviewer", required=True)
    panel_review.add_argument("--decision", choices=["approve", "reject"], required=True)
    panel_review.add_argument("--reason", required=True)
    panel_reconcile = sub.add_parser("panel-reconcile", help="Record an independently completed payment.")
    panel_reconcile.add_argument("assignment_id")
    panel_reconcile.add_argument("--reference", required=True)
    panel_report = sub.add_parser("panel-result", help="Read a panel's aggregate responses.")
    panel_report.add_argument("question_id")
    panel_quality = sub.add_parser("panel-quality", help="Read a participant's review and retest statistics.")
    panel_quality.add_argument("subject")

    cohort_grant = sub.add_parser("grant-cohort-consent", help="Permit bounded aggregate twin contributions.")
    cohort_grant.add_argument("consent", type=Path)
    cohort_revoke = sub.add_parser("revoke-cohort-consent", help="Stop using a twin's aggregate contributions.")
    cohort_revoke.add_argument("subject")
    cohort_revoke.add_argument("cohort_id")
    panel_eval = sub.add_parser("evaluate-cohort-panel", help="Compare simulations with reviewed panel responses.")
    panel_eval.add_argument("cohort_id")
    panel_eval.add_argument("question_ids", nargs="+")
    panel_eval.add_argument("--sample-size", type=int, default=500)
    panel_eval.add_argument("--allow-remote-model", action="store_true")

    subject_export = sub.add_parser("export-subject", help="Export subject records and provenance to JSON.")
    subject_export.add_argument("subject_id")
    erase = sub.add_parser("erase-subject", help="Erase subject data or selected decision IDs.")
    erase.add_argument("subject_id")
    erase.add_argument("--decision-id", action="append")
    erase.add_argument("--confirm-subject", required=True)

    inbox = sub.add_parser("serve-inbox", help="Start the local owner escalation inbox.")
    inbox.add_argument("--port", type=int, default=8084)
    inbox.add_argument("--subject", help="Bind the owner inbox to one subject.")
    inbox.add_argument("--panel-topic", help="Enable panel questions for this subject and topic.")

    res_p = sub.add_parser("resolve", help="Resolve a pending escalation ticket.")
    res_p.add_argument("ticket_id", type=str)
    res_p.add_argument("--action", required=True, type=str)
    res_p.add_argument("--rationale", required=True, type=str)

    issue = sub.add_parser("issue-capability", help="Issue a scoped read capability.")
    issue.add_argument("--subject", required=True)
    issue.add_argument("--client", required=True)
    issue.add_argument("--scope", action="append", required=True)
    issue.add_argument("--ttl", type=int, default=3600)
    issue.add_argument("--rate", type=int, default=60)
    revoke = sub.add_parser("revoke-capability", help="Revoke a capability using its public ID.")
    revoke.add_argument("token_id")

    mcp_p = sub.add_parser("serve-mcp", help="Start the AnthroMesh MCP server.")
    mcp_p.add_argument("--transport", choices=["stdio", "http"], default="http")
    mcp_p.add_argument("--host", default="127.0.0.1")
    mcp_p.add_argument("--port", type=int, default=8081)
    mcp_p.add_argument("--allow-network-bind", action="store_true")
    mcp_p.add_argument("--allow-capability-tokens", action="store_true")

    ros = sub.add_parser("serve-ros2", help="Run the ROS 2 fast preference-gate node.")
    ros.add_argument("--subject", required=True)
    ros.add_argument("--domain", action="append", required=True)
    ros.add_argument("--minimum-risk", choices=[tier.value for tier in RiskTier], default="low")
    ros.add_argument("--request-topic", default="/anthromesh/gate/request")
    ros.add_argument("--result-topic", default="/anthromesh/gate/result")
    ros.add_argument("--deadline-ms", type=int, default=50)

    a2a_p = sub.add_parser("serve-a2a", help="Start the AnthroMesh A2A v1.0 server.")
    a2a_p.add_argument("--host", default="127.0.0.1")
    a2a_p.add_argument("--port", type=int, default=8082)
    a2a_p.add_argument("--allow-network-bind", action="store_true")

    args = parser.parse_args(argv)
    cfg = AnthroMeshConfig()

    if args.command == "init":
        vault = _open_vault(cfg)
        print(f"Initialized Behavioral Vault at {vault.db_path}")
        return 0

    if args.command == "export-encrypted":
        if cfg.encryption_key is None:
            parser.error("Set ANTHROMESH_VAULT_KEY before exporting the encrypted vault.")
        export_encrypted_vault(args.source, args.destination, cfg.encryption_key)
        print(f"Exported encrypted vault to {args.destination}")
        return 0

    if args.command == "seed":
        vault, twin, oracle = _build_runtime(cfg)
        if args.decisions:
            items = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
            for item in items:
                twin.record_decision(DecisionLog.model_validate(item))
            print(f"Seeded {len(items)} decisions into {vault.db_path}")
        if args.cohort:
            profile = CohortProfile.model_validate_json(Path(args.cohort).read_text(encoding="utf-8"))
            oracle.register_cohort(profile)
            print(f"Registered cohort '{profile.cohort_id}' into {vault.db_path}")
        return 0

    if args.command == "import-decisions":
        source = json.loads(args.path.read_text(encoding="utf-8"))
        if args.source == "github-review":
            if not args.account:
                parser.error("GitHub review import requires --account.")
            decisions = github_reviews(source if isinstance(source, list) else [source],
                                       args.subject, args.account)
        else:
            decisions = calendar_responses(source if isinstance(source, list) else source["items"],
                                           args.subject)
        vault = _open_vault(cfg)
        count = vault.import_captured_decisions(decisions)
        print(f"Imported {count} of {len(decisions)} matching decisions.")
        return 0

    if args.command == "sync-graph":
        if args.poll_seconds is not None and args.poll_seconds < 10:
            parser.error("Graph polling intervals must be at least ten seconds.")
        vault = _open_vault(cfg)
        reset = args.reset
        try:
            while True:
                token = (args.token_file.read_text(encoding="utf-8").strip() if args.token_file
                         else os.getenv("ANTHROMESH_GRAPH_ACCESS_TOKEN"))
                if not token:
                    parser.error("Provide a delegated Microsoft Graph access token.")
                drafts = json.loads(args.drafts.read_text(encoding="utf-8")) if args.drafts else None
                with GraphCapture(vault, args.subject, args.account, token) as capture:
                    count = capture.sync(
                        args.stream, start=args.start, end=args.end, drafts=drafts,
                        track_reschedules=args.track_reschedules, reset=reset,
                    )
                print(f"Captured {count} new {args.stream} decisions.", flush=True)
                reset = False
                if args.poll_seconds is None:
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            pass
        return 0

    if args.command == "serve-github-capture":
        import uvicorn

        secret = os.getenv("ANTHROMESH_GITHUB_WEBHOOK_SECRET")
        if not secret:
            parser.error("Set ANTHROMESH_GITHUB_WEBHOOK_SECRET before starting capture.")
        vault = _open_vault(cfg)
        app = create_github_capture_app(
            vault, secret=secret, subject=args.subject, account=args.account,
            repositories=set(args.repository),
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return 0

    if args.command == "import-survey":
        vault = _open_vault(cfg)
        margins = json.loads(args.margins.read_text(encoding="utf-8")) if args.margins else None
        if not 0 <= args.twin_prior_weight <= 0.25:
            parser.error("Twin prior weight must be between 0 and 0.25.")
        inputs = dict(cohort_id=args.id, domain=args.domain, response=args.response,
                      weight=args.weight, margins=margins)
        if args.cluster_count is not None:
            profile = build_clustered_profile(
                read_survey(args.path), strata=args.trait, features=args.cluster_feature,
                clusters=args.cluster_count, **inputs,
            )
        else:
            if args.cluster_feature:
                parser.error("Provide --cluster-count with clustering features.")
            profile = build_empirical_profile(read_survey(args.path), traits=args.trait, **inputs)
        profile.twin_prior_weight = args.twin_prior_weight
        vault.save_cohort(profile)
        print(f"Imported {len(profile.archetypes)} survey strata into cohort {profile.cohort_id}.")
        return 0

    if args.command == "fit-calibration":
        values = json.loads(args.path.read_text(encoding="utf-8"))
        held_out = [DecisionLog.model_validate(item) for item in values]
        vault, twin, _ = _build_runtime(replace(cfg, calibration_path=None, managed_calibration=False))
        profiles = [profile.model_dump() for profile in fit_profiles(twin, held_out, args.alpha)]
        if args.store:
            vault.save_calibration_profiles(profiles)
            print(f"Stored {len(profiles)} calibration profiles.")
        else:
            print(json.dumps(profiles, indent=2))
        return 0

    if args.command == "register-study":
        record = register_study(args.plan, args.destination)
        print(json.dumps({"study_id": record.plan.study_id, "plan_hash": record.plan_hash}))
        return 0

    if args.command == "check-study":
        record = load_registration(args.registration)
        dataset, digest = load_dataset(args.dataset)
        prepared = prepare_study(record, dataset)
        print(json.dumps({
            "data_source": dataset.source, "dataset_hash": digest,
            "participants": len(prepared.participants), "train": len(prepared.train),
            "calibration": len(prepared.calibration), "test": len(prepared.test),
            "retests": len(prepared.retests),
        }, indent=2))
        return 0

    if args.command == "forecast-study":
        adapter = create_adapter(cfg)
        remote = _remote_model(cfg, adapter.adapter_name)
        outcome = forecast_study(
            load_registration(args.registration), args.dataset, adapter, args.destination,
            allow_remote=args.allow_remote_model, remote_backend=remote, vault_key=cfg.encryption_key,
        )
        print(f"Recorded {len(outcome.forecasts)} paired forecasts for {outcome.backend}.")
        return 0

    if args.command == "evaluate-study":
        print(json.dumps(evaluate_study(
            load_registration(args.registration), args.dataset, args.forecasts,
        ), indent=2))
        return 0

    if args.command == "download-public-twin":
        download_public_twin(args.folder)
        print(args.folder.resolve())
        return 0

    if args.command == "evaluate-public-twin":
        adapter = create_adapter(cfg)
        if _remote_model(cfg, adapter.adapter_name):
            raise ValueError("Public Twin-2K-500 participant rows are evaluated with a local adapter.")
        if args.output and args.output.exists():
            raise FileExistsError(args.output)
        result = evaluate_public_twin(args.folder, adapter, seed=args.seed, participants=args.participants)
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            with args.output.open("x", encoding="utf-8") as report:
                report.write(json.dumps(result, indent=2) + "\n")
        return 0

    if args.command == "audit-public-twin":
        print(json.dumps(audit_original_public_run(args.folder, args.original_report), indent=2))
        return 0

    if args.command == "validate-public-twin":
        print(json.dumps(run_public_validation(args.folder, args.original_report, args.frozen, phase=args.phase), indent=2))
        return 0

    if args.command == "draw-cohort-anchors":
        units = json.loads(args.frame.read_text(encoding="utf-8"))
        if not isinstance(units, list) or not all(isinstance(unit, str) for unit in units):
            raise ValueError("The sampling frame must be an array of string unit IDs.")
        print(json.dumps(draw_anchor_units(units, args.count, args.seed), indent=2))
        return 0

    if args.command == "calibrate-cohort":
        payload = json.loads(args.study.read_text(encoding="utf-8"))
        if "categories" in payload:
            output = analyze_categories(CategoricalAnchorStudy.model_validate(payload))
        else:
            output = analyze_anchors(AnchorStudy.model_validate(payload))
        print(json.dumps(output, indent=2))
        return 0

    if args.command == "score-study":
        print(json.dumps(score_study(args.path, args.mode, args.baseline), indent=2))
        return 0

    if args.command == "benchmark":
        adapter = create_adapter(cfg)
        report = BenchmarkRunner(adapter=adapter).run_all()
        print(json.dumps(report.model_dump(), indent=2))
        return 0

    if args.command == "escalations":
        vault, _, _ = _build_runtime(cfg)
        pending = vault.list_escalations(status=EscalationStatus.PENDING)
        print(json.dumps([t.model_dump() for t in pending], indent=2))
        return 0

    if args.command in {"configure-judgment", "assign-elicitation", "list-elicitation",
                        "respond-elicitation", "trust-map"}:
        vault = _open_vault(cfg)
        elicitation = Elicitation(vault)
        if args.command == "configure-judgment":
            settings = JudgmentSettings.model_validate_json(args.settings.read_text(encoding="utf-8"))
            save_judgment_settings(vault, settings)
            output = settings.model_dump(mode="json")
        elif args.command == "assign-elicitation":
            case = ElicitationCase.model_validate_json(args.case.read_text(encoding="utf-8"))
            output = elicitation.assign(case).model_dump(mode="json")
        elif args.command == "list-elicitation":
            output = elicitation.pending(args.subject)
        elif args.command == "respond-elicitation":
            output = elicitation.respond(
                args.subject, args.case_id, args.action, args.rationale, args.switch_condition,
            ).model_dump(mode="json")
        else:
            output = PredictionMonitor(vault).trust_map(args.domain)
        print(json.dumps(output, indent=2))
        return 0

    if args.command == "delegation-key":
        print(json.dumps({"public_key": delegation.create_private_key(args.path)}))
        return 0

    if args.command in {"issue-delegation", "revoke-delegation", "serve-delegation"}:
        issuer = delegation.DelegationIssuer(
            _open_vault(cfg), args.issuer, delegation.read_private_key(args.issuer_key),
        )
        if args.command == "issue-delegation":
            request = delegation.GrantRequest.model_validate_json(args.request.read_bytes())
            with args.destination.open("x", encoding="utf-8") as output:
                grant, token = issuer.issue(request)
                output.write(token + "\n")
            print(json.dumps({"grant_id": grant.jti, "expires_at": grant.exp}))
        elif args.command == "revoke-delegation":
            issuer.revoke(args.grant_id)
        else:
            import uvicorn

            uvicorn.run(delegation.create_status_app(issuer), host="127.0.0.1", port=args.port)
        return 0

    if args.command in {"delegation-challenge", "prove-delegated-action", "verify-delegated-action"}:
        import hashlib

        intent = delegation.Intent.model_validate_json(args.intent.read_bytes())
        content_digest = hashlib.sha256()
        with args.artifact.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                content_digest.update(chunk)
        artifact_hash = content_digest.hexdigest()
        if intent.artifact_sha256 != artifact_hash:
            parser.error("The actual artifact does not match the action intent.")
        if args.command == "delegation-challenge":
            value = delegation.issue_challenge(_open_vault(cfg), args.audience, intent)
            print(json.dumps({"challenge": value}))
        elif args.command == "prove-delegated-action":
            proof = delegation.sign_action(
                args.grant.read_text(encoding="utf-8").strip(), args.issuer_public_key,
                delegation.read_private_key(args.holder_key), args.challenge, intent,
            )
            with args.destination.open("x", encoding="utf-8") as output:
                output.write(proof + "\n")
        else:
            status = json.loads(args.status.read_bytes())["status"]
            outcome = delegation.verify_action(
                _open_vault(cfg), issuer=args.issuer, issuer_public_key=args.issuer_public_key,
                audience=args.audience, grant_token=args.grant.read_text(encoding="utf-8").strip(),
                status_token=status, proof_token=args.proof.read_text(encoding="utf-8").strip(),
                intent=intent,
            )
            print(json.dumps(outcome))
        return 0

    if args.command.startswith("panel-"):
        panel = HumanPanel(_open_vault(cfg))
        output = None
        if args.command == "panel-enroll":
            panel.enroll(
                PanelProfile.model_validate_json(args.profile.read_bytes()),
                args.attestation.read_text(encoding="utf-8").strip(),
                issuer=args.identity_issuer, audience=args.identity_audience,
                public_key=args.identity_public_key,
            )
        elif args.command == "panel-withdraw":
            panel.withdraw(args.subject)
        elif args.command == "panel-question":
            output = {"question_id": panel.submit_question(
                PanelQuestion.model_validate_json(args.question.read_bytes()),
            )}
        elif args.command == "panel-claim":
            output = panel.claim(args.subject, args.topic)
        elif args.command == "panel-answer":
            panel.answer(args.subject, args.assignment_id,
                         PanelAnswer.model_validate_json(args.answer.read_bytes()))
        elif args.command == "panel-review-queue":
            output = panel.pending_reviews(args.topic)
        elif args.command == "panel-review":
            panel.review(args.assignment_id, args.reviewer, args.decision == "approve", args.reason)
        elif args.command == "panel-reconcile":
            panel.reconcile(args.assignment_id, args.reference)
        elif args.command == "panel-result":
            output = panel.report(args.question_id)
        else:
            output = panel.quality(args.subject)
        print(json.dumps(output, indent=2))
        return 0

    if args.command == "grant-cohort-consent":
        grant_cohort_consent(
            _open_vault(cfg), TwinCohortConsent.model_validate_json(args.consent.read_bytes()),
        )
        return 0
    if args.command == "revoke-cohort-consent":
        revoke_cohort_consent(_open_vault(cfg), args.subject, args.cohort_id)
        return 0
    if args.command == "evaluate-cohort-panel":
        from anthromesh.benchmarks.panel_cohort import evaluate_against_panel

        adapter = create_adapter(cfg)
        if _remote_model(cfg, adapter.adapter_name) and not args.allow_remote_model:
            parser.error("Enable remote model inference explicitly for this evaluation.")
        oracle = ResearchCohortOracle(_open_vault(cfg), adapter)
        print(json.dumps(evaluate_against_panel(
            oracle, args.cohort_id, args.question_ids, sample_size=args.sample_size,
        ), indent=2))
        return 0

    if args.command in {"export-subject", "erase-subject"}:
        data = SubjectData(_open_vault(cfg))
        if args.command == "export-subject":
            result = data.export(args.subject_id)
        else:
            if args.confirm_subject != args.subject_id:
                parser.error("The confirmation must match the exact subject ID.")
            result = data.erase(
                args.subject_id, decision_ids=set(args.decision_id) if args.decision_id else None,
            )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "verify-audit":
        vault = _open_vault(cfg)
        valid = vault.verify_audit_chain(expected_head=args.expected_head)
        print("Audit chain verified." if valid else "Audit verification failed.")
        return 0 if valid else 1

    if args.command in {"resolve", "serve-inbox"}:
        vault, twin, oracle = _build_runtime(cfg)

        def resolve(ticket_id: str, action: str, rationale: str) -> dict:
            task = vault.get_a2a_task_for_ticket(ticket_id)
            if task is None:
                ticket, decision = twin.resolve_escalation(ticket_id, action, rationale)
                return {"ticket": ticket.model_dump(), "recorded_decision": decision.model_dump()}
            complete = A2AService(twin=twin, oracle=oracle).resolve_task(
                task["id"], action, rationale
            )
            return complete["artifacts"][-1]["data"]

        if args.command == "resolve":
            print(json.dumps(resolve(args.ticket_id, args.action, args.rationale), indent=2))
        else:
            import uvicorn

            password = os.getenv("ANTHROMESH_INBOX_PASSWORD")
            if not password:
                parser.error("Set ANTHROMESH_INBOX_PASSWORD before starting the inbox.")
            uvicorn.run(create_inbox_app(vault, resolve, password, subject=args.subject, panel_topic=args.panel_topic),
                        host="127.0.0.1", port=args.port)
        return 0

    if args.command in {"issue-capability", "revoke-capability"}:
        vault = _open_vault(cfg)
        if args.command == "revoke-capability":
            return 0 if vault.revoke_capability(args.token_id) else 1
        identifier, token = CapabilityTokens(vault).issue(
            args.subject, args.client, args.scope, args.ttl, args.rate,
        )
        print(json.dumps({"token_id": identifier, "token": token}))
        return 0

    if args.command == "serve-mcp":
        vault, twin, oracle = _build_runtime(cfg)
        tokens = CapabilityTokens(vault) if args.allow_capability_tokens else None
        server = MCPServer(twin=twin, oracle=oracle, capabilities=tokens)
        if args.transport == "stdio":
            server.serve_stdio()
        else:
            issuer = os.getenv("ANTHROMESH_OAUTH_ISSUER")
            resource = os.getenv("ANTHROMESH_OAUTH_RESOURCE")
            jwks = os.getenv("ANTHROMESH_OAUTH_JWKS")
            discover = os.getenv("ANTHROMESH_OAUTH_DISCOVERY") == "1"
            if issuer or resource or jwks or discover:
                if not issuer or not resource or bool(jwks) == discover:
                    parser.error("Set the OAuth issuer, resource URL, and a JWKS file or discovery mode.")
                ca_file = os.getenv("ANTHROMESH_OAUTH_CA_FILE")
                server.oauth = OAuthResource(
                    issuer, resource, Path(jwks) if jwks else None,
                    discover=discover, profile=os.getenv("ANTHROMESH_OAUTH_TOKEN_PROFILE", "rfc9068"),
                    ca_file=Path(ca_file) if ca_file else None,
                    clock_skew_seconds=int(os.getenv("ANTHROMESH_OAUTH_CLOCK_SKEW_SECONDS", "0")),
                )
            server.serve_http(host=args.host, port=args.port, allow_network_bind=args.allow_network_bind)
        return 0

    if args.command == "serve-ros2":
        from anthromesh.ros_gate import RosPreferenceGate, serve_ros

        _, twin, _ = _build_runtime(replace(cfg, fast_enabled=True))
        gate = RosPreferenceGate(
            twin.gate.evaluate_fast, subject_id=args.subject, domains=set(args.domain),
            minimum_risk=RiskTier(args.minimum_risk),
        )
        serve_ros(gate, request_topic=args.request_topic, result_topic=args.result_topic,
                  deadline_ms=args.deadline_ms)
        return 0

    if args.command == "serve-a2a":
        _, twin, oracle = _build_runtime(cfg)
        public_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        service = A2AService(twin=twin, oracle=oracle, base_url=f"http://{public_host}:{args.port}")
        service.serve_http(host=args.host, port=args.port, allow_network_bind=args.allow_network_bind)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
