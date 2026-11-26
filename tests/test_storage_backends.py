import os
import uuid

import pytest

from anthromesh.models import CandidateOption, DecisionLog, QueryContext
from anthromesh.vault.dense import DenseDecisionIndex
from anthromesh.vault.postgres import PostgresBehavioralVault
from anthromesh.vault.privacy import SubjectData
from anthromesh.vault.storage import BehavioralVault
from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.protocols.a2a_server import A2AService
from anthromesh.cohort.oracle import ResearchCohortOracle


def decisions(subject: str):
    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]
    query = QueryContext(subject_id=subject, domain="calendar", context="Recurring sync",
                         candidate_options=options)
    decision = DecisionLog(subject_id=subject, domain=query.domain, context=query.context,
                           options_considered=options, chosen_action="yes", rationale="Weekly schedule")
    return query, decision


@pytest.fixture(params=["sqlite", "postgres"])
def vault(request, tmp_path):
    if request.param == "sqlite":
        yield BehavioralVault(str(tmp_path / "test.db"))
        return
    psycopg = pytest.importorskip("psycopg")
    dsn = os.getenv("ANTHROMESH_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Start the disposable pgvector fixture and set ANTHROMESH_TEST_POSTGRES_DSN.")
    from psycopg import sql

    schema = "am_test_" + uuid.uuid4().hex[:16]
    yield PostgresBehavioralVault(dsn, schema=schema)
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_privacy_export_targeted_and_full_erasure(vault):
    pytest.importorskip("sqlite_vec")
    query, original = decisions("owner")
    _, other = decisions("other")
    vault.add_decision(original)
    vault.add_decision(other)
    adapter = HeuristicAdapter()
    twin = PersonalDecisionTwin(vault, PolicyEngine(), adapter, min_evidence_count=2)
    a2a = A2AService(twin, ResearchCohortOracle(vault, adapter))
    task = a2a.send_task({"skill": "twin_decision", "input": query.model_dump()})
    assert task["status"]["state"] == "input-required"
    assert vault.list_a2a_tasks(after="", limit=5)[1] == 1
    assert vault.get_a2a_task_for_ticket(task["escalation_ticket_id"]) == task
    vault.save_calibration_profiles([{"subject_id": "owner", "domain": "calendar",
                                      "adapter": "heuristic", "model": "test",
                                      "alpha": 0.1, "scores": [0.1]}])
    vector_index = DenseDecisionIndex(vault, "test", lambda texts: [[1.0, 0.0] for _ in texts])
    assert vector_index.rank("query", [(original, "original")], limit=1)[original.decision_id] > 0.9

    data = SubjectData(vault)
    before = data.export("owner")
    assert len(before["decisions"]) == len(before["a2a_tasks"]) == len(before["calibration_profiles"]) == 1
    assert data.export("other")["decisions"][0]["decision_id"] == other.decision_id
    with pytest.raises(ValueError):
        data.erase("owner", decision_ids={other.decision_id})
    partial = data.erase("owner", decision_ids={original.decision_id})
    assert partial["erased"]["decisions"] == 1
    assert vault.file_calibration_revoked("owner")
    assert vault.verify_audit_chain(partial["audit_head"])
    assert vector_index.rank("query", [(original, "original")], limit=1) == {}
    with pytest.raises(ValueError):
        vault.add_decision(original)

    _, fresh = decisions("owner")
    vault.add_decision(fresh)
    result = data.erase("owner")
    assert result["erased"]["decisions"] == 1
    assert vault.verify_audit_chain(result["audit_head"])
    assert data.export("owner")["decisions"] == []
    assert vault.get_decision(other.decision_id) is not None
    with pytest.raises(ValueError):
        vault.add_decision(fresh)


def test_exact_fast_path_never_calls_model_and_stops_after_conflict(vault):
    class NoInference(HeuristicAdapter):
        def predict_decision(self, *_args):
            raise AssertionError("Fast-tier tests never invoke model inference.")

    query, first = decisions("owner")
    _, second = decisions("owner")
    vault.add_decision(first)
    vault.add_decision(second)
    twin = PersonalDecisionTwin(vault, PolicyEngine(), NoInference(), fast_enabled=True)
    decision = twin.query_decision(query)
    assert decision.chosen_action == "yes"
    assert decision.provenance.inference_tier == "fast"
    assert twin.gate.evaluate_fast(query.model_copy(update={"monetary_value": 0})) is None
    _, conflict = decisions("owner")
    vault.add_decision(conflict.model_copy(update={"chosen_action": "no"}))
    assert twin.gate.evaluate_fast(query) is None
    assert vault.verify_audit_chain()


def test_postgres_preprovisioned_roles_isolate_tenants():
    psycopg = pytest.importorskip("psycopg")
    dsn = os.getenv("ANTHROMESH_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Start the disposable PostgreSQL fixture with role creation privileges.")
    from psycopg import sql

    prefix = "am_roles_" + uuid.uuid4().hex[:12]
    tenants = [prefix + "_a", prefix + "_b"]
    with psycopg.connect(dsn) as admin:
        for tenant in tenants:
            admin.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(tenant)))
            admin.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(tenant), sql.Identifier(tenant),
            ))
    try:
        vaults = [
            PostgresBehavioralVault(
                psycopg.conninfo.make_conninfo(dsn, options=f"-c role={tenant}"), schema=tenant,
            )
            for tenant in tenants
        ]
        _, first = decisions("shared-subject-id")
        _, second = decisions("shared-subject-id")
        vaults[0].add_decision(first)
        vaults[1].add_decision(second)
        assert vaults[0].get_decision(second.decision_id) is None
        assert vaults[1].get_decision(first.decision_id) is None
        with pytest.raises(psycopg.errors.InsufficientPrivilege), vaults[0]._connect() as conn:
            conn.raw.execute(sql.SQL("SELECT * FROM {}.decisions").format(sql.Identifier(tenants[1])))
        SubjectData(vaults[0]).erase("shared-subject-id")
        assert vaults[1].get_decision(second.decision_id) is not None
    finally:
        with psycopg.connect(dsn) as admin:
            for tenant in tenants:
                admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(tenant)))
                admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(tenant)))
