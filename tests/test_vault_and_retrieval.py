from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from anthromesh.models import CandidateOption, DecisionLog, QueryContext
from anthromesh.vault.encryption import export_encrypted_vault
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault


def test_vault_roundtrip_and_retrieval() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = BehavioralVault(db_path=str(Path(tmpdir) / "test.db"))
        opts = [
            CandidateOption(option_id="request_changes", label="Request Changes"),
            CandidateOption(option_id="approve", label="Approve"),
        ]
        d1 = DecisionLog(
            subject_id="u1",
            domain="code_review",
            context="PR modifies solver kernel without unit tests",
            options_considered=opts,
            chosen_action="request_changes",
            rationale="Require unit tests for numerical kernels.",
        )
        d2 = DecisionLog(
            subject_id="u1",
            domain="calendar",
            context="Lunch invite with visiting researcher",
            options_considered=[
                CandidateOption(option_id="accept", label="Accept"),
                CandidateOption(option_id="decline", label="Decline"),
            ],
            chosen_action="accept",
            rationale="Always prioritize visiting collaborators.",
        )
        vault.add_decision(d1)
        vault.add_decision(d2)

        stored = vault.list_decisions(subject_id="u1")
        assert len(stored) == 2
        assert d1.rejected_options == ["approve"]

        retriever = HybridDecisionRetriever()
        q = QueryContext(
            subject_id="u1",
            domain="code_review",
            context="PR edits solver convergence kernel and omits unit tests",
            candidate_options=opts,
        )
        ranked = retriever.retrieve(q, stored, top_k=2)
        assert ranked[0][0].decision_id == d1.decision_id
        assert ranked[0][1] > ranked[1][1]


def test_audit_chain_detects_modified_rows_and_truncated_tail(tmp_path: Path) -> None:
    vault = BehavioralVault(str(tmp_path / "audit.db"))
    for action in ("approve", "decline"):
        vault.log_audit("test", "agent", "subject", "calendar", action)

    head = vault.list_audit_logs()[0]["entry_hash"]
    assert vault.verify_audit_chain(expected_head=head)
    with sqlite3.connect(vault.db_path) as conn:
        conn.execute("DELETE FROM audit_logs WHERE id = 2")
    assert vault.verify_audit_chain()
    assert not vault.verify_audit_chain(expected_head=head)

    with sqlite3.connect(vault.db_path) as conn:
        conn.execute("UPDATE audit_logs SET action = 'changed' WHERE id = 1")
    assert not BehavioralVault(vault.db_path).verify_audit_chain()


def test_audit_chain_migrates_existing_rows(tmp_path: Path) -> None:
    db_path = str(tmp_path / "legacy.db")
    vault = BehavioralVault(db_path)
    vault.log_audit("test", "agent", "subject", "calendar", "approve")
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE audit_logs DROP COLUMN entry_hash")

    migrated = BehavioralVault(db_path)
    assert migrated.list_audit_logs()[0]["action"] == "approve"
    migrated.log_audit("test", "agent", "subject", "calendar", "decline")
    assert migrated.verify_audit_chain()


def test_encrypted_vault_requires_correct_key(tmp_path: Path) -> None:
    cipher = pytest.importorskip("sqlcipher3.dbapi2")
    db_path = tmp_path / "encrypted.db"
    vault = BehavioralVault(str(db_path), encryption_key="test-passphrase")
    vault.log_audit("test", "agent", "subject", "calendar", "accept")
    assert not db_path.read_bytes().startswith(b"SQLite format 3")
    assert BehavioralVault(str(db_path), encryption_key="test-passphrase").verify_audit_chain()
    with pytest.raises(cipher.DatabaseError):
        BehavioralVault(str(db_path), encryption_key="wrong-passphrase")

    plain = tmp_path / "plain.db"
    BehavioralVault(str(plain))
    with pytest.raises(ValueError, match="Export"):
        BehavioralVault(str(plain), encryption_key="test-passphrase")


def test_encrypted_export_preserves_original_and_audit_chain(tmp_path: Path) -> None:
    pytest.importorskip("sqlcipher3.dbapi2")
    source = tmp_path / "source.db"
    destination = tmp_path / "encrypted.db"
    vault = BehavioralVault(str(source))
    vault.log_audit("test", "agent", "user", "calendar", "approve")
    original = source.read_bytes()
    export_encrypted_vault(source, destination, "passphrase with 'quote'")
    assert source.read_bytes() == original
    exported = BehavioralVault(str(destination), encryption_key="passphrase with 'quote'")
    assert exported.list_audit_logs() == vault.list_audit_logs()
    with pytest.raises(FileExistsError):
        export_encrypted_vault(source, destination, "different")


def test_dense_retrieval_uses_one_embedding_batch() -> None:
    options = [CandidateOption(option_id="a", label="A"), CandidateOption(option_id="b", label="B")]
    decision = DecisionLog(
        subject_id="owner", domain="review", context="Solver convergence",
        options_considered=options, chosen_action="a", rationale="Prior choice",
    )
    query = QueryContext(
        subject_id="owner", domain="review", context="Numerical stability",
        candidate_options=options,
    )

    def embed_batch(texts):
        assert len(texts) == 2
        assert "Numerical stability" in texts[0] and "Solver convergence" in texts[1]
        return [[1.0, 0.0], [1.0, 0.0]]

    assert HybridDecisionRetriever(embed_batch=embed_batch).retrieve(query, [decision]) == [
        (decision, 1.0),
    ]
