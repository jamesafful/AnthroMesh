import pytest

from anthromesh.models import CandidateOption, DecisionLog
from anthromesh.vault.dense import DenseDecisionIndex
from anthromesh.vault.storage import BehavioralVault


def test_sqlite_vector_index_persists_and_refreshes_changed_documents(tmp_path):
    pytest.importorskip("sqlite_vec")
    db = str(tmp_path / "vectors.db")
    vault = BehavioralVault(db)
    options = [CandidateOption(option_id="a", label="A"), CandidateOption(option_id="b", label="B")]
    item = DecisionLog(subject_id="owner", domain="calendar", context="Old",
                       options_considered=options, chosen_action="a", rationale="Why")
    vault.add_decision(item)
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[0.0, 1.0] if text == "other" else [1.0, 0.0] for text in texts]

    assert DenseDecisionIndex(vault, "test", embed).rank(
        "query", [(item, "old")], limit=1,
    ) == {item.decision_id: 1.0}
    reopened = BehavioralVault(db)
    index = DenseDecisionIndex(reopened, "test", embed)
    assert index.rank("query", [(item, "old")], limit=1) == {item.decision_id: 1.0}
    assert calls == [["query"], ["old"], ["query"]]
    updated = item.model_copy(update={"context": "New"})
    reopened.add_decision(updated)
    assert index.rank("query", [(updated, "other")], limit=1) == {item.decision_id: 0.0}
    assert calls[-2:] == [["query"], ["other"]]
