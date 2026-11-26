"""Evaluate 72 authored exact, paraphrased, changed, and aged queries."""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from anthromesh.adapters.base import post_json
from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.models import CandidateOption, DecisionLog, GateAction, QueryContext
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault


def run(model):
    scenarios = json.loads(Path(__file__).with_name("near-repeat-cases.json").read_text(encoding="utf-8"))
    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]

    def embed(texts):
        return post_json("http://127.0.0.1:11434/api/embed", {"model": model, "input": texts})["embeddings"]

    lexical = HybridDecisionRetriever()
    semantic = HybridDecisionRetriever(embed_batch=embed) if model else None
    rows = []
    with TemporaryDirectory() as temporary:
        vault = BehavioralVault(str(Path(temporary) / "cases.db"))
        twin = PersonalDecisionTwin(vault, PolicyEngine(), HeuristicAdapter(), memory_only=True)
        for index, (exact, paraphrase, changed) in enumerate(scenarios, 1):
            for label, context, age in (("exact", exact, 1), ("paraphrase", paraphrase, 1),
                                        ("changed", changed, 1), ("age_89", exact, 89),
                                        ("age_91", exact, 91), ("age_120", exact, 120)):
                subject = f"case-{index}-{label}"
                stamp = (datetime.now(timezone.utc) - timedelta(days=age)).isoformat()
                evidence = [
                    DecisionLog(subject_id=subject, domain="benchmark", context=text,
                                options_considered=options, chosen_action=action,
                                rationale="Synthetic record", created_at=stamp)
                    for text, action in [(exact, "yes"), (exact, "yes"),
                                         ("Reschedule the recurring planning meeting.", "no"),
                                         ("Buy a pack of paper for the meeting.", "no")]
                ]
                for item in evidence:
                    vault.add_decision(item)
                query = QueryContext(subject_id=subject, domain="benchmark", context=context, candidate_options=options)
                lexical_top = lexical.retrieve(query, evidence, top_k=1)[0][0].context == exact
                semantic_top = semantic.retrieve(query, evidence, top_k=1)[0][0].context == exact if semantic else None
                decision = twin.query_decision(query)
                rows.append({
                    "scenario": index, "kind": label, "lexical_top_is_target": lexical_top,
                    "hybrid_top_is_target": semantic_top,
                    "fast_approved": decision.provenance.inference_tier == "fast",
                    "action": decision.action.value, "choice": decision.chosen_action,
                    "has_owner_ticket": bool(decision.escalation_ticket),
                })
    by_kind = {}
    for kind in dict.fromkeys(row["kind"] for row in rows):
        selected = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "cases": len(selected),
            "actions": dict(Counter(row["action"] for row in selected)),
            "fast_approvals": sum(row["fast_approved"] for row in selected),
            "lexical_top_matches": sum(row["lexical_top_is_target"] for row in selected),
            "hybrid_top_matches": sum(row["hybrid_top_is_target"] for row in selected) if semantic else None,
            "owner_tickets": sum(row["has_owner_ticket"] for row in selected),
            "incorrect_yes_approvals_if_changed": sum(
                row["action"] == GateAction.APPROVED.value and row["choice"] == "yes" for row in selected
            ) if kind == "changed" else None,
        }
    return {"design": "builder-authored synthetic examples; all changes stipulated consequential",
            "optional_local_embedding_model": model,
            "semantic_endpoint_used_for": "retrieval comparison only" if model else None,
            "summary": by_kind, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model")
    print(json.dumps(run(parser.parse_args().embedding_model), indent=2))
