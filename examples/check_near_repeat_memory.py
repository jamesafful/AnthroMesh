"""Synthetic checks of exact preference reuse and near-repeat abstention."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from anthromesh.gate.fast import FastDecisionGate
from anthromesh.models import CandidateOption, DecisionLog, PolicyEffect, PolicyVerdict, QueryContext
from anthromesh.vault.retrieval import HybridDecisionRetriever
from anthromesh.vault.storage import BehavioralVault

SCENARIOS = (
    ("Buy a refill pack for $8.", "Purchase the refill pack for eight dollars.", "Buy a refill pack for $18."),
    ("Keep the Friday sync if all owners can attend.", "Keep Friday's sync if every owner can make it.",
     "Keep the Friday sync if half of the owners can attend."),
    ("Use quiet mode in the shared office.", "Enable quiet mode in the office we share.",
     "Use quiet mode during the emergency drill."),
)


def run() -> dict:
    options = [CandidateOption(option_id="yes", label="Yes"), CandidateOption(option_id="no", label="No")]
    result = []
    with TemporaryDirectory() as temporary:
        vault = BehavioralVault(str(Path(temporary) / "decisions.db"))
        gate = FastDecisionGate(vault, minimum_evidence=2, threshold=.8)
        policy = PolicyVerdict(effect=PolicyEffect.ALLOW, reason="Synthetic low-risk case")
        retriever = HybridDecisionRetriever()
        for index, (exact, paraphrase, changed) in enumerate(SCENARIOS):
            subject = f"example-{index}"
            history = [DecisionLog(
                subject_id=subject, domain="example", context=exact, options_considered=options,
                chosen_action="yes", rationale="Synthetic earlier answer",
            ) for _ in range(2)]
            stale = datetime.now(timezone.utc) - timedelta(days=91)
            for item in history:
                vault.add_decision(item)
                vault.add_decision(DecisionLog(
                    subject_id=subject + "-stale", domain="example", context=exact,
                    options_considered=options, chosen_action="yes",
                    rationale="Synthetic expired evidence", created_at=stale.isoformat(),
                ))
            for kind, context in (("exact", exact), ("paraphrase", paraphrase),
                                  ("changed", changed), ("expired", exact)):
                query = QueryContext(
                    subject_id=subject + "-stale" if kind == "expired" else subject,
                    domain="example", context=context, candidate_options=options,
                )
                lexical = retriever.retrieve(query, history)
                approved = gate.evaluate(query, policy) is not None
                if approved != (kind == "exact"):
                    raise AssertionError(f"Unexpected fast-path result for {kind}")
                result.append({
                    "scenario": index, "kind": kind, "exact_memory_approved": approved,
                    "related_history_retrieved": bool(lexical),
                    "lexical_top_similarity": lexical[0][1] if lexical else None,
                })
    return {"design": "synthetic_fast_memory_path_only", "records_expire_after_days": 90,
            "changed_context_false_approvals": sum(row["kind"] == "changed" and row["exact_memory_approved"]
                                                   for row in result),
            "cases": result}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
