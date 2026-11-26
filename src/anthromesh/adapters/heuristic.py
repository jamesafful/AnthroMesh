from __future__ import annotations

from collections import defaultdict

from anthromesh.models import CandidateOption, CohortArchetype, DecisionLog, QueryContext
from anthromesh.vault.retrieval import tokenize
from .base import BaseModelAdapter, StructuredPrediction


class HeuristicAdapter(BaseModelAdapter):
    """Zero-dependency, deterministic evidence-grounded adapter for offline dev, tests, and CI benchmarks."""

    adapter_name = "heuristic"

    def __init__(self, model: str = "deterministic-v1") -> None:
        super().__init__(model=model)

    def predict_persona(self, query: QueryContext, persona: str) -> StructuredPrediction:
        return self.predict_decision(query, [])

    def predict_decision(
        self,
        query: QueryContext,
        evidence: list[tuple[DecisionLog, float]],
    ) -> StructuredPrediction:
        scores: dict[str, float] = {o.option_id: 0.0 for o in query.candidate_options}
        valid_ids = set(scores.keys())

        if not evidence:
            first = query.candidate_options[0].option_id
            return StructuredPrediction(
                chosen_option=first,
                rationale="No historical evidence found in vault; fallback prediction has low confidence.",
                self_reported_confidence=0.15,
                option_scores={o.option_id: round(1.0 / len(query.candidate_options), 4) for o in query.candidate_options},
            )

        supporting_ids: dict[str, list[str]] = defaultdict(list)
        for dec, sim in evidence:
            if dec.chosen_action in valid_ids:
                scores[dec.chosen_action] += sim * 1.25
                supporting_ids[dec.chosen_action].append(dec.decision_id)
            else:
                hist_tokens = set(tokenize(f"{dec.chosen_action} {dec.rationale}"))
                for opt in query.candidate_options:
                    opt_tokens = set(tokenize(f"{opt.option_id} {opt.label} {opt.description}"))
                    overlap = len(hist_tokens & opt_tokens) / max(len(opt_tokens), 1)
                    if overlap > 0:
                        scores[opt.option_id] += sim * overlap
                        supporting_ids[opt.option_id].append(dec.decision_id)

        total = sum(scores.values())
        if total <= 0.0:
            first = query.candidate_options[0].option_id
            return StructuredPrediction(
                chosen_option=first,
                rationale="Retrieved decisions did not align with current candidate options.",
                self_reported_confidence=0.2,
                option_scores={o.option_id: round(1.0 / len(query.candidate_options), 4) for o in query.candidate_options},
            )

        norm_scores = {k: round(v / total, 4) for k, v in scores.items()}
        best_opt = max(norm_scores, key=lambda k: norm_scores[k])
        sorted_vals = sorted(norm_scores.values(), reverse=True)
        margin = sorted_vals[0] - (sorted_vals[1] if len(sorted_vals) > 1 else 0.0)
        top_sim = evidence[0][1]
        conf = max(0.05, min(0.98, 0.5 * norm_scores[best_opt] + 0.35 * top_sim + 0.15 * margin))

        cites = ", ".join(list(dict.fromkeys(supporting_ids[best_opt]))[:3]) or evidence[0][0].decision_id
        rationale = (
            f"Selected '{best_opt}' based on weighted similarity to prior decisions ({cites}) "
            f"with normalized support {norm_scores[best_opt]:.2f}."
        )
        return StructuredPrediction(
            chosen_option=best_opt,
            rationale=rationale,
            self_reported_confidence=round(conf, 4),
            option_scores=norm_scores,
        )

    def simulate_cohort_archetype(
        self,
        stimulus: str,
        options: list[CandidateOption],
        archetype: CohortArchetype,
    ) -> StructuredPrediction:
        scores: dict[str, float] = {}
        stim_tokens = set(tokenize(stimulus))
        trait_text = " ".join(f"{k} {v}" for k, v in archetype.traits.items())
        rationale_text = " ".join(archetype.sample_rationales)
        arch_tokens = set(tokenize(f"{archetype.label} {trait_text} {rationale_text}"))

        for opt in options:
            prior = float(archetype.prior_tendencies.get(opt.option_id, 0.0))
            opt_tokens = set(tokenize(f"{opt.option_id} {opt.label} {opt.description}"))
            lexical = len(arch_tokens & opt_tokens) / max(len(opt_tokens), 1)
            stim_align = len(stim_tokens & opt_tokens) / max(len(opt_tokens), 1)
            scores[opt.option_id] = max(0.02, prior * 0.75 + lexical * 0.20 + stim_align * 0.05)

        total = sum(scores.values())
        norm_scores = {k: round(v / total, 4) for k, v in scores.items()}
        best_opt = max(norm_scores, key=lambda k: norm_scores[k])
        sample_rat = (
            archetype.sample_rationales[0]
            if archetype.sample_rationales
            else f"Aligned with {archetype.label} archetype priorities."
        )
        return StructuredPrediction(
            chosen_option=best_opt,
            rationale=f"[{archetype.label}] {sample_rat}",
            self_reported_confidence=round(norm_scores[best_opt], 4),
            option_scores=norm_scores,
        )
