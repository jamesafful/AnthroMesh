from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from anthromesh.adapters.base import BaseModelAdapter
from anthromesh.models import (
    CohortProfile,
    CohortQuery,
    CohortSimulationResult,
    DistributionMetrics,
    SimulatedResponse,
)
from anthromesh.vault.storage import BehavioralVault
from .twin_priors import blended_archetypes


class ResearchCohortOracle:
    """Generate stratified forecasts for comparison with empirical response distributions."""

    def __init__(self, vault: BehavioralVault, adapter: BaseModelAdapter) -> None:
        self.vault = vault
        self.adapter = adapter

    def register_cohort(self, profile: CohortProfile) -> CohortProfile:
        return self.vault.save_cohort(profile)

    def simulate(self, query: CohortQuery) -> CohortSimulationResult:
        profile = self.vault.get_cohort(query.cohort_id)
        if profile is None:
            raise KeyError(f"Cohort '{query.cohort_id}' not found in vault.")

        rng = random.Random(query.random_seed)
        option_ids = [option.option_id for option in query.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("Cohort options must have unique identifiers.")
        archetypes, blended = (blended_archetypes(self.vault, profile, set(option_ids))
                               if query.use_twin_priors else (profile.archetypes, False))
        if len(archetypes) > query.sample_size:
            raise ValueError("Sample size must provide at least one draw per cohort stratum.")
        total_weight = sum(a.weight for a in archetypes)
        norm_weights = [a.weight / total_weight for a in archetypes]

        allocations = [max(1, int(round(w * query.sample_size))) for w in norm_weights]
        if query.preserve_long_tail:
            for idx, arch in enumerate(archetypes):
                if arch.is_long_tail:
                    allocations[idx] = max(allocations[idx], max(2, int(round(0.05 * query.sample_size))))

        while sum(allocations) > query.sample_size:
            largest = max(range(len(allocations)), key=lambda i: allocations[i])
            if allocations[largest] > 1:
                allocations[largest] -= 1
            else:
                break
        while sum(allocations) < query.sample_size:
            largest_w = max(range(len(norm_weights)), key=lambda i: norm_weights[i])
            allocations[largest_w] += 1

        overall_counts: Counter[str] = Counter()
        archetype_counts: dict[str, Counter[str]] = defaultdict(Counter)
        sample_responses: list[SimulatedResponse] = []
        long_tail_count = 0

        for arch, n_samples in zip(archetypes, allocations):
            pred = self.adapter.simulate_cohort_archetype(query.stimulus, query.options, arch)
            probs = [max(0.01, float(pred.option_scores.get(oid, 0.05))) for oid in option_ids]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]

            for i in range(n_samples):
                chosen = rng.choices(option_ids, weights=probs, k=1)[0]
                overall_counts[chosen] += 1
                archetype_counts[arch.archetype_id][chosen] += 1
                if arch.is_long_tail:
                    long_tail_count += 1
                if i < 2:
                    sample_responses.append(
                        SimulatedResponse(
                            respondent_id=f"{arch.archetype_id}_{i+1:03d}",
                            archetype_id=arch.archetype_id,
                            archetype_label=arch.label,
                            is_long_tail=arch.is_long_tail,
                            chosen_option=chosen,
                            rationale=pred.rationale,
                            confidence=pred.self_reported_confidence,
                        )
                    )

        total_n = sum(overall_counts.values())
        effective_size = total_n
        if profile.population_weighted:
            dist = {
                oid: round(sum(
                    weight * archetype_counts[arch.archetype_id][oid] / allocation
                    for arch, weight, allocation in zip(archetypes, norm_weights, allocations)
                ), 4)
                for oid in option_ids
            }
            effective_size = min(total_n, round(1 / sum(
                weight ** 2 / allocation for weight, allocation in zip(norm_weights, allocations)
            )))
        else:
            dist = {oid: round(overall_counts[oid] / total_n, 4) for oid in option_ids}

        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
        max_entropy = math.log2(len(option_ids)) if len(option_ids) > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        archetype_breakdown = {
            arch_id: {
                oid: round(cnts[oid] / max(sum(cnts.values()), 1), 4)
                for oid in option_ids
            }
            for arch_id, cnts in archetype_counts.items()
        }

        metrics = DistributionMetrics(
            option_distribution=dist,
            shannon_entropy=round(entropy, 4),
            normalized_entropy=round(norm_entropy, 4),
            long_tail_representation_rate=round(long_tail_count / max(total_n, 1), 4),
            distinct_options_selected=sum(1 for v in dist.values() if v > 0),
            effective_sample_size=effective_size,
        )

        self.vault.log_audit(
            event_type="cohort_simulation",
            caller_agent=query.caller_agent,
            subject_or_cohort_id=query.cohort_id,
            domain=profile.domain,
            action="simulated",
            confidence=None,
            details={
                "simulation_id": query.simulation_id,
                "sample_size": total_n,
                "normalized_entropy": metrics.normalized_entropy,
                "distribution": dist,
            },
        )

        return CohortSimulationResult(
            simulation_id=query.simulation_id,
            cohort_id=query.cohort_id,
            stimulus=query.stimulus,
            sample_size=total_n,
            adapter_used=self.adapter.adapter_name,
            metrics=metrics,
            archetype_breakdown=archetype_breakdown,
            sample_responses=sample_responses,
            prior_sources=["cohort_profile", "consented_twin_aggregates"] if blended else ["cohort_profile"],
        )
