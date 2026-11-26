from __future__ import annotations

import tempfile
from pathlib import Path

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.benchmarks.datasets import build_cohort_benchmark_suite
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.gate.calibration import jensen_shannon_divergence
from anthromesh.models import CohortQuery
from anthromesh.vault.storage import BehavioralVault


def test_cohort_oracle_preserves_variance_and_long_tail() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = BehavioralVault(db_path=str(Path(tmpdir) / "cohort.db"))
        oracle = ResearchCohortOracle(vault=vault, adapter=HeuristicAdapter())
        profile, options, stimulus, gt_dist = build_cohort_benchmark_suite()
        oracle.register_cohort(profile)

        res = oracle.simulate(
            CohortQuery(
                cohort_id=profile.cohort_id,
                stimulus=stimulus,
                options=options,
                sample_size=200,
                preserve_long_tail=True,
                random_seed=42,
            )
        )
        assert res.metrics.effective_sample_size == 200
        assert res.metrics.distinct_options_selected == 4
        assert res.metrics.normalized_entropy > 0.75
        assert res.metrics.long_tail_representation_rate >= 0.25
        jsd = jensen_shannon_divergence(res.metrics.option_distribution, gt_dist)
        assert jsd < 0.05
