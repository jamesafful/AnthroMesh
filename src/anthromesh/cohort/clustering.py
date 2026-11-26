from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from anthromesh.models import CohortProfile

from .ingest import build_empirical_profile


def _distance(first: tuple[str, ...], second: tuple[str, ...]) -> int:
    return sum(left != right for left, right in zip(first, second))


def _modes(points: Counter, clusters: int, iterations: int) -> list[tuple[str, ...]]:
    candidates = sorted(points)
    if not 2 <= clusters <= min(100, len(points)):
        raise ValueError("Requested clusters exceed the available feature combinations.")
    first = min(candidates, key=lambda point: (-points[point], point))
    centers = [first]
    while len(centers) < clusters:
        centers.append(min((point for point in candidates if point not in centers),
                           key=lambda point: (-min(_distance(point, seed) for seed in centers),
                                              -points[point], point)))

    for _ in range(iterations):
        groups = [[] for _ in centers]
        for point in candidates:
            match = min(range(len(centers)), key=lambda i: (_distance(point, centers[i]), i))
            groups[match].append(point)
        if any(not members for members in groups):
            raise ValueError("Clustering collapsed under the chosen features. Use fewer clusters.")
        updated = []
        for group in groups:
            modes = []
            for column in range(len(first)):
                tally = defaultdict(float)
                for point in group:
                    tally[point[column]] += points[point]
                modes.append(min(tally, key=lambda entry: (-tally[entry], entry)))
            updated.append(tuple(modes))
        if updated == centers:
            return centers
        centers = updated
    raise ValueError("Survey clustering did not converge. Use fewer clusters or simplify the features.")


def build_clustered_profile(
    records: Iterable[dict], *, cohort_id: str, domain: str, strata: list[str],
    features: list[str], response: str, clusters: int, weight: str | None = None,
    margins: dict[str, dict[str, float]] | None = None,
) -> CohortProfile:
    """Fit weighted categorical modes, then rake the cluster-by-stratum response cells."""
    combined = [*strata, *features, response]
    if (not features or len(set(combined)) != len(combined)
            or "__cluster" in combined or "__mass" in combined):
        raise ValueError("Use separate clustering features, population strata, and response columns.")
    compressed = Counter()
    points = Counter()
    for record in records:
        values = [record.get(column) for column in combined]
        if any(value is None or not str(value).strip() for value in values):
            raise ValueError("Clustering requires complete categorical fields.")
        feature_values = tuple(str(value).strip() for value in values[len(strata):-1])
        cells = tuple(str(value).strip() for value in values[:len(strata)])
        outcome = str(values[-1]).strip()
        mass = float(record[weight]) if weight else 1.0
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("Survey weights must be positive and finite.")
        points[feature_values] += mass
        compressed[(feature_values, cells, outcome)] += mass
    centers = _modes(points, clusters, 100)
    labels = [f"cluster-{index + 1}" for index in range(len(centers))]
    assignments = {point: min(range(len(centers)), key=lambda i: (_distance(point, centers[i]), i))
                   for point in points}
    summarized = (
        {**dict(zip(strata, cell)), "__cluster": labels[assignments[features_tuple]],
         response: answer, "__mass": total}
        for (features_tuple, cell, answer), total in compressed.items()
    )
    result = build_empirical_profile(
        summarized, cohort_id=cohort_id, domain=domain, traits=[*strata, "__cluster"],
        response=response, weight="__mass", margins=margins,
    )
    for archetype in result.archetypes:
        cluster = archetype.traits["__cluster"]
        center = centers[labels.index(cluster)]
        archetype.traits = {key: value for key, value in archetype.traits.items() if key != "__cluster"}
        archetype.traits["cluster"] = cluster
        archetype.traits.update(dict(zip(features, center)))
        archetype.label = ", ".join(f"{key}={value}" for key, value in archetype.traits.items())
    return result
