from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from anthromesh.models import CohortArchetype, CohortProfile


def read_survey(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        for batch in parquet.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)


def build_empirical_profile(
    rows: Iterable[dict],
    *,
    cohort_id: str,
    domain: str,
    traits: list[str],
    response: str,
    weight: str | None = None,
    margins: dict[str, dict[str, float]] | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> CohortProfile:
    """Estimate weighted response priors by observed survey cell and rake cell weights."""
    if not traits or len(set(traits)) != len(traits) or response in traits:
        raise ValueError("Provide distinct trait columns separate from the response column.")
    tallies: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for row in rows:
        values = [row.get(column) for column in [*traits, response]]
        if any(value is None or not str(value).strip() for value in values):
            raise ValueError("Survey traits and responses must be present on every row.")
        cell = tuple(str(value).strip() for value in values[:-1])
        mass = float(row[weight]) if weight else 1.0
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("Survey weights must be positive and finite.")
        tallies[cell][str(values[-1]).strip()] += mass

    if len(tallies) < 2:
        raise ValueError("At least two observed trait combinations are required.")
    cell_mass = {cell: sum(counts.values()) for cell, counts in tallies.items()}
    total = sum(cell_mass.values())
    if not math.isfinite(total):
        raise ValueError("Total survey weight must be finite.")
    probabilities = {cell: mass / total for cell, mass in cell_mass.items()}

    groups = []
    for trait, targets in (margins or {}).items():
        if trait not in traits:
            raise ValueError(f"Unknown margin trait {trait!r}.")
        index = traits.index(trait)
        cells_by_value = defaultdict(list)
        for cell in tallies:
            cells_by_value[cell[index]].append(cell)
        if set(targets) != set(cells_by_value):
            raise ValueError(f"Margin categories for {trait!r} must match observed categories.")
        if any(not math.isfinite(value) or value <= 0 for value in targets.values()):
            raise ValueError(f"Margin targets for {trait!r} must be positive and finite.")
        if not math.isclose(sum(targets.values()), 1.0, abs_tol=tolerance):
            raise ValueError(f"Margin targets for {trait!r} must sum to one.")
        groups.extend((cells, targets[value]) for value, cells in cells_by_value.items())

    for _ in range(max_iterations):
        for cells, target in groups:
            current = sum(probabilities[cell] for cell in cells)
            if current == 0.0:
                raise ValueError("Raking exhausted support on a target margin.")
            for cell in cells:
                probabilities[cell] *= target / current
        if all(abs(sum(probabilities[cell] for cell in cells) - target) <= tolerance
               for cells, target in groups):
            break
    else:
        raise ValueError("Raking did not converge. Check the joint support of target margins.")

    archetypes = []
    for cell, counts in sorted(tallies.items()):
        attributes = dict(zip(traits, cell))
        digest = hashlib.sha256(json.dumps(attributes, sort_keys=True).encode()).hexdigest()
        fraction = probabilities[cell]
        archetypes.append(CohortArchetype(
            archetype_id=digest,
            label=", ".join(f"{trait}={value}" for trait, value in attributes.items()),
            traits=attributes,
            weight=fraction,
            is_long_tail=fraction < 0.10,
            prior_tendencies={answer: mass / cell_mass[cell] for answer, mass in counts.items()},
        ))
    return CohortProfile(
        cohort_id=cohort_id, name=cohort_id, domain=domain,
        population_weighted=True, archetypes=archetypes,
    )
