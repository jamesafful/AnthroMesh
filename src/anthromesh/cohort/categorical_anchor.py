"""Categorical prediction-powered estimates with fixed or independently tuned weights."""
from __future__ import annotations

from math import isclose, sqrt
from statistics import NormalDist, fmean, variance

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    unit: str = Field(min_length=1)
    group: str = Field(min_length=1)
    probabilities: dict[str, float]


class LabeledForecast(Forecast):
    outcome: str


class CategoricalAnchorStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    sampling_design: str = Field(min_length=1)
    categories: list[str] = Field(min_length=2)
    method: str = "power_tuned"
    tuning: list[LabeledForecast] = Field(default_factory=list)
    anchors: list[LabeledForecast] = Field(min_length=2)
    predictions: list[Forecast] = Field(min_length=2)
    alpha: float = Field(default=.05, gt=0, lt=1)
    minimum_subgroup_size: int = Field(default=10, ge=2)

    @model_validator(mode="after")
    def check_samples(self):
        if self.method not in {"power_tuned", "fixed"}:
            raise ValueError("Use power_tuned or fixed.")
        keys = set(self.categories)
        if "" in keys or len(keys) != len(self.categories):
            raise ValueError("Categories must be distinct and nonempty.")
        if self.method == "power_tuned" and len(self.tuning) < 2:
            raise ValueError("Power tuning requires a separate labeled sample.")
        rows = [*self.tuning, *self.anchors, *self.predictions]
        if len({row.unit for row in rows}) != len(rows):
            raise ValueError("Tuning, anchor, and forecast units must be disjoint and unique.")
        for row in rows:
            if (row.probabilities.keys() != keys
                    or not all(0 <= p <= 1 for p in row.probabilities.values())
                    or not isclose(sum(row.probabilities.values()), 1, abs_tol=1e-8)):
                raise ValueError("Every prediction must be a full probability vector summing to one.")
            if isinstance(row, LabeledForecast) and row.outcome not in keys:
                raise ValueError("Unknown human response category.")
        return self


def _weight(y, p, sample_ratio):
    y_mean, p_mean = fmean(y), fmean(p)
    total = sum((p_value - p_mean) ** 2 for p_value in p)
    if total == 0:
        return 0.0
    covariance = sum((a - y_mean) * (b - p_mean) for a, b in zip(y, p))
    return min(1.0, max(0.0, covariance / (total * (1 + sample_ratio))))


def _estimate(y, anchor_p, unseen_p, tuning_y, tuning_p, method, z):
    weight = 1.0 if method == "fixed" else _weight(tuning_y, tuning_p, len(y) / len(unseen_p))
    residuals = [a - weight * b for a, b in zip(y, anchor_p)]
    raw = fmean(unseen_p)
    human = fmean(y)
    corrected = fmean(residuals) + weight * raw
    uncertainty = sqrt(variance(residuals) / len(y) + weight**2 * variance(unseen_p) / len(unseen_p))
    if uncertainty < 1e-12:
        interval = [0.0, 1.0]
    else:
        interval = [min(1.0, max(0.0, corrected + sign * z * uncertainty)) for sign in (-1, 1)]

    # Wilson score interval, using the same simultaneous normal quantile.
    n = len(y)
    denom = 1 + z**2 / n
    center = (human + z**2 / (2 * n)) / denom
    half = z * sqrt(human * (1 - human) / n + z**2 / (4 * n**2)) / denom
    return {"weight": weight, "raw_forecast": raw, "human_only": human,
            "anchor_mean_signed_error": fmean(p - a for a, p in zip(y, anchor_p)),
            "anchor_mean_absolute_error": fmean(abs(p - a) for a, p in zip(y, anchor_p)),
            "estimate_unclipped": corrected, "interval": interval,
            "human_only_interval": [max(0.0, center - half), min(1.0, center + half)]}


def analyze_categories(study: CategoricalAnchorStudy) -> dict:
    groups = sorted({row.group for row in [*study.anchors, *study.predictions]})
    cells = [(None, study.anchors, study.predictions, study.tuning)]
    suppressed = []
    for group in groups:
        a, p, t = ([r for r in sample if r.group == group]
                   for sample in (study.anchors, study.predictions, study.tuning))
        if min(len(a), len(p)) < study.minimum_subgroup_size or (study.method == "power_tuned" and len(t) < 2):
            suppressed.append(group)
        else:
            cells.append((group, a, p, t))
    z = NormalDist().inv_cdf(1 - study.alpha / (2 * len(cells) * len(study.categories)))
    results = []
    for group, anchors, predictions, tuning in cells:
        marginals = {}
        for key in study.categories:
            marginals[key] = _estimate(
                [int(r.outcome == key) for r in anchors], [r.probabilities[key] for r in anchors],
                [r.probabilities[key] for r in predictions], [int(r.outcome == key) for r in tuning],
                [r.probabilities[key] for r in tuning], study.method, z,
            )
        results.append({"group": group, "anchors": len(anchors), "unlabeled": len(predictions),
                        "tuning": len(tuning), "marginals": marginals})
    return {"outcome_id": study.outcome_id, "model_revision": study.model_revision,
            "sampling_design": study.sampling_design, "confidence_level": 1 - study.alpha,
            "method": study.method, "interval_type": "asymptotic Bonferroni marginal intervals",
            "assumptions": "Fixed predictor; separate IID samples from the same target population.",
            "population": results[0], "subgroups": results[1:], "suppressed_subgroups": suppressed}
