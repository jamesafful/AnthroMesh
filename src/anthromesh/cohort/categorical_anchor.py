"""Categorical prediction-powered estimates with fixed, independently tuned, or cross-fitted forecasts."""
from __future__ import annotations

from hashlib import sha256
from math import isclose, sqrt
from random import Random
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
    method: str = "human_only"
    anchor_seed: int | None = None
    anchor_model_revision: str | None = None
    max_forecast_shift: float = Field(default=.15, gt=0, le=1)
    folds: int = Field(default=5, ge=2)
    fold_seed: int = 0
    tuning: list[LabeledForecast] = Field(default_factory=list)
    anchors: list[LabeledForecast] = Field(min_length=2)
    predictions: list[Forecast] = Field(min_length=2)
    alpha: float = Field(default=.05, gt=0, lt=1)
    minimum_subgroup_size: int = Field(default=10, ge=2)

    @model_validator(mode="before")
    @classmethod
    def preserve_implicit_tuning(cls, data):
        if isinstance(data, dict) and "method" not in data and data.get("tuning"):
            return {**data, "method": "power_tuned"}
        return data

    @model_validator(mode="after")
    def check_samples(self):
        if self.method not in {"human_only", "power_tuned", "fixed", "cross_ppi", "guarded_cross_ppi"}:
            raise ValueError("Unknown categorical calibration method.")
        keys = set(self.categories)
        if "" in keys or len(keys) != len(self.categories):
            raise ValueError("Categories must be distinct and nonempty.")
        if self.method == "power_tuned" and len(self.tuning) < 2:
            raise ValueError("Power tuning requires a separate labeled sample.")
        if self.method in {"cross_ppi", "guarded_cross_ppi"}:
            if self.tuning:
                raise ValueError("Cross-PPI uses the anchors for cross-fitting; leave tuning empty.")
            if self.method == "cross_ppi" and len(self.anchors) < 2 * self.folds:
                raise ValueError("Cross-PPI needs at least twice as many anchors as folds.")
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


def draw_anchor_units(units: list[str], count: int, seed: int) -> list[str]:
    """Draw without replacement from the frozen, unlabeled sampling frame."""
    if len(set(units)) != len(units) or any(not unit for unit in units):
        raise ValueError("Sampling frame unit IDs must be nonempty and unique.")
    if not 2 <= count <= len(units) - 2:
        raise ValueError("Retain at least two anchors and two unlabeled units.")
    return sorted(Random(seed).sample(sorted(units), count))


def analyze_guarded_categories(study: CategoricalAnchorStudy) -> dict:
    """Apply selection and forecast-drift checks before using Cross-PPI."""
    selection_verified = study.anchor_seed is not None and set(draw_anchor_units(
        [row.unit for row in (*study.anchors, *study.predictions)],
        len(study.anchors), study.anchor_seed,
    )) == {row.unit for row in study.anchors}
    shift = sum(abs(
        fmean(row.probabilities[key] for row in study.anchors)
        - fmean(row.probabilities[key] for row in study.predictions)
    ) for key in study.categories) / 2
    reasons = []
    if not selection_verified:
        reasons.append("anchor_draw_mismatch")
    if study.anchor_model_revision != study.model_revision:
        reasons.append("forecast_revision_mismatch")
    if shift > study.max_forecast_shift:
        reasons.append("forecast_shift")
    if len(study.anchors) < 2 * study.folds:
        reasons.append("insufficient_anchors_for_cross_fitting")
    if not reasons:
        result = analyze_categories(study.model_copy(update={"method": "cross_ppi"}))
        cells = [result["population"], *result["subgroups"]]
        if any(
            row["interval"][1] - row["interval"][0]
            >= row["human_only_interval"][1] - row["human_only_interval"][0]
            for cell in cells for row in cell["marginals"].values()
        ):
            reasons.append("intervals_not_uniformly_narrower")
    if reasons:
        result = analyze_categories(study.model_copy(update={"method": "human_only"}))
    if not selection_verified:
        # Selection failure invalidates population inference even for human-only.
        result["confidence_level"] = None
        result["interval_type"] = "descriptive selected-anchor proportions"
        for cell in [result["population"], *result["subgroups"]]:
            for row in cell["marginals"].values():
                row["interval"] = row["human_only_interval"] = None
    result["guard"] = {"selection_verified": selection_verified,
                       "forecast_mean_total_variation": shift, "fallback_reasons": reasons}
    return result


def _weight(y, p, sample_ratio):
    y_mean, p_mean = fmean(y), fmean(p)
    total = sum((p_value - p_mean) ** 2 for p_value in p)
    if total == 0:
        return 0.0
    covariance = sum((a - y_mean) * (b - p_mean) for a, b in zip(y, p))
    return min(1.0, max(0.0, covariance / (total * (1 + sample_ratio))))


def _estimate(y, anchor_p, unseen_p, tuning_y, tuning_p, method, z):
    if method == "human_only":
        weight = 0.0
    elif method == "fixed":
        weight = 1.0
    else:
        weight = _weight(tuning_y, tuning_p, len(y) / len(unseen_p))
    residuals = [a - weight * b for a, b in zip(y, anchor_p)]
    return _summarize(y, anchor_p, unseen_p, residuals,
                      [weight * p for p in unseen_p], weight, z)


def _cross_estimate(y, anchor_p, unseen_p, folds, z):
    residuals = [0.0] * len(y)
    average_weight = 0.0
    for held_out in folds:
        excluded = set(held_out)
        train = [i for i in range(len(y)) if i not in excluded]
        weight = _weight([y[i] for i in train], [anchor_p[i] for i in train],
                         len(y) / len(unseen_p))
        for i in held_out:
            residuals[i] = y[i] - weight * anchor_p[i]
        average_weight += weight * len(held_out) / len(y)
    return _summarize(y, anchor_p, unseen_p, residuals,
                      [average_weight * p for p in unseen_p], average_weight, z)


def _summarize(y, anchor_p, unseen_p, residuals, imputed, weight, z):
    raw = fmean(unseen_p)
    human = fmean(y)
    corrected = fmean(residuals) + fmean(imputed)
    uncertainty = sqrt(variance(residuals) / len(y) + variance(imputed) / len(imputed))
    if uncertainty < 1e-12:
        interval = [0.0, 1.0]
    else:
        interval = [min(1.0, max(0.0, corrected + sign * z * uncertainty)) for sign in (-1, 1)]

    # Wilson score interval, using the same simultaneous normal quantile.
    n = len(y)
    denom = 1 + z**2 / n
    center = (human + z**2 / (2 * n)) / denom
    half = z * sqrt(human * (1 - human) / n + z**2 / (4 * n**2)) / denom
    human_interval = [max(0.0, center - half), min(1.0, center + half)]
    return {"weight": weight, "raw_forecast": raw, "human_only": human,
            "anchor_mean_signed_error": fmean(p - a for a, p in zip(y, anchor_p)),
            "anchor_mean_absolute_error": fmean(abs(p - a) for a, p in zip(y, anchor_p)),
            "estimate_unclipped": corrected, "interval": interval,
            "human_only_interval": human_interval}


def analyze_categories(study: CategoricalAnchorStudy) -> dict:
    if study.method == "guarded_cross_ppi":
        return analyze_guarded_categories(study)
    groups = sorted({row.group for row in [*study.anchors, *study.predictions]})
    cells = [(None, study.anchors, study.predictions, study.tuning)]
    suppressed = []
    for group in groups:
        a, p, t = ([r for r in sample if r.group == group]
                   for sample in (study.anchors, study.predictions, study.tuning))
        minimum_anchors = max(study.minimum_subgroup_size, 2 * study.folds) if study.method == "cross_ppi" else study.minimum_subgroup_size
        if (len(a) < minimum_anchors or len(p) < study.minimum_subgroup_size
                or (study.method == "power_tuned" and len(t) < 2)):
            suppressed.append(group)
        else:
            cells.append((group, a, p, t))
    z = NormalDist().inv_cdf(1 - study.alpha / (2 * len(cells) * len(study.categories)))
    results = []
    for group, anchors, predictions, tuning in cells:
        folds = []
        if study.method == "cross_ppi":
            ordered = sorted(range(len(anchors)), key=lambda i: (
                sha256(f"{study.fold_seed}:{anchors[i].unit}".encode()).digest(), anchors[i].unit
            ))
            folds = [ordered[i::study.folds] for i in range(study.folds)]
        marginals = {}
        for key in study.categories:
            y = [int(r.outcome == key) for r in anchors]
            anchor_p = [r.probabilities[key] for r in anchors]
            unseen_p = [r.probabilities[key] for r in predictions]
            if folds:
                marginals[key] = _cross_estimate(y, anchor_p, unseen_p, folds, z)
            else:
                marginals[key] = _estimate(
                    y, anchor_p, unseen_p, [int(r.outcome == key) for r in tuning],
                    [r.probabilities[key] for r in tuning], study.method, z,
                )
        if study.method == "human_only":
            marginals = {
                key: {"human_only": row["human_only"], "estimate_unclipped": row["human_only"],
                      "interval": row["human_only_interval"], "human_only_interval": row["human_only_interval"]}
                for key, row in marginals.items()
            }
        results.append({"group": group, "anchors": len(anchors), "unlabeled": len(predictions),
                        "tuning": len(tuning), "marginals": marginals})
    assumptions = "Fixed predictor; separate IID samples from the same target population."
    metadata = {}
    if study.method == "human_only":
        assumptions = "Representative independent human anchors from the target population."
    if study.method == "cross_ppi":
        assumptions += " Cross-fitted scaling requires a stable fold learner for asymptotic inference."
        metadata = {"folds": study.folds, "fold_seed": study.fold_seed}
    return {"outcome_id": study.outcome_id, "model_revision": study.model_revision,
            "sampling_design": study.sampling_design, "confidence_level": 1 - study.alpha,
            "method": study.method, "interval_type": "asymptotic Bonferroni marginal intervals",
            "assumptions": assumptions, **metadata,
            "population": results[0], "subgroups": results[1:], "suppressed_subgroups": suppressed}
