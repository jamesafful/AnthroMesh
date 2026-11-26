"""Human-anchored estimation of a binary population proportion."""
from __future__ import annotations

from math import log, sqrt
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field


class ForecastRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    unit: str = Field(min_length=1)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    group: str = Field(min_length=1)


class AnchorRow(ForecastRow):
    outcome: int = Field(ge=0, le=1)


class AnchorStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    sampling_design: str = Field(min_length=1)
    anchors: list[AnchorRow] = Field(min_length=2)
    predictions: list[ForecastRow] = Field(min_length=2)
    alpha: float = Field(default=.05, gt=0, lt=1)
    minimum_subgroup_size: int = Field(default=10, ge=2)


def analyze_anchors(study: AnchorStudy) -> dict:
    """Rectify fixed predictions using disjoint representative IID samples."""
    units = [row.unit for row in (*study.anchors, *study.predictions)]
    if len(set(units)) != len(units):
        raise ValueError("Each independent sampling unit must occur once across both samples.")
    groups = sorted({row.group for row in (*study.anchors, *study.predictions)})
    cells = [(None, study.anchors, study.predictions)]
    suppressed = []
    for group in groups:
        anchors = [row for row in study.anchors if row.group == group]
        predictions = [row for row in study.predictions if row.group == group]
        if min(len(anchors), len(predictions)) >= study.minimum_subgroup_size:
            cells.append((group, anchors, predictions))
        else:
            suppressed.append(group)
    # Simultaneous bounds on prediction mean, residual mean and human-only mean.
    # Residuals are in [-1, 1]; other components have range length one.
    factor = log(6 * len(cells) / study.alpha) / 2
    reports = []
    for group, anchors, predictions in cells:
        predicted = fmean(row.probability for row in predictions)
        anchor_radius = sqrt(factor / len(anchors))
        prediction_radius = sqrt(factor / len(predictions))
        correction = fmean(row.outcome - row.probability for row in anchors)
        corrected = predicted + correction
        human = fmean(row.outcome for row in anchors)
        radius = 2 * anchor_radius + prediction_radius

        def interval(estimate: float, half_width: float) -> list[float]:
            return [min(1.0, max(0.0, estimate + sign * half_width)) for sign in (-1, 1)]

        residual_interval = [max(-1, correction - 2 * anchor_radius),
                             min(1, correction + 2 * anchor_radius)]
        reports.append({
            "group": group, "anchors": len(anchors), "unlabeled": len(predictions),
            "synthetic_mean": predicted, "mean_residual": correction,
            "residual_interval": residual_interval,
            "mean_absolute_anchor_error": fmean(abs(row.outcome - row.probability) for row in anchors),
            "rectified_unclipped": corrected,
            "rectified_proportion": max(0.0, min(1.0, corrected)),
            "rectified_interval": interval(corrected, radius),
            "human_only_proportion": human, "human_only_interval": interval(human, anchor_radius),
        })
    subgroup_reports = reports[1:]
    bias_gap = None
    if len(subgroup_reports) >= 2:
        bias_gap = {
            "point": max(r["mean_residual"] for r in subgroup_reports)
                     - min(r["mean_residual"] for r in subgroup_reports),
            "simultaneous_upper_bound": min(2.0, max(
                a["residual_interval"][1] - b["residual_interval"][0]
                for i, a in enumerate(subgroup_reports)
                for j, b in enumerate(subgroup_reports) if i != j
            )),
        }
    return {"outcome_id": study.outcome_id, "model_revision": study.model_revision,
            "sampling_design": study.sampling_design, "confidence_level": 1 - study.alpha,
            "method": "sample-split prediction-powered mean with simultaneous Hoeffding bounds",
            "assumptions": "Independent representative units from the same target; predictions fixed before viewing anchors.",
            "population": reports[0], "subgroups": subgroup_reports,
            "suppressed_subgroups": suppressed, "subgroup_signed_bias_range": bias_gap}
