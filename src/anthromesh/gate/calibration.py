from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from anthromesh.models import DecisionLog


def compute_evidence_coverage(
    evidence: list[tuple[DecisionLog, float]],
    min_evidence_count: int = 2,
) -> float:
    """Scores how well historical evidence covers the current query (0..1)."""
    if not evidence:
        return 0.0
    sims = [max(0.0, min(1.0, sim)) for _, sim in evidence]
    top_k = sims[: max(min_evidence_count, 1)]
    mean_sim = sum(top_k) / len(top_k)
    count_factor = min(1.0, len(evidence) / max(min_evidence_count, 1))
    return round(max(0.0, min(1.0, 0.72 * mean_sim + 0.28 * count_factor)), 4)


def compute_historical_agreement(
    chosen_option: str,
    evidence: list[tuple[DecisionLog, float]],
) -> float:
    """Weighted fraction of retrieved decisions supporting the predicted action."""
    if not evidence:
        return 0.0
    total_w = sum(max(sim, 0.01) for _, sim in evidence)
    agree_w = sum(max(sim, 0.01) for dec, sim in evidence if dec.chosen_action == chosen_option)
    return round(max(0.0, min(1.0, agree_w / total_w)), 4)


def compute_calibration_confidence(
    evidence_coverage: float,
    agreement_score: float,
    model_confidence: float,
    risk_severity: int = 1,
) -> float:
    """Fuses empirical evidence coverage, historical agreement, and model confidence with risk damping."""
    raw = 0.45 * evidence_coverage + 0.30 * agreement_score + 0.25 * model_confidence
    risk_penalty = max(0.0, (risk_severity - 1) * 0.08)
    return round(max(0.0, min(1.0, raw - risk_penalty)), 4)


def compute_ece(
    confidences: Sequence[float],
    correctness: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across equally spaced confidence bins."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    conf_arr = np.asarray(confidences, dtype=float)
    corr_arr = np.asarray(correctness, dtype=float)
    n = len(conf_arr)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf_arr >= lo) & (conf_arr <= hi if i == n_bins - 1 else conf_arr < hi)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(conf_arr[mask]))
            avg_acc = float(np.mean(corr_arr[mask]))
            ece += (count / n) * abs(avg_acc - avg_conf)
    return round(float(ece), 4)


def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Computes bounded Jensen-Shannon divergence (0..1 using base-2 log) between two discrete distributions."""
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    p_vec = np.array([max(0.0, float(p.get(k, 0.0))) for k in keys], dtype=float)
    q_vec = np.array([max(0.0, float(q.get(k, 0.0))) for k in keys], dtype=float)
    p_sum, q_sum = float(p_vec.sum()), float(q_vec.sum())
    if p_sum == 0.0 or q_sum == 0.0:
        return 1.0
    p_vec /= p_sum
    q_vec /= q_sum
    m = 0.5 * (p_vec + q_vec)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                total += float(ai) * math.log2(float(ai) / float(bi))
        return total

    jsd = 0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m)
    return round(max(0.0, min(1.0, jsd)), 4)
