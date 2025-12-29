from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


ArrayLike = Any


@dataclass(frozen=True)
class Confusion:
    """Confusion matrix counts for binary burned/unburned maps."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)

    @property
    def iou(self) -> float:
        # Intersection over Union for the positive (burned) class
        return _safe_div(self.tp, self.tp + self.fp + self.fn)


@dataclass(frozen=True)
class ToaErrorMetrics:
    """Regression-style metrics computed on cells where both TOAs are finite."""

    n: int
    bias: float
    mae: float
    rmse: float


@dataclass(frozen=True)
class FitnessComponents:
    """Components used to construct a single fitness score."""

    rmse: float
    fp_rate: float
    fn_rate: float
    score: float


def _as_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    return arr


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def burned_mask(toa: ArrayLike) -> np.ndarray:
    """Convert TOA grid to a burned/unburned boolean map.

    NaN/inf values are treated as unburned.
    """

    t = _as_float_array(toa)
    return np.isfinite(t)


def confusion_matrix(
    sim_toa: ArrayLike,
    real_toa: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> Confusion:
    """Compute TP/FP/FN/TN based on burned/unburned classification.

    Evaluation domain:
    - If `mask` is provided, counts are computed only where mask is True.
    - Otherwise, defaults to the *real burned mask* (finite real TOA cells).
      This matches the project assumption that we validate only inside the
      real-fire mask.
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    domain = burned_mask(real) if mask is None else np.asarray(mask, dtype=bool)

    sim_b = burned_mask(sim)
    real_b = burned_mask(real)

    tp = int(np.sum(domain & sim_b & real_b))
    fp = int(np.sum(domain & sim_b & ~real_b))
    fn = int(np.sum(domain & ~sim_b & real_b))
    tn = int(np.sum(domain & ~sim_b & ~real_b))

    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def toa_error_metrics(
    sim_toa: ArrayLike,
    real_toa: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> ToaErrorMetrics:
    """Compute bias/MAE/RMSE on cells where both TOAs are finite.

    - If mask is provided, it must be broadcastable to TOA shape and will be
      AND-ed with the finite mask.
    - Cells where either sim or real is NaN/inf are excluded.

    Returns n=0 and zeros for metrics if there are no comparable cells.
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    valid = np.isfinite(sim) & np.isfinite(real)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)

    if not np.any(valid):
        return ToaErrorMetrics(n=0, bias=0.0, mae=0.0, rmse=0.0)

    diff = sim[valid] - real[valid]
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    return ToaErrorMetrics(n=int(diff.size), bias=bias, mae=mae, rmse=rmse)


def calculate_fitness(
    sim_toa: ArrayLike,
    real_toa: ArrayLike,
    *,
    mode: Literal["rmse", "rmse+burn"] = "rmse+burn",
    fp_weight: float = 1.0,
    fn_weight: float = 1.0,
    rmse_weight: float = 1.0,
    mask: ArrayLike | None = None,
    return_components: bool = False,
) -> float | FitnessComponents:
    """Return a single-number fitness (lower is better).

    Intended as the team-facing API: `calculate_fitness(sim_data, real_data)`.

    Modes:
    - "rmse": RMSE computed only where both TOAs are finite.
    - "rmse+burn": RMSE + penalties for burned-map disagreement:
        fp_rate = FP / (FP + TN)  (sim burns where real doesn't)
        fn_rate = FN / (FN + TP)  (sim misses real burned cells)

    Notes:
    - By default we evaluate only inside the real-fire mask (finite real TOA).
      You can override this by passing an explicit `mask`.
    - NaNs are treated as "unburned" for the burned-map penalties.
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    domain = burned_mask(real) if mask is None else np.asarray(mask, dtype=bool)

    err = toa_error_metrics(sim, real, mask=domain)
    base = err.rmse

    if mode == "rmse":
        score = rmse_weight * base
        comps = FitnessComponents(rmse=base, fp_rate=0.0, fn_rate=0.0, score=score)
        return comps if return_components else score

    if mode != "rmse+burn":
        raise ValueError(f"Unknown mode: {mode}")

    sim_b = burned_mask(sim)
    real_b = burned_mask(real)

    tp = int(np.sum(domain & sim_b & real_b))
    fp = int(np.sum(domain & sim_b & ~real_b))
    fn = int(np.sum(domain & ~sim_b & real_b))
    tn = int(np.sum(domain & ~sim_b & ~real_b))

    fp_rate = _safe_div(fp, fp + tn)
    fn_rate = _safe_div(fn, fn + tp)

    score = (rmse_weight * base) + (fp_weight * fp_rate) + (fn_weight * fn_rate)
    comps = FitnessComponents(rmse=base, fp_rate=fp_rate, fn_rate=fn_rate, score=float(score))
    return comps if return_components else float(score)
