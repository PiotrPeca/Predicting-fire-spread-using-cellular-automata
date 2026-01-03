from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ArrayLike = Any


@dataclass(frozen=True)
class Confusion:
    """Confusion matrix counts for binary burned/unburned maps."""

    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class ToaErrorMetrics:
    """Regression-style metrics computed on cells where both TOAs are finite."""

    n: int
    bias: float
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
        - Otherwise, defaults to the full grid.

        Notes:
        - A cell is treated as "burned" if its TOA value is finite.
        - For meaningful FP/TN rates, prefer passing a study-area mask (e.g. boundary
            polygon mapped to grid). Using only the real-burned mask would make FP=0 by
            definition.
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    domain = np.ones_like(real, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

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
    """Compute bias and RMSD/RMSE on cells where both TOAs are finite.

    - If mask is provided, it must be broadcastable to TOA shape and will be
      AND-ed with the finite mask.
    - Cells where either sim or real is NaN/inf are excluded.

    Returns n=0 and zeros for metrics if there are no comparable cells.

        Important:
        - Units: results are in the same units as the input arrays.
            In this project, TOA is intended to be a Unix timestamp (seconds).
        - Terminology: in the papers you may see RMSD (root mean square deviation).
            For our use case RMSD == RMSE (same formula, different naming).
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    valid = np.isfinite(sim) & np.isfinite(real)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)

    if not np.any(valid):
        return ToaErrorMetrics(n=0, bias=0.0, rmse=0.0)

    diff = sim[valid] - real[valid]
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    return ToaErrorMetrics(n=int(diff.size), bias=bias, rmse=rmse)


def calculate_fitness(
    sim_toa: ArrayLike,
    real_toa: ArrayLike,
    *,
    fp_weight: float = 1.0,
    fn_weight: float = 1.0,
    rmse_weight: float = 1.0,
    mask: ArrayLike | None = None,
    return_components: bool = False,
) -> float | FitnessComponents:
    """Main fitness function for validation & grid-search (lower is better).

    This is the team-facing API used by the headless runner / parameter search:
    call it as `calculate_fitness(sim_toa, real_toa, ...)`.

    Inputs (expected):
    - `sim_toa`, `real_toa`: 2D grids (H x W). Values are TOA timestamps.
      Convention: a cell is considered "burned/reached" iff its TOA is finite
      (NaN/inf means "never reached" during the simulation / outside boundary).

    Time metric (TOA regression):
        - We compute RMSD/RMSE only on cells where BOTH TOAs are finite.
      (If sim never reaches a cell, there is no time to compare there.)

    Burned/unburned consistency (confusion penalties):
    - We compute a burned-mask confusion matrix in the evaluation domain and
      penalize disagreement:

        fp_rate = FP / (FP + TN)   # sim burns where real does not
        fn_rate = FN / (FN + TP)   # sim misses real burned cells

    Final score (linear, easy to tune in grid-search):
        score = rmse_weight * rmse + fp_weight * fp_rate + fn_weight * fn_rate

    Masking / evaluation domain:
    - By default the full grid is evaluated.
    - For meaningful FP/TN rates, pass `mask` as the study area / boundary mask
      (e.g. polygon boundary rasterized to grid). This prevents rewarding weird
      behavior outside the area of interest.

    Units / weights (important when using Unix timestamps):
        - If TOA values are Unix timestamps in SECONDS, RMSE will also be in seconds.
      Typical magnitude can be large (e.g. thousands of seconds), while fp_rate/fn_rate
      are in [0, 1]. You MUST scale with `rmse_weight`.
            Example: to effectively measure RMSE in hours, use `rmse_weight = 1/3600`.
      Alternatively, pre-normalize TOA inputs to "hours since t0" before calling.

    Returns:
    - By default: float score.
        - If `return_components=True`: `FitnessComponents` (rmse, fp_rate, fn_rate, score).
    """

    sim = _as_float_array(sim_toa)
    real = _as_float_array(real_toa)
    if sim.shape != real.shape:
        raise ValueError(f"Shape mismatch: sim={sim.shape} real={real.shape}")

    domain = np.ones_like(real, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    err = toa_error_metrics(sim, real, mask=domain)
    base = err.rmse

    cm = confusion_matrix(sim, real, mask=domain)
    fp_rate = _safe_div(cm.fp, cm.fp + cm.tn)
    fn_rate = _safe_div(cm.fn, cm.fn + cm.tp)

    score = (rmse_weight * base) + (fp_weight * fp_rate) + (fn_weight * fn_rate)
    comps = FitnessComponents(rmse=base, fp_rate=fp_rate, fn_rate=fn_rate, score=float(score))
    return comps if return_components else float(score)
