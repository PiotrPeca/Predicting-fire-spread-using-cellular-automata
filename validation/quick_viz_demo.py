from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on path when running from repo root (same style as validation_run.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fire_spread.validation.visualisation import GridVisualizer


# --- Simple config (edit these values if you want different outputs) ---
OUT_DIR = REPO_ROOT / "validation" / "_demo_outputs"
GRID_H = 300
GRID_W = 600
SEED = 7
T_MIN = 220.0
BURN_DURATION_MIN = 60.0

# Static validation indicators shown on the error map (easy to edit/extend).
ERROR_METRICS: list[tuple[str, float]] = [("rmsd", 10.9), ("bias", -4.4)]


def _majority_smooth(mask: np.ndarray, *, iterations: int = 6, threshold: int = 5) -> np.ndarray:
    """Simple morphological smoothing without extra deps.

    Uses a majority rule over a 3x3 neighborhood.
    """

    mask = mask.astype(bool)
    for _ in range(iterations):
        neigh = (
            mask
            + np.roll(mask, 1, 0)
            + np.roll(mask, -1, 0)
            + np.roll(mask, 1, 1)
            + np.roll(mask, -1, 1)
            + np.roll(np.roll(mask, 1, 0), 1, 1)
            + np.roll(np.roll(mask, 1, 0), -1, 1)
            + np.roll(np.roll(mask, -1, 0), 1, 1)
            + np.roll(np.roll(mask, -1, 0), -1, 1)
        )
        mask = neigh >= threshold
    return mask


def _make_irregular_burn_mask(*, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Irregular mask that roughly resembles a burned area footprint."""

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    # Broad envelope (ellipse) to keep the mask in a plausible region.
    cx, cy = width * 0.52, height * 0.58
    a, b = width * 0.22, height * 0.30
    envelope = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) < 1.0

    # Start with random noise inside the envelope and smooth it into blobs.
    seed = (rng.random((height, width)) < 0.56) & envelope
    blob = _majority_smooth(seed, iterations=6, threshold=5)

    # Safety: ensure the mask isn't empty (randomness + smoothing can wipe it out).
    min_area = int(0.02 * height * width)
    if int(blob.sum()) < min_area:
        blob = envelope

    # Carve a couple of "unburned" holes (like lakes / islands) for realism.
    hole_seed = (rng.random((height, width)) < 0.04) & blob
    holes = _majority_smooth(hole_seed, iterations=5, threshold=6)

    mask = blob & (~holes)
    if not mask.any():
        mask = blob
    return mask


def _make_synthetic_toa(
    *,
    height: int,
    width: int,
    ignition_xy: tuple[float, float],
    minutes_per_cell: float,
) -> np.ndarray:
    """Synthetic time-of-arrival map.

    Interpreted as: time (minutes from t0) when a cell FIRST becomes BURNING.
    """

    x0, y0 = ignition_xy
    xs = np.arange(width, dtype=float)
    ys = np.arange(height, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return dist * minutes_per_cell


def _burning_mask_from_toa(toa: np.ndarray, *, t_min: float, burn_duration_min: float) -> np.ndarray:
    """Return boolean mask for cells currently BURNING at time t_min."""

    return (t_min >= toa) & (t_min < (toa + burn_duration_min))


def main() -> int:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # 1) Define an analysis mask (e.g., final burned area boundary).
    # Here: an irregular blob that resembles a burned-area footprint.
    mask = _make_irregular_burn_mask(height=GRID_H, width=GRID_W, rng=rng)

    # 2) Create synthetic TOA fields and scale them to a paper-like range (~0..320 min).
    ign_real = (GRID_W * 0.30, GRID_H * 0.55)
    ign_sim = (GRID_W * 0.34, GRID_H * 0.50)

    raw_dist = _make_synthetic_toa(height=GRID_H, width=GRID_W, ignition_xy=ign_real, minutes_per_cell=1.0)
    if mask.any():
        max_dist = float(np.nanmax(raw_dist[mask]))
    else:
        max_dist = float(np.nanmax(raw_dist))
    minutes_per_cell = 320.0 / max(max_dist, 1.0)

    real_toa = _make_synthetic_toa(
        height=GRID_H,
        width=GRID_W,
        ignition_xy=ign_real,
        minutes_per_cell=minutes_per_cell,
    )

    sim_toa = _make_synthetic_toa(
        height=GRID_H,
        width=GRID_W,
        ignition_xy=ign_sim,
        minutes_per_cell=minutes_per_cell * 0.98,
    )

    # Add mild noise to simulation to mimic stochasticity.
    sim_toa = sim_toa + rng.normal(loc=0.0, scale=6.0, size=sim_toa.shape)

    # 2b) Simulate an interrupted run: the simulation didn't reach the whole mask.
    # We'll represent "not reached" as NaN (so those pixels stay uncolored).
    # Cut off by time (stopped early)...
    time_limit = float(np.nanpercentile(real_toa[mask], 65))
    reached = (sim_toa <= time_limit) & mask

    # ...and also carve an internal unreached pocket inside the mask.
    pocket = _make_irregular_burn_mask(height=GRID_H, width=GRID_W, rng=rng) & mask
    pocket = _majority_smooth(pocket, iterations=3, threshold=6)
    reached = reached & (~pocket)

    sim_toa = np.where(reached, sim_toa, np.nan)

    # Synthetic "terrain" background (grayscale): 0=water, 1=forest-ish.
    # This imitates a landcover layer drawn behind the TOA heatmap.
    # Synthetic "terrain" background (grayscale): imitates landcover underlay.
    yy, xx = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
    background = 0.35 + 0.25 * (np.sin(xx / 18.0) * np.cos(yy / 22.0))
    background = np.clip(background, 0.0, 1.0)
    # Darker outside the analysis region so the overlay pops.
    background = np.where(mask, background, background * 0.7)

    # 3) Derive example "currently burning" masks from TOA.
    real_burning = _burning_mask_from_toa(real_toa, t_min=T_MIN, burn_duration_min=BURN_DURATION_MIN) & mask
    sim_burning = _burning_mask_from_toa(sim_toa, t_min=T_MIN, burn_duration_min=BURN_DURATION_MIN) & mask

    # 4) Visualise.
    viz = GridVisualizer(show_colorbar=True, origin="upper")

    # A) TOA comparison (2 grids side-by-side)
    if mask.any():
        vmin = float(np.nanmin(real_toa[mask]))
        vmax = float(np.nanmax(real_toa[mask]))
    else:
        vmin = float(np.nanmin(real_toa))
        vmax = float(np.nanmax(real_toa))

    fig_toa = viz.plot_two_grids(
        real_toa,
        sim_toa,
        background=background,
        show_axes=True,
        tick_step=100,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        cbar_label="TOA (minutes from t0)",
    )
    viz.save(fig_toa, str(out_dir / "toa_compare.png"))

    # A2) Single TOA maps (paper-like look: warm colors + mask boundary)
    ax_single_real = viz.plot_toa(
        real_toa,
        title="TOA (Real)",
        background=background,
        mask=mask,
        show_axes=True,
        tick_step=100,
    )
    viz.save(ax_single_real.figure, str(out_dir / "toa_single_real.png"))

    ax_single_sim = viz.plot_toa(
        sim_toa,
        title="TOA (Sim)",
        background=background,
        mask=mask,
        show_axes=True,
        tick_step=100,
    )
    viz.save(ax_single_sim.figure, str(out_dir / "toa_single_sim.png"))

    # B) Error map (sim - real), masked to analysis region (paper-like: background + axes)
    fig_err = viz.plot_error_map(
        sim_toa,
        real_toa,
        mask=mask,
        background=background,
        metrics=ERROR_METRICS,
        show_axes=True,
        tick_step=100,
    )
    viz.save(fig_err, str(out_dir / "error_map.png"))

    # C) Burning masks comparison (boolean grids)
    fig_burn = viz.plot_two_grids(
        real_burning.astype(int),
        sim_burning.astype(int),
        background=background,
        show_axes=True,
        tick_step=100,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    viz.save(fig_burn, str(out_dir / "burning_compare.png"))

    print(f"Saved demo images to: {out_dir.resolve()}")
    print("Files: toa_compare.png, error_map.png, burning_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
