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
GRID_H = 120
GRID_W = 160
SEED = 7
T_MIN = 220.0
BURN_DURATION_MIN = 60.0


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

    # 1) Create a synthetic 'real' TOA field and a slightly different 'sim' TOA.
    real_toa = _make_synthetic_toa(
        height=GRID_H,
        width=GRID_W,
        ignition_xy=(GRID_W * 0.30, GRID_H * 0.55),
        minutes_per_cell=3.5,
    )

    sim_toa = _make_synthetic_toa(
        height=GRID_H,
        width=GRID_W,
        ignition_xy=(GRID_W * 0.34, GRID_H * 0.50),
        minutes_per_cell=3.3,
    )

    # Add mild noise to simulation to mimic stochasticity.
    sim_toa = sim_toa + rng.normal(loc=0.0, scale=8.0, size=sim_toa.shape)

    # 2) Define an analysis mask (e.g., final burned area boundary or study extent).
    # Here: a circle-like region.
    yy, xx = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
    cx, cy = GRID_W * 0.45, GRID_H * 0.55
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(GRID_W, GRID_H) * 0.42) ** 2

    # 3) Derive example "currently burning" masks from TOA.
    real_burning = _burning_mask_from_toa(real_toa, t_min=T_MIN, burn_duration_min=BURN_DURATION_MIN) & mask
    sim_burning = _burning_mask_from_toa(sim_toa, t_min=T_MIN, burn_duration_min=BURN_DURATION_MIN) & mask

    # 4) Visualise.
    viz = GridVisualizer(show_colorbar=True, origin="upper")

    # A) TOA comparison (2 grids side-by-side)
    fig_toa = viz.plot_two_grids(
        real_toa,
        sim_toa,
        cmap="YlOrRd",
        vmin=np.nanmin(real_toa[mask]),
        vmax=np.nanmax(real_toa[mask]),
        cbar_label="TOA (minutes from t0)",
    )
    viz.save(fig_toa, str(out_dir / "toa_compare.png"))

    # A2) Single TOA map (paper-like look: warm colors + mask boundary)
    ax_single = viz.plot_toa(
        real_toa,
        title="TOA (Real)",
        mask=mask,
    )
    viz.save(ax_single.figure, str(out_dir / "toa_single.png"))

    # B) Error map (sim - real), masked to analysis region
    fig_err = viz.plot_error_map(sim_toa, real_toa, mask=mask)
    viz.save(fig_err, str(out_dir / "error_map.png"))

    # C) Burning masks comparison (boolean grids)
    fig_burn = viz.plot_two_grids(
        real_burning.astype(int),
        sim_burning.astype(int),
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
