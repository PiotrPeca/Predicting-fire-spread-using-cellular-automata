from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on path when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fire_spread.ignition_processor import IgnitionProcessor  # type: ignore[import-not-found]
from fire_spread.terrain import Terrain  # type: ignore[import-not-found]
from fire_spread.validation.visualisation import GridVisualizer  # type: ignore[import-not-found]


# --- Config (edit these paths/values) ---
OUT_DIR = REPO_ROOT / "validation" / "_real_outputs"

TIF_PATH = REPO_ROOT / "data" / "las_20m_resolution.tif"
FIRE_DATA_PATH = REPO_ROOT / "data" / "fire_archive_M-C61_675224.json"
BOUNDARY_PATH = REPO_ROOT / "data" / "1a6cd4865f484fb48f8ba4ea97a6e0d1.json"  # GeoJSON FeatureCollection

# Keep this small so interpolation is fast. Increase if needed.
# The script will automatically scale this up if the boundary mask gets clipped.
TARGET_SIZE = (300, 600)  # (H, W)

# Time resolution of reconstructed ignition time.
# Using hours matches the colleague's example.
TIMEDELTA = pd.Timedelta(hours=1)

# If True, the plotted output is cropped to the fire mask extent (with padding).
# This does NOT affect the reconstructed grid itself.
CROP_TO_MASK = True
CROP_PADDING_PX = 12


def _normalize_background(biomass: np.ndarray) -> np.ndarray:
    biomass = biomass.astype(float)
    finite = np.isfinite(biomass)
    if not np.any(finite):
        return np.zeros_like(biomass, dtype=float)

    lo = float(np.nanpercentile(biomass[finite], 2))
    hi = float(np.nanpercentile(biomass[finite], 98))
    if hi <= lo:
        return np.zeros_like(biomass, dtype=float)

    bg = (biomass - lo) / (hi - lo)
    return np.clip(bg, 0.0, 1.0)


def _mask_touches_edges(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    m = mask.astype(bool)
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def _crop_to_finite_extent(
    toa: np.ndarray,
    background: np.ndarray,
    *,
    pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(toa)
    if not np.any(finite):
        return toa, background

    ys, xs = np.where(finite)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, toa.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, toa.shape[1])
    return toa[y0:y1, x0:x1], background[y0:y1, x0:x1]


def _build_real_toa(*, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    terrain = Terrain(str(TIF_PATH), target_size=target_size)
    ip = IgnitionProcessor(terrain)

    h, w = terrain.get_dimensions()
    ip.load_and_prepare_data(
        str(FIRE_DATA_PATH),
        bbox=(0, 0, w - 1, h - 1),
        timedelta=TIMEDELTA,
    )
    ip.create_boundary_mask(str(BOUNDARY_PATH))
    real_toa = ip.interpolate_ignition_time(method="linear")

    background = _normalize_background(terrain.biomass)
    return real_toa, background


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build TOA with automatic upscaling if the mask appears clipped.
    target = TARGET_SIZE
    for attempt in range(3):
        real_toa, background = _build_real_toa(target_size=target)

        # Heuristic: if the mask got clipped, the finite TOA will often touch the grid edges.
        finite = np.isfinite(real_toa)
        touches = False
        if np.any(finite):
            touches = bool(finite[0, :].any() or finite[-1, :].any() or finite[:, 0].any() or finite[:, -1].any())

        if not touches:
            break

        # Scale up and retry.
        if attempt < 2:
            target = (int(target[0] * 1.6), int(target[1] * 1.6))

    if CROP_TO_MASK:
        real_toa, background = _crop_to_finite_extent(real_toa, background, pad=CROP_PADDING_PX)

    viz = GridVisualizer(show_colorbar=True, origin="upper")

    ax = viz.plot_toa(
        real_toa,
        title="TOA (Real, reconstructed from hotspots)",
        background=background,
        cbar_label="Time since fire start (hours)",
        show_axes=True,
        tick_step=100,
    )
    viz.save(ax.figure, str(OUT_DIR / "toa_real.png"))

    print(f"Saved: {OUT_DIR / 'toa_real.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
