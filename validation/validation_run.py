"""Headless runner for fire-spread simulations (no Pygame).

Edit the CONFIG block to tweak parameters for validation runs. The script
prints the simulation step for each run; extend it later with metrics or
comparisons against ground truth.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np


# Ensure src/ is on path when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.insert(0, str(SRC_PATH))


def _is_quiet() -> bool:
	"""Control console output. Default quiet; set VALIDATION_VERBOSE=1 to see extra logs."""
	return os.environ.get("VALIDATION_VERBOSE", "0").strip() != "1"


_QUIET = _is_quiet()


@contextlib.contextmanager
def _silence_stdio(enabled: bool):
	if not enabled:
		yield
		return
	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		yield


# Mesa may print optional viz warnings on import; silence them in quiet mode.
with _silence_stdio(_QUIET):
	from fire_spread import FireModel, VegetationType, VegetationDensity, WindProvider
	from fire_spread.terrain import Terrain
	from fire_spread.validation.metrics import toa_error_metrics
	from fire_spread.validation.visualisation.grid_viz import GridTitles, GridVisualizer


# ---- User-configurable parameters (minimal surface for validation) ----
CONFIG: Dict[str, Any] = {
	# Grid
	"width": 600,
	"height": 300,
	"initial_fire_pos": None,  # (x, y) or None for center

	# Weather / wind
	"lat": 61.62453,
	"lon": 14.69939,
	"from_date": datetime(2018, 7, 5, 12, 0, 0),
	"to_date": datetime(2018, 7, 5, 12, 0, 0) + timedelta(days=30),
	"history_days": 30,       # for drought index (Sielianinow)
	"steps_per_h": 1,         # simulation steps per meteo hour

	# Model ignition / wind parameters
	"p0": 0.15,
	"wind_c1": 0.045,
	"wind_c2": 0.1,
	"spark_gust_threshold_kph": 40.0,
	"spark_ignition_prob": 0.2,

	# Fuel probability tweaks (optional)
	"p_veg_override": {},
	"p_dens_override": {},

	# Terrain
	"use_terrain": True,
	"terrain_path": REPO_ROOT / "data" / "las_20m_resolution.tif",
	"meters_per_pixel": 40,

	# Optional: ścieżka do siatki rzeczywistych czasów zapalenia (NumPy .npy)
	"real_ignition_grid_path": None,

	# Optional: generowanie siatki rzeczywistych czasów zapalenia z danych satelitarnych
	# (zgodnie z src/fire_spread/ignition_processor.py)
	"real_fire_archive_path": REPO_ROOT / "data" / "fire_archive_SV-C2_675228.json",
	"real_boundary_mask_path": REPO_ROOT / "data" / "1a6cd4865f484fb48f8ba4ea97a6e0d1.json",
	# bbox w indeksach siatki (x_min, y_min, x_max, y_max); None = bez filtrowania
	"real_fire_bbox": None,
	"real_interp_method": "linear",

	# Loss / fitness (lower is better)
	# When using TOA in "hours since start", RMSE is also in hours.
	"rmse_weight": 1.0,
	"fp_weight": 1.0,
	"fn_weight": 1.0,

	# Run control
	"max_steps": 3000,
	"runs": 1,

	# Optional: save per-run plots to outputs/run_<datetime>_<run>
	"save_run_outputs": True,
	"outputs_root": REPO_ROOT / "outputs",
	"outputs_crop_pad_px": 12,

	# If the real-fire boundary mask touches grid edges, the GeoTIFF crop is too small.
	# Auto-expand the simulation/terrain grid to avoid clipping the burned area.
	"auto_expand_grid_for_mask": True,
	"expand_factor": 1.6,
	"expand_attempts": 3,
}


def _terrain_biomass_background(terrain: Terrain | None) -> np.ndarray | None:
	"""Extract a grayscale-friendly biomass background from Terrain.

	Terrain stores cell info as an object grid of dicts; we derive a float biomass grid.
	"""
	if terrain is None:
		return None
	grid = terrain.get_grid("cartesian")
	h, w = grid.shape
	bg = np.empty((h, w), dtype=float)
	for y in range(h):
		for x in range(w):
			cell = grid[y, x]
			try:
				bg[y, x] = float(cell.get("biomass", np.nan))
			except Exception:
				bg[y, x] = np.nan
	return bg


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
	"""Heuristic: if the boundary mask reaches the raster edges, the crop is too small."""
	if mask.size == 0:
		return False
	m = np.asarray(mask, dtype=bool)
	return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def _save_run_outputs(
	*,
	cfg: Dict[str, Any],
	run_id: int,
	session_ts: str,
	background: np.ndarray | None,
	mask: np.ndarray,
	real_hours: np.ndarray,
	sim_hours: np.ndarray,
) -> None:
	"""Save TOA comparison + error map for a run."""
	if not bool(cfg.get("save_run_outputs")):
		return

	outputs_root = Path(cfg.get("outputs_root") or (REPO_ROOT / "outputs"))
	run_dir = outputs_root / f"run_{session_ts}_{run_id}"
	run_dir.mkdir(parents=True, exist_ok=True)

	# Crop everything to mask bbox for compact plots.
	pad = int(cfg.get("outputs_crop_pad_px", 0) or 0)
	bbox = _bbox_from_mask(mask.astype(bool), pad=pad)
	if bbox is not None:
		ys, xs = bbox
		mask_c = mask[ys, xs].astype(bool)
		real_c = real_hours[ys, xs].astype(float, copy=True)
		sim_c = sim_hours[ys, xs].astype(float, copy=True)
		bg_c = background[ys, xs].astype(float, copy=True) if background is not None else None
	else:
		mask_c = mask.astype(bool)
		real_c = real_hours.astype(float, copy=True)
		sim_c = sim_hours.astype(float, copy=True)
		bg_c = background.astype(float, copy=True) if background is not None else None

	# Apply mask (NaN outside) so visuals match the "validate only inside real mask" rule.
	real_c[~mask_c] = np.nan
	sim_c[~mask_c] = np.nan

	# Error map: sim - real, only where both are finite inside mask.
	err = sim_c - real_c
	valid = mask_c & np.isfinite(sim_c) & np.isfinite(real_c)
	err[~valid] = np.nan

	if bg_c is not None:
		bg_c = _normalize_background(bg_c)

	viz = GridVisualizer(show_colorbar=True, origin="upper")

	fig_toa = viz.plot_toa_compare(
		real_c,
		sim_c,
		titles=GridTitles(left="Real TOA", right="Sim TOA"),
		background=bg_c,
		cbar_label="Time since t0 (hours)",
		outline=True,
		outline_mask=mask_c,
		show_axes=True,
		tick_step=100,
	)
	viz.save(fig_toa, str(run_dir / "toa_compare.png"))

	fig_err = viz.plot_error_map(
		err,
		background=bg_c,
		cbar_label="Sim - Real (hours)",
		outline=True,
		outline_mask=mask_c,
		show_axes=True,
		tick_step=100,
	)
	viz.save(fig_err, str(run_dir / "toa_error.png"))


def _load_real_ignition_grid(cfg: Dict[str, Any]) -> np.ndarray | None:
	"""Wczytaj siatkę rzeczywistych czasów zapalenia jeśli podano ścieżkę."""
	path = cfg.get("real_ignition_grid_path")
	if not path:
		return None
	file_path = Path(path)
	if not file_path.exists():
		return None
	return np.load(file_path)


def _compute_real_ignition_grid(
	cfg: Dict[str, Any],
	terrain: Terrain | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
	"""Wylicz siatkę rzeczywistych czasów zapalenia przez IgnitionProcessor (Unix ts).

	Zwraca:
		(grid_unix_seconds, boundary_mask)
	"""
	archive_path = cfg.get("real_fire_archive_path")
	if not archive_path:
		return None, None
	if terrain is None:
		return None, None
	archive_path = Path(archive_path)
	if not archive_path.exists():
		return None, None

	try:
		from fire_spread.ignition_processor import IgnitionProcessor
	except Exception as exc:
		return None, None

	ip = IgnitionProcessor(terrain)
	bbox = cfg.get("real_fire_bbox")
	ip.load_and_prepare_data(str(archive_path), bbox=bbox)

	mask = None
	mask_path = cfg.get("real_boundary_mask_path")
	if mask_path:
		mask_path = Path(mask_path)
		if mask_path.exists():
			mask = ip.create_boundary_mask(str(mask_path))

	method = str(cfg.get("real_interp_method") or "linear")
	grid = ip.interpolate_ignition_time(method=method)
	return grid, mask


def _load_boundary_mask(cfg: Dict[str, Any], terrain: Terrain | None) -> np.ndarray | None:
	"""Wczytaj/utwórz maskę granicy (jeśli skonfigurowana) bez liczenia gridu czasów."""
	mask_path = cfg.get("real_boundary_mask_path")
	if not mask_path or terrain is None:
		return None
	mask_path = Path(mask_path)
	if not mask_path.exists():
		return None
	try:
		from fire_spread.ignition_processor import IgnitionProcessor
	except Exception:
		return None
	ip = IgnitionProcessor(terrain)
	return ip.create_boundary_mask(str(mask_path))


def _to_hours_from_start(unix_grid: np.ndarray | None, t0_unix: float) -> np.ndarray | None:
	"""Unix seconds -> godziny od t0 (NaN zostają NaN)."""
	if unix_grid is None:
		return None
	return (unix_grid - t0_unix) / 3600.0


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice] | None:
	"""(yslice, xslice) obejmujące True w masce (opcjonalnie z paddingiem)."""
	if mask.size == 0 or not np.any(mask):
		return None
	ys, xs = np.where(mask)
	y0 = max(int(ys.min()) - pad, 0)
	y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
	x0 = max(int(xs.min()) - pad, 0)
	x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
	return slice(y0, y1), slice(x0, x1)


def _mask_and_crop(grid: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
	"""Ustaw NaN poza maską i przytnij do bbox maski."""
	if grid is None or mask is None:
		return grid
	if grid.shape != mask.shape:
		return grid
	out = grid.astype(float, copy=True)
	out[~mask.astype(bool)] = np.nan
	bbox = _bbox_from_mask(mask.astype(bool), pad=0)
	if bbox is None:
		return out
	ys, xs = bbox
	return out[ys, xs]


def _print_grid(label: str, grid: np.ndarray | None) -> None:
	"""Pomocniczy druk macierzy (w tym godziny) z NaN jako '.' i przycięciem (jeśli zastosowane)."""
	print(f"=== {label} ===")
	if grid is None:
		print("(brak danych)")
		return
	with np.printoptions(precision=1, suppress=True, threshold=200, linewidth=160, nanstr="."):
		print(grid)


def build_wind_provider(cfg: Dict[str, Any]) -> WindProvider:
	"""Create provider; caller is responsible for fetch_data and drought map."""
	return WindProvider(
		lat=cfg["lat"],
		lon=cfg["lon"],
		steps_per_h=cfg["steps_per_h"],
	)


def build_terrain(cfg: Dict[str, Any]) -> Terrain | None:
	if not cfg.get("use_terrain"):
		return None
	return Terrain(
		tiff_path=cfg["terrain_path"],
		# Terrain expects target_size as (width, height) (same as pygame_run.py)
		target_size=(cfg["width"], cfg["height"]),
		meters_per_pixel=cfg["meters_per_pixel"],
	)


def prepare_weather(cfg: Dict[str, Any]) -> tuple[WindProvider, dict[str, float]]:
	"""Fetch wind data and compute drought (Sielianinow) schedule, like pygame_run."""
	wind = build_wind_provider(cfg)
	wind.fetch_data(
		from_date=cfg["from_date"],
		to_date=cfg["to_date"],
		history_days=cfg["history_days"],
	)
	htc = wind.get_daily_sielianinov_map(
		fire_start_date=cfg["from_date"],
		fire_end_date=cfg["to_date"],
		history_days=cfg["history_days"],
	)
	return wind, htc


def run_once(run_id: int, cfg: Dict[str, Any], *, session_ts: str) -> None:
	wind, htc_schedule = prepare_weather(cfg)

	# Auto-expand grid/terrain if the boundary mask looks clipped.
	run_cfg: Dict[str, Any] = dict(cfg)
	terrain = None
	real_mask = None

	use_loaded_real = bool(run_cfg.get("real_ignition_grid_path"))
	auto_expand = bool(run_cfg.get("auto_expand_grid_for_mask")) and (not use_loaded_real)
	attempts = int(run_cfg.get("expand_attempts", 3) or 3)
	factor = float(run_cfg.get("expand_factor", 1.6) or 1.6)

	for attempt in range(max(attempts, 1)):
		terrain = build_terrain(run_cfg)
		real_mask = _load_boundary_mask(run_cfg, terrain)
		if (not auto_expand) or (real_mask is None) or (not _mask_touches_edges(real_mask)):
			break
		if attempt >= attempts - 1:
			break
		# Expand width/height and retry.
		run_cfg["width"] = int(run_cfg["width"] * factor)
		run_cfg["height"] = int(run_cfg["height"] * factor)

	model = FireModel(
		width=run_cfg["width"],
		height=run_cfg["height"],
		wind_provider=wind,
		initial_fire_pos=run_cfg["initial_fire_pos"],
		terrain=terrain,
		p_veg_override=run_cfg.get("p_veg_override"),
		p_dens_override=run_cfg.get("p_dens_override"),
	)

	# Inject drought schedule (Sielianinow) computed from real weather
	model.htc_schedule = htc_schedule

	# Apply ignition/wind params
	model.p0 = run_cfg["p0"]
	model.wind_parametr_c1 = run_cfg["wind_c1"]
	model.wind_parametr_c2 = run_cfg["wind_c2"]
	model.spark_gust_threshold_kph = run_cfg["spark_gust_threshold_kph"]
	model.spark_ignition_prob = run_cfg["spark_ignition_prob"]

	# Print once at start to confirm applied params
	model.print_parameters()

	max_steps = run_cfg["max_steps"]
	for step_idx in range(1, max_steps + 1):
		model.step()

		if not model.burning_cells:
			break

	# Po zakończeniu symulacji: wczytaj/wylicz real TOA (Unix seconds)
	real_grid = _load_real_ignition_grid(run_cfg)
	if real_grid is None:
		real_grid, computed_mask = _compute_real_ignition_grid(run_cfg, terrain)
		if real_mask is None:
			real_mask = computed_mask

	sim_grid = model.ignition_time_grid

	# --- Walidacja (symulacja ograniczona) ---
	# Analizujemy WYŁĄCZNIE komórki spalone w rzeczywistości.
	# Definicja "spalone w real" pochodzi z maski granicy pożaru (GeoJSON), a nie z TOA.
	if real_mask is None:
		raise ValueError(
			"Brak real_boundary_mask_path/real_mask: nie da się wykonać walidacji w trybie 'symulacji ograniczonej' (tylko komórki spalone w real)."
		)

	domain_mask = np.asarray(real_mask, dtype=bool)

	# Choose time-zero as the earliest REAL ignition inside the domain.
	# This makes Real TOA start at 0h (so you will see the yellow early-ignition areas).
	default_t0 = float(run_cfg["from_date"].timestamp())
	t0_unix = default_t0
	if real_grid is not None:
		finite = domain_mask & np.isfinite(real_grid)
		if np.any(finite):
			t0_unix = float(np.nanmin(real_grid[finite]))

	# Konwersja do godzin od t0
	real_hours = _to_hours_from_start(real_grid, t0_unix)
	sim_hours = _to_hours_from_start(sim_grid, t0_unix)

	sim_h = sim_hours if sim_hours is not None else np.full((run_cfg["height"], run_cfg["width"]), np.nan)
	real_h = real_hours if real_hours is not None else np.full((run_cfg["height"], run_cfg["width"]), np.nan)

	# RMSE tylko tam, gdzie oba TOA są skończone (w obrębie domain_mask)
	err = toa_error_metrics(sim_h, real_h, mask=domain_mask)

	# FN w trybie ograniczonym: real burned = domain_mask
	sim_burned = np.isfinite(sim_h)
	real_burned_count = int(np.sum(domain_mask))
	fn = int(np.sum(domain_mask & ~sim_burned))
	fn_rate = 0.0 if real_burned_count == 0 else float(fn) / float(real_burned_count)

	# FP w trybie ograniczonym nie jest sensowne (brak real-unburned w domenie), więc ustawiamy 0
	fp_rate = 0.0

	fps_w = float(run_cfg.get("fp_weight", 1.0))
	fns_w = float(run_cfg.get("fn_weight", 1.0))
	rmse_w = float(run_cfg.get("rmse_weight", 1.0))
	fitness_score = (rmse_w * float(err.rmse)) + (fps_w * fp_rate) + (fns_w * fn_rate)
	print(
		f"[run {run_id}] fitness={fitness_score:.6f} (bias={float(err.bias):.3f}h, rmse={float(err.rmse):.3f}h, n_rmse={err.n}, fp_rate={fp_rate:.4f}, fn_rate={fn_rate:.4f})"
	)

	# Wydruki przycinamy do tej samej domeny (real burned mask) i obie macierze mają identyczny crop
	print_mask = domain_mask
	real_hours_masked = _mask_and_crop(real_hours, print_mask)
	sim_hours_cropped = _mask_and_crop(sim_hours, print_mask) if print_mask is not None else sim_hours

	_print_grid(
		"Czasy zapalenia (rzeczywiste) [h od t0=min(real w masce), tylko spalone w real]",
		real_hours_masked,
	)
	_print_grid("Czasy zapalenia (symulacja) [h od t0]", sim_hours_cropped)

	# Optional: save per-run plots (TOA compare + error map)
	background = _terrain_biomass_background(terrain)
	_save_run_outputs(
		cfg=run_cfg,
		run_id=run_id,
		session_ts=session_ts,
		background=background,
		mask=domain_mask,
		real_hours=real_h,
		sim_hours=sim_h,
	)


def main():
	runs = int(CONFIG.get("runs", 1))
	session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	for run_id in range(1, runs + 1):
		run_once(run_id, CONFIG, session_ts=session_ts)


if __name__ == "__main__":
	main()
