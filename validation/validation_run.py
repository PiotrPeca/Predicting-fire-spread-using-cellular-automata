"""Headless runner for fire-spread simulations (no Pygame).

Edit the CONFIG block to tweak parameters for validation runs. The script
prints the simulation step for each run; extend it later with metrics or
comparisons against ground truth.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any


# Ensure src/ is on path when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.insert(0, str(SRC_PATH))

from fire_spread import FireModel, VegetationType, VegetationDensity, WindProvider
from fire_spread.terrain import Terrain


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
	"p0": 0.1,
	"wind_c1": 0.05,
	"wind_c2": 0.1,
	"spark_gust_threshold_kph": 40.0,
	"spark_ignition_prob": 0.1,

	# Fuel probability tweaks (optional)
	"p_veg_override": {},
	"p_dens_override": {},

	# Terrain
	"use_terrain": True,
	"terrain_path": REPO_ROOT / "data" / "las_20m_resolution.tif",
	"meters_per_pixel": 40,

	# Run control
	"max_steps": 3000,
	"runs": 1,
}


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
		target_size=(cfg["height"], cfg["width"]),
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


def run_once(run_id: int, cfg: Dict[str, Any]) -> None:
	wind, htc_schedule = prepare_weather(cfg)
	terrain = build_terrain(cfg)

	model = FireModel(
		width=cfg["width"],
		height=cfg["height"],
		wind_provider=wind,
		initial_fire_pos=cfg["initial_fire_pos"],
		terrain=terrain,
		p_veg_override=cfg.get("p_veg_override"),
		p_dens_override=cfg.get("p_dens_override"),
	)

	# Inject drought schedule (Sielianinow) computed from real weather
	model.htc_schedule = htc_schedule

	# Apply ignition/wind params
	model.p0 = cfg["p0"]
	model.wind_parametr_c1 = cfg["wind_c1"]
	model.wind_parametr_c2 = cfg["wind_c2"]
	model.spark_gust_threshold_kph = cfg["spark_gust_threshold_kph"]
	model.spark_ignition_prob = cfg["spark_ignition_prob"]

	# Print once at start to confirm applied params
	model.print_parameters()

	max_steps = cfg["max_steps"]
	for step_idx in range(1, max_steps + 1):
		model.step()
		print(f"[run {run_id}] step {step_idx}")

		if not model.burning_cells:
			print(f"[run {run_id}] fire extinguished at step {step_idx}")
			break


def main():
	runs = int(CONFIG.get("runs", 1))
	for run_id in range(1, runs + 1):
		run_once(run_id, CONFIG)


if __name__ == "__main__":
	main()
