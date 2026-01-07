"""Headless runner for fire-spread simulations (no Pygame).

Edit the CONFIG block to tweak parameters for validation runs. The script
prints the simulation step for each run; extend it later with metrics or
comparisons against ground truth.
"""

from __future__ import annotations

import contextlib
import io
import json
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
   from fire_spread import FireModel, VegetationType, VegetationDensity, WindProvider, CellState
   from fire_spread.terrain import Terrain
   from fire_spread.validation.metrics import toa_error_metrics
   from fire_spread.validation.visualisation.grid_viz import GridTitles, GridVisualizer


class InvertedWindProvider(WindProvider):
   """WindProvider that flips wind direction across E-W axis (for inverted Y terrain)."""
   def get_next_wind(self) -> dict | None:
      wind = super().get_next_wind()
      if wind:
         w = wind.copy()
         wd = w.get('windDir', 0)
         # Reflect across E-W axis: 0->180, 90->90
         w['windDir'] = (180 - wd) % 360
         return w
      return None


# ---- User-configurable parameters (minimal surface for validation) ----
CONFIG: Dict[str, Any] = {
   # Grid
   "width": 600,
   "height": 300,
   "initial_fire_pos": None,  # (x, y) or None; if None and seed_from_real_mask=True, seeds are auto-picked
   "seed_from_real_mask": True,
   "seed_points": 3,
   "align_sim_start_to_real_t0": True,

   # Weather / wind
   "lat": 61.62453,
   "lon": 14.69939,
   "from_date": datetime(2018, 7, 5, 12, 0, 0),
   "to_date": datetime(2018, 8, 1, 12, 0, 0),
   "history_days": 30,       # for drought index (Sielianinow)
   "steps_per_h": 1,         # simulation steps per meteo hour

   # Model ignition / wind parameters
   "p0": 0.4,
   "wind_c1": 0.045,
   "wind_c2": 0.1,
   "spark_gust_threshold_kph": 40.0,
   "spark_ignition_prob": 0.2,

   # Fuel probability tweaks (optional)
   "p_veg_override": {

   },
   "p_dens_override": {

   },

   # Terrain
   "use_terrain": True,
   "terrain_path": REPO_ROOT / "data" / "las_20m_resolution.tif",
   "meters_per_pixel": 40,
   "biomass_thresholds": {
    #   "sparse": 60,
    #   "normal": 80,
   },
   "vegkvot_thresholds": {
    #   "shrub": 30,
    #   "forest": 80,
   },

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
   # Plot orientation. Use "lower" if maps appear flipped vertically.
   "viz_origin": "upper",

   # If the real-fire boundary mask touches grid edges, the GeoTIFF crop is too small.
   # Auto-expand the simulation/terrain grid to avoid clipping the burned area.
   "auto_expand_grid_for_mask": True,
   "expand_factor": 1.6,
   "expand_attempts": 3,

   # Arbitrary metadata describing the run (e.g., experiment tags, git commit, notes)
   "additional_info": {"""  

   """},
}


def _jsonify(value: Any):
   """Convert complex objects (Paths, datetimes, numpy types) into JSON-safe values."""
   if isinstance(value, dict):
      return {str(k): _jsonify(v) for k, v in value.items()}
   if isinstance(value, (list, tuple, set)):
      return [_jsonify(v) for v in value]
   if isinstance(value, datetime):
      return value.isoformat()
   if isinstance(value, timedelta):
      return value.total_seconds()
   if isinstance(value, Path):
      return str(value)
   if isinstance(value, np.generic):
      return value.item()
   if isinstance(value, np.ndarray):
      return value.tolist()
   if isinstance(value, (str, int, float, bool)) or value is None:
      return value
   return str(value)


def _pick_seed_points_from_mask(
   mask: np.ndarray,
   *,
   real_grid: np.ndarray | None = None,
   max_points: int = 3,
) -> list[tuple[int, int]]:
   """Pick up to max_points seed positions, one per connected component in mask.

   Heuristic:
   - Find connected components (4-neighborhood) in the boundary mask.
   - For each component:
      - If real_grid is provided and has finite values inside the component, pick the earliest ignition cell.
      - Otherwise pick the component centroid (rounded).
   - Return seeds from the largest components first.
   """
   m = np.asarray(mask, dtype=bool)
   if m.ndim != 2 or not np.any(m):
      return []
   use_real = real_grid is not None and isinstance(real_grid, np.ndarray) and real_grid.shape == m.shape
   rg = real_grid if use_real else None

   h, w = m.shape
   visited = np.zeros((h, w), dtype=bool)
   components: list[dict[str, Any]] = []

   # Iterate only on True pixels.
   true_ys, true_xs = np.where(m)
   for y0, x0 in zip(true_ys.tolist(), true_xs.tolist()):
      if visited[y0, x0]:
         continue
      # BFS/DFS for the component.
      stack = [(y0, x0)]
      visited[y0, x0] = True
      size = 0
      sum_x = 0
      sum_y = 0
      best_t = float("inf")
      best_pos: tuple[int, int] | None = None
      while stack:
         y, x = stack.pop()
         size += 1
         sum_x += x
         sum_y += y
         if rg is not None:
            t = float(rg[y, x])
            if np.isfinite(t) and t < best_t:
               best_t = t
               best_pos = (x, y)
         # 4-neighborhood
         if y > 0 and m[y - 1, x] and not visited[y - 1, x]:
            visited[y - 1, x] = True
            stack.append((y - 1, x))
         if y + 1 < h and m[y + 1, x] and not visited[y + 1, x]:
            visited[y + 1, x] = True
            stack.append((y + 1, x))
         if x > 0 and m[y, x - 1] and not visited[y, x - 1]:
            visited[y, x - 1] = True
            stack.append((y, x - 1))
         if x + 1 < w and m[y, x + 1] and not visited[y, x + 1]:
            visited[y, x + 1] = True
            stack.append((y, x + 1))

      centroid = (int(round(sum_x / max(size, 1))), int(round(sum_y / max(size, 1))))
      seed = best_pos if best_pos is not None else centroid
      components.append({"size": size, "seed": seed})

   components.sort(key=lambda c: int(c["size"]), reverse=True)
   return [tuple(c["seed"]) for c in components[: max(0, int(max_points))]]


def _ignite_seed(model: FireModel, seed_pos: tuple[int, int]) -> tuple[int, int] | None:
   """Ignite a single seed position, falling back to nearest burnable if needed."""
   x, y = seed_pos
   if not (0 <= x < model.grid.width and 0 <= y < model.grid.height):
      return None
   cell = model.grid[x][y]
   current_ts = model.wind.get("timestamp", 0)
   if getattr(cell, "is_burnable", None) and cell.is_burnable():
      cell.state = CellState.Burning
      cell.next_state = CellState.Burning
      cell.burn_timer = int(cell.fuel.burn_time)
      model.burning_cells.add((x, y))
      model.ignition_time_grid[y, x] = current_ts
      return (x, y)
   # Fallback: nearest burnable
   fire_cell = model._find_nearest_burnable((x, y))
   if fire_cell is None:
      return None
   fire_cell.state = CellState.Burning
   fire_cell.next_state = CellState.Burning
   fire_cell.burn_timer = int(fire_cell.fuel.burn_time)
   model.burning_cells.add(fire_cell.pos)
   fx, fy = fire_cell.pos
   model.ignition_time_grid[fy, fx] = current_ts
   return (fx, fy)


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


def _prepare_run_directory(cfg: Dict[str, Any], run_id: int, session_ts: str) -> Path:
   """Create (if needed) and return the directory for a single run's artifacts."""
   outputs_root = Path(cfg.get("outputs_root") or (REPO_ROOT / "outputs"))
   run_dir = outputs_root / f"run_{session_ts}_{run_id}"
   run_dir.mkdir(parents=True, exist_ok=True)
   return run_dir


def _save_run_outputs(
   *,
   cfg: Dict[str, Any],
   run_dir: Path,
   background: np.ndarray | None,
   mask: np.ndarray,
   real_hours: np.ndarray,
   sim_hours: np.ndarray,
) -> None:
   """Save TOA comparison + error map for a run."""
   if not bool(cfg.get("save_run_outputs")):
      return

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

   origin = str(cfg.get("viz_origin") or "upper")
   if origin not in {"upper", "lower"}:
      origin = "upper"
   viz = GridVisualizer(show_colorbar=True, origin=origin)

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
      cbar_label="|Sim - Real| (hours)",
      outline=True,
      outline_mask=mask_c,
      show_axes=True,
      tick_step=100,
   )
   viz.save(fig_err, str(run_dir / "toa_error.png"))


def _save_run_summary(
   *,
   run_dir: Path,
   cfg: Dict[str, Any],
   metrics: Dict[str, Any],
) -> None:
   """Persist the effective configuration and metrics for a run alongside visual outputs."""
   run_dir.mkdir(parents=True, exist_ok=True)
   summary = {
      "config": _jsonify(cfg),
      "metrics": _jsonify(metrics),
   }
   summary_path = run_dir / "summary.json"
   with summary_path.open("w", encoding="utf-8") as fh:
      json.dump(summary, fh, indent=2, ensure_ascii=False)


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
   return InvertedWindProvider(
      lat=cfg["lat"],
      lon=cfg["lon"],
      steps_per_h=cfg["steps_per_h"],
   )


def build_terrain(cfg: Dict[str, Any]) -> Terrain | None:
   if not cfg.get("use_terrain"):
      return None
   tt = Terrain(
      tiff_path=cfg["terrain_path"],
      target_size=(cfg["width"], cfg["height"]),
      meters_per_pixel=cfg["meters_per_pixel"],
      biomass_thresholds=cfg.get("biomass_thresholds"),
      vegkvot_thresholds=cfg.get("vegkvot_thresholds"),
   )
   tt.grid_data = np.flip(tt.grid_data, axis=0)
   return tt


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

   # Pre-load/compute real ignition grid BEFORE simulation so we can align start time and seed points.
   real_grid = _load_real_ignition_grid(run_cfg)
   if real_grid is None:
      real_grid, computed_mask = _compute_real_ignition_grid(run_cfg, terrain)
      if real_mask is None:
         real_mask = computed_mask

   if real_mask is None:
      raise ValueError(
         "Brak real_boundary_mask_path/real_mask: nie da się dobrać punktów startowych i wykonać walidacji. "
         "Ustaw real_boundary_mask_path albo podaj real_ignition_grid_path + maskę granicy."
      )

   domain_mask = np.asarray(real_mask, dtype=bool)

   # Choose time-zero as the earliest REAL ignition inside the domain.
   default_t0 = float(run_cfg["from_date"].timestamp())
   t0_unix = default_t0
   if real_grid is not None:
      finite = domain_mask & np.isfinite(real_grid)
      if np.any(finite):
         t0_unix = float(np.nanmin(real_grid[finite]))

   # Optionally align simulation start (wind timestamps) to real t0
   if bool(run_cfg.get("align_sim_start_to_real_t0", True)) and np.isfinite(t0_unix):
      old_from = run_cfg["from_date"]
      old_to = run_cfg["to_date"]
      duration = old_to - old_from
      run_cfg["from_date"] = datetime.fromtimestamp(t0_unix)
      run_cfg["to_date"] = run_cfg["from_date"] + duration

   # Now prepare weather with the (possibly) adjusted start date.
   wind, htc_schedule = prepare_weather(run_cfg)

   # Pick 3 seed points from the real burned mask components (one per fragment).
   seed_points: list[tuple[int, int]] = []
   if bool(run_cfg.get("seed_from_real_mask", True)) and domain_mask is not None:
      seed_points = _pick_seed_points_from_mask(
         domain_mask,
         real_grid=real_grid,
         max_points=int(run_cfg.get("seed_points", 3) or 3),
      )

   # Decide primary initial seed
   initial_fire_pos = run_cfg.get("initial_fire_pos")
   if initial_fire_pos is None and seed_points:
      initial_fire_pos = seed_points[0]

   model = FireModel(
      width=run_cfg["width"],
      height=run_cfg["height"],
      wind_provider=wind,
      initial_fire_pos=initial_fire_pos,
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

   # Ignite additional seed points (if any) so each real fragment can start burning.
   # FireModel already ignited the primary seed in __init__.
   if seed_points:
      for extra_seed in seed_points[1:]:
         _ignite_seed(model, extra_seed)

   # Print once at start to confirm applied params
   model.print_parameters()

   max_steps = run_cfg["max_steps"]
   for step_idx in range(1, max_steps + 1):
      model.step()

      if not model.burning_cells:
         break

   sim_grid = model.ignition_time_grid

   # --- Walidacja (symulacja ograniczona) ---
   # Analizujemy WYŁĄCZNIE komórki spalone w rzeczywistości.
   # Definicja "spalone w real" pochodzi z maski granicy pożaru (GeoJSON), a nie z TOA.

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

   metrics_payload = {
      "fitness": fitness_score,
      "bias_hours": float(err.bias),
      "rmse_hours": float(err.rmse),
      "rmse_samples": int(err.n),
      "fp_rate": fp_rate,
      "fn_rate": fn_rate,
   }

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
   run_dir = _prepare_run_directory(run_cfg, run_id, session_ts)
   _save_run_outputs(
      cfg=run_cfg,
      run_dir=run_dir,
      background=background,
      mask=domain_mask,
      real_hours=real_h,
      sim_hours=sim_h,
   )
   _save_run_summary(run_dir=run_dir, cfg=run_cfg, metrics=metrics_payload)


def main():
   runs = int(CONFIG.get("runs", 1))
   session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
   for run_id in range(1, runs + 1):
      run_once(run_id, CONFIG, session_ts=session_ts)


if __name__ == "__main__":
   main()
