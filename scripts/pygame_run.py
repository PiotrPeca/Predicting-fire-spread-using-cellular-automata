#!/usr/bin/env python3
"""Pygame visualization launcher for the fire spread simulation.

This script provides an interactive Pygame-based visualization of the
cellular automaton fire spread model. It includes a configuration menu,
real-time rendering, and interactive controls for simulation speed.

Usage:
    python scripts/pygame_run.py
"""

import pygame
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import numpy as np

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from fire_spread import FireModel, CellState, WindProvider
from fire_spread.terrain import Terrain

from visualization import (
    GridRenderer,
    InfoPanel,
    SpeedSlider,
    WindRose,
    SetupMenu,
    WHITE,
    DEFAULT_FPS,
    MIN_FPS,
    MAX_FPS,
)


# --- Ignition control (no UI changes) ---
# ignition_mode: 0 = manual (current behavior), 1 = validation-like multi-seed from mask
DEFAULT_IGNITION_MODE = 0
# How many seed points to ignite in validation mode (typically 1 or 3)
DEFAULT_VALIDATION_SEED_POINTS = 3


class SimulationRunner:
    """Main simulation runner with Pygame visualization.

    Handles the main game loop, event processing, and coordination between
    the fire spread model and visualization components.

    Attributes:
        model: The fire spread simulation model.
        screen: Pygame display surface.
        clock: Pygame clock for FPS control.
        renderer: Grid renderer for drawing cells.
        info_panel: UI panel for displaying simulation info.
        slider: Speed control slider.
        paused: Whether the simulation is paused.
        current_fps: Current frames per second setting.
        first_frame: Flag to skip the first step (show initial state).
        dragging_slider: Whether the user is dragging the speed slider.
    """

    def __init__(
            self,
            width: int,
            height: int,
            cell_size: int,
            fire_pos: Optional[tuple[int, int]],
            ignition_mode: int = 0,
            seed_points: int = 3,
    ) -> None:
        """Initialize the simulation runner.

        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            cell_size: Size of each cell in pixels.
            fire_pos: Initial fire position (x, y) or None for center.
        """
        # Window setup
        window_width = max(width * cell_size + 400, 1600)
        window_height = max(height * cell_size + 400, 950)
        
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Symulacja Pożaru Lasu")
        self.clock = pygame.time.Clock()

        # Store initial parameters for reset
        self.width = width
        self.height = height
        self.ignition_mode = int(ignition_mode or 0)
        self.seed_points = int(seed_points or 3)

        # 1. Inicjalizacja Providera
        self.wind_provider = WindProvider(61.62453, 14.69939)

        # 2. Definicja dat (Pożar w Szwecji 2018)
        # 12:00 to bezpieczna godzina startu
        from_date = datetime.fromisoformat("2018-07-05T12:00:00")
        to_date = from_date + timedelta(days=30)

        # --- NOWOŚĆ: Pobranie harmonogramu suszy (Sielianinowa) ---
        print("Generowanie mapy suszy (Sielianinow)...")
        self.htc_schedule = self.wind_provider.get_daily_sielianinov_map(
            fire_start_date=from_date,
            fire_end_date=to_date,
            history_days=30
        )
        # ----------------------------------------------------------

        # 3. Pobranie danych wiatrowych (standardowo)
        print("Pobieranie danych wiatrowych...")
        self.wind_provider.fetch_data(from_date=from_date, to_date=to_date)

        self.initial_fire_pos = fire_pos

        # Validation-like ignition seeds (computed once; reused on reset)
        self.validation_seed_points: list[tuple[int, int]] = []

        # 4. Tworzenie modelu
        self.terrain = self._load_terrain(width, height)

        if self.ignition_mode == 1:
            self.validation_seed_points = self._compute_validation_seed_points(max_points=self.seed_points)
            if self.validation_seed_points:
                self.initial_fire_pos = self.validation_seed_points[0]

        self._reset_model()

        # --- NOWOŚĆ: Wstrzyknięcie mapy suszy do modelu ---
        # Dzięki temu model wie, jak bardzo jest sucho w danym dniu
        self.model.htc_schedule = self.htc_schedule
        # --------------------------------------------------

        # Visualization components
        self.renderer = GridRenderer(cell_size)
        self.slider = SpeedSlider(
            0, 0,
            width=400,
            height=20,
            min_val=MIN_FPS,
            max_val=MAX_FPS
        )
        self.info_panel = InfoPanel(self.slider)

        self.wind_rose = WindRose(0, 0)
        
        # Simulation state
        self.paused = False
        self.current_fps = DEFAULT_FPS
        self.first_frame = True
        self.dragging_slider = False

        # Grid dimensions (for info panel)
        self.grid_height = height
        self.cell_size = cell_size

    def _compute_validation_seed_points(self, max_points: int = 3) -> list[tuple[int, int]]:
        """Compute ignition seed points like validation_run.

        Uses IgnitionProcessor to rasterize the real boundary mask and (optionally)
        compute a real ignition time grid. Then picks up to max_points seed points
        (one per connected component) prioritizing earliest real ignition within
        each component.

        Falls back to empty list if required deps/files are missing.
        """
        if self.terrain is None:
            return []

        # Match defaults from validation/validation_run.py
        fire_archive_path = project_root / "data" / "fire_archive_SV-C2_675228.json"
        boundary_mask_path = project_root / "data" / "1a6cd4865f484fb48f8ba4ea97a6e0d1.json"

        if not fire_archive_path.exists() or not boundary_mask_path.exists():
            print("Brak plików walidacji (fire_archive/maska). Używam trybu ręcznego.")
            return []

        try:
            from fire_spread.ignition_processor import IgnitionProcessor
        except Exception as exc:
            print(f"Nie można załadować IgnitionProcessor (brak zależności typu geopandas/scipy?): {exc}")
            return []

        try:
            ip = IgnitionProcessor(self.terrain)
            ip.load_and_prepare_data(str(fire_archive_path))
            mask = ip.create_boundary_mask(str(boundary_mask_path)).astype(bool)
            real_grid = ip.interpolate_ignition_time(method="linear")
        except Exception as exc:
            print(f"Nie udało się wyliczyć seedów walidacyjnych: {exc}. Używam trybu ręcznego.")
            return []

        seeds = self._pick_seed_points_from_mask(mask, real_grid=real_grid, max_points=max_points)

        # Coordinate-system fix:
        # - IgnitionProcessor / rasterio use row index with origin at top-left (row=0 at the top).
        # - Mesa model uses (x, y) with origin at bottom-left.
        # Convert seeds so they land in the same places you see in validation plots.
        h = int(self.height)
        converted = [(x, (h - 1 - y)) for (x, y) in seeds]

        if converted:
            print(f"[validation seeds] raw(top-left)={seeds} -> model(bottom-left)={converted}")
        return converted

    @staticmethod
    def _pick_seed_points_from_mask(
        mask: np.ndarray,
        *,
        real_grid: np.ndarray | None = None,
        max_points: int = 3,
    ) -> list[tuple[int, int]]:
        """Pick up to max_points seed positions, one per connected component in mask.

        Ported from validation/validation_run.py to keep seed selection consistent.
        """
        m = np.asarray(mask, dtype=bool)
        if m.ndim != 2 or not np.any(m):
            return []

        use_real = (
            real_grid is not None
            and isinstance(real_grid, np.ndarray)
            and real_grid.shape == m.shape
        )
        rg = real_grid if use_real else None

        h, w = m.shape
        visited = np.zeros((h, w), dtype=bool)
        components: list[dict[str, object]] = []

        true_ys, true_xs = np.where(m)
        for y0, x0 in zip(true_ys.tolist(), true_xs.tolist()):
            if visited[y0, x0]:
                continue
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
            components.append({"size": int(size), "seed": seed})

        components.sort(key=lambda c: int(c["size"]), reverse=True)
        return [tuple(c["seed"]) for c in components[: max(0, int(max_points))]]

    def _ignite_seed(self, seed_pos: tuple[int, int]) -> tuple[int, int] | None:
        """Ignite a single seed position, falling back to nearest burnable if needed."""
        x, y = seed_pos
        if not (0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height):
            return None
        cell = self.model.grid[x][y]
        current_ts = self.model.wind.get("timestamp", 0)

        if getattr(cell, "is_burnable", None) and cell.is_burnable():
            cell.state = CellState.Burning
            cell.next_state = CellState.Burning
            cell.burn_timer = int(cell.fuel.burn_time)
            self.model.burning_cells.add((x, y))
            self.model.ignition_time_grid[y, x] = current_ts
            return (x, y)

        fire_cell = self.model._find_nearest_burnable((x, y))
        if fire_cell is None:
            return None

        fire_cell.state = CellState.Burning
        fire_cell.next_state = CellState.Burning
        fire_cell.burn_timer = int(fire_cell.fuel.burn_time)
        self.model.burning_cells.add(fire_cell.pos)
        fx, fy = fire_cell.pos
        self.model.ignition_time_grid[fy, fx] = current_ts
        return (fx, fy)

    def _reset_model(self) -> None:
        """(Re)create FireModel using current settings and re-apply extra ignition seeds."""
        self.model = FireModel(
            width=self.width,
            height=self.height,
            wind_provider=self.wind_provider,
            initial_fire_pos=self.initial_fire_pos,
            terrain=self.terrain,
        )
        self.model.htc_schedule = self.htc_schedule

        if self.ignition_mode == 1 and self.validation_seed_points:
            for extra_seed in self.validation_seed_points[1:]:
                self._ignite_seed(extra_seed)
    
    def _handle_keyboard_events(self, event: pygame.event.Event) -> bool:
        """Handle keyboard input events.
        
        Args:
            event: The keyboard event to process.
            
        Returns:
            False if the simulation should quit, True otherwise.
        """
        if event.key == pygame.K_ESCAPE:
            return False
        
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
        
        elif event.key == pygame.K_r:
            # Reset simulation with original parameters
            self._reset_model()
            self.paused = False
            self.first_frame = True

        elif event.key == pygame.K_v:
            # Toggle ignition mode (manual <-> validation-like) and reset
            self.ignition_mode = 0 if self.ignition_mode == 1 else 1
            if self.ignition_mode == 1 and not self.validation_seed_points:
                self.validation_seed_points = self._compute_validation_seed_points(max_points=self.seed_points)
                if self.validation_seed_points:
                    self.initial_fire_pos = self.validation_seed_points[0]
            self._reset_model()
            self.paused = False
            self.first_frame = True

        elif event.key == pygame.K_1:
            # In validation mode: ignite only 1 seed (or prepare for it) and reset
            self.seed_points = 1
            if self.ignition_mode == 1:
                self.validation_seed_points = self._compute_validation_seed_points(max_points=self.seed_points)
                if self.validation_seed_points:
                    self.initial_fire_pos = self.validation_seed_points[0]
            self._reset_model()
            self.paused = False
            self.first_frame = True

        elif event.key == pygame.K_3:
            # In validation mode: ignite 3 seeds and reset
            self.seed_points = 3
            if self.ignition_mode == 1:
                self.validation_seed_points = self._compute_validation_seed_points(max_points=self.seed_points)
                if self.validation_seed_points:
                    self.initial_fire_pos = self.validation_seed_points[0]
            self._reset_model()
            self.paused = False
            self.first_frame = True
        
        return True

    def _load_terrain(self, width: int, height: int) -> Optional[Terrain]:
        terrain_path = project_root / "data" / "las_20m_resolution.tif"
        if not terrain_path.exists():
            print(f"Brak pliku terenu pod ścieżką {terrain_path}, używam losowego rozmieszczenia.")
            return None
        try:
            return Terrain(
                tiff_path=str(terrain_path),
                target_size=(width, height),
                meters_per_pixel=40,
            )
        except Exception as exc:
            print(f"Nie udało się wczytać terenu: {exc}. Używam losowego rozmieszczenia.")
            return None
    
    def _handle_slider_events(self, event: pygame.event.Event) -> None:
        """Handle slider interaction events.
        
        Args:
            event: The mouse event to process.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                new_fps = self.slider.handle_click(*event.pos)
                if new_fps is not None:
                    self.dragging_slider = True
                    self.current_fps = new_fps
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging_slider = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_slider:
                new_fps = self.slider.handle_click(*event.pos)
                if new_fps is not None:
                    self.current_fps = new_fps
    
    def _update_simulation(self) -> None:
        """Update the simulation by one step if not paused."""
        if self.paused or self.first_frame:
            return
        
        if self.model.running:
            self.model.step()
            
            # Check if fire has extinguished
            is_burning = any(
                agent.state == CellState.Burning
                for agent in self.model.agents
            )
            if not is_burning:
                self.paused = True

        print(self.model.ignition_time_grid)
    
    def _render(self) -> None:
        """Render all visual components to the screen."""
        window_width, window_height = self.screen.get_size()

        # Calculating offset to center elements
        grid_width_px = self.width * self.cell_size
        grid_height_px = self.height * self.cell_size

        offset_x = (window_width - grid_width_px) // 2
        offset_y = (window_height - (grid_height_px + 300)) // 2

        self.offset_x = max(0, offset_x)
        self.offset_y = max(0, offset_y)

        # Right-bottom corner wind_rose
        self.wind_rose.center_x, self.wind_rose.center_y = self.wind_rose.get_offset(10, window_width, window_height)


        self.screen.fill(WHITE)
        
        # LAYER 0 — Background
        window_width, window_height = self.screen.get_size()
        self.renderer.draw_background(self.screen, window_width, window_height)


        # LAYER 1 — Base grid
        self.renderer.draw_base(self.screen, self.model, self.offset_x, self.offset_y)

        # LAYER 2 — Fire effects / sprites / smoke
        self.renderer.draw_effects(self.screen, self.model)

        # LAYER 3 — UI is drawn below normally
        
        # Draw UI elements
        window_width, window_height = self.screen.get_size()
        self.info_panel.draw(
            self.screen,
            self.model,
            self.paused,
            self.current_fps,
            self.grid_height,
            self.cell_size,
            window_width,
            window_height
        )

        # Draw wind rose
        wind_direction = 0
        wind_speed = 0
        if hasattr(self.model, 'wind') and isinstance(self.model.wind, dict):
            wind_direction = self.model.wind.get('direction', 0)
            wind_speed = self.model.wind.get('speed', 0)

        self.wind_rose.draw(self.screen, wind_direction, wind_speed)
        
        pygame.display.flip()
    
    def run(self) -> None:
        """Run the main simulation loop.
        
        Handles rendering, event processing, and simulation updates
        until the user quits.
        """
        running = True
        
        while running:
            # Render first (to show initial state)
            self._render()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if not self._handle_keyboard_events(event):
                        running = False
                
                else:
                    self._handle_slider_events(event)

                # Obsługa kliknięcia w przycisk WRÓĆ
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.info_panel.back_button_rect.collidepoint(event.pos):
                        return "BACK_TO_MENU"
                    if self.info_panel.pause_button_rect.collidepoint(event.pos):
                        self.paused = not self.paused
                    if hasattr(self.info_panel, "reset_button_rect") and \
                            self.info_panel.reset_button_rect.collidepoint(event.pos):

                        self._reset_model()
                        self.paused = False
                        self.first_frame = True
            
            # Update simulation
            self._update_simulation()
            
            # Mark first frame as complete
            if self.first_frame:
                self.first_frame = False
            
            # Control frame rate
            self.clock.tick(self.current_fps)
        
        pygame.quit()
        sys.exit()


def main() -> None:
    """Main entry point for the Pygame visualization.
    
    Shows the setup menu, then runs the simulation with the
    configured parameters.
    """
    while True:
        # Show configuration menu
        menu = SetupMenu()
        params = menu.run()

        # Extract parameters
        width = params['width']
        height = params['height']
        cell_size = params['cell_size']
        ignition_mode = DEFAULT_IGNITION_MODE
        seed_points = DEFAULT_VALIDATION_SEED_POINTS
        # Handle fire position (None means auto-center in model)
        fire_pos = None
        if params['fire_x'] is not None and params['fire_y'] is not None:
            fire_pos = (params['fire_x'], params['fire_y'])

        # Run simulation
        runner = SimulationRunner(
            width,
            height,
            cell_size,
            fire_pos,
            ignition_mode=ignition_mode,
            seed_points=seed_points,
        )
        result = runner.run()

        if result != "BACK_TO_MENU":
            break

if __name__ == "__main__":
    main()

