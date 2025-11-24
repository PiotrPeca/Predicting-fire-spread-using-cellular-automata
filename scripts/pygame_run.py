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

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from fire_spread import FireModel, CellState, WindProvider

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
        fire_pos: Optional[tuple[int, int]]
    ) -> None:
        """Initialize the simulation runner.
        
        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            cell_size: Size of each cell in pixels.
            wind: Wind vector [x, y].
            fire_pos: Initial fire position (x, y) or None for center.
        """
        # Window setup
        window_width = width * cell_size
        window_height = height * cell_size + 300  # Extra space for UI
        
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Symulacja Pożaru Lasu")
        self.clock = pygame.time.Clock()
        
        # Store initial parameters for reset
        self.width = width
        self.height = height
        self.wind_provider = WindProvider(61.62453, 14.69939)

        from_date = datetime.fromisoformat("2018-07-05T14:00:00")
        to_date = from_date + timedelta(days=10)
        self.wind_provider.fetch_data(from_date=from_date, to_date=to_date)

        self.initial_fire_pos = fire_pos
        
        # Create model
        self.model = FireModel(
            width=width,
            height=height,
            wind_provider=self.wind_provider,
            initial_fire_pos=fire_pos
        )
        
        # Visualization components
        self.renderer = GridRenderer(cell_size)
        self.info_panel = InfoPanel()
        self.slider = SpeedSlider(
            x=200,
            y=height * cell_size + 55,
            width=400,
            height=20,
            min_val=MIN_FPS,
            max_val=MAX_FPS
        )
        self.wind_rose = WindRose(
            center_x=window_width - 120,
            center_y=height * cell_size + 120,
            radius=70
        )
        
        # Simulation state
        self.paused = False
        self.current_fps = DEFAULT_FPS
        self.first_frame = True
        self.dragging_slider = False
        
        # Grid dimensions (for info panel)
        self.grid_height = height
        self.cell_size = cell_size
        self.window_width = window_width
    
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
            self.model = FireModel(
                width=self.width,
                height=self.height,
                wind_provider=self.wind_provider,
                initial_fire_pos=self.initial_fire_pos
            )
            self.paused = False
            self.first_frame = True
        
        return True
    
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
    
    def _render(self) -> None:
        """Render all visual components to the screen."""
        self.screen.fill(WHITE)
        
        # LAYER 0 — Background
        self.renderer.draw_background(self.screen, self.window_width, self.grid_height * self.cell_size + 300)

        # LAYER 1 — Base grid
        self.renderer.draw_base(self.screen, self.model)

        # LAYER 2 — Fire effects / sprites / smoke
        self.renderer.draw_effects(self.screen, self.model)

        # LAYER 3 — UI is drawn below normally
        
        # Draw UI elements
        self.info_panel.draw(
            self.screen,
            self.model,
            self.paused,
            self.current_fps,
            self.grid_height,
            self.cell_size,
            self.window_width
        )
        self.slider.draw(self.screen, self.current_fps)

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
        # Handle fire position (None means auto-center in model)
        fire_pos = None
        if params['fire_x'] is not None and params['fire_y'] is not None:
            fire_pos = (params['fire_x'], params['fire_y'])

        # Run simulation
        runner = SimulationRunner(width, height, cell_size, fire_pos)
        result = runner.run()

        if result != "BACK_TO_MENU":
            break

if __name__ == "__main__":
    main()

