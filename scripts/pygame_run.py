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

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from fire_spread import FireModel
from fire_spread import CellState

from visualization import (
    GridRenderer,
    InfoPanel,
    SpeedSlider,
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
        wind: list[int],
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
        window_height = height * cell_size + 100  # Extra space for UI
        
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Symulacja Pożaru Lasu")
        self.clock = pygame.time.Clock()
        
        # Store initial parameters for reset
        self.width = width
        self.height = height
        self.wind = wind
        self.initial_fire_pos = fire_pos
        
        # Create model
        self.model = FireModel(
            width=width,
            height=height,
            wind=wind,
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
                wind=self.wind,
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
        
        # Draw grid
        self.renderer.draw(self.screen, self.model)
        
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
    # Show configuration menu
    menu = SetupMenu()
    params = menu.run()
    
    # Extract parameters
    width = params['width']
    height = params['height']
    cell_size = params['cell_size']
    wind = [params['wind_x'], params['wind_y']]
    fire_pos = (
        (params['fire_x'], params['fire_y'])
        if params['fire_x'] is not None
        else None
    )
    
    # Run simulation
    runner = SimulationRunner(width, height, cell_size, wind, fire_pos)
    runner.run()


if __name__ == "__main__":
    main()

