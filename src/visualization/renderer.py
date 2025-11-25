"""Grid rendering functionality for the fire spread simulation.

This module provides the GridRenderer class which handles drawing the
cellular automaton grid with proper colors for each cell state.
"""

from typing import TYPE_CHECKING

import pygame
import random
import math
import os

from fire_spread.cell import CellState, VegetationType, VegetationDensity
from .colors import (
    BLACK,
    BURNING_COLOR,
    BURNED_COLOR,
    # Cultivated
    CULTIVATED_SPARSE_COLOR,
    CULTIVATED_NORMAL_COLOR,
    CULTIVATED_DENSE_COLOR,
    # Forest
    FOREST_SPARSE_COLOR,
    FOREST_NORMAL_COLOR,
    FOREST_DENSE_COLOR,
    # Shrub
    SHRUB_SPARSE_COLOR,
    SHRUB_NORMAL_COLOR,
    SHRUB_DENSE_COLOR,
    # Barriers
    WATER_COLOR,
    ROAD_PRIMARY_COLOR,
    ROAD_SECONDARY_COLOR,
    ROAD_TERTIARY_COLOR,
)

if TYPE_CHECKING:
    from fire_spread.model import FireModel


class GridRenderer:
    """Renders the fire spread grid onto a Pygame surface.
    
    This class is responsible for drawing the grid of cells, where each
    cell is colored based on its current state (Fuel, Burning, Burned, etc.).
    
    Attributes:
        cell_size: Size of each cell in pixels.
    """

    # Mapping of (VegetationType, VegetationDensity) to colors
    FUEL_COLORS = {
        # Cultivated
        (VegetationType.CULTIVATED, VegetationDensity.SPARSE): CULTIVATED_SPARSE_COLOR,
        (VegetationType.CULTIVATED, VegetationDensity.NORMAL): CULTIVATED_NORMAL_COLOR,
        (VegetationType.CULTIVATED, VegetationDensity.DENSE): CULTIVATED_DENSE_COLOR,
        
        # Forest
        (VegetationType.FORESTS, VegetationDensity.SPARSE): FOREST_SPARSE_COLOR,
        (VegetationType.FORESTS, VegetationDensity.NORMAL): FOREST_NORMAL_COLOR,
        (VegetationType.FORESTS, VegetationDensity.DENSE): FOREST_DENSE_COLOR,
        
        # Shrub
        (VegetationType.SHRUB, VegetationDensity.SPARSE): SHRUB_SPARSE_COLOR,
        (VegetationType.SHRUB, VegetationDensity.NORMAL): SHRUB_NORMAL_COLOR,
        (VegetationType.SHRUB, VegetationDensity.DENSE): SHRUB_DENSE_COLOR,
        
        # Barriers (density doesn't matter, using WATER as default)
        (VegetationType.WATER, VegetationDensity.WATER): WATER_COLOR,
        (VegetationType.ROAD_PRIMARY, VegetationDensity.ROAD_PRIMARY): ROAD_PRIMARY_COLOR,
        (VegetationType.ROAD_SECONDARY, VegetationDensity.ROAD_SECONDARY): ROAD_SECONDARY_COLOR,
        (VegetationType.ROAD_TERTIARY, VegetationDensity.ROAD_TERTIARY): ROAD_TERTIARY_COLOR,
    }
    
    def __init__(self, cell_size: int) -> None:
        """Initialize the grid renderer.
        
        Args:
            cell_size: Size of each cell in pixels.
        """
        self.cell_size = cell_size
        image_path = os.path.join(
            os.path.dirname(__file__), "images", "wildfire-background-blurred.jpg"
        )
        self.background_img = pygame.image.load(image_path).convert()    

    def get_cell_color(self, agent) -> tuple[int, int, int]:
        """Get the RGB color for a given cell based on state and fuel type.
        
        Args:
            agent: The ForestCell agent to get color for.
            
        Returns:
            RGB color tuple for the given cell.
        """
        # Priority 1: Burning state
        if agent.state == CellState.Burning:
            pulse = 1.0 + 0.4 * math.sin(pygame.time.get_ticks() * 0.01 + agent.pos[0] * 0.3)
            return self._apply_glow(BURNING_COLOR, pulse)
        
        # Priority 2: Burned state
        elif agent.state == CellState.Burned:
            return BURNED_COLOR
        
        # Priority 3: Fuel state - color based on vegetation type and density
        elif agent.state == CellState.Fuel:
            fuel_key = (agent.fuel.veg_type, agent.fuel.veg_density)
            return self.FUEL_COLORS.get(fuel_key, FOREST_NORMAL_COLOR)  # Default fallback
        
        # Priority 4: Empty or unknown state
        else:
            return BLACK
        
    def _apply_glow(self, color: tuple[int, int, int], strength: float) -> tuple[int, int, int]:
        """Apply glow to a base color by increasing brightness."""
        r, g, b = color

        r = min(255, int(r * strength))
        g = min(255, int(g * strength))
        b = min(255, int(b * strength))

        return (r, g, b)

    def draw_background(self, screen: pygame.Surface, window_width: int, window_height: int):
        """Layer 0 — background image for simulation."""
        bg = pygame.transform.scale(self.background_img, (window_width, window_height))
        screen.blit(bg, (0, 0))


    def draw_base(self, screen: pygame.Surface, model: "FireModel", offset_x, offset_y):
        """Layer 1 — grid terrain."""
        grid_height = model.grid.height

        for agent in model.agents:
            x, y = agent.pos
            visual_y = grid_height - 1 - y  # bottom-left → top-left conversion
            color = self.get_cell_color(agent)

            pygame.draw.rect(
                screen,
                color,
                (
                    offset_x + x * self.cell_size,
                    offset_y + visual_y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
            )

    def draw_effects(self, screen: pygame.Surface, model: "FireModel"):
        """Layer 2 — currently unused."""
        pass