"""Grid rendering functionality for the fire spread simulation.

This module provides the GridRenderer class which handles drawing the
cellular automaton grid with proper colors for each cell state.
"""

from typing import TYPE_CHECKING

import pygame

from fire_spread.cell import CellState
from .colors import GREEN, RED, GRAY, BLUE, BLACK

if TYPE_CHECKING:
    from fire_spread.model import FireModel


class GridRenderer:
    """Renders the fire spread grid onto a Pygame surface.
    
    This class is responsible for drawing the grid of cells, where each
    cell is colored based on its current state (Fuel, Burning, Burned, etc.).
    
    Attributes:
        cell_size: Size of each cell in pixels.
    """
    
    def __init__(self, cell_size: int) -> None:
        """Initialize the grid renderer.
        
        Args:
            cell_size: Size of each cell in pixels.
        """
        self.cell_size = cell_size
    
    def get_cell_color(self, state: CellState) -> tuple[int, int, int]:
        """Get the RGB color for a given cell state.
        
        Args:
            state: The current state of the cell.
            
        Returns:
            RGB color tuple for the given state.
        """
        if state == CellState.Fuel:
            return GREEN
        elif state == CellState.Burning:
            return RED
        elif state == CellState.Burned:
            return GRAY
        else:
            return BLUE
    
    def draw(self, screen: pygame.Surface, model: "FireModel") -> None:
        """Draw the grid of cells onto the screen.
        
        Iterates through all agents in the model and draws each as a
        colored rectangle with a black border.
        
        Args:
            screen: The Pygame surface to draw on.
            model: The fire spread model containing all cell agents.
        """
        for agent in model.agents:
            x, y = agent.pos
            color = self.get_cell_color(agent.state)
            
            # Draw filled cell
            pygame.draw.rect(
                screen,
                color,
                (x * self.cell_size, y * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
            
            # Draw cell border
            pygame.draw.rect(
                screen,
                BLACK,
                (x * self.cell_size, y * self.cell_size,
                 self.cell_size, self.cell_size),
                1  # Border width
            )
