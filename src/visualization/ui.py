"""UI components for the fire spread visualization.

This module contains interactive UI elements like the info panel
showing simulation status and the speed control slider.
"""

from typing import TYPE_CHECKING, Optional

import pygame

from .colors import WHITE, BLACK, BURNING_COLOR

if TYPE_CHECKING:
    from fire_spread.model import FireModel


class InfoPanel:
    """Displays simulation information at the bottom of the screen.
    
    Shows current step number, pause status, keyboard shortcuts,
    and current FPS setting.
    
    Attributes:
        font: Main font for primary information.
        small_font: Smaller font for secondary information.
    """
    
    def __init__(self) -> None:
        """Initialize the info panel with fonts."""
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)
    
    def draw(
        self,
        screen: pygame.Surface,
        model: "FireModel",
        paused: bool,
        fps: int,
        grid_height: int,
        cell_size: int,
        window_width: int
    ) -> None:
        """Draw the information panel.
        
        Args:
            screen: The Pygame surface to draw on.
            model: The fire spread model.
            paused: Whether the simulation is paused.
            fps: Current frames per second setting.
            grid_height: Height of the grid in cells.
            cell_size: Size of each cell in pixels.
            window_width: Width of the window in pixels.
        """
        panel_y = grid_height * cell_size
        
        # Background
        pygame.draw.rect(
            screen,
            WHITE,
            (0, panel_y, window_width, 100)
        )
        
        # Step counter
        step_text = self.font.render(f"Krok: {model.steps}", True, BLACK)
        screen.blit(step_text, (10, panel_y + 10))
        
        # Status (paused/playing)
        status = "PAUZA" if paused else "GRAJ"
        status_text = self.font.render(status, True, BLACK)
        screen.blit(status_text, (200, panel_y + 10))
        
        # Instructions
        help_text = self.small_font.render(
            "SPACJA=Pauza | R=Reset | ESC=Wyjście",
            True,
            BLACK
        )
        screen.blit(help_text, (400, panel_y + 10))
        
        # Speed label
        speed_label = self.small_font.render(f"Prędkość: {fps} FPS", True, BLACK)
        screen.blit(speed_label, (10, panel_y + 50))


class SpeedSlider:
    """Interactive slider for controlling simulation speed.
    
    Allows the user to adjust the FPS (frames per second) by clicking
    and dragging a circular handle along a horizontal bar.
    
    Attributes:
        x: X coordinate of the slider's left edge.
        y: Y coordinate of the slider's center.
        width: Width of the slider bar in pixels.
        height: Height of the slider bar in pixels.
        min_val: Minimum value (FPS).
        max_val: Maximum value (FPS).
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        min_val: int,
        max_val: int
    ) -> None:
        """Initialize the speed slider.
        
        Args:
            x: X coordinate of the slider's left edge.
            y: Y coordinate of the slider's center.
            width: Width of the slider bar.
            height: Height of the slider bar.
            min_val: Minimum FPS value.
            max_val: Maximum FPS value.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
    
    def draw(self, screen: pygame.Surface, current_val: int) -> None:
        """Draw the slider with handle at the current value position.
        
        Args:
            screen: The Pygame surface to draw on.
            current_val: Current FPS value to display.
        """
        # Draw slider bar background
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))
        
        # Calculate handle position
        ratio = (current_val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.x + int(ratio * self.width)
        
        # Draw circular handle
        pygame.draw.circle(
            screen,
            BURNING_COLOR,
            (handle_x, self.y + self.height // 2),
            10  # Handle radius
        )
    
    def handle_click(self, mouse_x: int, mouse_y: int) -> Optional[int]:
        """Check if the slider was clicked and return new value.
        
        Args:
            mouse_x: X coordinate of mouse click.
            mouse_y: Y coordinate of mouse click.
            
        Returns:
            New FPS value if slider was clicked, None otherwise.
        """
        # Check if click is within slider bounds (with some vertical tolerance)
        if self.x <= mouse_x <= self.x + self.width:
            if self.y - 10 <= mouse_y <= self.y + self.height + 10:
                # Calculate new value based on click position
                ratio = (mouse_x - self.x) / self.width
                new_val = self.min_val + ratio * (self.max_val - self.min_val)
                # Clamp to valid range
                return max(self.min_val, min(self.max_val, int(new_val)))
        
        return None
