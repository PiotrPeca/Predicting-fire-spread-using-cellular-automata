"""UI components for the fire spread visualization.

This module contains interactive UI elements like the info panel
showing simulation status and the speed control slider.
"""

from typing import TYPE_CHECKING, Optional

import pygame
import math

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
        self.back_button_rect = pygame.Rect(0, 0, 150, 40)
    
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
            (0, panel_y, window_width, 300)
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

        # Button "WRÓĆ"
        button_x = 200
        button_y = panel_y + 180
        self.back_button_rect.topleft = (button_x, button_y)

        pygame.draw.rect(screen, (200, 200, 200), self.back_button_rect)
        pygame.draw.rect(screen, (0, 0, 0), self.back_button_rect, 2)

        font = pygame.font.SysFont(None, 32)
        label = font.render("WRÓĆ", True, (0, 0, 0))
        screen.blit(label, (button_x + 40, button_y + 5))

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

class WindRose:
    """Displays wind direction and speed as a compass rose.
    
    Shows a circular compass with 16 cardinal directions and a bi-directional 
    arrow indicating current wind direction and speed.
    
    Attributes:
        center_x: X coordinate of the compass center.
        center_y: Y coordinate of the compass center.
        radius: Radius of the compass circle.
        font: Font for direction labels.
    """
    
    def __init__(self, center_x: int, center_y: int, radius: int = 70):
        """Initialize the wind rose.
        
        Args:
            center_x: X coordinate of the compass center.
            center_y: Y coordinate of the compass center.
            radius: Radius of the compass circle (increased for 16 directions).
        """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.font = pygame.font.Font(None, 18)  # Smaller font for more labels
    
    def draw(self, screen: pygame.Surface, wind_direction: float, wind_speed: float) -> None:
        """Draw the wind rose with current wind direction and speed.
        
        Args:
            screen: The Pygame surface to draw on.
            wind_direction: Wind direction in geographic degrees (0° = North, clockwise).
            wind_speed: Wind speed in km/h.
        """
        # Draw circle background
        pygame.draw.circle(screen, (240, 240, 240), (self.center_x, self.center_y), self.radius)
        pygame.draw.circle(screen, (50, 50, 50), (self.center_x, self.center_y), self.radius, 2)
        
        # Draw all 16 compass directions
        directions = [
            ("N", 0), ("NNE", 22.5), ("NE", 45), ("ENE", 67.5),
            ("E", 90), ("ESE", 112.5), ("SE", 135), ("SSE", 157.5),
            ("S", 180), ("SSW", 202.5), ("SW", 225), ("WSW", 247.5),
            ("W", 270), ("WNW", 292.5), ("NW", 315), ("NNW", 337.5)
        ]
        
        for label, angle_deg in directions:
            # Convert to radians (0° = North = up)
            angle_rad = math.radians(angle_deg)
            
            # Calculate position (add offset for text)
            text_distance = self.radius + 25
            text_x = self.center_x + text_distance * math.sin(angle_rad)
            text_y = self.center_y - text_distance * math.cos(angle_rad)
            
            text_surface = self.font.render(label, True, BLACK)
            text_rect = text_surface.get_rect(center=(text_x, text_y))
            screen.blit(text_surface, text_rect)
        
        # Draw wind direction arrow
        # Wind direction means WHERE wind is COMING FROM
        # Arrow should point WHERE wind is GOING TO (opposite direction)
        # So we add 180° to reverse the arrow
        blowing_to_direction = wind_direction + 180

        # Convert geography degrees to math radians
        # Geography: 0° = North, clockwise
        # Math: 0° = East, counter-clockwise
        # So: math_angle = 90° - geo_angle
        math_angle_deg = 90 - blowing_to_direction
        math_angle_rad = math.radians(math_angle_deg)
        
        # Arrow length proportional to wind speed (but capped)
        arrow_length = self.radius - 15
        
        # Calculate arrow tip (red end - shows WHERE wind is blowing TO)
        tip_x = self.center_x + arrow_length * math.cos(math_angle_rad)
        tip_y = self.center_y - arrow_length * math.sin(math_angle_rad)
        
        # Calculate arrow tail (black end - shows WHERE wind is coming FROM)
        tail_x = self.center_x - arrow_length * math.cos(math_angle_rad)
        tail_y = self.center_y + arrow_length * math.sin(math_angle_rad)
        
        # Draw arrow shaft (black to red gradient)
        pygame.draw.line(screen, (100, 100, 100), (tail_x, tail_y), 
                        (self.center_x, self.center_y), 4)
        pygame.draw.line(screen, (200, 0, 0), (self.center_x, self.center_y), 
                        (tip_x, tip_y), 4)
        
        # Draw arrowhead (red triangle at tip)
        arrowhead_size = 10
        arrowhead_angle1 = math_angle_rad + math.radians(150)
        arrowhead_angle2 = math_angle_rad - math.radians(150)
        
        arrowhead_point1 = (
            tip_x + arrowhead_size * math.cos(arrowhead_angle1),
            tip_y - arrowhead_size * math.sin(arrowhead_angle1)
        )
        arrowhead_point2 = (
            tip_x + arrowhead_size * math.cos(arrowhead_angle2),
            tip_y - arrowhead_size * math.sin(arrowhead_angle2)
        )
        
        pygame.draw.polygon(screen, (200, 0, 0), 
                            [(tip_x, tip_y), arrowhead_point1, arrowhead_point2])
            
        # Draw wind speed display box below compass
        box_width = 140
        box_height = 40
        box_x = self.center_x - box_width // 2
        box_y = self.center_y + self.radius + 35
        
        # Draw box background and border
        pygame.draw.rect(screen, (255, 255, 255), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (50, 50, 50), (box_x, box_y, box_width, box_height), 2)
        
        # Draw speed text centered in box
        speed_text = f"Wiatr: {wind_speed:.1f} km/h"
        speed_surface = self.font.render(speed_text, True, BLACK)
        speed_rect = speed_surface.get_rect(center=(self.center_x, box_y + box_height // 2))
        screen.blit(speed_surface, speed_rect)