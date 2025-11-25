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
    
    def __init__(self, slider) -> None:
        """Initialize the info panel with fonts."""
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)
        self.back_button_rect = pygame.Rect(0, 0, 150, 40)
        self.pause_button_rect = pygame.Rect(0, 0, 150, 40)
        self.slider = slider
    
    def draw(
        self,
        screen: pygame.Surface,
        model: "FireModel",
        paused: bool,
        fps: int,
        grid_height: int,
        cell_size: int,
        window_width: int,
        window_height: int
    ) -> None:
        """Draw unified UI panel block beneath the grid."""

        # === PANEL DIMENSIONS ===
        PANEL_HEIGHT = 200
        PANEL_PADDING = 20
        PANEL_ALPHA = 160
        PANEL_WIDTH = min(1100, window_width - 20)   # 10px margins
        panel_x = (window_width - PANEL_WIDTH) // 2
        panel_y = window_height - PANEL_HEIGHT - 10  # 10px bottom margin

        # === PANEL BACKGROUND ===
        panel_surface = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT), pygame.SRCALPHA)
        panel_surface.fill((80, 0, 0, PANEL_ALPHA))
        screen.blit(panel_surface, (panel_x, panel_y))

        # === TEXTS ===
        step_text = self.font.render(f"Krok: {model.steps}", True, WHITE)
        screen.blit(step_text, (panel_x + PANEL_PADDING, panel_y + 15))

        status = "PAUZA" if paused else "SYMULACJA"
        status_text = self.font.render(status, True, WHITE)
        screen.blit(status_text, (panel_x + PANEL_WIDTH//2 - status_text.get_width()//2, panel_y + 15))

        inst1 = self.small_font.render("SPACJA = Pauza / Wznów", True, WHITE)
        inst2 = self.small_font.render("R = Reset", True, WHITE)
        inst3 = self.small_font.render("ESC = Wyjście", True, WHITE)
        INSTR_PAD = 20  # odstęp od prawej krawędzi panelu

        inst_x = panel_x + PANEL_WIDTH - inst1.get_width() - INSTR_PAD

        screen.blit(inst1, (inst_x, panel_y + 10))
        screen.blit(inst2, (inst_x, panel_y + 35))
        screen.blit(inst3, (inst_x, panel_y + 60))


        fps_text = self.small_font.render(f"Prędkość: {fps} FPS", True, WHITE)
        screen.blit(fps_text, (panel_x + PANEL_PADDING, panel_y + 55))

        # === SLIDER ===
        slider_x = panel_x + PANEL_WIDTH//2 - self.slider.width//2
        slider_y = panel_y + 110
        self.slider.x = slider_x
        self.slider.y = slider_y
        self.slider.draw(screen, fps)

                # === BUTTON DIMENSIONS ===
        BUTTON_W = 150
        BUTTON_H = 40
        SPACING = 20   # odstęp między przyciskami

        # Now we have 3 buttons: RESET, PAUZA, WRÓĆ
        self.reset_button_rect = pygame.Rect(0, 0, BUTTON_W, BUTTON_H)
        self.pause_button_rect.size = (BUTTON_W, BUTTON_H)
        self.back_button_rect.size = (BUTTON_W, BUTTON_H)

        # === BUTTON POSITIONS (RESET - PAUZA - WRÓĆ) ===
        center_x = panel_x + PANEL_WIDTH // 2
        button_y = panel_y + PANEL_HEIGHT - 60

        total_width = BUTTON_W * 3 + SPACING * 2  # total width of 3 buttons + spacing

        # LEFT BUTTON — RESET
        reset_x = center_x - total_width // 2
        self.reset_button_rect.topleft = (reset_x, button_y)

        # CENTER BUTTON — PAUZA / WZNÓW
        pause_x = reset_x + BUTTON_W + SPACING
        self.pause_button_rect.topleft = (pause_x, button_y)

        # RIGHT BUTTON — WRÓĆ
        back_x = pause_x + BUTTON_W + SPACING
        self.back_button_rect.topleft = (back_x, button_y)

        # === RESET BUTTON ===
        pygame.draw.rect(screen, (180, 0, 0), self.reset_button_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, self.reset_button_rect, 2, border_radius=10)
        lbl = self.font.render("RESET", True, WHITE)
        screen.blit(lbl, lbl.get_rect(center=self.reset_button_rect.center))

        # === PAUSE BUTTON ===
        pause_label = "WZNÓW" if paused else "PAUZA"
        pygame.draw.rect(screen, (180, 0, 0), self.pause_button_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, self.pause_button_rect, 2, border_radius=10)
        lbl = self.font.render(pause_label, True, WHITE)
        screen.blit(lbl, lbl.get_rect(center=self.pause_button_rect.center))

        # === BACK BUTTON ===
        pygame.draw.rect(screen, (180, 0, 0), self.back_button_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, self.back_button_rect, 2, border_radius=10)
        lbl = self.font.render("WRÓĆ", True, WHITE)
        screen.blit(lbl, lbl.get_rect(center=self.back_button_rect.center))



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
    
    @staticmethod
    def draw_fire_icon(screen, x, y, scale=1.0):
        """Draw small fire-shaped icon centered at (x, y)."""

        # Flame polygon points (stylized flame)
        pts = [
            (x, y - 12 * scale),
            (x + 6 * scale, y - 4 * scale),
            (x + 4 * scale, y + 6 * scale),
            (x, y + 10 * scale),
            (x - 4 * scale, y + 6 * scale),
            (x - 6 * scale, y - 4 * scale)
        ]

        pygame.draw.polygon(screen, (255, 80, 0), pts)       # main flame
        pygame.draw.polygon(screen, (255, 150, 0), pts, 2)   # outline

    def draw(self, screen: pygame.Surface, current_val: int) -> None:
        """Draw smooth slider with fire icon as handle."""
        
        # === BAR BACKGROUND ===
        bar_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # metal-like bar
        pygame.draw.rect(screen, (20, 20, 20), bar_rect, border_radius=6)
        pygame.draw.rect(screen, (70, 70, 70), bar_rect, 2, border_radius=6)

        # === HANDLE POSITION ===
        ratio = (current_val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.x + int(ratio * self.width)
        handle_y = self.y + self.height // 2

        # === FIRE ICON HANDLE ===
        self.draw_fire_icon(screen, handle_x, handle_y, scale=1.0)

    def handle_click(self, mouse_x: int, mouse_y: int) -> Optional[int]:
        """Smooth FPS calculation from mouse, used for both click + drag."""

        # easier grab area (more forgiving)
        grab_margin = 20
        if not (self.x - grab_margin <= mouse_x <= self.x + self.width + grab_margin):
            return None
        if not (self.y - grab_margin <= mouse_y <= self.y + self.height + grab_margin):
            return None

        # proportional value
        ratio = (mouse_x - self.x) / self.width
        new_val = self.min_val + ratio * (self.max_val - self.min_val)

        return max(self.min_val, min(self.max_val, int(new_val)))

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

    def get_full_size(self):
        """Return full width/height of whole element: panel + labels + speedbox."""
        top_text = 25      # distance from circle to top labels
        bottom_text = 25   # distance from circle to bottom labels
        spacing = 15       # space before speedbox
        box_h = 45         # speedbox height (matches draw())
        panel_padding = 40 # padding inside background

        full_width = (self.radius * 2) + panel_padding * 2
        full_height = (
            (self.radius * 2) +
            top_text +
            bottom_text +
            spacing +
            box_h +
            panel_padding * 2
        )
        return full_width, full_height


    def get_offset(self, margin: int, window_width: int, window_height: int):
        """Compute ideal center_x, center_y so entire wind rose fits bottom-right."""
        
        full_width, full_height = self.get_full_size()

        panel_left = window_width - full_width - margin
        panel_top = window_height - full_height - margin

        # circle center
        center_x = panel_left + full_width // 2
        center_y = panel_top + (full_height // 2) - 5

        return center_x, center_y


    def draw(self, screen: pygame.Surface, wind_direction: float, wind_speed: float) -> None:
        """Draw the wind rose with arrow and speed box."""

        # === CONFIGURABLE SETTINGS ===
        PADDING = 40            # padding around compass
        SPEEDBOX_EXTRA = 60     # space for wind speed box at bottom

        # === CALCULATE FULL ELEMENT SIZE ===
        full_width = (self.radius * 2) + PADDING * 2
        full_height = (self.radius * 2) + PADDING * 2 + SPEEDBOX_EXTRA

        # === DETERMINE TOP-LEFT OF THE PANEL ===
        panel_x = self.center_x - full_width // 2
        panel_y = self.center_y - self.radius - PADDING

        # === DRAW BACKGROUND PANEL (semi-transparent dark red) ===
        panel_surface = pygame.Surface((full_width, full_height), pygame.SRCALPHA)
        panel_surface.fill((120, 0, 0, 160))  # dark red with alpha
        screen.blit(panel_surface, (panel_x, panel_y))

        # === COMPASS CIRCLE ===
        pygame.draw.circle(screen, (240, 240, 240), (self.center_x, self.center_y), self.radius)
        pygame.draw.circle(screen, (50, 50, 50), (self.center_x, self.center_y), self.radius, 2)

        # === 16 DIRECTIONS ===
        directions = [
            ("N", 0), ("NNE", 22.5), ("NE", 45), ("ENE", 67.5),
            ("E", 90), ("ESE", 112.5), ("SE", 135), ("SSE", 157.5),
            ("S", 180), ("SSW", 202.5), ("SW", 225), ("WSW", 247.5),
            ("W", 270), ("WNW", 292.5), ("NW", 315), ("NNW", 337.5)
        ]

        for label, angle_deg in directions:
            angle_rad = math.radians(angle_deg)
            text_distance = self.radius + 25
            text_x = self.center_x + text_distance * math.sin(angle_rad)
            text_y = self.center_y - text_distance * math.cos(angle_rad)
            text_surface = self.font.render(label, True, WHITE)
            text_rect = text_surface.get_rect(center=(text_x, text_y))
            screen.blit(text_surface, text_rect)

        # === WIND ARROW ===
        blowing_to = wind_direction + 180
        math_angle = math.radians(90 - blowing_to)
        arrow_length = self.radius - 15

        tail_x = self.center_x - arrow_length * math.cos(math_angle)
        tail_y = self.center_y + arrow_length * math.sin(math_angle)
        tip_x = self.center_x + arrow_length * math.cos(math_angle)
        tip_y = self.center_y - arrow_length * math.sin(math_angle)

        pygame.draw.line(screen, (100, 100, 100), (tail_x, tail_y),
                        (self.center_x, self.center_y), 4)
        pygame.draw.line(screen, (200, 0, 0), (self.center_x, self.center_y),
                        (tip_x, tip_y), 4)

        head_size = 12
        a1 = math_angle + math.radians(140)
        a2 = math_angle - math.radians(140)
        p1 = (tip_x + head_size * math.cos(a1), tip_y - head_size * math.sin(a1))
        p2 = (tip_x + head_size * math.cos(a2), tip_y - head_size * math.sin(a2))
        pygame.draw.polygon(screen, (200, 0, 0), [(tip_x, tip_y), p1, p2])

        # === WIND SPEED BOX (centered below panel) ===
        box_w, box_h = 150, 45
        box_x = self.center_x - box_w // 2
        box_y = panel_y + full_height - box_h - 10

        pygame.draw.rect(screen, (255, 255, 255), (box_x, box_y, box_w, box_h))
        pygame.draw.rect(screen, (50, 50, 50), (box_x, box_y, box_w, box_h), 2)

        txt = f"Wiatr: {wind_speed:.1f} km/h"
        surf = self.font.render(txt, True, BLACK)
        rect = surf.get_rect(center=(self.center_x, box_y + box_h // 2))
        screen.blit(surf, rect)