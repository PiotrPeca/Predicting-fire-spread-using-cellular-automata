"""Setup menu for configuring simulation parameters.

This module provides an interactive menu where users can configure
grid size, wind direction, fire starting position, and other parameters
before starting the simulation.
"""

from typing import TypedDict, Optional
import sys

import pygame
import os

from .colors import WHITE, BURNING_COLOR
from .colors import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CELL_SIZE


class SimulationParams(TypedDict):
    """Type definition for simulation parameters.
    
    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        cell_size: Size of each cell in pixels.
        fire_x: Initial fire X position (None for auto-center).
        fire_y: Initial fire Y position (None for auto-center).
    """
    width: int
    height: int
    cell_size: int
    fire_x: Optional[int]
    fire_y: Optional[int]


class SetupMenu:
    """Interactive setup menu for simulation configuration.
    
    Displays a GUI where users can click on parameter values to edit them.
    Supports numeric input, negative numbers for wind, and "None" for
    auto-centering the fire position.
    
    Attributes:
        screen: Pygame surface for the menu.
        title_font: Large font for title.
        font: Regular font for labels and values.
        small_font: Small font for instructions.
        params: Dictionary of current parameter values.
        active_field: Currently selected field for editing (if any).
        input_text: Text being entered by the user.
    """
    
    def __init__(self) -> None:
        """Initialize the setup menu with default parameters."""
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Ustawienia Symulacji Pożaru")
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Default parameters
        self.params: SimulationParams = {
            'width': DEFAULT_WIDTH,
            'height': DEFAULT_HEIGHT,
            'cell_size': DEFAULT_CELL_SIZE,
            'fire_x': None,
            'fire_y': None,
        }
        
        # Input state
        self.active_field: Optional[str] = None
        self.input_text: str = ""

        # Background
        bg_path = os.path.join(os.path.dirname(__file__), "images", "forest-background-blurred.jpg")
        self.background_img = pygame.image.load(bg_path).convert()

        # Start button
        self.start_button_rect = pygame.Rect(300, 500, 200, 50)

        self.cursor_timer = 0
        self.cursor_visible = True
        self.clock = pygame.time.Clock()

        self.mouse_pos = (0, 0)


    def _draw_panel(self, rect, color=(0, 100, 0, 150)):
        """Draw semi-transparent rectangle."""
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(color)
        self.screen.blit(panel, rect.topleft)

    def _draw_instructions(self, offset_y) -> None:
        """Draw instruction text at the top of the menu."""
        lines = [
            "Kliknij na wartość aby ją zmienić",
            "None = automatyczny środek siatki"
        ]

        y = offset_y
        for line in lines:
            text = self.font.render(line, True, WHITE)
            x = 400 - text.get_width() // 2  # wyśrodkowanie
            self.screen.blit(text, (x, y))
            y += 28

        return y

    def _draw_fields(self, start_y: int) -> list[tuple[str, str, any]]:
        fields = []

        params_list = [
            ("Szerokość siatki:", "width", self.params["width"]),
            ("Wysokość siatki:", "height", self.params["height"]),
            ("Rozmiar komórki (px):", "cell_size", self.params["cell_size"]),
            ("Pożar X:", "fire_x", self.params["fire_x"] if self.params["fire_x"] is not None else "None"),
            ("Pożar Y:", "fire_y", self.params["fire_y"] if self.params["fire_y"] is not None else "None"),
        ]

        y = start_y
        center_x = 400

        label_x = center_x - 200
        value_x = center_x + 100

        for label, key, value in params_list:

            # label
            label_surf = self.font.render(label, True, WHITE)
            self.screen.blit(label_surf, (label_x, y))

            # value (dynamic)
            display_value = self.input_text if self.active_field == key else str(value)
            value_color = (255, 0, 0) if self.active_field == key else WHITE
            value_surf = self.font.render(display_value, True, value_color)

            # --- BOX ---
            box_width = max(120, value_surf.get_width() + 20)
            box_height = 34
            box_rect = pygame.Rect(value_x - 10, y - 2, box_width, box_height)

            # highlight on hover
            if box_rect.collidepoint(self.mouse_pos):
                pygame.draw.rect(self.screen, (0, 150, 0), box_rect, border_radius=6)  # jaśniejszy zielony
            else:
                pygame.draw.rect(self.screen, (0, 80, 0), box_rect, border_radius=6)

            # border
            pygame.draw.rect(self.screen, (0, 0, 0), box_rect, width=2, border_radius=6)


            # value text
            text_y = y + 5
            value_rect = value_surf.get_rect(topleft=(value_x, text_y))
            self.screen.blit(value_surf, value_rect.topleft)

            # cursor
            if self.active_field == key and self.cursor_visible:
                cursor_surf = self.font.render("|", True, (255, 0, 0))
                self.screen.blit(cursor_surf, (value_rect.right + 5, text_y))


            # hitbox IS STILL NEEDED → now tied to box_rect
            fields.append((label_x, y, key, label_surf.get_width(), value_x, box_width))

            y += 40

        return fields

    
    def _draw_screen(self) -> list[tuple[str, str, any]]:
        """Draw the entire menu screen.
        
        Returns:
            List of field definitions for hit detection.
        """
        # Draw menu background
        bg = pygame.transform.scale(self.background_img, (800, 600))
        self.screen.blit(bg, (0, 0))
        
        # HEADER BLOCK
        title_text = "KONFIGURACJA SYMULACJI"
        title_surface = self.title_font.render(title_text, True, WHITE)

        # wyśrodkowanie
        title_x = 400 - title_surface.get_width() // 2
        title_y = 40

        # panel za headerem
        title_rect = pygame.Rect(
            title_x - 20,
            title_y - 10,
            title_surface.get_width() + 40,
            title_surface.get_height() + 20
        )
        self._draw_panel(title_rect)

        self.screen.blit(title_surface, (title_x, title_y))

        # CONTENT BLOCK (instructions + parameter fields)
        content_top = 130
        content_left = 100
        content_width = 600
        content_height = 330  # dostosujesz jak zechcesz

        content_rect = pygame.Rect(content_left, content_top, content_width, content_height)
        self._draw_panel(content_rect)

        next_y = self._draw_instructions(content_top + 20)
        fields = self._draw_fields(next_y + 20)

        # START BUTTON
        if self.start_button_rect.collidepoint(self.mouse_pos):
            pygame.draw.rect(self.screen, (255, 255, 255), self.start_button_rect, border_radius=12)  # jaśniejszy
        else:
            pygame.draw.rect(self.screen, (230, 230, 230), self.start_button_rect, border_radius=12)
        pygame.draw.rect(self.screen, (0, 0, 0), self.start_button_rect, 2, border_radius=12)

        start_surf = self.title_font.render("START", True, (0, 0, 0))
        sx = self.start_button_rect.centerx - start_surf.get_width() // 2
        sy = self.start_button_rect.centery - start_surf.get_height() // 2
        self.screen.blit(start_surf, (sx, sy))

        
        pygame.display.flip()
        
        return fields
    
    def _handle_text_input(self, event: pygame.event.Event) -> None:
        """Handle text input events for the active field.
        
        Args:
            event: The keyboard event to process.
        """
        if not self.active_field:
            return
        
        if event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key == pygame.K_MINUS or event.unicode == '-':
            self.input_text += '-'
        elif event.unicode.isdigit():
            self.input_text += event.unicode
        elif event.unicode.lower() in 'none':
            self.input_text += event.unicode.lower()
    
    def _save_field_value(self) -> None:
        """Save the current input text to the active field's parameter."""
        if not self.active_field or not self.input_text:
            return
        
        if self.input_text.lower() == 'none':
            self.params[self.active_field] = None  # type: ignore
        else:
            try:
                self.params[self.active_field] = int(self.input_text)  # type: ignore
            except ValueError:
                pass  # Ignore invalid input
        
        self.input_text = ""
        self.active_field = None

    def _handle_mouse_click(self, pos, fields):
        clicked_field = False

        for (label_x, y, key, label_w, value_x, value_w) in fields:

            label_rect = pygame.Rect(label_x, y, label_w, 30)
            value_rect = pygame.Rect(value_x, y, value_w, 30)

            if label_rect.collidepoint(pos) or value_rect.collidepoint(pos):
                # jeśli klikamy pole które już było aktywne → nic nie zapisuj
                if self.active_field and self.active_field != key:
                    self._save_field_value()

                self.active_field = key
                self.input_text = ""
                clicked_field = True
                return

        # jeśli kliknięto poza polami → zapisz wartość
        if self.active_field and not clicked_field:
            self._save_field_value()
            self.active_field = None
            self.input_text = ""


    
    def run(self) -> SimulationParams:
        """Run the setup menu and return configured parameters.
        
        Displays the menu and waits for user input. Returns when the
        user presses ENTER to start the simulation.
        
        Returns:
            Dictionary containing all configured simulation parameters.
        """
        running = True
        
        while running:
            self.cursor_timer += self.clock.get_time()
            if self.cursor_timer > 500:  # 500 ms blink
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0

            fields = self._draw_screen()

            self.mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        # Save current field or start simulation
                        if self.active_field and self.input_text:
                            self._save_field_value()
                        else:
                            running = False
                    
                    elif event.key == pygame.K_ESCAPE:
                        # Cancel editing or quit
                        if self.active_field:
                            self.active_field = None
                            self.input_text = ""
                        else:
                            pygame.quit()
                            sys.exit()
                    
                    elif self.active_field:
                        self._handle_text_input(event)
                
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        
                        # 1) obsługa kliknięcia START
                        if self.start_button_rect.collidepoint(event.pos):
                            # jeśli aktualnie edytujesz pole → zapisz wartość
                            if self.active_field and self.input_text:
                                self._save_field_value()
                            return self.params  # ← uruchamiamy symulację
                        
                        # 2) inne kliknięcia (pola formularza)
                        self._handle_mouse_click(event.pos, fields)
            
            self.clock.tick(60)
        
        return self.params
