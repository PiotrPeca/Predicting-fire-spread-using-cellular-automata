"""Setup menu for configuring simulation parameters.

This module provides an interactive menu where users can configure
grid size, wind direction, fire starting position, and other parameters
before starting the simulation.
"""

from typing import TypedDict, Optional
import sys

import pygame

from .colors import WHITE, BLACK, RED, BLUE, GRAY, GREEN
from .colors import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CELL_SIZE


class SimulationParams(TypedDict):
    """Type definition for simulation parameters.
    
    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        cell_size: Size of each cell in pixels.
        wind_x: Wind component in X direction.
        wind_y: Wind component in Y direction.
        fire_x: Initial fire X position (None for auto-center).
        fire_y: Initial fire Y position (None for auto-center).
    """
    width: int
    height: int
    cell_size: int
    wind_x: int
    wind_y: int
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
            'wind_x': 1,
            'wind_y': 0,
            'fire_x': None,
            'fire_y': None,
        }
        
        # Input state
        self.active_field: Optional[str] = None
        self.input_text: str = ""
    
    def _draw_instructions(self) -> None:
        """Draw instruction text at the top of the menu."""
        instructions = [
            "Kliknij na wartość aby ją zmienić",
            "None = automatyczny środek siatki",
            "Wiatr: dodatni = wschód/północ, ujemny = zachód/południe",
            "",
            "Naciśnij ENTER aby rozpocząć symulację"
        ]
        
        y_offset = 100
        for i, line in enumerate(instructions):
            text = self.small_font.render(line, True, GRAY)
            self.screen.blit(text, (50, y_offset + i * 25))
    
    def _draw_fields(self) -> list[tuple[str, str, any]]:
        """Draw all parameter fields with labels and values.
        
        Returns:
            List of tuples (label, key, value) for each field.
        """
        y_offset = 250
        
        fields = [
            ('Szerokość siatki:', 'width', self.params['width']),
            ('Wysokość siatki:', 'height', self.params['height']),
            ('Rozmiar komórki (px):', 'cell_size', self.params['cell_size']),
            ('Wiatr X (→/←):', 'wind_x', self.params['wind_x']),
            ('Wiatr Y (↑/↓):', 'wind_y', self.params['wind_y']),
            ('Pożar X:', 'fire_x', 
             self.params['fire_x'] if self.params['fire_x'] is not None else 'None'),
            ('Pożar Y:', 'fire_y',
             self.params['fire_y'] if self.params['fire_y'] is not None else 'None'),
        ]
        
        for i, (label, key, value) in enumerate(fields):
            # Draw label
            label_text = self.font.render(label, True, BLACK)
            self.screen.blit(label_text, (100, y_offset + i * 40))
            
            # Draw value (red if active, blue otherwise)
            display_value = self.input_text if self.active_field == key else str(value)
            value_color = RED if self.active_field == key else BLUE
            value_text = self.font.render(display_value, True, value_color)
            value_rect = value_text.get_rect(topleft=(450, y_offset + i * 40))
            self.screen.blit(value_text, value_rect.topleft)
            
            # Draw cursor if field is active
            if self.active_field == key:
                cursor = self.font.render("_", True, RED)
                self.screen.blit(cursor, (value_rect.right + 5, y_offset + i * 40))
        
        return fields
    
    def _draw_screen(self) -> list[tuple[str, str, any]]:
        """Draw the entire menu screen.
        
        Returns:
            List of field definitions for hit detection.
        """
        self.screen.fill(WHITE)
        
        # Title
        title = self.title_font.render("KONFIGURACJA SYMULACJI", True, BLACK)
        self.screen.blit(title, (150, 30))
        
        # Instructions
        self._draw_instructions()
        
        # Parameter fields
        fields = self._draw_fields()
        
        # Start button
        start_text = self.title_font.render("ENTER - START", True, GREEN)
        self.screen.blit(start_text, (250, 520))
        
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
    
    def _handle_mouse_click(
        self,
        mouse_pos: tuple[int, int],
        fields: list[tuple[str, str, any]]
    ) -> None:
        """Handle mouse click for field selection.
        
        Args:
            mouse_pos: The (x, y) position of the mouse click.
            fields: List of field definitions for hit detection.
        """
        mouse_x, mouse_y = mouse_pos
        y_offset = 250
        
        for i, (label, key, value) in enumerate(fields):
            value_rect = pygame.Rect(450, y_offset + i * 40, 200, 35)
            if value_rect.collidepoint(mouse_x, mouse_y):
                self.active_field = key
                # Load current value into input text
                param_value = self.params[key]  # type: ignore
                self.input_text = str(param_value) if param_value is not None else ""
                break
    
    def run(self) -> SimulationParams:
        """Run the setup menu and return configured parameters.
        
        Displays the menu and waits for user input. Returns when the
        user presses ENTER to start the simulation.
        
        Returns:
            Dictionary containing all configured simulation parameters.
        """
        running = True
        
        while running:
            fields = self._draw_screen()
            
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
                        self._handle_mouse_click(event.pos, fields)
        
        return self.params
