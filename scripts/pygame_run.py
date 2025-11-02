#!/usr/bin/env python3
"""Pygame visualization for the fire spread simulation."""

import pygame
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from fire_spread.model import FireModel
from fire_spread.cell import CellState

# Kolory
GREEN = (0, 128, 0)      # Las (paliwo)
RED = (255, 0, 0)        # Pożar
GRAY = (64, 64, 64)      # Spalone
BLUE = (0, 0, 255)       # Woda
BLACK = (0, 0, 0)        # Linie siatki
WHITE = (255, 255, 255)  # Tło

# Parametry domyślne
DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 10
DEFAULT_CELL_SIZE = 40
DEFAULT_FPS = 5


def get_cell_color(state):
    """Zwraca kolor dla danego stanu komórki."""
    if state == CellState.Fuel:
        return GREEN
    elif state == CellState.Burning:
        return RED
    elif state == CellState.Burned:
        return GRAY
    else:
        return BLUE


def draw_grid(screen, model, cell_size):
    """Rysuje siatkę z komórkami."""
    # Rysuj komórki
    for agent in model.agents:
        x, y = agent.pos
        color = get_cell_color(agent.state)
        
        # Rysuj prostokąt komórki
        pygame.draw.rect(
            screen,
            color,
            (x * cell_size, y * cell_size, cell_size, cell_size)
        )
        
        # Rysuj obramowanie komórki
        pygame.draw.rect(
            screen,
            BLACK,
            (x * cell_size, y * cell_size, cell_size, cell_size),
            1  # Grubość linii
        )


def draw_info(screen, model, paused, fps, grid_height, cell_size, window_width):
    """Rysuje informacje na dole ekranu."""
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)
    
    # Tło dla tekstu
    pygame.draw.rect(
        screen,
        WHITE,
        (0, grid_height * cell_size, window_width, 100)
    )
    
    # Tekst z krokiem (Mesa inicjalizuje steps=0, ale zwiększa go w step())
    display_step = model.steps
    step_text = font.render(f"Krok: {display_step}", True, BLACK)
    screen.blit(step_text, (10, grid_height * cell_size + 10))
    
    # Status (pauza/play)
    status = "PAUZA" if paused else "GRAJ"
    status_text = font.render(status, True, BLACK)
    screen.blit(status_text, (200, grid_height * cell_size + 10))
    
    # Instrukcje
    help_text = small_font.render("SPACJA=Pauza | R=Reset | ESC=Wyjście", True, BLACK)
    screen.blit(help_text, (400, grid_height * cell_size + 10))
    
    # Etykieta prędkości
    speed_label = small_font.render(f"Prędkość: {fps} FPS", True, BLACK)
    screen.blit(speed_label, (10, grid_height * cell_size + 50))


def draw_slider(screen, slider_x, slider_y, slider_width, slider_height, min_val, max_val, current_val):
    """Rysuje suwak do kontroli prędkości."""
    # Tło suwaka
    pygame.draw.rect(screen, GRAY, (slider_x, slider_y, slider_width, slider_height))
    
    # Pozycja uchwytu
    handle_x = slider_x + int((current_val - min_val) / (max_val - min_val) * slider_width)
    
    # Uchwyt suwaka
    pygame.draw.circle(screen, RED, (handle_x, slider_y + slider_height // 2), 10)
    
    return slider_x, slider_y, slider_width, slider_height


def handle_slider_click(mouse_x, mouse_y, slider_x, slider_y, slider_width, slider_height, min_val, max_val):
    """Sprawdza czy kliknięto suwak i zwraca nową wartość."""
    # Sprawdź czy kliknięto w obszar suwaka
    if slider_x <= mouse_x <= slider_x + slider_width:
        if slider_y - 10 <= mouse_y <= slider_y + slider_height + 10:
            # Oblicz nową wartość
            ratio = (mouse_x - slider_x) / slider_width
            new_val = min_val + ratio * (max_val - min_val)
            return max(min_val, min(max_val, int(new_val)))
    return None


def show_setup_menu():
    """Pokazuje menu konfiguracyjne i zwraca wybrane parametry."""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Ustawienia Symulacji Pożaru")
    
    # Parametry do konfiguracji
    params = {
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'cell_size': DEFAULT_CELL_SIZE,
        'wind_x': 1,
        'wind_y': 0,
        'fire_x': None,  # None = środek
        'fire_y': None
    }
    
    # Fonty
    title_font = pygame.font.Font(None, 48)
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)
    
    # Pole aktywne (które jest edytowane)
    active_field = None
    input_text = ""
    
    running = True
    while running:
        screen.fill(WHITE)
        
        # Tytuł
        title = title_font.render("KONFIGURACJA SYMULACJI", True, BLACK)
        screen.blit(title, (150, 30))
        
        # Instrukcje
        instructions = [
            "Kliknij na wartość aby ją zmienić",
            "None = automatyczny środek siatki",
            "Wiatr: dodatni = wschód/północ, ujemny = zachód/południe",
            "",
            "Naciśnij ENTER aby rozpocząć symulację"
        ]
        
        y_offset = 100
        for i, line in enumerate(instructions):
            text = small_font.render(line, True, GRAY)
            screen.blit(text, (50, y_offset + i * 25))
        
        # Parametry do edycji
        y_offset = 250
        fields = [
            ('Szerokość siatki:', 'width', params['width']),
            ('Wysokość siatki:', 'height', params['height']),
            ('Rozmiar komórki (px):', 'cell_size', params['cell_size']),
            ('Wiatr X (→/←):', 'wind_x', params['wind_x']),
            ('Wiatr Y (↑/↓):', 'wind_y', params['wind_y']),
            ('Pożar X:', 'fire_x', params['fire_x'] if params['fire_x'] is not None else 'None'),
            ('Pożar Y:', 'fire_y', params['fire_y'] if params['fire_y'] is not None else 'None'),
        ]
        
        for i, (label, key, value) in enumerate(fields):
            # Label
            label_text = font.render(label, True, BLACK)
            screen.blit(label_text, (100, y_offset + i * 40))
            
            # Wartość (klikalna) - jeśli pole aktywne, pokaż input_text
            display_value = input_text if active_field == key else str(value)
            value_text = font.render(display_value, True, RED if active_field == key else BLUE)
            value_rect = value_text.get_rect(topleft=(450, y_offset + i * 40))
            screen.blit(value_text, value_rect.topleft)
            
            # Jeśli to aktywne pole, pokaż kursor
            if active_field == key:
                cursor = font.render("_", True, RED)
                screen.blit(cursor, (value_rect.right + 5, y_offset + i * 40))
        
        # Przycisk Start
        start_text = title_font.render("ENTER - START", True, GREEN)
        screen.blit(start_text, (250, 520))
        
        pygame.display.flip()
        
        # Obsługa zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Zapisz wartość aktywnego pola jeśli coś wpisano
                    if active_field and input_text:
                        if input_text.lower() == 'none':
                            params[active_field] = None
                        else:
                            try:
                                params[active_field] = int(input_text)
                            except ValueError:
                                pass
                        input_text = ""
                        active_field = None
                    else:
                        # Rozpocznij symulację
                        running = False
                
                elif event.key == pygame.K_ESCAPE:
                    if active_field:
                        active_field = None
                        input_text = ""
                    else:
                        pygame.quit()
                        sys.exit()
                
                elif active_field:
                    # Obsługa wpisywania tekstu
                    if event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_MINUS or event.unicode == '-':
                        input_text += '-'
                    elif event.unicode.isdigit():
                        input_text += event.unicode
                    elif event.unicode.lower() in 'none':
                        input_text += event.unicode.lower()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Lewy przycisk
                    mouse_x, mouse_y = event.pos
                    
                    # Sprawdź czy kliknięto na którąś wartość
                    y_offset = 250
                    for i, (label, key, value) in enumerate(fields):
                        value_rect = pygame.Rect(450, y_offset + i * 40, 200, 35)
                        if value_rect.collidepoint(mouse_x, mouse_y):
                            active_field = key
                            input_text = str(params[key]) if params[key] is not None else ""
                            break
    
    return params


def main():
    """Główna funkcja uruchamiająca wizualizację."""
    # Pokaż menu konfiguracyjne
    params = show_setup_menu()
    
    # Wyciągnij parametry
    width = params['width']
    height = params['height']
    cell_size = params['cell_size']
    wind = [params['wind_x'], params['wind_y']]
    fire_pos = (params['fire_x'], params['fire_y']) if params['fire_x'] is not None else None
    
    # Oblicz rozmiar okna
    window_width = width * cell_size
    window_height = height * cell_size + 100
    
    # Inicjalizacja Pygame
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Symulacja Pożaru Lasu")
    clock = pygame.time.Clock()
    
    # Tworzymy model
    model = FireModel(width=width, height=height, wind=wind, initial_fire_pos=fire_pos)
    
    # Stan symulacji
    paused = False
    running = True
    current_fps = DEFAULT_FPS
    first_frame = True  # Flaga do pominięcia step() w pierwszej klatce
    
    # Parametry suwaka
    slider_x = 200
    slider_y = height * cell_size + 55
    slider_width = 400
    slider_height = 20
    min_fps = 1
    max_fps = 30
    dragging_slider = False
    
    # Główna pętla
    while running:
        # Rysowanie NAJPIERW (żeby zobaczyć krok 0)
        screen.fill(WHITE)
        draw_grid(screen, model, cell_size)
        draw_info(screen, model, paused, current_fps, height, cell_size, window_width)
        draw_slider(screen, slider_x, slider_y, slider_width, slider_height, 
                   min_fps, max_fps, current_fps)
        
        # Odświeżanie ekranu
        pygame.display.flip()
        
        # Obsługa zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                # ESC - wyjście
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # SPACJA - pauza/wznów
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                
                # R - reset
                elif event.key == pygame.K_r:
                    model = FireModel(width=width, height=height, wind=wind, initial_fire_pos=fire_pos)
                    paused = False
                    first_frame = True  # Resetuj flagę przy resecie
            
            # Obsługa suwaka myszką
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Lewy przycisk myszy
                    mouse_x, mouse_y = event.pos
                    new_fps = handle_slider_click(
                        mouse_x, mouse_y, slider_x, slider_y, 
                        slider_width, slider_height, min_fps, max_fps
                    )
                    if new_fps is not None:
                        dragging_slider = True
                        current_fps = new_fps
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging_slider:
                    mouse_x, mouse_y = event.pos
                    new_fps = handle_slider_click(
                        mouse_x, mouse_y, slider_x, slider_y,
                        slider_width, slider_height, min_fps, max_fps
                    )
                    if new_fps is not None:
                        current_fps = new_fps
        
        # Wykonaj krok symulacji (jeśli nie pauza i nie pierwsza klatka)
        if not paused and model.running and not first_frame:
            model.step()
            
            # Sprawdź czy pożar wygasł
            is_burning = any(agent.state == CellState.Burning for agent in model.agents)
            if not is_burning:
                paused = True
        
        # Po pierwszej klatce wyłącz flagę
        if first_frame:
            first_frame = False
        
        # Clock tick
        clock.tick(current_fps)
    
    # Zamknięcie
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
