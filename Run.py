# --- Plik: run.py ---

from FireModel import FireModel
from ForestCell import CellState


# --- Funkcja pomocnicza do wizualizacji (bez zmian) ---
def print_grid(model):
    """Drukuje prostą reprezentację siatki w konsoli."""
    grid_str = ""
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            cell = model.grid[x][y]
            if cell.state == CellState.Fuel:
                grid_str += "🌲"
            elif cell.state == CellState.Burning:
                grid_str += "🔥"
            elif cell.state == CellState.Burned:
                grid_str += "⬛"
            else:
                grid_str += "🌊"
        grid_str += "\n"
    print(grid_str)


# --- Główny blok uruchomieniowy ---
if __name__ == "__main__":

    # --- Parametry symulacji ---
    WIDTH = 10
    HEIGHT = 10
    STEPS = 50

    # --- Inicjalizacja modelu ---
    print("--- TWORZENIE MODELU ---")
    model = FireModel(WIDTH, HEIGHT)

    # --- ZMIANA: Logika podpalania przeniesiona tutaj ---
    # Ustawiamy punkt startowy pożaru
    x_start, y_start = WIDTH // 2, HEIGHT // 2

    # Dostęp do agenta na siatce przez model.grid[x][y]
    center_cell = model.grid[x_start][y_start]

    if center_cell.is_burnable():
        center_cell.state = CellState.Burning
        center_cell.next_state = CellState.Burning  # Ważne na wszelki wypadek
        center_cell.burn_timer = int(center_cell.fuel.burn_time)
        print(f"Podpalono komórkę na pozycji ({x_start}, {y_start})")
    else:
        print("Nie można podpalić komórki startowej.")
    # --- KONIEC ZMIANY ---

    print("--- STAN POCZĄTKOWY (PO PODPALENIU) ---")
    print_grid(model)

    # --- Główna pętla symulacji ---
    for i in range(STEPS):
        print(f"\n--- KROK {i + 1} ---")
        model.step()
        print_grid(model)

        is_burning = any(c.state == CellState.Burning for c in model.agents)
        if not is_burning:
            print("\nPożar zgasł.")
            break