#!/usr/bin/env python3
"""Main script to run the fire spread simulation."""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from fire_spread.model import FireModel
from fire_spread.cell import CellState


def print_grid(model: FireModel) -> None:
    """
    Print a simple representation of the grid to console.
    
    Args:
        model: The FireModel instance to visualize
    """
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


def main():
    """Run the fire spread simulation."""
    # Simulation parameters
    WIDTH = 10
    HEIGHT = 10
    STEPS = 50

    # Initialize model
    print("--- CREATING MODEL ---")
    model = FireModel(WIDTH, HEIGHT)

    # Set starting fire point
    x_start, y_start = WIDTH // 2, HEIGHT // 2
    center_cell = model.grid[x_start][y_start]

    if center_cell.is_burnable():
        center_cell.state = CellState.Burning
        center_cell.next_state = CellState.Burning
        center_cell.burn_timer = int(center_cell.fuel.burn_time)
        print(f"Ignited cell at position ({x_start}, {y_start})")
    else:
        print("Cannot ignite starting cell.")

    print("--- INITIAL STATE (AFTER IGNITION) ---")
    print_grid(model)

    # Main simulation loop
    for i in range(STEPS):
        print(f"\n--- STEP {i + 1} ---")
        model.step()
        print_grid(model)

        # Check if fire is still burning
        is_burning = any(c.state == CellState.Burning for c in model.agents)
        if not is_burning:
            print("\nFire has been extinguished.")
            break


if __name__ == "__main__":
    main()
