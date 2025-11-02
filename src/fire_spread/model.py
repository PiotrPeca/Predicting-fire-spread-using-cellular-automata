"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid

from .cell import ForestCell, FuelType, CellState


class FireModel(Model):
    """Main model for fire spread simulation using cellular automata."""

    def __init__(self, width: int, height: int, wind: list[int, int], initial_fire_pos: tuple[int, int] | None = None):
        """
        Initialize the fire spread model.
        
        Args:
            width: Width of the grid (number of cells)
            height: Height of the grid (number of cells)
            wind: Wind direction on x and y axes as a list [wind_x, wind_y]
            initial_fire_pos: Optional tuple specifying the initial fire position (x, y).
        """
        super().__init__()
        self.grid = SingleGrid(width, height, torus=False)
        self.wind = wind
        
        # Temporary fuel type setup
        # TODO: Load fuel types from configuration
        self.fuel_grass = FuelType(name="grass", burn_time=10, color="green")

        # Initialize grid with fuel cells
        for content, (x, y) in self.grid.coord_iter():
            fuel = self.fuel_grass
            state = CellState.Fuel
            cell = ForestCell((x, y), self, fuel, state)
            self.grid.place_agent(cell, (x, y))
            self.agents.add(cell)

        # Default to igniting the center of the grid if no initial position is provided
        if initial_fire_pos is None:
            initial_fire_pos = (width // 2, height // 2)
        
        # Ignite the initial fire position
        x_start, y_start = initial_fire_pos
        center_cell = self.grid[x_start][y_start]
        if center_cell.is_burnable():
            center_cell.state = CellState.Burning
            center_cell.next_state = CellState.Burning
            center_cell.burn_timer = int(center_cell.fuel.burn_time)

    def step(self):
        """
        Execute one step of the simulation.
        
        Uses a two-phase update: first all agents calculate their next state,
        then all agents update to their next state. This ensures synchronous updates.
        """
        # Phase 1: Calculate next states
        for agent in self.agents.shuffle():
            agent.step()
        
        # Phase 2: Apply next states
        for agent in self.agents.shuffle():
            agent.advance()
