"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid

from .cell import ForestCell, FuelType, CellState


class FireModel(Model):
    """Main model for fire spread simulation using cellular automata."""

    def __init__(self, width: int, height: int, wind: list[int], initial_fire_pos: tuple[int, int] | None = None):
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
        self.ignite_prob: dict[tuple[int, int], float] = {}
        
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
            self.ignite_prob[(x, y)] = 0.0

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
        print("Ignite probabilities at this step:")
        for pos, prob in sorted(self.ignite_prob.items()):
            if prob > 0:
                print(f"  {pos}: {prob:.3f}")
        self._prepare_ignite_probabilities()

        # Phase 1: Calculate next states
        for agent in self.agents.shuffle():
            agent.step()
        
        # Phase 2: Apply next states
        for agent in self.agents.shuffle():
            agent.advance()
    
    def _prepare_ignite_probabilities(self) -> None:
        """Compute ignition probabilities for the current step."""
        # Resetting ignite_prob for next step
        for key in self.ignite_prob:
            self.ignite_prob[key] = 0.0

        base_prob = 0.1
        for agent in self.agents:
            if isinstance(agent, ForestCell) and agent.state == CellState.Burning:
                neighbours = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
                for neighbour in neighbours:
                    if isinstance(neighbour, ForestCell) and neighbour.is_burnable():
                        pos = neighbour.pos
                        prev_prob = self.ignite_prob.get(pos, 0.0)
                        next_prob = 1.0 - (1.0 - prev_prob) * (1.0 - base_prob)
                        self.ignite_prob[pos] = max(0.0, min(1.0, next_prob))