"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid
import math

from .cell import ForestCell, FuelType, CellState
from .wind_provider import WindProvider


class FireModel(Model):
    """Main model for fire spread simulation using cellular automata."""

    def __init__(self, width: int, height: int, wind_provider: WindProvider, initial_fire_pos: tuple[int, int] | None = None):
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
        self.wind_provider = wind_provider
        self.wind: dict
        self.ignite_prob: dict[tuple[int, int], float] = {}
        self.p0 = 0.05 #base_ignition_prob
        self.wind_parametr_c1 = 0.05
        self.wind_parametr_c2 = 0.1
        
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
        self.update_wind()
        self.__str__()
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

        for agent in self.agents:
            if isinstance(agent, ForestCell) and agent.state == CellState.Burning:
                neighbours = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
                for neighbour in neighbours:
                    if isinstance(neighbour, ForestCell) and neighbour.is_burnable():
                        pw = self.calculate_wind_ignition_prob(agent.pos, neighbour.pos)
                        pburn = self.p0 * pw
                        prev_prob = self.ignite_prob.get(neighbour.pos, 0.0)
                        next_prob = 1.0 - (1.0 - prev_prob) * (1.0 - pburn)
                        self.ignite_prob[neighbour.pos] = max(0.0, min(1.0, next_prob))
    def update_wind(self):
        next_wind = rec = self.wind_provider.get_next_wind()
        if not next_wind:
            self.wind = {'x': 0, 'y': 0, 'speed': 0, 'direction': 0}
            return
        direction_degrees = next_wind.get('wind_degree', 0)
        speed = next_wind.get('wind_kph', 0)

        # Conversion from geography to Math grades
        math_degrees = (450 - direction_degrees) % 360
        math_radians = math.radians(math_degrees)

        wind_x = speed * math.cos(math_radians)
        wind_y = speed * math.sin(math_radians)

        self.wind = {
            'wind_x': wind_x,
            'wind_y': wind_y,
            'speed': speed,
            'direction': direction_degrees,
        }
        return
    def calculate_wind_ignition_prob(self, burning_cell_position: tuple[int, int],neighbor_cell_position: tuple[int, int]) -> float:
        '''calculates wind effect in the main formula '''
        V = self.wind.get('speed', 0)
        if V == 0:
            return 1.0
        wind_x = self.wind.get('wind_x', 0)
        wind_y = self.wind.get('wind_y', 0)


        dx = neighbor_cell_position[0] - burning_cell_position[0]
        dy = neighbor_cell_position[1] - burning_cell_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        dot_product = (wind_x * dx) + (wind_y * dy)

        cos_theta = dot_product / (V * distance)

        c1 = self.wind_parametr_c1
        c2 = self.wind_parametr_c2

        exponent_content = V * (c1 + c2 * (cos_theta - 1))

        return math.exp(exponent_content)

    def __str__(self):
        print("Model state:")
        print(self.wind)
        for pos, prob in sorted(self.ignite_prob.items()):
            if prob > 0:
                print(f"  {pos}: {prob:.3f}")

