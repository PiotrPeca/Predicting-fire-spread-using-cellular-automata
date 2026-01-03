"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid
import math
import numpy as np

from .cell import (
    ForestCell,
    FuelType,
    FUEL_TYPES,
    CellState,
    VegetationType,
    VegetationDensity,
    configure_fuel_prob_maps,
)
from .wind_provider import WindProvider
from .terrain import Terrain
from collections import deque
from datetime import datetime
from typing import Optional

class FireModel(Model):
    """Main model for fire spread simulation using cellular automata."""

    def __init__(
        self,
        width: int,
        height: int,
        wind_provider: WindProvider,
        initial_fire_pos: tuple[int, int] | None = None,
        terrain: Optional[Terrain] = None,
        p_veg_override: dict[VegetationType, float] | None = None,
        p_dens_override: dict[VegetationDensity, float] | None = None,
    ):
        super().__init__()

        # Allow external overrides of fuel probability modifiers before cells are created
        configure_fuel_prob_maps(p_veg_override, p_dens_override)

        self.grid = SingleGrid(width, height, torus=False)
        
        # Track burning cells for performance optimization
        self.burning_cells = set()
        
        self.terrain = terrain
        
        if self.terrain:
            terrain_height, terrain_width = self.terrain.get_dimensions()
            if (terrain_width, terrain_height) != (width, height):
                raise ValueError(
                    "Terrain dimensions do not match provided grid size. "
                    "Create the Terrain with target_size=(width, height) before passing it to FireModel."
                )
        self.wind_provider = wind_provider
        self.wind: dict
        self.update_wind()

        self.ignite_prob: dict[tuple[int, int], float] = {}
        self.htc_schedule = {}  # Tu wpadnie harmonogram współczynników sielianowa
        self.drought_multiplier = 1.0
        self.p0 = 0.1 #base_ignition_prob
        self.spark_gust_threshold_kph = 40.0
        self.spark_ignition_prob = 0.1
        self.wind_parametr_c1 = 0.05
        self.wind_parametr_c2 = 0.1
        self.ignition_time_grid = np.full((height, width), np.nan)
        
        # Define fuel types
        self.fuel_types = FUEL_TYPES
        terrain_grid = self.terrain.get_grid() if self.terrain else None

        for content, (x, y) in self.grid.coord_iter():
            if terrain_grid is not None:
                terrain_cell = terrain_grid[y, x]
                fuel_type_key = terrain_cell["fuel_type"]
                fuel = self.fuel_types.get(fuel_type_key, self.fuel_types["water"])
            else:
                fuel = self._generate_random_fuel()

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

        current_ts = self.wind.get('timestamp', 0)

        if center_cell.is_burnable():
            center_cell.state = CellState.Burning
            center_cell.next_state = CellState.Burning
            center_cell.burn_timer = int(center_cell.fuel.burn_time)
            self.burning_cells.add((x_start, y_start))

            self.ignition_time_grid[y_start, x_start] = current_ts
        else:
            # Find nearest burnable cell using BFS
            fire_cell = self._find_nearest_burnable(initial_fire_pos)
            if fire_cell:
                fire_cell.state = CellState.Burning
                fire_cell.next_state = CellState.Burning
                fire_cell.burn_timer = int(fire_cell.fuel.burn_time)
                self.burning_cells.add(fire_cell.pos)

                fx, fy = fire_cell.pos
                self.ignition_time_grid[fy, fx] = current_ts

                print(f"Warning: Initial position {initial_fire_pos} is not burnable. "
                      f"Igniting nearest burnable cell at {fire_cell.pos} instead.")
            else:
                print(f"Critical: No burnable cells found in the entire grid!")

    def _generate_random_fuel(self) -> FuelType:
        rand_val = self.random.random()
        if rand_val < 0.05:
            return self.fuel_types["water"]
        elif rand_val < 0.08:
            road_type = self.random.choice(["road_primary", "road_secondary", "road_tertiary"])
            return self.fuel_types[road_type]
        elif rand_val < 0.48:
            density = self.random.choice(["sparse", "normal", "dense"])
            return self.fuel_types[f"forest_{density}"]
        elif rand_val < 0.8:
            density = self.random.choice(["sparse", "normal", "dense"])
            return self.fuel_types[f"shrub_{density}"]
        else:
            density = self.random.choice(["sparse", "normal", "dense"])
            return self.fuel_types[f"cultivated_{density}"]

    def _find_nearest_burnable(self, start_pos: tuple[int, int]) -> ForestCell | None:
        """
        Find the nearest burnable cell using BFS (Breadth-First Search).
        
        Args:
            start_pos: Starting position (x, y)
            
        Returns:
            Nearest ForestCell that is burnable, or None if none found
        """
        
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        
        while queue:
            current_pos = queue.popleft()
            current_cell = self.grid[current_pos[0]][current_pos[1]]
            
            # Check if current cell is burnable
            if isinstance(current_cell, ForestCell) and current_cell.is_burnable():
                return current_cell
            
            # Add neighbors to queue
            neighbors = self.grid.get_neighborhood(
                current_pos, 
                moore=True, 
                include_center=False
            )
            
            for neighbor_pos in neighbors:
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    queue.append(neighbor_pos)
        
        return None

    def step(self):
        """
        Execute one step of the simulation.
        
        Uses a two-phase update: first all agents calculate their next state,
        then all agents update to their next state. This ensures synchronous updates.
        """
        self.update_wind()
        self._update_drought_multiplier()
        self._prepare_ignite_probabilities()

        # Phase 1: Calculate next states
        for agent in self.agents: #before for agent in self.agents.shuffle(): change to increase model speed
            agent.step()
        
        # Phase 2: Apply next states
        for agent in self.agents: #before for agent in self.agents.shuffle(): change to increase model speed
            agent.advance()
        #self.__str__() # Debug print current model state slows down the model a lot
    
    def _prepare_ignite_probabilities(self) -> None:
        """Compute ignition probabilities for the current step."""
        # Reset all probabilities from previous step (maintains original logic)
        for key in self.ignite_prob:
            self.ignite_prob[key] = 0.0
        
        # Iterate only over burning cells (performance optimization)
        for burning_pos in list(self.burning_cells):  # list() because set may change during iteration
            burning_cell = self.grid[burning_pos[0]][burning_pos[1]]
            
            # Skip cells that are no longer burning (they will be removed in advance())
            if burning_cell.state != CellState.Burning:
                continue
            
            x, y = burning_pos
            neighbours = self.grid.get_neighbors(burning_pos, moore=True, include_center=False)
            
            for neighbour in neighbours:
                if isinstance(neighbour, ForestCell) and neighbour.is_burnable():
                    # Calculate p_burn using formula from article:
                    # p_burn = p0 * (1 + p_veg) * (1 + p_dens) * pw
                    p_veg = neighbour.fuel.p_veg
                    p_dens = neighbour.fuel.p_dens
                    p_w = self.calculate_wind_ignition_prob(burning_pos, neighbour.pos)
                    p_burn = self.p0 * (1.0 + p_veg) * (1.0 + p_dens) * p_w * self.drought_multiplier
                    p_burn = max(0.0, min(1.0, p_burn))

                    # Combine with previous probability using independent sources formula
                    prev_prob = self.ignite_prob.get(neighbour.pos, 0.0)
                    next_prob = 1.0 - (1.0 - prev_prob) * (1.0 - p_burn)
                    self.ignite_prob[neighbour.pos] = max(0.0, min(1.0, next_prob))
            
            self.apply_spark_probability(x, y)


    def update_wind(self):
        next_wind = rec = self.wind_provider.get_next_wind()
        if not next_wind:
            self.wind = {'x': 0, 'y': 0, 'speed': 0, 'direction': 0}
            return
        direction_degrees_from = next_wind.get('windDir', 0)
        speed = next_wind.get('windSpeedKPH', 0)
        gust = next_wind.get('windGustKPH', 0)
        timestamp = next_wind.get('timestamp', 0)

        # Conversion from geography to Math grades
        dir_to = (direction_degrees_from + 180.0) % 360.0
        math_dir_to = 90.0 - dir_to
        math_radians = math.radians(math_dir_to)

        wind_x = speed * math.cos(math_radians)
        wind_y = speed * math.sin(math_radians)

        self.wind = {
            'wind_x': wind_x,
            'wind_y': wind_y,
            'speed': speed,
            'direction': direction_degrees_from,
            'gust': gust,
            'timestamp': timestamp
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

    def main_gust_direction(self):
        wind_x = self.wind.get('wind_x', 0)
        wind_y = self.wind.get('wind_y', 0)

        if wind_x == 0 and wind_y == 0:
            return (0,0)

        if wind_x > 0:
            dx = 1
        elif wind_x < 0:
            dx = -1
        else:
            dx = 0

        if wind_y > 0:
            dy = 1
        elif wind_y < 0:
            dy = -1
        else:
            dy = 0
        return (dx, dy)
    
    def apply_spark_probability(self, x: int, y: int, affected_cells: set = None) -> None:
        """Apply spark ignition probability to cells 2 steps away in wind direction.
        
        Args:
            x, y: Position of burning cell
            affected_cells: Deprecated parameter, kept for compatibility
        """
        gust = float(self.wind.get('gust', 0.0))

        # za słaby poryw → brak iskier
        if gust <= self.spark_gust_threshold_kph:
            return

        dx, dy = self.main_gust_direction()
        if dx == 0 and dy == 0:
            return

        far_x = x + 2 * dx
        far_y = y + 2 * dy
        #print(f"Iskraaa z ({x},{y}) -> ({far_x},{far_y})") debug print slows model

        # jeśli poza planszą → nic nie robimy
        if not (0 <= far_x < self.grid.width and 0 <= far_y < self.grid.height):
            return

        target = self.grid[far_x][far_y]

        # musi być ForestCell i burnable
        if not (isinstance(target, ForestCell) and target.is_burnable()):
            return

        prev = self.ignite_prob.get((far_x, far_y), 0.0)
        p = self.spark_ignition_prob

        next_p = 1.0 - (1.0 - prev) * (1.0 - p)
        self.ignite_prob[(far_x, far_y)] = next_p

    def _update_drought_multiplier(self):
        """
        Sprawdza datę i aktualizuje mnożnik ryzyka (suszy).
        """
        # Pobieramy czas z aktualnych danych wiatrowych
        current_ts = self.wind.get('timestamp')

        if current_ts:
            dt = datetime.fromtimestamp(current_ts)
            date_key = dt.strftime('%Y-%m-%d')

            # Sprawdzamy czy mamy obliczony Sielianinow dla tego dnia
            if date_key in self.htc_schedule:
                htc = self.htc_schedule[date_key]
                # Zabezpieczenie: Mnożnik = 1 / HTC (im mniej wilgoci, tym większy mnożnik)
                safe_htc = max(0.1, htc)
                self.drought_multiplier = 1.0 / safe_htc
            else:
                self.drought_multiplier = 1.0


    def __str__(self):
        print("Model state:")
        print(self.wind)
        print(f' Współczynnik sielianowa {1/self.drought_multiplier}')
        for pos, prob in sorted(self.ignite_prob.items()):
            if prob > 0:
                print(f"  {pos}: {prob:.3f}")

    def describe_parameters(self) -> str:
        """Return a human-readable snapshot of the current model settings."""
        wind_steps = getattr(self.wind_provider, "stemps_per_h", None)
        parts = [
            f"Grid: {self.grid.width} x {self.grid.height}",
            f"Base ignition p0: {self.p0}",
            f"Wind c1: {self.wind_parametr_c1}",
            f"Wind c2: {self.wind_parametr_c2}",
            f"Spark gust threshold [kph]: {self.spark_gust_threshold_kph}",
            f"Spark ignition prob: {self.spark_ignition_prob}",
            f"Terrain: {'enabled' if self.terrain else 'disabled'}",
            f"Wind provider lat/lon: {getattr(self.wind_provider, 'lat', 'n/a')}, {getattr(self.wind_provider, 'lon', 'n/a')}",
            f"Wind steps per hour: {wind_steps if wind_steps is not None else 'n/a'}",
        ]
        return "\n".join(parts)

    def print_parameters(self) -> None:
        """Print model parameters for quick validation in headless runs."""
        print("=== FireModel parameters ===")
        print(self.describe_parameters())
