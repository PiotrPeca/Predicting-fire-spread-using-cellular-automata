"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid
from collections import deque

from .cell import ForestCell, FuelType, CellState, VegetationType, VegetationDensity


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
        
        # Define fuel types
        self.fuel_types = {
            # Cultivated vegetation with different densities
            "cultivated_sparse": FuelType(
                name="cultivated_sparse",
                burn_time=6,
                color="lightyellow",
                veg_type=VegetationType.CULTIVATED,
                veg_density=VegetationDensity.SPARSE
            ),
            "cultivated_normal": FuelType(
                name="cultivated_normal",
                burn_time=8,
                color="yellow",
                veg_type=VegetationType.CULTIVATED,
                veg_density=VegetationDensity.NORMAL
            ),
            "cultivated_dense": FuelType(
                name="cultivated_dense",
                burn_time=10,
                color="gold",
                veg_type=VegetationType.CULTIVATED,
                veg_density=VegetationDensity.DENSE
            ),
            
            # Forest vegetation with different densities
            "forest_sparse": FuelType(
                name="forest_sparse",
                burn_time=12,
                color="lightgreen",
                veg_type=VegetationType.FORESTS,
                veg_density=VegetationDensity.SPARSE
            ),
            "forest_normal": FuelType(
                name="forest_normal",
                burn_time=15,
                color="green",
                veg_type=VegetationType.FORESTS,
                veg_density=VegetationDensity.NORMAL
            ),
            "forest_dense": FuelType(
                name="forest_dense",
                burn_time=18,
                color="darkgreen",
                veg_type=VegetationType.FORESTS,
                veg_density=VegetationDensity.DENSE
            ),
            
            # Shrub vegetation with different densities
            "shrub_sparse": FuelType(
                name="shrub_sparse",
                burn_time=8,
                color="lightseagreen",
                veg_type=VegetationType.SHRUB,
                veg_density=VegetationDensity.SPARSE
            ),
            "shrub_normal": FuelType(
                name="shrub_normal",
                burn_time=10,
                color="seagreen",
                veg_type=VegetationType.SHRUB,
                veg_density=VegetationDensity.NORMAL
            ),
            "shrub_dense": FuelType(
                name="shrub_dense",
                burn_time=12,
                color="darkseagreen",
                veg_type=VegetationType.SHRUB,
                veg_density=VegetationDensity.DENSE
            ),
            
            # Non-burnable types (barriers)
            "water": FuelType(
                name="water",
                burn_time=0,
                color="blue",
                veg_type=VegetationType.WATER,
                veg_density=VegetationDensity.WATER
            ),
            "road_primary": FuelType(
                name="road_primary",
                burn_time=0,
                color="gray",
                veg_type=VegetationType.ROAD_PRIMARY,
                veg_density=VegetationDensity.ROAD_PRIMARY
            ),
            "road_secondary": FuelType(
                name="road_secondary",
                burn_time=0,
                color="lightgray",
                veg_type=VegetationType.ROAD_SECONDARY,
                veg_density=VegetationDensity.ROAD_SECONDARY
            ),
            "road_tertiary": FuelType(
                name="road_tertiary",
                burn_time=0,
                color="silver",
                veg_type=VegetationType.ROAD_TERTIARY,
                veg_density=VegetationDensity.ROAD_TERTIARY
            ),
        }


        # Initialize grid with fuel cells with random terrain generation
        for content, (x, y) in self.grid.coord_iter():
            # Random terrain generation with weighted probabilities
            rand_val = self.random.random()
            
            if rand_val < 0.05:  # 5% water
                fuel = self.fuel_types["water"]
            elif rand_val < 0.08:  # 3% roads
                road_type = self.random.choice(["road_primary", "road_secondary", "road_tertiary"])
                fuel = self.fuel_types[road_type]
            elif rand_val < 0.48:  # 40% forest (various densities)
                density = self.random.choice(["sparse", "normal", "dense"])
                fuel = self.fuel_types[f"forest_{density}"]
            elif rand_val < 0.8:  # 32% shrub (various densities)
                density = self.random.choice(["sparse", "normal", "dense"])
                fuel = self.fuel_types[f"shrub_{density}"]
            else:  # 20% cultivated (various densities)
                density = self.random.choice(["sparse", "normal", "dense"])
                fuel = self.fuel_types[f"cultivated_{density}"]

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
        else:
            # Find nearest burnable cell using BFS
            fire_cell = self._find_nearest_burnable(initial_fire_pos)
            if fire_cell:
                fire_cell.state = CellState.Burning
                fire_cell.next_state = CellState.Burning
                fire_cell.burn_timer = int(fire_cell.fuel.burn_time)
                print(f"Warning: Initial position {initial_fire_pos} is not burnable. "
                      f"Igniting nearest burnable cell at {fire_cell.pos} instead.")
            else:
                print(f"Critical: No burnable cells found in the entire grid!")

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

        base_prob = 0.1 #p0
        for agent in self.agents:
            if isinstance(agent, ForestCell) and agent.state == CellState.Burning:
                neighbours = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
                for neighbour in neighbours:
                    if isinstance(neighbour, ForestCell) and neighbour.is_burnable():
                        pos = neighbour.pos

                        # Calculate p_burn using formula from article:
                        # p_burn = p0 * (1 + p_veg) * (1 + p_dens)
                        p_veg = neighbour.fuel.p_veg
                        p_dens = neighbour.fuel.p_dens
                        p_burn = base_prob * (1.0 + p_veg) * (1.0 + p_dens)
                        p_burn = max(0.0, min(1.0, p_burn))

                        # Combine with previous probability using independent sources formula
                        prev_prob = self.ignite_prob.get(pos, 0.0)
                        next_prob = 1.0 - (1.0 - prev_prob) * (1.0 - p_burn)
                        self.ignite_prob[pos] = max(0.0, min(1.0, next_prob))