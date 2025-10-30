"""Fire spread model implementation."""

from mesa import Model
from mesa.space import SingleGrid

from .cell import ForestCell, FuelType, CellState


class FireModel(Model):
    """Main model for fire spread simulation using cellular automata."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize the fire spread model.
        
        Args:
            width: Width of the grid (number of cells)
            height: Height of the grid (number of cells)
        """
        super().__init__()
        self.grid = SingleGrid(width, height, torus=False)
        
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
