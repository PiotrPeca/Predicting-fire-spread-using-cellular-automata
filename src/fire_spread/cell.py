"""Forest cell agent implementation for fire spread simulation."""

from enum import Enum
from typing import Tuple

from mesa import Agent


class CellState(Enum):
    """Possible states of a forest cell."""
    Empty = 0
    Fuel = 1
    Burning = 2
    Burned = 3


class FuelType:
    """Represents a type of fuel with specific burning properties."""
    
    def __init__(self, name: str, burn_time: int, color: str):
        """
        Initialize a fuel type.
        
        Args:
            name: Name of the fuel type (e.g., "grass", "tree", "water")
            burn_time: Duration the fuel burns for (in simulation steps)
            color: Color representation for visualization
        """
        self.name = name
        self.burn_time = burn_time
        self.color = color

    def __str__(self) -> str:
        return f"Fuel type: {self.name}, burn time: {self.burn_time}"


class ForestCell(Agent):
    """Agent representing a single cell in the forest grid."""
    
    def __init__(
        self, 
        pos: Tuple[int, int], 
        model, 
        fuel: FuelType, 
        state: CellState
    ):
        """
        Initialize a forest cell.
        
        Args:
            pos: (x, y) coordinates of the cell
            model: The FireModel instance this cell belongs to
            fuel: FuelType object defining burning properties
            state: Initial CellState of the cell
        """
        self.pos = None  # Required by Mesa library
        self.unique_id = pos  # Unique ID required by Mesa library
        self.model = model
        self.fuel = fuel
        self.state = state
        self.burn_timer = int(fuel.burn_time) if state == CellState.Burning else 0
        self.next_state = state

    def is_burnable(self) -> bool:
        """
        Check if the cell can catch fire.
        
        Returns:
            True if the cell contains fuel and is not already burning/burned
        """
        if self.fuel.name == "water" or self.state != CellState.Fuel:
            return False
        return True

    def burning_chance(self) -> float:
        """
        Calculate the probability of this cell catching fire.
        
        Returns:
            Probability value between 0 and 1 based on burning neighbors
        """
        burning_chance = 0.0
        neighbors = self.model.grid.get_neighbors(
            self.pos, 
            moore=True,  # Check all 8 directions
            include_center=False
        )
        
        for neighbour in neighbors:
            if isinstance(neighbour, ForestCell) and neighbour.state == CellState.Burning:
                burning_chance += 0.1
                
        return burning_chance

    def step(self):
        """
        Calculate the next state of the cell.
        
        This method is called first in the simulation step to determine
        what the cell's next state should be.
        """
        self.next_state = self.state
        
        # Handle burning cells
        if self.state == CellState.Burning:
            self.burn_timer -= 1
            if self.burn_timer <= 0:
                self.next_state = CellState.Burned
            return
        
        # Handle fuel cells that might catch fire
        if self.is_burnable():
            p = self.burning_chance()
            if self.model.random.random() < p:
                self.next_state = CellState.Burning

    def advance(self):
        """
        Apply the next state calculated in step().
        
        This two-phase update ensures all cells calculate their next state
        before any state changes are applied.
        """
        prev = self.state
        self.state = self.next_state

        # Reset burn timer when transitioning to burning
        if prev != CellState.Burning and self.state == CellState.Burning:
            self.burn_timer = int(self.fuel.burn_time)
