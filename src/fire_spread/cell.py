"""Forest cell agent implementation for fire spread simulation."""

from enum import Enum
from typing import Tuple, Mapping

from mesa import Agent


class CellState(Enum):
    """Possible states of a forest cell."""
    Empty = 0
    Fuel = 1
    Burning = 2
    Burned = 3


class VegetationType(Enum):
    """Types of vegetation."""
    NO_VEGETATION = "no_vegetation"
    CULTIVATED = "cultivated"
    FORESTS = "forests"
    SHRUB = "shrub"
    WATER = "water"
    ROAD_PRIMARY = "road_primary"
    ROAD_SECONDARY = "road_secondary"
    ROAD_TERTIARY = "road_tertiary"


class VegetationDensity(Enum):
    """Density of vegetation."""
    NO_VEGETATION = "no_vegetation"
    SPARSE = "sparse"
    NORMAL = "normal"
    DENSE = "dense"
    WATER = "water"
    ROAD_PRIMARY = "road_primary"
    ROAD_SECONDARY = "road_secondary"
    ROAD_TERTIARY = "road_tertiary"


class FuelType:
    """Represents a type of fuel with specific burning properties."""
    
    # Vegetation type probability factors (p_veg)
    P_VEG_VALUES = {
        VegetationType.NO_VEGETATION: -1.0,
        VegetationType.CULTIVATED: -0.2,
        VegetationType.FORESTS: 0.4,
        VegetationType.SHRUB: 0.4,
        VegetationType.WATER: -0.4,
        VegetationType.ROAD_PRIMARY: -0.8,
        VegetationType.ROAD_SECONDARY: -0.7,
        VegetationType.ROAD_TERTIARY: -0.4,
    }
    
    # Vegetation density probability factors (p_dens)
    P_DENS_VALUES = {
        VegetationDensity.NO_VEGETATION: -1.0,
        VegetationDensity.SPARSE: -0.3,
        VegetationDensity.NORMAL: 0.0,
        VegetationDensity.DENSE: 0.3,
        VegetationDensity.WATER: -0.4,
        VegetationDensity.ROAD_PRIMARY: -0.8,
        VegetationDensity.ROAD_SECONDARY: -0.7,
        VegetationDensity.ROAD_TERTIARY: -0.4,
    }


    def __init__(
        self, 
        name: str, 
        burn_time: int, 
        color: str, 
        veg_type: VegetationType, 
        veg_density: VegetationDensity
    ):
        """
        Initialize a fuel type.
        
        Args:
            name: Name of the fuel type (e.g., "grass", "tree", "water")
            burn_time: Duration the fuel burns for (in simulation steps)
            color: Color representation for visualization
            veg_type: Type of vegetation
            veg_density: Density of vegetation
        """
        self.name = name
        self.burn_time = burn_time
        self.color = color
        self.veg_type = veg_type
        self.veg_density = veg_density
        self.p_veg = self.P_VEG_VALUES[veg_type]
        self.p_dens = self.P_DENS_VALUES[veg_density]

    def __str__(self) -> str:
        return (f"Fuel type: {self.name}, burn time: {self.burn_time}, "
                f"veg_type: {self.veg_type.value}, veg_density: {self.veg_density.value}, "
                f"p_veg: {self.p_veg}, p_dens: {self.p_dens}")
    
FUEL_TYPES = {
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


def configure_fuel_prob_maps(
    p_veg_values: Mapping[VegetationType, float] | None = None,
    p_dens_values: Mapping[VegetationDensity, float] | None = None,
) -> None:
    """
    Allow overriding global p_veg/p_dens maps used by FuelType from an external script.
    Call before instantiating FuelType objects to ensure new values propagate.
    """
    if p_veg_values:
        FuelType.P_VEG_VALUES.update(p_veg_values)
    if p_dens_values:
        FuelType.P_DENS_VALUES.update(p_dens_values)


class ForestCell(Agent):
    """Agent representing a single cell in the forest grid."""
    
    def __init__(
        self, 
        pos: tuple[int, int], 
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
        # Non-burnable types: water, roads, no vegetation, or already burning/burned
        non_burnable_types = {
            VegetationType.WATER,
            VegetationType.ROAD_PRIMARY,
            VegetationType.ROAD_SECONDARY,
            VegetationType.ROAD_TERTIARY,
            VegetationType.NO_VEGETATION
        }

        if self.fuel.veg_type in non_burnable_types or self.state != CellState.Fuel:
            return False
        return True

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
            ignition_prob = self.model.ignite_prob.get(self.pos, 0.0)
            if ignition_prob > 0 and self.model.random.random() < ignition_prob:
                self.next_state = CellState.Burning
                # burn_timer will be set in advance()

    def advance(self):
        """
        Apply the next state calculated in step().
        
        This two-phase update ensures all cells calculate their next state
        before any state changes are applied.
        """
        prev_state = self.state
        self.state = self.next_state
        
        # Update burning cells tracking for performance optimization
        if prev_state != CellState.Burning and self.state == CellState.Burning:
            # Cell started burning - add to tracking and reset burn timer
            self.burn_timer = int(self.fuel.burn_time)
            self.model.burning_cells.add(self.pos)

            current_ts = self.model.wind.get('timestamp')
            if current_ts is not None:
                x, y = self.pos
                self.model.ignition_time_grid[y, x] = current_ts

        elif prev_state == CellState.Burning and self.state != CellState.Burning:
            # Cell stopped burning - remove from tracking
            self.model.burning_cells.discard(self.pos)
