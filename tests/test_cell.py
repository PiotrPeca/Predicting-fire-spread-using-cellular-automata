"""Unit tests for ForestCell class."""

import pytest
from fire_spread.cell import ForestCell, CellState, FuelType
from fire_spread.model import FireModel


class TestFuelType:
    """Test cases for FuelType class."""
    
    def test_fuel_type_creation(self):
        """Test creating a fuel type."""
        fuel = FuelType(name="grass", burn_time=10, color="green")
        assert fuel.name == "grass"
        assert fuel.burn_time == 10
        assert fuel.color == "green"
    
    def test_fuel_type_str(self):
        """Test string representation of fuel type."""
        fuel = FuelType(name="tree", burn_time=20, color="brown")
        assert str(fuel) == "Fuel type: tree, burn time: 20"


class TestForestCell:
    """Test cases for ForestCell class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return FireModel(width=5, height=5)
    
    @pytest.fixture
    def grass_fuel(self):
        """Create grass fuel type."""
        return FuelType(name="grass", burn_time=10, color="green")
    
    @pytest.fixture
    def water_fuel(self):
        """Create water fuel type."""
        return FuelType(name="water", burn_time=0, color="blue")
    
    def test_cell_creation(self, model, grass_fuel):
        """Test creating a forest cell."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Fuel)
        assert cell.unique_id == (0, 0)
        assert cell.state == CellState.Fuel
        assert cell.fuel == grass_fuel
        assert cell.burn_timer == 0
    
    def test_burnable_fuel_cell(self, model, grass_fuel):
        """Test that fuel cells are burnable."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Fuel)
        assert cell.is_burnable() is True
    
    def test_not_burnable_water(self, model, water_fuel):
        """Test that water cells are not burnable."""
        cell = ForestCell((0, 0), model, water_fuel, CellState.Fuel)
        assert cell.is_burnable() is False
    
    def test_not_burnable_burning_cell(self, model, grass_fuel):
        """Test that burning cells are not burnable."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Burning)
        assert cell.is_burnable() is False
    
    def test_not_burnable_burned_cell(self, model, grass_fuel):
        """Test that burned cells are not burnable."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Burned)
        assert cell.is_burnable() is False
    
    def test_burning_timer_decreases(self, model, grass_fuel):
        """Test that burn timer decreases each step."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Burning)
        initial_timer = cell.burn_timer
        cell.step()
        cell.advance()
        assert cell.burn_timer == initial_timer - 1
    
    def test_cell_burns_out(self, model, grass_fuel):
        """Test that cell transitions to burned when timer reaches 0."""
        cell = ForestCell((0, 0), model, grass_fuel, CellState.Burning)
        cell.burn_timer = 1
        cell.step()
        cell.advance()
        assert cell.state == CellState.Burned


class TestCellState:
    """Test cases for CellState enum."""
    
    def test_cell_states_exist(self):
        """Test that all expected cell states exist."""
        assert CellState.Empty
        assert CellState.Fuel
        assert CellState.Burning
        assert CellState.Burned
    
    def test_cell_state_values(self):
        """Test cell state values."""
        assert CellState.Empty.value == 0
        assert CellState.Fuel.value == 1
        assert CellState.Burning.value == 2
        assert CellState.Burned.value == 3
