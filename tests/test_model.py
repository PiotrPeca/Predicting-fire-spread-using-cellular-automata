"""Unit tests for FireModel class."""

import pytest
from fire_spread.model import FireModel
from fire_spread.cell import CellState, ForestCell


class TestFireModel:
    """Test cases for FireModel class."""
    
    def test_model_creation(self):
        """Test creating a fire model."""
        model = FireModel(width=10, height=10)
        assert model.grid.width == 10
        assert model.grid.height == 10
        assert not model.grid.torus  # Grid should not wrap around
    
    def test_grid_initialized_with_cells(self):
        """Test that grid is initialized with forest cells."""
        model = FireModel(width=5, height=5)
        assert len(model.agents) == 25  # 5x5 grid
        
        # Check that all cells are ForestCell instances
        for agent in model.agents:
            assert isinstance(agent, ForestCell)
    
    def test_all_cells_start_as_fuel(self):
        """Test that all cells start in Fuel state."""
        model = FireModel(width=5, height=5)
        for agent in model.agents:
            assert agent.state == CellState.Fuel
    
    def test_model_step(self):
        """Test that model can execute a step."""
        model = FireModel(width=5, height=5)
        # Set one cell on fire
        center_cell = model.grid[2][2]
        center_cell.state = CellState.Burning
        center_cell.burn_timer = 10
        
        # Execute step
        model.step()
        
        # Verify step executed (burn timer should decrease)
        assert center_cell.burn_timer == 9
    
    def test_fire_spreads(self):
        """Test that fire spreads to neighboring cells."""
        model = FireModel(width=5, height=5)
        
        # Set center cell on fire
        center_cell = model.grid[2][2]
        center_cell.state = CellState.Burning
        center_cell.burn_timer = 10
        
        # Run multiple steps to allow fire to spread
        burning_count_before = sum(1 for c in model.agents if c.state == CellState.Burning)
        
        for _ in range(20):
            model.step()
        
        burning_count_after = sum(1 for c in model.agents if c.state == CellState.Burning)
        burned_count = sum(1 for c in model.agents if c.state == CellState.Burned)
        
        # Fire should have spread (more cells burning or burned)
        assert burning_count_after + burned_count > burning_count_before
