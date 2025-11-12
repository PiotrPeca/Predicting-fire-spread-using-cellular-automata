"""Visualization package for fire spread simulation using Pygame."""

from .colors import *
from .renderer import GridRenderer
from .ui import InfoPanel, SpeedSlider
from .menu import SetupMenu

__all__ = [
    # Renderer and UI components
    'GridRenderer',
    'InfoPanel',
    'SpeedSlider',
    'SetupMenu',
    
    # Fuel type colors - Cultivated
    'CULTIVATED_SPARSE_COLOR',
    'CULTIVATED_NORMAL_COLOR',
    'CULTIVATED_DENSE_COLOR',
    
    # Fuel type colors - Forest
    'FOREST_SPARSE_COLOR',
    'FOREST_NORMAL_COLOR',
    'FOREST_DENSE_COLOR',
    
    # Fuel type colors - Shrub
    'SHRUB_SPARSE_COLOR',
    'SHRUB_NORMAL_COLOR',
    'SHRUB_DENSE_COLOR',
    
    # Fuel type colors - Barriers
    'WATER_COLOR',
    'ROAD_PRIMARY_COLOR',
    'ROAD_SECONDARY_COLOR',
    'ROAD_TERTIARY_COLOR',
    
    # Cell state colors
    'BURNING_COLOR',
    'BURNED_COLOR',
    
    # UI colors
    'BLACK',
    'WHITE',
    
    # Default parameters
    'DEFAULT_WIDTH',
    'DEFAULT_HEIGHT',
    'DEFAULT_CELL_SIZE',
    'DEFAULT_FPS',
    
    # FPS limits
    'MIN_FPS',
    'MAX_FPS',
]