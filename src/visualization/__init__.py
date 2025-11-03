"""Visualization package for fire spread simulation using Pygame."""

from .colors import *
from .renderer import GridRenderer
from .ui import InfoPanel, SpeedSlider
from .menu import SetupMenu

__all__ = [
    'GridRenderer',
    'InfoPanel',
    'SpeedSlider',
    'SetupMenu',
    # Colors
    'GREEN',
    'RED',
    'GRAY',
    'BLUE',
    'BLACK',
    'WHITE',
    # Defaults
    'DEFAULT_WIDTH',
    'DEFAULT_HEIGHT',
    'DEFAULT_CELL_SIZE',
    'DEFAULT_FPS',
]
