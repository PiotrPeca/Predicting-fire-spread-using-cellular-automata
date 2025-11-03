"""Color definitions and constants for the fire spread visualization.

This module contains all RGB color tuples and default configuration values
used throughout the Pygame visualization.
"""

from typing import Tuple

# Type alias for RGB color tuples
Color = Tuple[int, int, int]

# Cell state colors
GREEN: Color = (0, 128, 0)      # Forest (fuel)
RED: Color = (255, 0, 0)        # Fire (burning)
GRAY: Color = (64, 64, 64)      # Burned
BLUE: Color = (0, 0, 255)       # Water/Empty

# UI colors
BLACK: Color = (0, 0, 0)        # Grid lines
WHITE: Color = (255, 255, 255)  # Background

# Default simulation parameters
DEFAULT_WIDTH: int = 20         # Grid width in cells
DEFAULT_HEIGHT: int = 10        # Grid height in cells
DEFAULT_CELL_SIZE: int = 40     # Cell size in pixels
DEFAULT_FPS: int = 5            # Default frames per second

# FPS slider limits
MIN_FPS: int = 1
MAX_FPS: int = 30
