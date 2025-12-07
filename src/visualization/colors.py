"""Color definitions and constants for the fire spread visualization.

This module contains all RGB color tuples and default configuration values
used throughout the Pygame visualization.
"""

from typing import Tuple

# Type alias for RGB color tuples
Color = Tuple[int, int, int]

# ============================================================================
# FUEL TYPE COLORS (matching model.py fuel_types definitions)
# ============================================================================

# Cultivated vegetation colors
CULTIVATED_SPARSE_COLOR: Color = (245, 228, 118)    # lightyellow
CULTIVATED_NORMAL_COLOR: Color = (255, 255, 0)      # yellow
CULTIVATED_DENSE_COLOR: Color = (255, 215, 0)       # gold

# Forest vegetation colors
FOREST_SPARSE_COLOR: Color = (144, 238, 144)        # lightgreen
FOREST_NORMAL_COLOR: Color = (2, 168, 2)            # green
FOREST_DENSE_COLOR: Color = (0, 100, 0)             # darkgreen

# Shrub vegetation colors
SHRUB_SPARSE_COLOR: Color = (6, 209, 125)          # lightseagreen
SHRUB_NORMAL_COLOR: Color = (3, 173, 103)           # seagreen
SHRUB_DENSE_COLOR: Color = (19, 133, 86)          # darkseagreen

# Non-burnable barrier colors
WATER_COLOR: Color = (0, 0, 255)                    # blue
ROAD_PRIMARY_COLOR: Color = (105, 105, 105)         # gray
ROAD_SECONDARY_COLOR: Color = (150, 150, 150)       # lightgray
ROAD_TERTIARY_COLOR: Color = (201, 181, 181)        # silver

# ============================================================================
# CELL STATE COLORS (burning and burned)
# ============================================================================

BURNING_COLOR: Color = (255, 0, 0)                  # red (on fire)
BURNED_COLOR: Color = (0, 0, 0)                     # black (burned out)

# ============================================================================
# UI COLORS
# ============================================================================

BLACK: Color = (0, 0, 0)                            # Grid lines, text
WHITE: Color = (255, 255, 255)                      # Background

# ============================================================================
# DEFAULT SIMULATION PARAMETERS
# ============================================================================

DEFAULT_WIDTH: int = 600                            # Grid width in cells
DEFAULT_HEIGHT: int = 300                           # Grid height in cells
DEFAULT_CELL_SIZE: int = 2                          # Cell size in pixels
DEFAULT_FPS: int = 5                                # Default frames per second

# ============================================================================
# FPS SLIDER LIMITS
# ============================================================================

MIN_FPS: int = 1                                    # Minimum simulation speed
MAX_FPS: int = 30                                   # Maximum simulation speed