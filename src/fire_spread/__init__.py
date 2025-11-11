"""
Fire Spread Simulation using Cellular Automata.

A model of Discrete Event Simulation which visualizes and predicts 
how fire spread using Stochastic Cellular Automata.
"""

from .cell import ForestCell, CellState, FuelType, VegetationType, VegetationDensity
from .model import FireModel

__version__ = "0.1.0"

__all__ = [
    "ForestCell",
    "CellState",
    "FuelType",
    "VegetationType",
    "VegetationDensity",
    "FireModel",
]
