# Fire Spread Simulation using Cellular Automata

A Discrete Event Simulation model that visualizes and predicts how fire spreads in Biebrza National Park using Stochastic Cellular Automata, built with the Mesa agent-based modeling framework.

##  Project Overview

This project implements a cellular automaton to simulate fire spread through forested areas. Each cell in the grid can be in one of several states (fuel, burning, burned, or empty), and fire spreads probabilistically to neighboring cells based on their fuel type and current state.

##  Project Structure

```
Predicting-fire-spread-using-cellular-automata/
├── src/
│   └── fire_spread/          # Main package
│       ├── __init__.py       # Package initialization
│       ├── cell.py           # ForestCell agent implementation
│       └── model.py          # FireModel implementation
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_cell.py          # Tests for cell logic
│   └── test_model.py         # Tests for model logic
├── scripts/                  # Utility scripts
│   └── run_simulation.py     # Main simulation runner
├── data/                     # Data files
│   ├── fire_archive_J1V-C2_675226.json
│   ├── fire_archive_M-C61_675224.json
│   └── fire_archive_SV-C2_675228.json
├── docs/                     # Documentation
├── .gitignore               # Git ignore rules (IDE-agnostic)
├── pyproject.toml           # Project configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

