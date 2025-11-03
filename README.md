# Fire Spread Simulation using Cellular Automata

A Discrete Event Simulation model that visualizes and predicts how fire spreads in Biebrza National Park using Stochastic Cellular Automata, built with the Mesa agent-based modeling framework.

##  Project Overview

This project implements a cellular automaton to simulate fire spread through forested areas. Each cell in the grid can be in one of several states (fuel, burning, burned, or empty), and fire spreads probabilistically to neighboring cells based on their fuel type and current state.

##  Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PiotrPeca/Predicting-fire-spread-using-cellular-automata.git
cd Predicting-fire-spread-using-cellular-automata
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux:

# Create virtual environment
python3 -m venv .venv

# Activate it 
source .venv/bin/activate

# On Windows:

# Create virtual environment
python -m venv .venv

# Activate it 
.venv\Scripts\activate
```

3. Install dependencies:
```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Simulation

#### Interactive Pygame Visualization (Recommended)

Run the interactive visualization with a configuration menu:
```bash
python scripts/pygame_viz.py
```

This will launch a graphical interface where you can:
- Configure grid size, cell size, and wind direction
- Set initial fire position
- Control simulation speed with a slider
- Pause/resume with SPACE
- Reset with R
- View real-time fire spread with color-coded cells

**Legend:**
- 🟢 Green: Fuel (unburned vegetation)
- 🔴 Red: Burning cells
- ⬛ Gray: Burned out areas
- 🔵 Blue: Empty/water cells

### Running Tests

Install development dependencies and run tests:
```bash
pip install -r requirements-dev.txt
pytest
```

For coverage report:
```bash
pytest --cov=fire_spread --cov-report=term-missing
```

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

