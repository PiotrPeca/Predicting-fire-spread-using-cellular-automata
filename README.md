# Fire Spread Simulation using Cellular Automata

A Discrete Event Simulation model that visualizes and predicts how fire spreads using Stochastic Cellular Automata, built with the Mesa agent-based modeling framework.

##  Project Overview

This project implements a cellular automaton to simulate fire spread through forested areas. Each cell in the grid can be in one of several states (fuel, burning, burned, or empty), and fire spreads probabilistically to neighboring cells based on their fuel type and current state.

### Features
- Agent-based fire spread simulation using Mesa framework
- Real-world weather data integration via Xweather API
- Intelligent caching system to minimize API calls
- Professional logging using Python's logging module
- Configurable simulation parameters
- Comprehensive test suite

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
# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# On Windows, use:
# .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Weather API Setup (Optional)

To use real weather data in your simulations:

1. Sign up for a free account at [Xweather](https://www.xweather.com/)
2. Get your API credentials (Client ID and Client Secret)
3. Copy the constants template:
```bash
cp src/fire_spread/constants.py.example src/fire_spread/constants.py
```
4. Edit `src/fire_spread/constants.py` and add your credentials

See [docs/WEATHER_API.md](docs/WEATHER_API.md) for detailed weather API documentation.

### Running the Simulation

Run the simulation using the provided script:
```bash
python scripts/run_simulation.py
```

The simulation will display the fire spread in the terminal using emojis:
- 🌲 Fuel (unburned vegetation)
- 🔥 Burning cells
- ⬛ Burned out areas
- 🌊 Empty/water cells

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
│       ├── model.py          # FireModel implementation
│       ├── weather.py        # Weather API wrapper with caching
│       ├── logging_config.py # Logging configuration
│       └── constants.py.example  # Template for API keys
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_cell.py          # Tests for cell logic
│   └── test_model.py         # Tests for model logic
├── scripts/                  # Utility scripts
│   └── run_simulation.py     # Main simulation runner
├── data/                     # Data files
│   ├── fire_archive_J1V-C2_675226.json
│   ├── fire_archive_M-C61_675224.json
│   ├── fire_archive_SV-C2_675228.json
│   └── weather_cache/        # Cached weather API responses
├── docs/                     # Documentation
│   ├── WEATHER_API.md        # Weather API integration guide
│   └── LOGGING.md            # Logging configuration guide
├── .gitignore               # Git ignore rules (IDE-agnostic)
├── pyproject.toml           # Project configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── README.md               # This file
```

## Documentation

- [Weather API Integration Guide](docs/WEATHER_API.md) - How to set up and use the Xweather API for real weather data
- [Logging Guide](docs/LOGGING.md) - How to configure and use logging in the project

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Mesa](https://github.com/projectmesa/mesa) - Agent-based modeling framework
- Weather data from [Xweather](https://www.xweather.com/)

