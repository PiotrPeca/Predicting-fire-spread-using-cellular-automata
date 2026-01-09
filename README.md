# Fire Spread Simulation using Cellular Automata

**Predicting fire spread in Biebrza National Park using Stochastic Cellular Automata**

A comprehensive fire spread simulation and validation framework that combines agent-based modeling with real-world weather data and satellite imagery. This project simulates wildfire propagation through forested areas using cellular automata principles, validated against actual fire observations from Biebrza National Park in Poland.

## Key Features

- **Stochastic Cellular Automata Model**: Probabilistic fire spread based on fuel type, wind, and weather conditions
- **Real Weather Integration**: Incorporates actual meteorological data (temperature, humidity, wind)
- **Terrain Awareness**: Uses GeoTIFF terrain data for realistic fire spread simulation
- **Validation Framework**: Compare simulations against real fire observations from satellite data
- **Interactive Visualization**: Real-time Pygame-based visualization with configurable parameters
- **Headless Simulation**: Batch processing with metrics calculation for parameter tuning
- **Drought Index**: Sielianinow-based drought calculations for historical weather context
- **Wind Physics**: Wind-enhanced ignition with customizable gust thresholds and spark probabilities

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)
- Virtual environment recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PiotrPeca/Predicting-fire-spread-using-cellular-automata.git
cd Predicting-fire-spread-using-cellular-automata
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running Simulations

### Interactive Visualization

Launch the Pygame interface for real-time simulation:
```bash
python scripts/pygame_run.py
```

**Features:**
- Configure grid dimensions and cell size
- Set initial fire position(s)
- Adjust simulation speed
- Toggle wind effects
- Real-time fire progression visualization

**Controls:**
- `SPACE` - Pause/Resume
- `R` - Reset simulation
- Slider - Adjust simulation speed

**Cell Legend:**
- **Green**: Fuel (unburned vegetation)
- **Red**: Burning cells
- **Gray**: Burned areas
- **Blue**: Empty/water cells

### Validation & Batch Processing

Run headless simulations with real weather data and validation metrics:

```bash
python validation/validation_run.py
```

The validation framework:
- Loads satellite-derived fire boundary masks
- Computes real ignition time grids from fire archives
- Picks seed points automatically from real burned areas
- Aligns simulation start time to real fire ignition
- Calculates RMSE, false positives, and false negatives
- Generates comparison visualizations

**Key Configuration Options** (in `validation_run.py`):
```python
CONFIG = {
    # Domain
    "width": 600,
    "height": 300,
    
    # Weather
    "lat": 61.62453,
    "lon": 14.69939,
    "from_date": datetime(2018, 7, 5, 12, 0, 0),
    "to_date": datetime(2018, 7, 5, 12, 0, 0) + timedelta(days=30),
    
    # Model parameters
    "p0": 0.1,                      # Base ignition probability
    "wind_c1": 0.045 * 10,         # Wind effect coefficient 1
    "wind_c2": 0.1 * 10,           # Wind effect coefficient 2
    "spark_gust_threshold_kph": 40,  # Gust threshold for spark ignition
    "spark_ignition_prob": 0.2,    # Spark ignition probability
    
    # Real fire data
    "real_fire_archive_path": REPO_ROOT / "data" / "fire_archive_SV-C2_675228.json",
    "real_boundary_mask_path": REPO_ROOT / "data" / "1a6cd4865f484fb48f8ba4ea97a6e0d1.json",
    
    # Terrain
    "use_terrain": True,
    "terrain_path": REPO_ROOT / "data" / "las_20m_resolution.tif",
    
    # Output
    "save_run_outputs": True,
    "outputs_root": REPO_ROOT / "outputs",
}
```

## Testing

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

Run all tests:
```bash
pytest
```

With coverage report:
```bash
pytest --cov=fire_spread --cov-report=term-missing
```

Run specific test file:
```bash
pytest tests/test_validation_metrics.py -v
```

## Project Structure

```
Predicting-fire-spread-using-cellular-automata/
├── src/
│   └── fire_spread/              # Main simulation package
│       ├── __init__.py           # Package exports (FireModel, etc.)
│       ├── cell.py               # ForestCell agent implementation
│       ├── model.py              # FireModel main class
│       ├── terrain.py            # Terrain and GeoTIFF handling
│       ├── wind_provider.py      # Weather data integration
│       ├── validation/
│       │   ├── metrics.py        # TOA error metrics (RMSE, FP, FN)
│       │   ├── visualisation/
│       │   │   └── grid_viz.py   # Grid visualization utilities
│       │   └── ignition_processor.py  # Real fire data processing
│       └── vegetation.py         # Vegetation types and densities
│
├── validation/
│   └── validation_run.py         # Headless batch simulation runner
│
├── scripts/
│   └── pygame_run.py             # Interactive Pygame visualization
│
├── tests/
│   ├── conftest.py               # Pytest configuration
│   ├── test_validation_metrics.py
│   └── __pycache__/
│
├── data/
│   ├── fire_archive_*.json       # Satellite fire observations
│   ├── 1a6cd4865f484fb48f8ba4ea97a6e0d1.json  # Fire boundary mask
│   ├── wind_data.json            # Cached weather data
│   ├── weather_cache/            # Hourly weather cache
│   └── las_20m_resolution.tif    # Terrain GeoTIFF
│
├── outputs/
│   ├── run_YYYYMMDD_HHMMSS_N/   # Per-run outputs
│   │   ├── toa_compare.png       # Real vs simulated TOA
│   │   └── toa_error.png         # Error magnitude map
│   └── p0_*/                     # Parameter sweep results
│
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
└── README.md                     # This file
```

## Model Details

### Cellular Automaton Rules

1. **Cell States**: FUEL → BURNING → BURNED | EMPTY
2. **Fire Spread**: Probabilistic to 4-neighbors (von Neumann) based on:
   - Target cell fuel type
   - Current weather (wind, drought)
   - Distance decay
3. **Ignition Probability**: $p = p_0 \cdot f(\text{fuel}) \cdot g(\text{wind}) \cdot h(\text{drought})$

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p0` | 0.1 | Base ignition probability |
| `wind_c1` | 0.45 | Wind effect scaling factor 1 |
| `wind_c2` | 1.0 | Wind effect scaling factor 2 |
| `spark_gust_threshold_kph` | 40 | Wind speed threshold for long-range ignition |
| `spark_ignition_prob` | 0.2 | Probability of spark-induced ignition |

### Weather Data

- **Source**: Xweather meteorological data
- **Variables**: Temperature, relative humidity, wind speed, wind direction
- **Frequency**: Hourly
- **Caching**: Automatic local cache to reduce API calls

### Terrain Integration

- **Format**: GeoTIFF raster data
- **Resolution**: 40m per pixel (configurable)
- **Use**: Vegetation coverage and biomass mapping per cell

## Validation Metrics

The validation framework computes:

- **RMSE**: Root Mean Square Error of Time-of-Arrival (TOA) in hours
- **False Negatives**: Real-burned cells not detected by simulation
- **False Positives**: Simulation-burned cells with no real fire record
- **Bias**: Systematic over/under-estimation of fire arrival time

Metrics are computed only within the real fire boundary mask for fair comparison.

## Real Fire Data

The project includes fire observations from **VIIRS and EFFIS** (Ljusdal):

- Satellite-derived fire boundaries
- Time-of-Arrival grids from multispectral analysis


**Data files**:
- `fire_archive_*.json` - Fire perimeters at different times
- `1a6cd4865f484fb48f8ba4ea97a6e0d1.json` - Boundary mask
- Weather cache in `data/weather_cache/`

## Development

### Adding New Features

1. Extend `FireModel` in `src/fire_spread/model.py`
2. Add cell logic to `src/fire_spread/cell.py`
3. Write tests in `tests/`
4. Update validation runner if needed

### Custom Wind Providers

Subclass `WindProvider`:
```python
class MyWindProvider(WindProvider):
    def get_next_wind(self) -> dict | None:
        # Return {"windDir": degrees, "windSpd": kph, "timestamp": unix_ts}
        pass
```

## Authors

**Piotr Peca** 

**Jakub Mikołajczyk** [GitHub](https://github.com/Haltie13)

**Mateusz Sobiech**


**Contributors welcome!** Please open issues or pull requests.

## License

This project is provided for educational purposes.

