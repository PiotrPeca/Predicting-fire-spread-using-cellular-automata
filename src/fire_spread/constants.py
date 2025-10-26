"""
Configuration constants for the fire spread simulation.

Copy this file to constants.py and fill in your actual values.
constants.py is gitignored to keep your API keys private.
"""

# Xweather API Configuration
# Get your API keys from: https://www.xweather.com/
XWEATHER_CLIENT_ID = "your_client_id_here"
XWEATHER_CLIENT_SECRET = "your_client_secret_here"

# Xweather API Base URL
XWEATHER_BASE_URL = "https://data.api.xweather.com"

# Cache settings
CACHE_DIR = "data/weather_cache"  # Directory to store cached weather data
CACHE_EXPIRY_HOURS = 24  # How long to keep cached data before refreshing

# Simulation location (example coordinates - update with your location)
DEFAULT_LATITUDE = 53.5833
DEFAULT_LONGITUDE = 22.5000

# Time range constraints
MIN_YEAR = 2004
MAX_DAYS_FUTURE = 15
