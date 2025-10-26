"""
Weather API integration for fire spread simulation.

This module provides a wrapper for the Xweather API with intelligent caching.

Caching Strategy:
- Data is stored in daily chunks (one file per location per day)
- All timestamps are rounded to full hours for consistency
- Cache is loaded into memory on initialization for fast lookups
- Cache never expires (weather history doesn't change)
- Only missing data triggers API calls
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

import requests

# Set up logger for this module
logger = logging.getLogger(__name__)

try:
    from .constants import (
        XWEATHER_CLIENT_ID,
        XWEATHER_CLIENT_SECRET,
        XWEATHER_BASE_URL,
        CACHE_DIR,
        MIN_YEAR,
        MAX_DAYS_FUTURE,
    )
except ImportError:
    raise ImportError(
        "constants.py not found."
    )


def round_to_hour(dt: datetime) -> datetime:
    """Round datetime to nearest hour (floor)."""
    return dt.replace(minute=0, second=0, microsecond=0)


class WeatherAPIWrapper:
    """
    Wrapper for Xweather Conditions API with intelligent in-memory caching.
    
    Caching Strategy:
    - Stores data in daily chunks (one file per location per day)
    - Loads all existing cache into memory on initialization
    - Timestamps rounded to full hours for consistency
    - Cache never expires (historical weather doesn't change)
    - Only fetches missing hourly data from API
    
    Data Structure:
    - In-memory: {(lat, lon, timestamp_hour): weather_data_dict}
    - On-disk: location_YYYY-MM-DD.json per day
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Weather API wrapper and load existing cache.

        Args:
            lat: Latitude of the simulation location (-90 to 90)
            lon: Longitude of the simulation location (-180 to 180)
            client_id: Xweather client ID (defaults to constants.py value)
            client_secret: Xweather client secret (defaults to constants.py value)
            base_url: Base URL for API (defaults to constants.py value)
            cache_dir: Directory for caching weather data (defaults to constants.py value)
        """
        if lat is None or lon is None:
            logger.error("Location variables cannot be left empty")
            raise ValueError("Location variables cannot be left empty.")
        
        # Store location for all requests
        self.lat = round(lat, 4)  # Round to ~11m precision
        self.lon = round(lon, 4)
        
        self.client_id = client_id or XWEATHER_CLIENT_ID
        self.client_secret = client_secret or XWEATHER_CLIENT_SECRET
        self.base_url = base_url or XWEATHER_BASE_URL
        
        # Setup cache directory - resolve relative to project root
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Get project root (3 levels up: weather.py -> fire_spread -> src -> project_root)
            project_root = Path(__file__).parent.parent.parent
            self.cache_dir = project_root / CACHE_DIR
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate credentials
        if not self.client_id or self.client_id == "your_client_id_here":
            raise ValueError(
                "Invalid Xweather credentials. Please update constants.py with your API keys."
            )
        
        # In-memory cache: key = (lat, lon, timestamp_hour), value = weather_data
        self.cache: Dict[Tuple[float, float, int], Dict[str, Any]] = {}
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        # Load existing cache into memory
        self._load_cache_from_disk()
        logger.info(f"Initialized with {len(self.cache)} cached hourly records for location ({self.lat}, {self.lon})")

    def _get_cache_filename(self, date: datetime) -> str:
        """
        Generate cache filename for a specific day.
        
        Format: lat_lon_YYYY-MM-DD.json
        Example: 53.5833_22.5000_2024-07-01.json
        """
        return f"{self.lat}_{self.lon}_{date.strftime('%Y-%m-%d')}.json"
    
    def _load_cache_from_disk(self) -> None:
        """
        Load all existing cache files for this location into memory.
        
        Scans cache directory for files matching this location and loads them
        into the in-memory cache dictionary.
        """
        pattern = f"{self.lat}_{self.lon}_*.json"
        cache_files = list(self.cache_dir.glob(pattern))
        
        if not cache_files:
            logger.info(f"No existing cache found for location ({self.lat}, {self.lon})")
            return
        
        loaded_count = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    day_data = json.load(f)
                
                # Load each hourly record into memory
                for record in day_data.get("hours", []):
                    timestamp = record.get("timestamp")
                    if timestamp:
                        # Use Unix timestamp (already in hours) as key
                        cache_key = (self.lat, self.lon, timestamp)
                        self.cache[cache_key] = record
                        loaded_count += 1
                        
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.warning(f"Error loading cache file {cache_file.name}: {e}")
        
        logger.info(f"Loaded {loaded_count} hourly records from {len(cache_files)} cache files")
    
    def _save_day_to_disk(self, date: datetime, hourly_data: List[Dict[str, Any]]) -> None:
        """
        Save a day's worth of hourly data to disk.
        
        Args:
            date: The date (day) to save
            hourly_data: List of hourly weather records for that day
        """
        if not hourly_data:
            return
        
        filename = self._get_cache_filename(date)
        filepath = self.cache_dir / filename
        
        cache_data = {
            "location": {
                "lat": self.lat,
                "lon": self.lon
            },
            "date": date.strftime('%Y-%m-%d'),
            "cached_at": datetime.now().isoformat(),
            "hours": hourly_data
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(hourly_data)} hourly records to {filename}")
        except IOError as e:
            logger.error(f"Error saving cache file {filename}: {e}")
    
    def _group_by_day(self, hourly_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group hourly weather records by day.
        
        Args:
            hourly_data: List of hourly records
            
        Returns:
            Dictionary mapping date string (YYYY-MM-DD) to list of hourly records
        """
        days = defaultdict(list)
        
        for record in hourly_data:
            timestamp = record.get("timestamp")
            if timestamp:
                # Convert Unix timestamp to datetime and extract date
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime('%Y-%m-%d')
                days[date_str].append(record)
        
        return days

    def _validate_date_range(self, from_date: datetime, to_date: datetime) -> None:
        """
        Validate that date range is within API limits.

        Args:
            from_date: Start date
            to_date: End date

        Raises:
            ValueError: If dates are outside allowed range
        """
        min_date = datetime(MIN_YEAR, 1, 1)
        max_date = datetime.now() + timedelta(days=MAX_DAYS_FUTURE)
        
        if from_date < min_date:
            raise ValueError(f"from_date cannot be before {MIN_YEAR}")
        
        if to_date > max_date:
            raise ValueError(f"to_date cannot be more than {MAX_DAYS_FUTURE} days in the future")
        
        if from_date > to_date:
            raise ValueError("from_date must be before to_date")

    def get_conditions(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get hourly weather conditions for the configured location and time range.

        This method checks the in-memory cache first. Only missing hours trigger API calls.
        All timestamps are rounded to full hours.

        Args:
            from_date: Start datetime for weather data (will be rounded to hour)
            to_date: End datetime for weather data (will be rounded to hour)

        Returns:
            List of hourly weather condition dictionaries, or None on error

        Raises:
            ValueError: If date range is invalid
        """
        # Round to hours
        from_date = round_to_hour(from_date)
        to_date = round_to_hour(to_date)
        
        # Validate inputs
        self._validate_date_range(from_date, to_date)
        
        # Generate list of all hours needed
        hours_needed = []
        current = from_date
        while current <= to_date:
            hours_needed.append(current)
            current += timedelta(hours=1)
        
        # Check which hours are missing from cache
        missing_hours = []
        cached_data = []
        
        for hour_dt in hours_needed:
            timestamp = int(hour_dt.timestamp())
            cache_key = (self.lat, self.lon, timestamp)
            
            if cache_key in self.cache:
                # Found in cache
                cached_data.append(self.cache[cache_key])
            else:
                # Missing - need to fetch
                missing_hours.append(hour_dt)
        
        logger.info(f"Requested {len(hours_needed)} hours: {len(cached_data)} cached, {len(missing_hours)} missing")
        
        # Fetch missing data if needed
        if missing_hours:
            # Group missing hours into continuous day ranges for efficient API calls
            day_ranges = self._group_hours_into_day_ranges(missing_hours)
            
            for start, end in day_ranges:
                logger.info(f"Fetching missing data from {start} to {end}")
                fetched_data = self._fetch_from_api(start, end)
                
                if fetched_data:
                    # Add to in-memory cache
                    for record in fetched_data:
                        timestamp = record.get("timestamp")
                        if timestamp:
                            cache_key = (self.lat, self.lon, timestamp)
                            self.cache[cache_key] = record
                            
                            # Add to result if in requested range
                            record_dt = datetime.fromtimestamp(timestamp)
                            if from_date <= record_dt <= to_date:
                                cached_data.append(record)
                    
                    # Save to disk grouped by day
                    days_grouped = self._group_by_day(fetched_data)
                    for date_str, day_records in days_grouped.items():
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                        self._save_day_to_disk(date, day_records)
        
        # Sort by timestamp
        cached_data.sort(key=lambda x: x.get("timestamp", 0))
        
        logger.info(f"Returning {len(cached_data)} hourly records")
        return cached_data if cached_data else None
    
    def _group_hours_into_day_ranges(self, hours: List[datetime]) -> List[Tuple[datetime, datetime]]:
        """
        Group hours into continuous day ranges for efficient API calls.
        
        Args:
            hours: List of datetimes (hourly)
            
        Returns:
            List of (start, end) tuples representing continuous date ranges
        """
        if not hours:
            return []
        
        # Sort hours
        sorted_hours = sorted(hours)
        
        ranges = []
        range_start = sorted_hours[0]
        range_end = sorted_hours[0]
        
        for hour in sorted_hours[1:]:
            # If this hour is within 1 day of range_end, extend the range
            if (hour - range_end).total_seconds() <= 86400:  # 24 hours
                range_end = hour
            else:
                # Gap found, save current range and start new one
                ranges.append((range_start, range_end))
                range_start = hour
                range_end = hour
        
        # Don't forget the last range
        ranges.append((range_start, range_end))
        
        return ranges
    
    def _fetch_from_api(
        self, 
        from_date: datetime, 
        to_date: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch weather data from Xweather API.
        
        Args:
            from_date: Start datetime (rounded to hour)
            to_date: End datetime (rounded to hour)
            
        Returns:
            List of hourly weather records with rounded timestamps
        """
        # Format dates as Unix timestamps 
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        logger.info(f"[API] Fetching data for ({self.lat}, {self.lon}) from {from_timestamp} to {to_timestamp}")
        
        endpoint = f"{self.base_url}/conditions/{self.lat},{self.lon}"
        params = {
            "format": "json",
            "from": from_timestamp,
            "to": to_timestamp,
            "filter": "1hr",  # Hourly data
            "plimit": 2000,   # Max results
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if response is successful
            if not data.get("success", False):
                error_msg = data.get("error", {}).get("description", "Unknown error")
                logger.error(f"[API] Error: {error_msg}")
                return None
            
            # Extract periods (hourly data)
            periods = data.get("response", [{}])[0].get("periods", [])
            
            if not periods:
                logger.warning("[API] No weather data returned")
                return None
            
            # Round all timestamps to full hours and filter to metric data only
            filtered_periods = []
            for period in periods:
                if "timestamp" in period:
                    dt = datetime.fromtimestamp(period["timestamp"])
                    rounded_dt = round_to_hour(dt)
                    period["timestamp"] = int(rounded_dt.timestamp())
                
                # Filter to keep only metric data
                filtered_period = self.filter_metric_data(period)
                filtered_periods.append(filtered_period)
            
            logger.info(f"[API] Retrieved and filtered {len(filtered_periods)} hourly records")
            return filtered_periods
            
        except requests.exceptions.Timeout:
            logger.error("[API] Request timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[API] Request error: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"[API] Response parse error: {e}")
            return None

    def get_wind_data(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract wind speed and direction from weather conditions (metric only).

        Args:
            from_date: Start datetime
            to_date: End datetime

        Returns:
            List of dicts with 'timestamp', 'dateTimeISO', 'speed_kph', 'speed_ms', 
            'gust_kph', 'gust_ms', 'direction_deg', 'direction'
        """
        conditions = self.get_conditions(from_date, to_date)
        
        if not conditions:
            return None
        
        wind_data = []
        for period in conditions:
            wind_info = {
                "timestamp": period.get("timestamp"),
                "dateTimeISO": period.get("dateTimeISO"),
                "speed_kph": period.get("windSpeedKPH"),
                "speed_ms": period.get("windSpeedMS"),
                "gust_kph": period.get("windGustKPH"),
                "gust_ms": period.get("windGustMS"),
                "direction_deg": period.get("windDir"),
                "direction": period.get("windDirENG"),
            }
            wind_data.append(wind_info)
        
        return wind_data

    def get_temperature_humidity(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract temperature and humidity from weather conditions (metric only).

        Args:
            from_date: Start datetime
            to_date: End datetime

        Returns:
            List of dicts with 'timestamp', 'dateTimeISO', 'temp_c', 'humidity', 
            'dewpoint_c', 'feelslike_c'
        """
        conditions = self.get_conditions(from_date, to_date)
        
        if not conditions:
            return None
        
        temp_humidity_data = []
        for period in conditions:
            info = {
                "timestamp": period.get("timestamp"),
                "dateTimeISO": period.get("dateTimeISO"),
                "temp_c": period.get("tempC"),
                "humidity": period.get("humidity"),
                "dewpoint_c": period.get("dewpointC"),
                "feelslike_c": period.get("feelslikeC"),
            }
            temp_humidity_data.append(info)
        
        return temp_humidity_data
        
        return temp_humidity_data

    def clear_cache(self, before_date: Optional[datetime] = None) -> int:
        """
        Clear cached weather data.

        Args:
            before_date: Only clear cache before this date (None = clear all for this location)

        Returns:
            Number of files deleted
        """
        deleted = 0
        pattern = f"{self.lat}_{self.lon}_*.json"
        
        for cache_file in self.cache_dir.glob(pattern):
            should_delete = False
            
            if before_date is None:
                should_delete = True
            else:
                # Extract date from filename
                try:
                    date_part = cache_file.stem.split('_')[-1]  # YYYY-MM-DD
                    file_date = datetime.strptime(date_part, '%Y-%m-%d')
                    should_delete = file_date < before_date
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse date from {cache_file.name}")
                    continue
            
            if should_delete:
                try:
                    cache_file.unlink()
                    deleted += 1
                except OSError as e:
                    logger.error(f"Error deleting cache file {cache_file.name}: {e}")
        
        # Clear from memory too
        if before_date is None:
            # Clear all for this location
            keys_to_delete = [
                key for key in self.cache.keys() 
                if key[0] == self.lat and key[1] == self.lon
            ]
        else:
            # Clear before date
            cutoff_timestamp = int(before_date.timestamp())
            keys_to_delete = [
                key for key in self.cache.keys()
                if key[0] == self.lat and key[1] == self.lon and key[2] < cutoff_timestamp
            ]
        
        for key in keys_to_delete:
            del self.cache[key]
        
        logger.info(f"Deleted {deleted} cache files and {len(keys_to_delete)} memory entries")
        return deleted
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count entries for this location
        location_entries = [
            key for key in self.cache.keys()
            if key[0] == self.lat and key[1] == self.lon
        ]
        
        if not location_entries:
            return {
                "location": f"({self.lat}, {self.lon})",
                "hourly_records": 0,
                "date_range": None,
                "days_covered": 0
            }
        
        # Get timestamp range
        timestamps = [key[2] for key in location_entries]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        min_date = datetime.fromtimestamp(min_ts)
        max_date = datetime.fromtimestamp(max_ts)
        
        # Count unique days
        unique_days = len(set(
            datetime.fromtimestamp(ts).date() 
            for ts in timestamps
        ))
        
        return {
            "location": f"({self.lat}, {self.lon})",
            "hourly_records": len(location_entries),
            "date_range": f"{min_date.date()} to {max_date.date()}",
            "days_covered": unique_days
        }
    
    def filter_metric_data(self, period: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter weather data to keep only metric values and relevant fields.
        
        Removes:
        - Imperial units (F, mph, inches, etc.)
        - Coded weather fields (cloudsCoded, weatherCoded, etc.)
        - Solar radiation data
        - Icon references
        - Redundant weather descriptions
        
        Keeps:
        - Metric units (C, kph, mm, mb, etc.)
        - Timestamps
        - Essential weather conditions
        
        Args:
            period: Raw weather data period from API
            
        Returns:
            Filtered dictionary with only metric and essential data
        """
        # Fields to keep (metric and essential data only)
        keep_fields = {
            # Time
            "timestamp",
            "dateTimeISO",
            
            # Temperature (Celsius only)
            "tempC",
            "feelslikeC",
            "dewpointC",
            "minTempC",
            "maxTempC",
            
            # Wind (kph and meters/sec only)
            "windSpeedKPH",
            "windSpeedMS",
            "windGustKPH",
            "windGustMS",
            "windDir",
            "windDirENG",
            
            # Pressure (millibars only)
            "pressureMB",
            "spressureMB",  # Sea level pressure
            
            # Precipitation (mm only)
            "precipMM",
            "precipRateMM",
            "snowDepthMM",
            
            # Humidity & clouds
            "humidity",
            "humidityRH",
            "sky",
            
            # Visibility (km only)
            "visibilityKM",
            
            # UV index
            "uvi",
        }
        
        # Create filtered data
        filtered = {}
        
        for key, value in period.items():
            if key in keep_fields:
                filtered[key] = value
        
        return filtered
    
    def get_conditions_filtered(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get filtered hourly weather conditions (metric only, essential fields).
        
        Note: As of the current implementation, all data from get_conditions()
        is already filtered to metric values only when fetched from the API.
        This method is provided for backward compatibility and clarity.
        
        Args:
            from_date: Start datetime for weather data (will be rounded to hour)
            to_date: End datetime for weather data (will be rounded to hour)
            
        Returns:
            List of filtered hourly weather dictionaries, or None on error
        """
        # Data is already filtered when fetched from API and cached
        return self.get_conditions(from_date, to_date)


# Example usage
if __name__ == "__main__":
    # Configure logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example location coordinates
    lat, lon = 53.5833, 22.5000
    
    # Initialize the wrapper (loads existing cache automatically)
    print(f"\n{'='*60}")
    print(f"Initializing WeatherAPIWrapper for ({lat}, {lon})")
    print(f"{'='*60}\n")
    
    api = WeatherAPIWrapper(lat=lat, lon=lon)
    
    # Show cache stats
    stats = api.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Location: {stats['location']}")
    print(f"  Hourly records: {stats['hourly_records']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Days covered: {stats['days_covered']}")
    
    # Get data for a specific time range
    from_date = datetime(2024, 7, 1, 0, 0)
    to_date = datetime(2024, 7, 2, 23, 0)
    
    print(f"\n{'='*60}")
    print(f"Fetching weather data")
    print(f"Time range: {from_date} to {to_date}")
    print(f"{'='*60}\n")
    
    # First call - will check cache, fetch missing data
    conditions = api.get_conditions(from_date, to_date)
    
    if conditions:
        print(f"\nRetrieved {len(conditions)} hourly records")
        print(f"\nFirst record:")
        print(f"  Time: {conditions[0].get('dateTimeISO')}")
        print(f"  Temp: {conditions[0].get('tempC')}°C")
        print(f"  Wind: {conditions[0].get('windSpeedKPH')} km/h from {conditions[0].get('windDirENG')}")
        print(f"  Humidity: {conditions[0].get('humidity')}%")
    
    print(f"\n{'='*60}")
    print("Fetching overlapping range (should be all cached)...")
    print(f"{'='*60}\n")
    
    # Second call with overlapping range - should use cache
    from_date2 = datetime(2024, 7, 1, 12, 0)
    to_date2 = datetime(2024, 7, 2, 12, 0)
    conditions = api.get_conditions(from_date2, to_date2)
    
    print(f"\n{'='*60}")
    print("Getting wind data specifically...")
    print(f"{'='*60}\n")
    
    # Get just wind data
    wind_data = api.get_wind_data(from_date, to_date)
    if wind_data:
        print(f"Wind records: {len(wind_data)}")
        print(f"Sample wind record: {wind_data[0]}")
    
    # Show updated cache stats
    stats = api.get_cache_stats()
    print(f"\nUpdated Cache Statistics:")
    print(f"  Hourly records: {stats['hourly_records']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Days covered: {stats['days_covered']}")