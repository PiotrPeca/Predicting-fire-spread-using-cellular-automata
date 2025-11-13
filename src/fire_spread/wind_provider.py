from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json
import logging

from .weather import WeatherAPIWrapper

logger = logging.getLogger(__name__)

compass_degree = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

class WindProvider:
    wind_fields = [ "timestamp", "windDir", "windSpeedKPH", "windGustKPH"]

    def __init__(
        self,
        lat: float,
        lon: float,
        steps_per_h: int = 1,
    ):
        self.lat = lat
        self.lon = lon
        self.weatherApiWrapper = WeatherAPIWrapper(lat, lon)
        self.stemps_per_h = steps_per_h
        self.steps_index = 0
        self.index = 0
        self.data = []

    def fetch_data(
        self,
        from_date: datetime,
        to_date: datetime,
    ):
        result = self.weatherApiWrapper.get_data(
            from_date, 
            to_date,
            self.wind_fields,
        )
        self.data = result if result is not None else []
        if not self.data:
            logger.warning(f"No wind data fetched for {from_date} to {to_date}")
        else:
            logger.info(f"Fetched {len(self.data)} wind records")
            self.steps_index = 0
            self.index = 0


    def get_next_wind(self) -> Optional[dict[str, Any]]:
        if not self.data or self.index >= len(self.data):
            return None

        r = self.data[self.index]
        wind_dir_str = str(r.get("windDir", "")).upper().strip()
        wind_dir_degrees = int(compass_degree.get(wind_dir_str, 0))

        if self.steps_index % self.stemps_per_h == 0:
            self.index += 1
        self.steps_index += 1

        return {
            "timestamp": r.get("timestamp"),
            "windDir": wind_dir_degrees,
            "windSpeedKPH": r.get("windSpeedKPH"),
            "windGustKPH": r.get("windGustKPH")
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    wp = WindProvider(61.62453, 14.69939, 3)
    from_date = datetime.fromisoformat("2018-07-05T14:00:00")
    to_date = from_date + timedelta(days=5)
    
    logger.info(f"Fetching data from {from_date} to {to_date}")
    wp.fetch_data(from_date, to_date)
    
    logger.info(f"Retrieved {len(wp.data)} records")
    
    i = 0
    while i < 24 * 3:
        i = i + 1
        w = wp.get_next_wind()
        if w is None:
            continue
        date_str = datetime.fromtimestamp(w.get("timestamp")).isoformat()
        print(date_str, w)