from pathlib import Path
import json

compass_degree = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

class WindProvider:
    def __init__(self,data_folder: str = "data", filename: str = "wind_data.json"):
        base_dir = Path(__file__).resolve().parents[2]
        self.file_path = base_dir / data_folder / filename

        if not self.file_path.exists():
            raise FileNotFoundError(f"Could not find wind data file: {self.file_path}")

        with self.file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.rows = raw_data.get("hours", []) if isinstance(raw_data, dict) else raw_data
        self.index = 0
    def get_next_wind(self) -> dict:
        if self.index >= len(self.rows):
            return {"wind_degree": int(round(0)), "wind_kph": 0}
        r = self.rows[self.index]
        self.index += 1
        dir_str = str(r.get("windDir", "")).upper().strip()
        wind_kph = float(r.get("windSpeedKPH", 0.0))
        wind_degree = compass_degree.get(dir_str, 0)
        return {"wind_degree": int(wind_degree), "wind_kph": wind_kph}

if __name__ == "__main__":
    wp = WindProvider()
    i = 0
    while i < 100:
        i = i + 1
        w = wp.get_next_wind()
        if w is None:
            break
        print(w)