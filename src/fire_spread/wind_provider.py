from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict

try:
    # Dla symulacji (pakiet)
    from .weather import WeatherAPIWrapper
except ImportError:
    # Dla testów (bezpośrednie uruchomienie)
    from weather import WeatherAPIWrapper

logger = logging.getLogger(__name__)

compass_degree = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

class WindProvider:
    wind_fields = [ "timestamp", "windDir", "windSpeedKPH", "windGustKPH", "tempC", "precipMM"]

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
        history_days = 30
    ):
        real_fetch_start = from_date - timedelta(days=history_days)
        logger.info(
            f"Pobieranie bufora danych (symulacja + {history_days} dni historii): od {real_fetch_start} do {to_date}")
        result = self.weatherApiWrapper.get_data(
            real_fetch_start,
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
        target_timestamp = from_date.timestamp()
        self.index = 0
        self.steps_index = 0
        found_start = False
        for i, record in enumerate(self.data):
            # Szukamy pierwszego rekordu, którego czas jest >= czasowi startu symulacji
            if record.get("timestamp") >= target_timestamp:
                self.index = i
                found_start = True
                break

        if found_start:
            logger.info(
                f"Index startowy ustawiony na {self.index}. Symulacja ruszy od {datetime.fromtimestamp(self.data[self.index]['timestamp'])}")
            logger.info(f"Dostępny bufor historyczny w pamięci: {self.index} godzin wstecz.")
        else:
            logger.warning("UWAGA: Nie znaleziono daty startu symulacji w pobranych danych! Ustawiono index na 0.")


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
#funckja do sprawdzenia

    def get_daily_sielianinov_map(self, fire_start_date: datetime, fire_end_date: datetime,
                                  history_days: int = 30) -> Dict[str, float]:
        """
        Generuje mapę współczynników Sielianinowa dla każdego dnia symulacji.

        Zwraca:
            Słownik { 'YYYY-MM-DD': float_value }
        """
        # 1. Musimy pobrać dane od (Start - 30 dni) aż do (Koniec pożaru)
        # Dzięki temu dla każdego dnia pożaru będziemy mieli jego własne 30 dni historii.
        fetch_start = fire_start_date - timedelta(days=history_days)

        logger.info(f"Generowanie mapy suszy (Sielianinow). Analiza danych: {fetch_start} -> {fire_end_date}")

        # Pobieramy dane TYLKO do obliczeń (nie nadpisujemy self.data używanego w symulacji!)
        raw_data = self.weatherApiWrapper.get_data(
            fetch_start,
            fire_end_date,
            ["timestamp", "tempC", "precipMM"]
        )

        if not raw_data:
            logger.warning("Brak danych do obliczenia wskaźnika suszy!")
            return {}

        # Sortujemy chronologicznie
        raw_data.sort(key=lambda x: x['timestamp'])

        htc_map = {}

        # 2. Pętla po każdym dniu trwania pożaru
        current_date = fire_start_date

        # Pętla działa dopóki nie przekroczymy daty końca
        while current_date <= fire_end_date:
            date_str = current_date.strftime('%Y-%m-%d')

            # Wyznaczamy okno czasowe: od (Dziś - 30 dni) do (Dziś koniec dnia)
            window_end_ts = (current_date + timedelta(days=1)).timestamp()
            window_start_ts = (current_date - timedelta(days=history_days)).timestamp()

            # Filtrujemy rekordy pasujące do tego okna
            window = [
                r for r in raw_data
                if window_start_ts <= r['timestamp'] < window_end_ts
            ]

            # 3. Obliczamy Sielianinowa (K) dla tego okna
            if not window:
                htc_map[date_str] = 0.0
            else:
                total_precip = sum(r.get("precipMM", 0) or 0 for r in window)

                # Sumujemy temperatury tylko dodatnie (uproszczenie metody)
                total_temp_sum = sum(r.get("tempC", 0) or 0 for r in window if (r.get("tempC", 0) or 0) > 0)

                # Przeliczenie sumy godzinowej na sumę średnich dobowych
                sum_daily_temps = total_temp_sum / 24.0

                if sum_daily_temps > 0:
                    k_factor = (total_precip * 10) / sum_daily_temps
                    htc_map[date_str] = round(k_factor, 2)
                else:
                    htc_map[date_str] = 0.0  # Jeśli zimno/brak danych, ryzyko "zerowe" lub inna logika

            # Przechodzimy do następnego dnia w kalendarzu
            current_date += timedelta(days=1)

        return htc_map


if __name__ == "__main__":
    # Konfiguracja logowania, żeby widzieć co się dzieje (np. "Fetching data...")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1. Inicjalizacja (Współrzędne pożaru w Szwecji)
    wp = WindProvider(61.62453, 14.69939, steps_per_h=1)

    # 2. Ustawienie dat testowych (np. pierwszy tydzień pożaru)
    fire_start = datetime.fromisoformat("2018-07-05T12:00:00")
    fire_end = fire_start + timedelta(days=7)  # Testujemy dla 7 dni

    print(f"\n=== TEST GENEROWANIA MAPY SIELIANINOWA ===")
    print(f"Zakres symulacji: {fire_start.date()} do {fire_end.date()}")
    print("Pobieranie danych i obliczanie... (może chwilę potrwać przy pierwszym razie)")

    # 3. WYWOŁANIE FUNKCJI
    htc_schedule = wp.get_daily_sielianinov_map(
        fire_start_date=fire_start,
        fire_end_date=fire_end,
        history_days=30
    )

    # 4. Wyświetlenie wyników
    print("\n=== WYNIKI (Harmonogram Suszy) ===")
    if not htc_schedule:
        print("BŁĄD: Otrzymano pusty słownik! Sprawdź klucze API lub połączenie.")
    else:
        # Sortujemy po dacie, żeby ładnie wyświetlić
        for date_str, htc_val in sorted(htc_schedule.items()):
            # Interpretacja wyniku (dla Twojej wygody)
            status = ""
            if htc_val < 0.5:
                status = "(Susza)"
            elif htc_val < 1.0:
                status = "(Niedostateczna wilgoć)"
            else:
                status = "(Wilgotno)"

            print(f"Data: {date_str} | HTC: {htc_val:.2f} {status}")