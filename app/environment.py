"""Live & Simulated Environmental Model for UAV Missions.

Provides automatic real-time environmental condition retrieval for the UAV's
current 3D GPS coordinates and altitude.

Architecture:
  - Layer 1 (Live Data): Real-time weather querying via Open-Meteo API with
    asynchronous caching, retrieving surface temperature, pressure, humidity,
    wind speed, wind direction, and weather phenomena.
  - Layer 2 (Simulated Fallback): Deterministic International Standard Atmosphere (ISA)
    thermodynamic model with tropospheric temperature lapse rate and altitude-scaled
    wind fields when offline.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class EnvironmentState:
    """Standardized atmospheric & weather state at UAV's current 3D position."""
    source: str  # "live" or "simulated"
    ambient_c: float
    pressure_hpa: float
    pressure_inhg: float
    air_density_ratio: float  # sigma = rho / rho0
    wind_speed_kt: float
    wind_direction_deg: float
    headwind_kt: float
    crosswind_kt: float
    humidity_pct: float
    visibility_km: float
    precipitation_mm: float
    weather_condition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnvironmentService:
    """
    Environmental conditions provider for AeroPulse-X.
    
    Automatically resolves real-time live meteorological data or falls back seamlessly
    to the deterministic ISA thermodynamic model without interrupting simulation.
    """

    def __init__(self, enable_live: bool = True, cache_ttl_seconds: float = 300.0):
        self.enable_live = enable_live
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, dict]] = {}

    def _get_cache_key(self, lat: float, lon: float) -> str:
        # 0.1 degree resolution (~11 km grid) for efficient local weather caching
        return f"{round(lat, 1)}_{round(lon, 1)}"

    def _fetch_live_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Queries Open-Meteo public free meteorological API with timeout."""
        if not self.enable_live:
            return None

        cache_key = self._get_cache_key(lat, lon)
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return data

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat:.4f}&longitude={lon:.4f}&"
                f"current=temperature_2m,relative_humidity_2m,surface_pressure,"
                f"wind_speed_10m,wind_direction_10m,precipitation,weather_code&"
                f"forecast_days=1"
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AeroPulse-X-DigitalTwin/2.0 (SIH26054 Research)"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    current = data.get("current", {})
                    self._cache[cache_key] = (now, current)
                    return current
        except Exception:
            # Graceful fallback to deterministic ISA model without crashing
            return None
        return None

    @staticmethod
    def _weather_code_to_condition(code: int) -> str:
        if code == 0:
            return "Clear Sky"
        if code in {1, 2}:
            return "Partly Cloudy"
        if code == 3:
            return "Overcast"
        if code in {45, 48}:
            return "Fog / Mist"
        if code in {51, 53, 55, 61, 63}:
            return "Light Rain"
        if code in {65, 80, 81, 82}:
            return "Heavy Rain"
        if code in {71, 73, 75}:
            return "Snow Showers"
        if code in {95, 96, 99}:
            return "Thunderstorm / High Turbulence"
        return "Fair Flight Conditions"

    def get_environment(
        self,
        latitude: float,
        longitude: float,
        altitude_ft: float,
        heading_deg: float = 0.0,
        manual_override: Optional[dict] = None
    ) -> EnvironmentState:
        """
        Calculates or retrieves the exact atmospheric & weather state at UAV location.
        """
        h = max(0.0, float(altitude_ft))
        heading = float(heading_deg) % 360.0

        # Check manual overrides first if explicitly activated for benchmark testing
        if manual_override and manual_override.get("manual_override") is True:
            t_amb = float(manual_override.get("ambient_c", 35.0))
            sigma = math.exp(-h / 30000.0)
            press_inhg = 29.92 * (sigma ** 1.2)
            press_hpa = press_inhg * 33.8639
            return EnvironmentState(
                source="manual_override",
                ambient_c=round(t_amb, 1),
                pressure_hpa=round(press_hpa, 1),
                pressure_inhg=round(press_inhg, 2),
                air_density_ratio=round(sigma, 4),
                wind_speed_kt=10.0,
                wind_direction_deg=270.0,
                headwind_kt=round(10.0 * math.cos(math.radians(270.0 - heading)), 1),
                crosswind_kt=round(10.0 * math.sin(math.radians(270.0 - heading)), 1),
                humidity_pct=45.0,
                visibility_km=10.0,
                precipitation_mm=0.0,
                weather_condition="Manual Test Override",
            )

        live_data = self._fetch_live_weather(latitude, longitude)

        if live_data is not None:
            source = "live"
            t_surface = float(live_data.get("temperature_2m", 28.0))
            p_surface_hpa = float(live_data.get("surface_pressure", 1013.25))
            wind_kmh = float(live_data.get("wind_speed_10m", 15.0))
            wind_kt = wind_kmh * 0.539957
            wind_dir = float(live_data.get("wind_direction_10m", 240.0))
            humidity = float(live_data.get("relative_humidity_2m", 50.0))
            precip = float(live_data.get("precipitation", 0.0))
            weather_code = int(live_data.get("weather_code", 0))
            condition = self._weather_code_to_condition(weather_code)
            visibility_km = 10.0 if weather_code < 40 else 5.0
        else:
            source = "simulated"
            # Deterministic geographic & climate profile
            lat_factor = math.sin(latitude * 0.1)
            lon_factor = math.cos(longitude * 0.1)
            t_surface = 30.0 + 3.0 * lat_factor + 2.0 * lon_factor
            p_surface_hpa = 1013.25 + 5.0 * lon_factor
            wind_kt = max(4.0, 10.0 + 4.0 * lat_factor + (h / 6000.0))
            wind_dir = (240.0 + 35.0 * lon_factor) % 360.0
            humidity = max(20.0, min(85.0, 52.0 + 15.0 * lat_factor))
            precip = 0.0
            condition = "Simulated ISA Standard Atmosphere"
            visibility_km = 12.0

        # Apply tropospheric temperature lapse rate for actual UAV altitude
        # Standard lapse rate: ~1.98 °C per 1,000 ft (6.5 °C / 1000m)
        lapse_c = (h / 1000.0) * 1.98
        t_amb = max(-55.0, min(50.0, t_surface - lapse_c))

        # Barometric formula for pressure at altitude
        press_hpa = p_surface_hpa * ((1.0 - 6.87558e-6 * h) ** 5.2559) if h < 36000 else p_surface_hpa * 0.22
        press_hpa = max(150.0, min(1080.0, press_hpa))
        press_inhg = press_hpa * 0.02953

        # Air density ratio sigma: rho(h) / rho_0
        temp_k = t_amb + 273.15
        rho_amb = (press_hpa * 100.0) / (287.058 * max(100.0, temp_k))
        density_ratio = max(0.55, min(1.15, rho_amb / 1.225))

        # Aerodynamic wind vector components relative to UAV heading
        # wind_direction is the direction FROM which wind blows
        delta_rad = math.radians(wind_dir - heading)
        headwind_kt = wind_kt * math.cos(delta_rad)
        crosswind_kt = wind_kt * math.sin(delta_rad)

        return EnvironmentState(
            source=source,
            ambient_c=round(t_amb, 1),
            pressure_hpa=round(press_hpa, 1),
            pressure_inhg=round(press_inhg, 2),
            air_density_ratio=round(density_ratio, 4),
            wind_speed_kt=round(wind_kt, 1),
            wind_direction_deg=round(wind_dir, 1),
            headwind_kt=round(headwind_kt, 1),
            crosswind_kt=round(crosswind_kt, 1),
            humidity_pct=round(humidity, 1),
            visibility_km=round(visibility_km, 1),
            precipitation_mm=round(precip, 2),
            weather_condition=condition,
        )


_DEFAULT_ENV_SERVICE = EnvironmentService(enable_live=True)
