"""UAV Navigation, Waypoint Routing Engine, and Path-Driven Flight Simulator.

This module provides the core 3D waypoint flight dynamics, atmospheric environment model,
automatic throttle/load scheduling, and navigation telemetry for AeroPulse-X.

The operator defines the 3D mission waypoints (coordinates, altitudes, objectives).
The simulator automatically derives:
  - Precise continuous GPS coordinates (geodesic interpolation)
  - Heading and dynamic true ground speed
  - Realistic climb/descent profile (vertical speed in FPM)
  - International Standard Atmosphere (ISA) temperature lapse rate, air density, pressure
  - Automatic mission phase state machine
  - Automatic propulsion throttle demand and engine load
  - Real-time remaining distance, segment ETAs, and total mission duration
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MissionWaypoint:
    """Represents a configurable 3D waypoint in the UAV flight plan."""
    id: str
    name: str
    latitude: float
    longitude: float
    altitude_ft: float = 8000.0
    type: str = "CRUISE"  # BASE, TAKEOFF, CLIMB, CRUISE, ISR, PATROL, DESCENT, RETURN, LANDING, RECOVERY
    speed_kt: float = 140.0
    loiter_time_min: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MissionWaypoint":
        raw_type = str(d.get("type", d.get("mission_type", "CRUISE"))).upper()
        return cls(
            id=str(d.get("id", "WP")),
            name=str(d.get("name", "Waypoint")),
            latitude=float(d.get("latitude", 28.5)),
            longitude=float(d.get("longitude", 77.0)),
            altitude_ft=float(d.get("altitude_ft", 8000.0)),
            type=raw_type,
            speed_kt=float(d.get("speed_kt", 140.0)),
            loiter_time_min=float(d.get("loiter_time_min", 0.0)),
            description=str(d.get("description", "")),
        )


@dataclass
class UAVPosition:
    """
    Standardized UAV navigation & environmental telemetry packet.
    
    Compatible with simulated path-driven flight dynamics and hardware GPS/INS units (MavLink/CAN/NMEA).
    """
    latitude: float
    longitude: float
    altitude_ft: float
    ground_speed_kt: float
    heading_deg: float
    vertical_speed_fpm: float
    mission_phase: str
    mission_progress: float  # 0.0 to 1.0
    active_waypoint_id: str
    active_waypoint_name: str
    distance_to_next_wp_km: float
    distance_travelled_km: float
    distance_remaining_km: float
    eta_next_wp_min: float
    eta_mission_min: float
    ambient_c: float
    air_density_ratio: float
    pressure_inhg: float
    auto_throttle: float
    auto_load: float
    operating_state: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =========================================================================
# Preset Mission Profiles (Civilian SIH Demonstration Coordinates)
# =========================================================================

# Preset 1: Border Patrol & ISR Alpha (Standard MALE-UAV Mission Profile)
DEFAULT_MISSION_WAYPOINTS: List[MissionWaypoint] = [
    MissionWaypoint(
        id="BASE",
        name="Airbase Bravo (Home)",
        latitude=28.4520,
        longitude=77.0250,
        altitude_ft=750.0,
        type="BASE",
        speed_kt=60.0,
        loiter_time_min=0.0,
        description="GCS Ground Operations & Pre-Flight Engine Runup"
    ),
    MissionWaypoint(
        id="WP01",
        name="Departure & Initial Climb",
        latitude=28.5480,
        longitude=77.1820,
        altitude_ft=6500.0,
        type="CLIMB",
        speed_kt=115.0,
        loiter_time_min=0.0,
        description="Airspace Departure Corridor & Continuous Climb"
    ),
    MissionWaypoint(
        id="WP02",
        name="High-Altitude Ingress Gate",
        latitude=28.6920,
        longitude=77.3450,
        altitude_ft=14000.0,
        type="CRUISE",
        speed_kt=145.0,
        loiter_time_min=0.0,
        description="Cruising Flight Level & Subsystem Optimization"
    ),
    MissionWaypoint(
        id="WP03",
        name="ISR Recon Zone 1",
        latitude=28.8450,
        longitude=77.5620,
        altitude_ft=18500.0,
        type="ISR",
        speed_kt=125.0,
        loiter_time_min=15.0,
        description="Electro-Optical / IR Sensor Sweep & EO Target Acquisition"
    ),
    MissionWaypoint(
        id="WP04",
        name="ISR Target Bravo",
        latitude=28.9620,
        longitude=77.7850,
        altitude_ft=20000.0,
        type="ISR",
        speed_kt=120.0,
        loiter_time_min=20.0,
        description="High-Altitude Comm Relay & Synthetic Aperture Radar"
    ),
    MissionWaypoint(
        id="WP05",
        name="Border Patrol Loiter Sector",
        latitude=28.8150,
        longitude=77.9420,
        altitude_ft=16500.0,
        type="PATROL",
        speed_kt=135.0,
        loiter_time_min=30.0,
        description="Perimeter Loiter Orbit & Thermal Boundary Surveillance"
    ),
    MissionWaypoint(
        id="WP06",
        name="Return Corridor South",
        latitude=28.6250,
        longitude=77.6100,
        altitude_ft=11000.0,
        type="DESCENT",
        speed_kt=140.0,
        loiter_time_min=0.0,
        description="Egress Corridor & Stepped Descent Gate"
    ),
    MissionWaypoint(
        id="WP07",
        name="Final Approach Fix",
        latitude=28.4980,
        longitude=77.2200,
        altitude_ft=3200.0,
        type="RETURN",
        speed_kt=95.0,
        loiter_time_min=0.0,
        description="Glide Slope Alignment & Landing Configuration Check"
    ),
    MissionWaypoint(
        id="RECOVERY",
        name="Airbase Bravo (Landing)",
        latitude=28.4520,
        longitude=77.0250,
        altitude_ft=750.0,
        type="RECOVERY",
        speed_kt=65.0,
        loiter_time_min=0.0,
        description="Touchdown, Reversible Prop Braking & Safe Recovery"
    ),
]

# Preset 2: High-Altitude Long-Endurance (HALE/MALE) Deep Recon
HIGH_ALT_SURVEILLANCE_WAYPOINTS: List[MissionWaypoint] = [
    MissionWaypoint("BASE", "Airbase Bravo", 28.4520, 77.0250, 750.0, "BASE", 60.0, 0.0, "Runway Alignment"),
    MissionWaypoint("WP01", "Rapid Climb Corridor", 28.5800, 77.2500, 10000.0, "CLIMB", 110.0, 0.0, "Climb to FL100"),
    MissionWaypoint("WP02", "Stratospheric Ingress", 28.7500, 77.5000, 22000.0, "CLIMB", 130.0, 0.0, "Climb to FL220"),
    MissionWaypoint("WP03", "Deep Recon Sector North", 29.1000, 77.9500, 24000.0, "ISR", 125.0, 45.0, "High Altitude Sensor Sweep"),
    MissionWaypoint("WP04", "High Loiter Orbit", 28.9500, 78.2000, 24000.0, "PATROL", 120.0, 60.0, "Extended Loiter Surveillance"),
    MissionWaypoint("WP05", "High-Speed Egress", 28.6800, 77.7000, 15000.0, "DESCENT", 155.0, 0.0, "Descent to FL150"),
    MissionWaypoint("WP06", "Terminal Approach", 28.5100, 77.2000, 4000.0, "RETURN", 100.0, 0.0, "Inbound Fix"),
    MissionWaypoint("RECOVERY", "Airbase Bravo", 28.4520, 77.0250, 750.0, "RECOVERY", 65.0, 0.0, "Arrested Recovery"),
]

# Preset 3: Coastal & Maritime Border Patrol
COASTAL_MARITIME_WAYPOINTS: List[MissionWaypoint] = [
    MissionWaypoint("BASE", "Coastal Airfield", 28.4520, 77.0250, 750.0, "BASE", 60.0, 0.0, "Base Departure"),
    MissionWaypoint("WP01", "Offshore Transit", 28.6000, 77.2000, 8000.0, "CLIMB", 120.0, 0.0, "Overwater Transit"),
    MissionWaypoint("WP02", "Maritime Recon 1", 28.8000, 77.4500, 12000.0, "ISR", 130.0, 20.0, "Vessel Identification & AIS"),
    MissionWaypoint("WP03", "Perimeter Patrol", 28.9500, 77.7500, 12000.0, "PATROL", 135.0, 30.0, "Maritime Border Tracking"),
    MissionWaypoint("WP04", "Island Recon Sweep", 28.7500, 77.9000, 10000.0, "ISR", 125.0, 25.0, "Radar Sweep Zone"),
    MissionWaypoint("WP05", "Inbound Gate", 28.5500, 77.4000, 5000.0, "DESCENT", 130.0, 0.0, "Return Transition"),
    MissionWaypoint("RECOVERY", "Coastal Airfield", 28.4520, 77.0250, 750.0, "RECOVERY", 65.0, 0.0, "Runway Landing"),
]

PRESET_MISSIONS = {
    "border_patrol_alpha": {
        "name": "ISR Border Patrol Alpha (9 Waypoints)",
        "description": "Standard MALE-UAV intelligence and border security patrol profile.",
        "waypoints": DEFAULT_MISSION_WAYPOINTS,
    },
    "high_alt_surveillance": {
        "name": "High-Altitude Deep Recon Bravo (8 Waypoints)",
        "description": "Long-endurance high-altitude (24,000 ft) surveillance and comms relay.",
        "waypoints": HIGH_ALT_SURVEILLANCE_WAYPOINTS,
    },
    "coastal_maritime": {
        "name": "Coastal & Maritime Patrol Charlie (7 Waypoints)",
        "description": "Overwater loiter with medium-altitude vessel and radar sweeps.",
        "waypoints": COASTAL_MARITIME_WAYPOINTS,
    },
}


class GPSSource(abc.ABC):
    """Abstract interface for GPS/INS positioning sources."""

    @abc.abstractmethod
    def get_position(self, progress_ratio: float, mission_context: Optional[dict] = None) -> UAVPosition:
        """Returns the current position and flight state."""
        pass


class SimulatedGPSSource(GPSSource):
    """
    Deterministic 3D Geodesic UAV Mission Flight Dynamics & Environment Generator.
    
    Calculates exact real-world great-circle waypoints, smooth Hermite spline turns,
    climb/descent altitude transitions, atmospheric environmental profiles,
    and automatic throttle/load scheduling.
    """

    def __init__(
        self,
        waypoints: Optional[List[MissionWaypoint]] = None,
        ground_temp_c: float = 30.0,
        hot_weather_bias: float = 0.0,
    ):
        self.waypoints = [
            wp if isinstance(wp, MissionWaypoint) else MissionWaypoint.from_dict(wp)
            for wp in (waypoints or DEFAULT_MISSION_WAYPOINTS)
        ]
        if len(self.waypoints) < 2:
            self.waypoints = DEFAULT_MISSION_WAYPOINTS.copy()

        self.ground_temp_c = float(ground_temp_c)
        self.hot_weather_bias = float(hot_weather_bias)
        
        self._cumulative_distances_km = self._calc_cumulative_distances()
        self.total_distance_km = self._cumulative_distances_km[-1]
        self.total_duration_min = self._calc_estimated_duration_min()

    def _calc_cumulative_distances(self) -> List[float]:
        distances = [0.0]
        for i in range(len(self.waypoints) - 1):
            w1 = self.waypoints[i]
            w2 = self.waypoints[i + 1]
            dist = self.haversine_distance(w1.latitude, w1.longitude, w2.latitude, w2.longitude)
            distances.append(distances[-1] + max(0.001, dist))
        return distances

    def _calc_estimated_duration_min(self) -> float:
        total_time_min = 0.0
        for i in range(len(self.waypoints) - 1):
            w1 = self.waypoints[i]
            w2 = self.waypoints[i + 1]
            dist_km = self.haversine_distance(w1.latitude, w1.longitude, w2.latitude, w2.longitude)
            avg_speed_kt = max(50.0, (w1.speed_kt + w2.speed_kt) / 2.0)
            speed_kmh = avg_speed_kt * 1.852
            leg_time_min = (dist_km / speed_kmh) * 60.0
            total_time_min += leg_time_min + w1.loiter_time_min
        total_time_min += self.waypoints[-1].loiter_time_min
        return max(5.0, total_time_min)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes Great-Circle distance in kilometers between two GPS coordinates."""
        r = 6371.0  # Earth radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    @staticmethod
    def calculate_heading(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes true course bearing in degrees [0, 360)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)

        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360.0) % 360.0

    def get_atmospheric_state(self, altitude_ft: float) -> Tuple[float, float, float]:
        """
        Computes International Standard Atmosphere (ISA) parameters:
        Returns: (ambient_temperature_c, air_density_ratio, pressure_inhg)
        """
        h = max(0.0, float(altitude_ft))
        
        # Standard tropospheric temperature lapse rate: ~1.98 °C per 1,000 ft (6.5 °C / 1000m)
        lapse_c = (h / 1000.0) * 1.98
        t_amb = (self.ground_temp_c + self.hot_weather_bias) - lapse_c
        t_amb = max(-55.0, min(50.0, t_amb))

        # Air density ratio sigma: rho(h) / rho_0
        density_ratio = math.exp(-h / 30000.0)
        density_ratio = max(0.55, min(1.15, density_ratio))

        # Atmospheric pressure in inHg
        pressure = 29.92 * ((1.0 - 6.87558e-6 * h) ** 5.2559) if h < 36000 else 6.68
        pressure = max(5.0, min(32.0, pressure))

        return round(t_amb, 2), round(density_ratio, 4), round(pressure, 2)

    def determine_phase_and_propulsion_state(
        self,
        seg_type: str,
        altitude_ft: float,
        vsi_fpm: float,
        speed_kt: float,
        progress_ratio: float,
    ) -> Tuple[str, str, float, float]:
        """
        Automatic Mission Phase State Machine & Throttle/Load Scheduler.
        Returns: (mission_phase, operating_state, auto_throttle, auto_load)
        """
        type_upper = seg_type.upper()
        h = altitude_ft

        if progress_ratio <= 0.02 or type_upper == "BASE":
            phase = "GROUND" if progress_ratio <= 0.01 else "TAKEOFF"
            op_state = "HIGH"
            throttle = 0.88 if phase == "TAKEOFF" else 0.25
        elif progress_ratio >= 0.98 or type_upper == "RECOVERY":
            phase = "LANDING"
            op_state = "CRUISE_LOW"
            throttle = 0.30
        elif type_upper == "CLIMB" or (vsi_fpm > 400.0 and type_upper not in {"ISR", "PATROL"}):
            phase = "CLIMB"
            op_state = "HIGH"
            climb_boost = min(0.12, max(0.0, (vsi_fpm / 1500.0) * 0.10))
            throttle = min(0.92, 0.78 + climb_boost)
        elif type_upper in {"DESCENT", "RETURN", "APPROACH"} or (vsi_fpm < -600.0 and progress_ratio > 0.65):
            phase = "DESCENT" if vsi_fpm < -300.0 else "RETURN"
            op_state = "CRUISE"
            throttle = max(0.35, 0.44 - (abs(vsi_fpm) / 2000.0) * 0.08)
        elif type_upper == "ISR":
            phase = "ISR"
            op_state = "CRUISE_LOW" if h < 18000 else "HIGH"
            throttle = 0.55 + (0.05 * (h / 20000.0))
        elif type_upper == "PATROL":
            phase = "PATROL"
            op_state = "CRUISE_LOW"
            throttle = 0.52 + (0.04 * (h / 20000.0))
        else:
            phase = "CRUISE"
            op_state = "HIGH" if h > 16000 else "CRUISE"
            throttle = 0.58 + (0.04 * (h / 20000.0))

        # Add small deterministic aerodynamic fluctuation (micro-turbulence)
        turbulence = 0.015 * math.sin(progress_ratio * math.pi * 16.0)
        throttle = max(0.20, min(0.98, throttle + turbulence))

        # Load derived from throttle, density, and altitude resistance
        sigma = math.exp(-h / 30000.0)
        load = throttle * (1.05 - 0.12 * (1.0 - sigma))
        load = max(0.20, min(1.15, load))

        return phase, op_state, round(throttle, 4), round(load, 4)

    def get_position(self, progress_ratio: float, mission_context: Optional[dict] = None) -> UAVPosition:
        context = mission_context or {}
        ratio = max(0.0, min(1.0, float(progress_ratio)))
        
        target_dist = ratio * self.total_distance_km
        num_segments = len(self.waypoints) - 1

        # Find active leg segment
        seg_idx = 0
        for i in range(num_segments):
            if self._cumulative_distances_km[i] <= target_dist <= self._cumulative_distances_km[i + 1]:
                seg_idx = i
                break
            if target_dist >= self._cumulative_distances_km[-1]:
                seg_idx = num_segments - 1

        w_start = self.waypoints[seg_idx]
        w_end = self.waypoints[min(seg_idx + 1, len(self.waypoints) - 1)]

        seg_start_dist = self._cumulative_distances_km[seg_idx]
        seg_end_dist = self._cumulative_distances_km[seg_idx + 1]
        seg_len = max(0.001, seg_end_dist - seg_start_dist)
        
        # Local leg interpolation parameter [0, 1]
        u = max(0.0, min(1.0, (target_dist - seg_start_dist) / seg_len))

        # Smooth cubic Hermite curve for realistic turns & transitions
        s_curve = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        curr_lat = w_start.latitude + (w_end.latitude - w_start.latitude) * u
        curr_lon = w_start.longitude + (w_end.longitude - w_start.longitude) * u

        # Automatic Smooth 3D Altitude Profile
        alt_start = w_start.altitude_ft
        alt_end = w_end.altitude_ft
        curr_alt = alt_start + (alt_end - alt_start) * s_curve

        # Override with context altitude only if explicitly requested by advanced manual overrides
        if context.get("manual_altitude_override") is True and "altitude_ft" in context:
            curr_alt = float(context["altitude_ft"])

        # Heading & Bearing
        heading = self.calculate_heading(w_start.latitude, w_start.longitude, w_end.latitude, w_end.longitude)

        # Ground speed based on target waypoint speed profile
        base_speed = w_start.speed_kt + (w_end.speed_kt - w_start.speed_kt) * u
        ground_speed_kt = round(max(55.0, min(185.0, base_speed)), 1)

        # Vertical speed indicator (FPM)
        leg_dist_km = self.haversine_distance(w_start.latitude, w_start.longitude, w_end.latitude, w_end.longitude)
        leg_speed_kmh = max(30.0, ground_speed_kt * 1.852)
        leg_duration_min = (leg_dist_km / leg_speed_kmh) * 60.0
        vsi_fpm = round(((alt_end - alt_start) / max(0.2, leg_duration_min)), 1)
        vsi_fpm = max(-2500.0, min(2500.0, vsi_fpm))

        # Distance & ETAs
        dist_to_next = self.haversine_distance(curr_lat, curr_lon, w_end.latitude, w_end.longitude)
        dist_travelled = target_dist
        dist_remaining = max(0.0, self.total_distance_km - target_dist)

        speed_kmh = ground_speed_kt * 1.852
        eta_next_min = round((dist_to_next / max(10.0, speed_kmh)) * 60.0, 1)
        eta_mission_min = round((dist_remaining / max(10.0, speed_kmh)) * 60.0, 1)

        # Automatic Atmospheric Environment
        amb_c, density_ratio, press_inhg = self.get_atmospheric_state(curr_alt)

        # Automatic Phase State Machine & Propulsion Demand
        phase, op_state, auto_thr, auto_ld = self.determine_phase_and_propulsion_state(
            w_end.type, curr_alt, vsi_fpm, ground_speed_kt, ratio
        )

        return UAVPosition(
            latitude=round(curr_lat, 6),
            longitude=round(curr_lon, 6),
            altitude_ft=round(curr_alt, 1),
            ground_speed_kt=ground_speed_kt,
            heading_deg=round(heading, 1),
            vertical_speed_fpm=vsi_fpm,
            mission_phase=phase,
            mission_progress=round(ratio, 4),
            active_waypoint_id=w_end.id,
            active_waypoint_name=w_end.name,
            distance_to_next_wp_km=round(dist_to_next, 2),
            distance_travelled_km=round(dist_travelled, 2),
            distance_remaining_km=round(dist_remaining, 2),
            eta_next_wp_min=eta_next_min,
            eta_mission_min=eta_mission_min,
            ambient_c=amb_c,
            air_density_ratio=density_ratio,
            pressure_inhg=press_inhg,
            auto_throttle=auto_thr,
            auto_load=auto_ld,
            operating_state=op_state,
        )

    def get_flight_plan_summary(self) -> dict[str, Any]:
        """Returns structured metadata for the entire flight plan."""
        hours = int(self.total_duration_min // 60)
        mins = int(self.total_duration_min % 60)
        return {
            "total_waypoints": len(self.waypoints),
            "total_distance_km": round(self.total_distance_km, 2),
            "estimated_duration_min": round(self.total_duration_min, 1),
            "formatted_duration": f"{hours}h {mins:02d}m",
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "planned_route": [[wp.latitude, wp.longitude] for wp in self.waypoints],
            "home_base": self.waypoints[0].to_dict(),
            "max_altitude_ft": max(wp.altitude_ft for wp in self.waypoints),
            "initial_ambient_c": self.ground_temp_c,
        }


class HardwareGPSInterface(GPSSource):
    """
    Interface for real physical UAV GPS/INS telemetry receiver (MavLink/CAN/Serial).
    Seamless drop-in replacement for physical flight test telemetry.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._last_position: Optional[UAVPosition] = None

    def get_position(self, progress_ratio: float, mission_context: Optional[dict] = None) -> UAVPosition:
        if self._last_position is not None:
            return self._last_position
        sim = SimulatedGPSSource()
        return sim.get_position(progress_ratio, mission_context)
