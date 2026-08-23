"""UAV Navigation, Waypoint Engine, and GPS Telemetry Subsystem.

This module provides the navigation abstractions, waypoint routing, and GPS/INS
telemetry generation for the AeroPulse-X MALE-UAV Digital Twin.

It defines an extensible GPSSource interface that enables seamless switching
between the simulated deterministic mission generator and real hardware GPS/INS
NMEA/MavLink/CAN streams.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MissionWaypoint:
    """Represents a planned 3D waypoint in the UAV flight plan."""
    id: str
    name: str
    latitude: float
    longitude: float
    altitude_ft: float
    type: str  # BASE, CLIMB, CRUISE, ISR, PATROL, RETURN, APPROACH, RECOVERY
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UAVPosition:
    """
    Standardized UAV navigation telemetry packet.
    
    Compatible with simulated GPS/INS and hardware navigation units (NMEA / ArduPilot / MavLink / CAN).
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
    eta_next_wp_min: float
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default SIH Research Prototype MALE-UAV Mission Waypoints
# (Safe demonstration coordinates - non-operational civilian test corridor)
DEFAULT_MISSION_WAYPOINTS: List[MissionWaypoint] = [
    MissionWaypoint(
        id="WP00",
        name="Airbase Bravo (Home)",
        latitude=28.4520,
        longitude=77.0250,
        altitude_ft=750.0,
        type="BASE",
        description="GCS Ground Operations & Runway Line"
    ),
    MissionWaypoint(
        id="WP01",
        name="Departure Transition",
        latitude=28.5480,
        longitude=77.1820,
        altitude_ft=6500.0,
        type="CLIMB",
        description="Climb Corridor & Airspace Transition"
    ),
    MissionWaypoint(
        id="WP02",
        name="Transit Gate Alpha",
        latitude=28.6920,
        longitude=77.3450,
        altitude_ft=14000.0,
        type="CRUISE",
        description="High-Altitude Ingress Route"
    ),
    MissionWaypoint(
        id="WP03",
        name="ISR Recon Zone 1",
        latitude=28.8450,
        longitude=77.5620,
        altitude_ft=18500.0,
        type="ISR",
        description="Electro-Optical / IR Sensor Sweep Area"
    ),
    MissionWaypoint(
        id="WP04",
        name="ISR Target Bravo",
        latitude=28.9620,
        longitude=77.7850,
        altitude_ft=20000.0,
        type="ISR",
        description="Communication Relay & High-Res Imagery"
    ),
    MissionWaypoint(
        id="WP05",
        name="Border Patrol Orbit",
        latitude=28.8150,
        longitude=77.9420,
        altitude_ft=16500.0,
        type="PATROL",
        description="Perimeter Loiter & Sector Surveillance"
    ),
    MissionWaypoint(
        id="WP06",
        name="Return Corridor South",
        latitude=28.6250,
        longitude=77.6100,
        altitude_ft=11000.0,
        type="RETURN",
        description="Descent Gate & Egress Transit"
    ),
    MissionWaypoint(
        id="WP07",
        name="Final Approach Fix",
        latitude=28.4980,
        longitude=77.2200,
        altitude_ft=3200.0,
        type="APPROACH",
        description="Glide Slope Alignment & Pre-Landing Check"
    ),
    MissionWaypoint(
        id="WP08",
        name="Airbase Bravo (Recovery)",
        latitude=28.4520,
        longitude=77.0250,
        altitude_ft=750.0,
        type="RECOVERY",
        description="Arresting Gear & Safe Mission Recovery"
    ),
]


class GPSSource(abc.ABC):
    """Abstract interface for GPS/INS positioning sources."""

    @abc.abstractmethod
    def get_position(self, progress_ratio: float, mission_context: dict) -> UAVPosition:
        """Returns the current position based on mission state."""
        pass


class SimulatedGPSSource(GPSSource):
    """
    Deterministic geodesic UAV mission flight path generator.
    
    Interpolates smoothly across 3D waypoints, calculates dynamic heading,
    ground speed based on engine propulsion state, and estimates waypoint ETAs.
    """

    def __init__(self, waypoints: Optional[List[MissionWaypoint]] = None):
        self.waypoints = waypoints or DEFAULT_MISSION_WAYPOINTS
        self._cumulative_distances_km = self._calc_cumulative_distances()
        self.total_distance_km = self._cumulative_distances_km[-1]

    def _calc_cumulative_distances(self) -> List[float]:
        distances = [0.0]
        for i in range(len(self.waypoints) - 1):
            w1 = self.waypoints[i]
            w2 = self.waypoints[i + 1]
            dist = self.haversine_distance(w1.latitude, w1.longitude, w2.latitude, w2.longitude)
            distances.append(distances[-1] + dist)
        return distances

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

    def get_position(self, progress_ratio: float, mission_context: Optional[dict] = None) -> UAVPosition:
        context = mission_context or {}
        ratio = max(0.0, min(1.0, float(progress_ratio)))
        
        target_dist = ratio * self.total_distance_km

        # Find active leg segment
        num_segments = len(self.waypoints) - 1
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

        # Smooth cubic hermite interpolation for realistic non-linear turns
        s_curve = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        curr_lat = w_start.latitude + (w_end.latitude - w_start.latitude) * u
        curr_lon = w_start.longitude + (w_end.longitude - w_start.longitude) * u

        # Altitude from mission context or waypoint profile
        if "altitude_ft" in context:
            curr_alt = float(context["altitude_ft"])
        else:
            curr_alt = w_start.altitude_ft + (w_end.altitude_ft - w_start.altitude_ft) * s_curve

        heading = self.calculate_heading(w_start.latitude, w_start.longitude, w_end.latitude, w_end.longitude)

        # Ground speed derived from throttle, RPM, and altitude
        throttle = float(context.get("throttle", 0.60))
        base_speed = 95.0 + 75.0 * throttle + (curr_alt / 1000.0) * 1.8
        ground_speed_kt = round(max(60.0, min(190.0, base_speed)), 1)

        # Distance to next waypoint & ETA
        dist_to_next = self.haversine_distance(curr_lat, curr_lon, w_end.latitude, w_end.longitude)
        speed_kmh = ground_speed_kt * 1.852
        eta_min = round((dist_to_next / max(10.0, speed_kmh)) * 60.0, 1)

        phase = str(context.get("mission_phase", w_end.type))

        return UAVPosition(
            latitude=round(curr_lat, 6),
            longitude=round(curr_lon, 6),
            altitude_ft=round(curr_alt, 1),
            ground_speed_kt=ground_speed_kt,
            heading_deg=round(heading, 1),
            vertical_speed_fpm=round((w_end.altitude_ft - w_start.altitude_ft) * 0.1, 1),
            mission_phase=phase,
            mission_progress=round(ratio, 4),
            active_waypoint_id=w_end.id,
            active_waypoint_name=w_end.name,
            distance_to_next_wp_km=round(dist_to_next, 2),
            eta_next_wp_min=eta_min,
        )


class HardwareGPSInterface(GPSSource):
    """
    Interface for real physical UAV GPS/INS telemetry receiver (MavLink/CAN/Serial).
    Acts as placeholder for future flight-test integration.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._last_position: Optional[UAVPosition] = None

    def get_position(self, progress_ratio: float, mission_context: dict) -> UAVPosition:
        if self._last_position is not None:
            return self._last_position
        # Fallback to simulated source if hardware connection is offline
        sim = SimulatedGPSSource()
        return sim.get_position(progress_ratio, mission_context)
