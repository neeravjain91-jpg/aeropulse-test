"""Unit and integration tests for UAV Navigation, Waypoints, and Simulated GPS."""
import pytest
from app.navigation import (
    DEFAULT_MISSION_WAYPOINTS,
    MissionWaypoint,
    SimulatedGPSSource,
    HardwareGPSInterface,
    UAVPosition,
)
from app.replay import run_replay
from app.inference import AeroTwinAI


def test_default_waypoints_are_valid_and_non_empty():
    assert len(DEFAULT_MISSION_WAYPOINTS) >= 6
    for wp in DEFAULT_MISSION_WAYPOINTS:
        assert isinstance(wp, MissionWaypoint)
        assert -90.0 <= wp.latitude <= 90.0
        assert -180.0 <= wp.longitude <= 180.0
        assert wp.altitude_ft >= 0.0
        assert len(wp.id) > 0
        assert len(wp.type) > 0


def test_haversine_distance_calculation():
    gps = SimulatedGPSSource()
    # Distance between New Delhi (28.6139, 77.2090) and Agra (27.1767, 78.0081) is approx 180-200 km
    dist = gps.haversine_distance(28.6139, 77.2090, 27.1767, 78.0081)
    assert 170.0 < dist < 210.0


def test_heading_calculation_cardinal_directions():
    gps = SimulatedGPSSource()
    # Due North: lat increases, lon constant -> ~0 deg
    north_heading = gps.calculate_heading(28.0, 77.0, 29.0, 77.0)
    assert abs(north_heading - 0.0) < 1.0 or abs(north_heading - 360.0) < 1.0

    # Due East: lat constant, lon increases -> ~90 deg
    east_heading = gps.calculate_heading(28.0, 77.0, 28.0, 78.0)
    assert abs(east_heading - 90.0) < 2.0


def test_simulated_gps_movement_is_continuous_and_bounded():
    gps = SimulatedGPSSource()
    steps = 50
    prev_pos = None

    for i in range(steps):
        progress = i / (steps - 1)
        pos = gps.get_position(progress, {"altitude_ft": 8000.0, "throttle": 0.60})
        assert isinstance(pos, UAVPosition)
        assert 28.0 <= pos.latitude <= 30.0
        assert 76.5 <= pos.longitude <= 78.5
        assert 0.0 <= pos.heading_deg <= 360.0
        assert 60.0 <= pos.ground_speed_kt <= 220.0
        assert pos.mission_progress == pytest.approx(progress, abs=0.01)

        if prev_pos is not None:
            # Step jump must be continuous, no random teleportation
            step_jump = gps.haversine_distance(
                prev_pos.latitude, prev_pos.longitude,
                pos.latitude, pos.longitude
            )
            assert step_jump < 25.0  # smooth movement between consecutive steps

        prev_pos = pos


def test_hardware_gps_fallback_interface():
    hw = HardwareGPSInterface(port="/dev/ttyUSB0")
    pos = hw.get_position(0.5, {"altitude_ft": 12000.0, "throttle": 0.65})
    assert isinstance(pos, UAVPosition)
    assert pos.latitude > 0
    assert pos.longitude > 0


def test_replay_contains_uav_navigation_and_waypoints():
    ai = AeroTwinAI()
    base = {
        "Engine_RPM": 3000.0, "EGT1": 1000.0, "EGT2": 1005.0, "EGT3": 995.0,
        "CHT": 180.0, "Fuel_Flow": 20.0, "Oil_Temp": 90.0, "Oil_Pressure": 60.0,
        "Battery_Voltage": 27.0, "Battery_Current": 2.0, "Alternator_Temp": 80.0,
        "EFI_Fuel_Temp": 30.0, "EFI_Water_Temp": 85.0, "MAP_Injector": 20.0,
        "Operating_State": "CRUISE",
    }
    scenario = {
        "fault": "none", "severity": 0.0, "altitude_ft": 8000.0,
        "ambient_c": 25.0, "duration_h": 4.0, "rapid_throttle": False,
        "operating_state": "CRUISE"
    }
    result = run_replay(ai, base, scenario, steps=24, step_minutes=5.0, fault_onset_ratio=0.35)
    
    assert "waypoints" in result
    assert len(result["waypoints"]) >= 6
    assert "planned_route" in result
    assert "home_base" in result
    assert "timeline" in result
    assert len(result["timeline"]) == 24

    for point in result["timeline"]:
        assert "uav" in point
        uav = point["uav"]
        assert "latitude" in uav
        assert "longitude" in uav
        assert "altitude_ft" in uav
        assert "ground_speed_kt" in uav
        assert "heading_deg" in uav
        assert "mission_phase" in uav
        assert "mission_progress" in uav
