"""Unit and Integration Tests for Live & Simulated Environmental Coupling and Wind Dynamics."""
import pytest
from app.environment import EnvironmentService, EnvironmentState, _DEFAULT_ENV_SERVICE
from app.navigation import (
    DEFAULT_MISSION_WAYPOINTS,
    HIGH_ALT_SURVEILLANCE_WAYPOINTS,
    SimulatedGPSSource,
    MissionWaypoint,
)
from app.engine_model import ReducedOrderPistonEngine
from app.inference import AeroTwinAI
from app.replay import run_replay


@pytest.fixture(autouse=True)
def disable_live_weather_for_unit_tests():
    orig_live = _DEFAULT_ENV_SERVICE.enable_live
    _DEFAULT_ENV_SERVICE.enable_live = False
    yield
    _DEFAULT_ENV_SERVICE.enable_live = orig_live


def test_environment_service_simulated_fallback():
    # Force simulated mode by disabling live fetch
    env_service = EnvironmentService(enable_live=False)
    
    # Test low altitude environment
    env_low = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=1000.0, heading_deg=90.0)
    assert env_low.source == "simulated"
    assert 25.0 <= env_low.ambient_c <= 35.0
    assert 0.90 <= env_low.air_density_ratio <= 1.05
    assert env_low.pressure_hpa > 900.0
    assert env_low.wind_speed_kt >= 4.0

    # Test high altitude environment (20,000 ft)
    env_high = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=20000.0, heading_deg=90.0)
    assert env_high.ambient_c < env_low.ambient_c - 30.0
    assert env_high.air_density_ratio < env_low.air_density_ratio
    assert env_high.pressure_hpa < env_low.pressure_hpa


def test_wind_vector_decomposition_headwind_and_crosswind():
    env_service = EnvironmentService(enable_live=False)
    
    # 1. Test pure headwind: heading aircraft directly into the wind
    env_sample = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=5000.0, heading_deg=0.0)
    wind_dir = env_sample.wind_direction_deg
    
    env_headwind = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=5000.0, heading_deg=wind_dir)
    assert env_headwind.headwind_kt > 0.0
    assert abs(env_headwind.crosswind_kt) < 0.1
    assert abs(env_headwind.headwind_kt - env_headwind.wind_speed_kt) < 0.1

    # 2. Test pure crosswind: heading 90 degrees offset from wind direction
    env_crosswind = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=5000.0, heading_deg=(wind_dir + 90.0) % 360.0)
    assert abs(env_crosswind.headwind_kt) < 0.1
    assert abs(abs(env_crosswind.crosswind_kt) - env_crosswind.wind_speed_kt) < 0.1


def test_wind_affects_ground_speed_and_mission_duration():
    # Construct a straight east-west leg
    waypoints = [
        MissionWaypoint("WP0", "Origin", 28.5, 77.0, 8000.0, "CRUISE", 140.0),
        MissionWaypoint("WP1", "East Fix", 28.5, 78.5, 8000.0, "CRUISE", 140.0),
    ]
    gps = SimulatedGPSSource(waypoints)
    pos = gps.get_position(0.5)

    # True Airspeed should match cruise target
    assert pos.airspeed_kt == 140.0
    # Ground speed should reflect airspeed minus headwind component
    expected_gs = max(35.0, round(pos.airspeed_kt - pos.environment["headwind_kt"], 1))
    assert abs(pos.ground_speed_kt - expected_gs) < 0.1
    assert pos.track_deg >= 0.0


def test_environment_feeds_engine_model():
    env_service = EnvironmentService(enable_live=False)
    env_cold_high = env_service.get_environment(latitude=28.5, longitude=77.0, altitude_ft=22000.0, heading_deg=0.0)
    
    engine = ReducedOrderPistonEngine()
    engine_data = engine.simulate(
        rpm=3100.0,
        throttle=0.75,
        altitude_ft=22000.0,
        ambient_c=env_cold_high.ambient_c,
        load=0.85,
    )
    
    assert engine_data["Air_Density_Ratio"] < 0.65
    assert engine_data["EGT1"] > 700.0
    assert engine_data["CHT"] > 90.0


def test_flight_plan_summary_includes_route_risk_projections():
    gps = SimulatedGPSSource(HIGH_ALT_SURVEILLANCE_WAYPOINTS)
    summary = gps.get_flight_plan_summary()

    assert "route_risk_segments" in summary
    assert len(summary["route_risk_segments"]) == len(HIGH_ALT_SURVEILLANCE_WAYPOINTS) - 1
    
    # High altitude legs (24,000 ft) should be flagged with MEDIUM or HIGH risk projection
    risk_levels = [seg["predicted_risk"] for seg in summary["route_risk_segments"]]
    assert "MEDIUM" in risk_levels or "HIGH" in risk_levels


def test_replay_contains_live_environment_and_wind_vectors():
    from app.environment import _DEFAULT_ENV_SERVICE
    orig_live = _DEFAULT_ENV_SERVICE.enable_live
    _DEFAULT_ENV_SERVICE.enable_live = False
    try:
        ai = AeroTwinAI()
        base = {
            "Engine_RPM": 3000.0, "EGT1": 1000.0, "EGT2": 1005.0, "EGT3": 995.0,
            "CHT": 180.0, "Fuel_Flow": 20.0, "Oil_Temp": 90.0, "Oil_Pressure": 60.0,
            "Battery_Voltage": 27.0, "Battery_Current": 2.0, "Alternator_Temp": 80.0,
            "EFI_Fuel_Temp": 30.0, "EFI_Water_Temp": 85.0, "MAP_Injector": 20.0,
            "Operating_State": "CRUISE",
        }
        scenario = {
            "fault": "none",
            "severity": 0.0,
            "simulation_mode": "automatic",
            "waypoints": [wp.to_dict() for wp in DEFAULT_MISSION_WAYPOINTS],
        }
        result = run_replay(ai, base, scenario, steps=15, step_minutes=5.0, fault_onset_ratio=0.35)
        
        for item in result["timeline"]:
            uav = item["uav"]
            env = item["environment"]
            assert "airspeed_kt" in uav
            assert "ground_speed_kt" in uav
            assert "track_deg" in uav
            assert "wind_speed_kt" in env
            assert "headwind_kt" in env
            assert "source" in env
    finally:
        _DEFAULT_ENV_SERVICE.enable_live = orig_live
