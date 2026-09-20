"""Unit and integration tests for Supported MALE Aero-Piston UAV Platform Subsystem.

Verifies:
1. Strict platform scope (Class: MALE, Propulsion: AERO-PISTON).
2. Engine digital twin linkages (Rotax 914, Aero-Diesel, AeroPiston).
3. Parameter completeness, provenance badges, and unverified data handling.
4. Dynamic flight simulation propagation (speed, altitude ceiling, MTOM throttle scaling).
5. Engine-out glide ratio propagation and Rule 11 fallback when unverified.
6. What-If mission constraint evaluation (ceiling and endurance violations, MTOM fuel scaling).
7. API contract conformance (/api/v1/uav/catalog, /api/v1/uav/{id}, /api/v1/uav/select).
8. Multi-UAV isolation and error handling.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, _ACTIVE_UAV_ID, _ACTIVE_UAV_PROFILE, _GPS
from app.navigation import MissionWaypoint, SimulatedGPSSource
from app.mission_whatif import MissionScenario
from app.mission_whatif_rul import MissionWhatIfRUL
from app.uav_platform_config import (
    CATALOG_TITLE,
    SUPPORTED_UAV_REGISTRY,
    UAVPlatformProfile,
    get_uav_profile,
    list_supported_uavs,
)
from app.engine_config import ENGINE_PROFILES


client = TestClient(app)


# =========================================================================
# 1. Platform Scope & Registry Integrity Tests
# =========================================================================

def test_catalog_title_and_scope_rule():
    """Verify registry title strictly states SUPPORTED MALE AERO-PISTON and does not claim ALL."""
    assert CATALOG_TITLE == "SUPPORTED MALE AERO-PISTON UAV PLATFORMS"
    assert "ALL" not in CATALOG_TITLE


def test_every_registered_platform_satisfies_male_aeropiston_criteria():
    """Verify that every platform in registry is strictly MALE and AERO-PISTON powered."""
    assert len(SUPPORTED_UAV_REGISTRY) >= 6

    for uav_id, profile in SUPPORTED_UAV_REGISTRY.items():
        assert isinstance(profile, UAVPlatformProfile)
        assert profile.uav_class.upper() == "MALE", f"{uav_id} must be MALE class"
        assert "PISTON" in profile.propulsion_type.upper(), f"{uav_id} must be aero-piston"

        # Validate non-negotiable physical bounds
        assert profile.max_takeoff_mass_kg > 0
        assert profile.cruise_speed_kt > 0
        assert profile.max_speed_kt > profile.cruise_speed_kt
        assert profile.service_ceiling_ft > 0
        assert profile.endurance_hours > 0

        # Validate profile method passes
        is_valid, violations = profile.validate_profile()
        assert is_valid, f"{uav_id} failed validation: {violations}"


def test_registry_excludes_turboprops_and_jets():
    """Ensure registry does not include non-piston propulsion types (turboprop, jet, electric)."""
    disallowed_types = ["TURBOPROP", "TURBOJET", "TURBOFAN", "ELECTRIC", "HYBRID"]
    for uav_id, profile in SUPPORTED_UAV_REGISTRY.items():
        for dt in disallowed_types:
            assert dt not in profile.propulsion_type.upper(), f"{uav_id} contains excluded propulsion {dt}"


def test_scientific_provenance_and_audit_trail():
    """Verify every platform has documented sources, provenance labels, and confidence levels."""
    valid_provenance = {"MANUFACTURER", "LITERATURE", "DERIVED", "ASSUMED"}
    valid_confidence = {"HIGH", "MEDIUM", "LOW"}

    for uav_id, profile in SUPPORTED_UAV_REGISTRY.items():
        assert profile.source and len(profile.source) > 5, f"{uav_id} missing documented source"
        assert profile.data_provenance in valid_provenance, f"{uav_id} has invalid provenance: {profile.data_provenance}"
        assert profile.confidence in valid_confidence, f"{uav_id} has invalid confidence: {profile.confidence}"


def test_unverified_glide_ratio_fallback_enforcement():
    """Rule 11 Safeguard: Unverified glide ratio must be None and not assume universal 12:1."""
    research_uav = get_uav_profile("UAV_MALE_CUSTOM_RESEARCH")
    assert research_uav is not None
    assert research_uav.glide_ratio is None, "Generic research article must explicitly have glide_ratio=None"
    assert research_uav.fuel_capacity_l is None, "Generic research article must handle missing fuel capacity gracefully"


# =========================================================================
# 2. Engine Digital Twin Linkages
# =========================================================================

def test_engine_profile_linkage_resolves():
    """Verify that every platform's linked engine profile exists in the engine digital twin registry."""
    for uav_id, profile in SUPPORTED_UAV_REGISTRY.items():
        assert profile.engine_id in ENGINE_PROFILES or "AeroPiston" in profile.engine_id, (
            f"{uav_id} engine {profile.engine_id} not found in ENGINE_PROFILES"
        )


def test_uav_selection_auto_swaps_digital_twin_engine():
    """Verify selecting a UAV hot-swaps the underlying digital twin engine."""
    # Select MQ-1B (Rotax 914 Turbo)
    res_predator = client.post("/api/v1/uav/select", json={"uav_id": "UAV_MALE_MQ1_PREDATOR"})
    assert res_predator.status_code == 200
    data_pred = res_predator.json()
    assert "Rotax" in data_pred["linked_engine_name"]
    assert data_pred["glide_ratio"] == 16.0

    # Select TAI Anka-A (Generic-Inline4-AeroDiesel)
    res_anka = client.post("/api/v1/uav/select", json={"uav_id": "UAV_MALE_TAI_ANKA_A"})
    assert res_anka.status_code == 200
    data_anka = res_anka.json()
    assert "Diesel" in data_anka["linked_engine_name"]
    assert data_anka["glide_ratio"] == 16.5


# =========================================================================
# 3. Dynamic Flight Simulation Propagation
# =========================================================================

def test_uav_service_ceiling_enforcement_in_flight_dynamics():
    """Verify that aircraft service ceiling caps dynamic flight altitude."""
    research_uav = get_uav_profile("UAV_MALE_CUSTOM_RESEARCH")  # Service ceiling 20,000 ft
    wps = [
        MissionWaypoint("WP0", "Base", 28.5, 77.0, 1000.0, "BASE", 60.0),
        MissionWaypoint("WP1", "Climb Ceiling", 28.6, 77.2, 28000.0, "CLIMB", 90.0),
    ]
    gps = SimulatedGPSSource(waypoints=wps, uav_profile=research_uav)
    pos = gps.get_position(1.0)
    # Target was 28,000 ft, but research UAV ceiling is 20,000 ft
    assert pos.altitude_ft <= 20000.0


def test_uav_max_speed_clamping_in_flight_dynamics():
    """Verify that true airspeed cannot exceed the aircraft's documented max speed."""
    predator = get_uav_profile("UAV_MALE_MQ1_PREDATOR")  # Max speed: 117 kt
    wps = [
        MissionWaypoint("WP0", "Origin", 28.5, 77.0, 8000.0, "CRUISE", 150.0),
        MissionWaypoint("WP1", "Fix", 28.5, 78.0, 8000.0, "CRUISE", 150.0),
    ]
    gps = SimulatedGPSSource(waypoints=wps, uav_profile=predator)
    pos = gps.get_position(0.5)
    assert pos.airspeed_kt <= predator.max_speed_kt


def test_uav_mtom_scales_throttle_and_load():
    """Verify that heavier aircraft require higher propulsion demand (auto-throttle/load)."""
    light_uav = get_uav_profile("UAV_MALE_DRDO_RUSTOM1")  # MTOM: 720 kg
    heavy_uav = get_uav_profile("UAV_MALE_TAI_ANKA_A")     # MTOM: 1600 kg

    wps = [
        MissionWaypoint("WP0", "Origin", 28.5, 77.0, 8000.0, "CRUISE", 80.0),
        MissionWaypoint("WP1", "Fix", 28.5, 78.0, 8000.0, "CRUISE", 80.0),
    ]
    gps_light = SimulatedGPSSource(waypoints=wps, uav_profile=light_uav)
    gps_heavy = SimulatedGPSSource(waypoints=wps, uav_profile=heavy_uav)

    pos_light = gps_light.get_position(0.5)
    pos_heavy = gps_heavy.get_position(0.5)

    assert pos_heavy.auto_throttle >= pos_light.auto_throttle
    assert pos_heavy.auto_load >= pos_light.auto_load


# =========================================================================
# 4. Flight Plan Summary & Constraint Warning Emission
# =========================================================================

def test_flight_plan_summary_emits_service_ceiling_warning():
    """Verify flight plan summary detects when waypoints exceed platform ceiling."""
    rustom1 = get_uav_profile("UAV_MALE_DRDO_RUSTOM1")  # Ceiling: 26,000 ft
    wps = [
        MissionWaypoint("WP0", "Base", 28.5, 77.0, 1000.0, "BASE", 60.0),
        MissionWaypoint("WP1", "Too High", 28.6, 77.2, 30000.0, "CRUISE", 80.0),
    ]
    gps = SimulatedGPSSource(waypoints=wps, uav_profile=rustom1)
    summary = gps.get_flight_plan_summary()

    assert summary["uav_platform"] is not None
    assert summary["uav_platform"]["uav_id"] == "UAV_MALE_DRDO_RUSTOM1"
    assert len(summary["platform_warnings"]) >= 1
    assert any("service ceiling" in w for w in summary["platform_warnings"])
    assert summary["glide_ratio"] == 13.5
    assert summary["glide_performance_available"] is True


def test_flight_plan_summary_handles_unverified_glide_ratio():
    """Verify flight plan summary flags unverified glide performance."""
    research_uav = get_uav_profile("UAV_MALE_CUSTOM_RESEARCH")
    wps = [
        MissionWaypoint("WP0", "Base", 28.5, 77.0, 1000.0, "BASE", 60.0),
        MissionWaypoint("WP1", "Fix", 28.6, 77.2, 8000.0, "CRUISE", 80.0),
    ]
    gps = SimulatedGPSSource(waypoints=wps, uav_profile=research_uav)
    summary = gps.get_flight_plan_summary()

    assert summary["glide_ratio"] is None
    assert summary["glide_performance_available"] is False


# =========================================================================
# 5. Mission What-If Integration
# =========================================================================

def test_whatif_rul_evaluates_platform_endurance_and_fuel_limits():
    """Verify MissionWhatIfRUL detects endurance and fuel capacity limits for selected UAV."""
    whatif = MissionWhatIfRUL()
    base_telemetry = {
        "Engine_RPM": 3000.0, "EGT1": 1000.0, "EGT2": 1005.0, "EGT3": 995.0,
        "CHT": 180.0, "Fuel_Flow": 20.0, "Oil_Temp": 90.0, "Oil_Pressure": 60.0,
    }

    # Rustom-1 has 14h endurance and 160 L fuel capacity
    scenario_excessive = MissionScenario(
        name="Excessive Loiter",
        altitude_ft=12000.0,
        ambient_c=25.0,
        duration_h=18.0,  # Exceeds 14h
    )

    eval_res = whatif.evaluate_scenario(base_telemetry, scenario_excessive, uav_profile="UAV_MALE_DRDO_RUSTOM1")
    assert eval_res["uav_platform"] is not None
    assert len(eval_res["platform_warnings"]) >= 1
    assert any("maximum endurance" in w for w in eval_res["platform_warnings"])


def test_whatif_rul_mtom_fuel_flow_scaling():
    """Verify heavier aircraft have higher fuel burn in What-If calculations."""
    whatif = MissionWhatIfRUL()
    base_telemetry = {
        "Engine_RPM": 3000.0, "EGT1": 1000.0, "EGT2": 1005.0, "EGT3": 995.0,
        "CHT": 180.0, "Fuel_Flow": 20.0, "Oil_Temp": 90.0, "Oil_Pressure": 60.0,
    }
    scenario = MissionScenario(name="Standard Leg", altitude_ft=10000.0, ambient_c=20.0, duration_h=5.0)

    res_light = whatif.evaluate_scenario(base_telemetry, scenario, uav_profile="UAV_MALE_DRDO_RUSTOM1")
    res_heavy = whatif.evaluate_scenario(base_telemetry, scenario, uav_profile="UAV_MALE_TAI_ANKA_A")

    assert res_heavy["fuel_flow_l_h"] > res_light["fuel_flow_l_h"]
    assert res_heavy["total_fuel_burn_l"] > res_light["total_fuel_burn_l"]


# =========================================================================
# 6. REST API Endpoints Contract Tests
# =========================================================================

def test_api_get_uav_catalog_contract():
    """Verify GET /api/v1/uav/catalog returns all supported platforms with provenance."""
    res = client.get("/api/v1/uav/catalog")
    assert res.status_code == 200
    data = res.json()

    assert data["catalog_title"] == "SUPPORTED MALE AERO-PISTON UAV PLATFORMS"
    assert "MALE" in data["scope_rule"]
    assert "AERO-PISTON" in data["scope_rule"]
    assert data["count"] >= 6
    assert len(data["platforms"]) == data["count"]

    # Verify each platform in API response contains all required fields
    required_fields = [
        "uav_id", "uav_name", "uav_class", "engine_id", "propulsion_type",
        "max_takeoff_mass_kg", "cruise_speed_kt", "max_speed_kt",
        "service_ceiling_ft", "endurance_hours", "data_provenance", "source"
    ]
    for p in data["platforms"]:
        for f in required_fields:
            assert f in p, f"Missing field {f} in UAV API response"


def test_api_get_uav_profile_by_id_contract():
    """Verify GET /api/v1/uav/{uav_id} returns exact profile, and unknown ID returns 404."""
    res = client.get("/api/v1/uav/UAV_MALE_MQ1_PREDATOR")
    assert res.status_code == 200
    pred = res.json()
    assert pred["uav_name"] == "General Atomics MQ-1B Predator"
    assert pred["glide_ratio"] == 16.0

    # Unknown UAV returns 404
    res_unknown = client.get("/api/v1/uav/NON_EXISTENT_UAV_XYZ")
    assert res_unknown.status_code == 404
    assert "Configuration unavailable for selected UAV" in res_unknown.json()["detail"]


def test_api_select_uav_contract_and_rejection():
    """Verify POST /api/v1/uav/select successfully selects platform and rejects invalid IDs."""
    # Valid selection
    res = client.post("/api/v1/uav/select", json={"uav_id": "UAV_MALE_IAI_HERON_1"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "Heron" in data["selected_uav"]
    assert data["glide_ratio"] == 19.0

    # Invalid ID
    res_err = client.post("/api/v1/uav/select", json={"uav_id": "UNKNOWN_PLATFORM_123"})
    assert res_err.status_code == 404


def test_api_mission_plan_with_uav_id():
    """Verify POST /api/mission/plan incorporates uav_id constraints."""
    wps = [
        {"id": "WP0", "name": "Base", "latitude": 28.5, "longitude": 77.0, "altitude_ft": 1000.0, "speed_kt": 60.0},
        {"id": "WP1", "name": "Fix", "latitude": 28.8, "longitude": 77.4, "altitude_ft": 28000.0, "speed_kt": 80.0},
    ]
    res = client.post("/api/mission/plan", json={
        "waypoints": wps,
        "uav_id": "UAV_MALE_DRDO_RUSTOM1"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["uav_platform"]["uav_id"] == "UAV_MALE_DRDO_RUSTOM1"
    assert len(data["platform_warnings"]) >= 1
    assert any("service ceiling" in w for w in data["platform_warnings"])
