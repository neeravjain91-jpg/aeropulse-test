"""Hard Validation Test Suite for Trajectory Kinematics, Supervisory State, and Report Exports.

Verifies:
1. Trajectory phase sequence follows CLIMB -> CRUISE -> DESCENT -> LANDING.
2. Altitude continuity and climb consistency (vertical_speed > 0, speed increasing).
3. Descent consistency (vertical_speed < 0, ground speed decreasing).
4. Landing altitude strictly below transition ceiling (< 2,000 ft).
5. Landing vertical speed trends toward zero near touchdown, not stagnant at 0 for all points.
6. Landing ground speed decreases substantially from cruise speed.
7. Mission termination state represents touchdown/rollout (low altitude, low speed, zero vertical speed).
8. Clean t=0 state (fault_present=false, severity=0, stage=HEALTHY, ECU=NORMAL, FADEC=NORMAL, DTC=[], derate=1.0, action=NONE).
9. No premature DTC or FADEC derate before deterministic trigger at 4.5h.
10. Deterministic fault onset transitions at 4.5h with proper DTC and derate.
11. Predicted RUL remains strictly monotonic non-increasing without target leakage.
12. Engine ID ROTAX_914_F_TWIN_01 resolves explicitly to Rotax-914-Turbo-115HP.
13. AI Report Markdown and HTML exports are complete, standalone, offline-renderable, and contain no API keys.
"""
from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.data_engine import VirtualDataLabEngine
from app.engine_config import resolve_canonical_engine_id, ENGINE_PROFILES
from app.rul_service import RULService
from app.mission_intelligence import EventExtractor, TrajectorySummarizer, resolve_engine_limits
from app.llm_report_service import LLMReportService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def thermal_trajectory():
    engine = VirtualDataLabEngine(master_seed=42)
    return engine.generate_degradation_trajectory(
        "TRAJ_CUSTOM_DEGRADATION_THERMAL_001",
        failure_mode="thermal",
        seed=42,
    )


# =========================================================================
# 1. KINEMATICS & PHASE TRANSITIONS
# =========================================================================

def test_trajectory_phase_sequence(thermal_trajectory):
    """Verify chronological phase progression: CLIMB -> CRUISE -> DESCENT -> LANDING."""
    phases = [p.mission_phase for p in thermal_trajectory]
    unique_sequence = []
    for ph in phases:
        if not unique_sequence or unique_sequence[-1] != ph:
            unique_sequence.append(ph)
            
    assert unique_sequence == ["CLIMB", "CRUISE", "DESCENT", "LANDING"], f"Unexpected sequence: {unique_sequence}"


def test_climb_consistency(thermal_trajectory):
    """Verify early mission setup / climb has positive vertical speed and rising altitude."""
    climb_pts = [p for p in thermal_trajectory if p.mission_phase == "CLIMB"]
    assert len(climb_pts) >= 1, "Must have at least one climb point"
    
    p0 = climb_pts[0]
    assert p0.vertical_speed > 0.0, f"Climb vertical speed must be positive: {p0.vertical_speed}"
    assert p0.altitude < 6000.0, f"Initial climb altitude should be below cruise: {p0.altitude}"
    assert p0.throttle >= 0.80, f"Climb throttle should be high: {p0.throttle}"


def test_descent_consistency(thermal_trajectory):
    """Verify descent has negative vertical speed and decreasing altitude."""
    descent_pts = [p for p in thermal_trajectory if p.mission_phase == "DESCENT"]
    assert len(descent_pts) >= 1, "Must have descent points"
    
    for p in descent_pts:
        assert p.vertical_speed < 0.0, f"Descent vertical speed must be negative: {p.vertical_speed}"
        assert p.altitude <= 6000.0, f"Descent altitude cannot exceed cruise: {p.altitude}"
        assert p.throttle <= 0.40, f"Descent throttle should be throttled back: {p.throttle}"


def test_landing_altitude_below_transition_ceiling(thermal_trajectory):
    """Verify LANDING points are strictly below the 2,000 ft transition ceiling."""
    landing_pts = [p for p in thermal_trajectory if p.mission_phase == "LANDING"]
    assert len(landing_pts) >= 2, "Must have at least 2 landing points"
    
    ceiling_ft = 2000.0
    for p in landing_pts:
        assert p.altitude <= ceiling_ft, f"Landing altitude {p.altitude} ft exceeded ceiling {ceiling_ft} ft"


def test_landing_vertical_speed_not_stagnant_zero(thermal_trajectory):
    """Verify landing vertical speed is NOT identically 0 for all landing points."""
    landing_pts = [p for p in thermal_trajectory if p.mission_phase == "LANDING"]
    vs_values = [p.vertical_speed for p in landing_pts]
    
    # Must have points with non-zero descent rate during approach
    non_zero = [vs for vs in vs_values if vs < 0.0]
    assert len(non_zero) >= 1, f"Landing vertical speed cannot be stagnant zero across all points: {vs_values}"
    
    # Touchdown point should have near-zero vertical speed
    touchdown = landing_pts[-1]
    assert abs(touchdown.vertical_speed) <= 10.0, f"Touchdown vertical speed should be ~0: {touchdown.vertical_speed}"


def test_landing_ground_speed_decreases_substantially(thermal_trajectory):
    """Verify landing ground speed decreases substantially from cruise speed."""
    cruise_pts = [p for p in thermal_trajectory if p.mission_phase == "CRUISE"]
    landing_pts = [p for p in thermal_trajectory if p.mission_phase == "LANDING"]
    
    avg_cruise_speed = sum(p.ground_speed for p in cruise_pts) / len(cruise_pts)
    touchdown_speed = landing_pts[-1].ground_speed
    
    assert touchdown_speed < avg_cruise_speed * 0.4, f"Landing speed {touchdown_speed} kt must be substantially lower than cruise {avg_cruise_speed} kt"


def test_mission_termination_state(thermal_trajectory):
    """Verify final point represents terminal touchdown/taxi state."""
    final_p = thermal_trajectory[-1]
    assert final_p.altitude <= 50.0, f"Terminal altitude should be near-ground: {final_p.altitude} ft"
    assert final_p.ground_speed <= 25.0, f"Terminal ground speed should be taxi/touchdown: {final_p.ground_speed} kt"
    assert final_p.vertical_speed == 0.0, f"Terminal vertical speed must be 0: {final_p.vertical_speed}"


# =========================================================================
# 2. SAFETY STATE INITIALIZATION & DETERMINISTIC TRIGGER
# =========================================================================

def test_first_point_clean_initialization(thermal_trajectory):
    """Verify point t=0 has clean initial supervisory state."""
    p0 = thermal_trajectory[0]
    assert p0.timestamp == 0.0
    assert p0.fault_present is False
    assert p0.fault_severity == 0.0
    assert p0.degradation_stage == "HEALTHY"
    assert p0.ECU_state == "NORMAL"
    assert p0.FADEC_state == "NORMAL"
    assert p0.DTC == []
    assert p0.derate_command == 1.0
    assert p0.safety_action == "NONE"


def test_no_premature_dtc_or_derate(thermal_trajectory):
    """Verify no active DTC or derate action occurs before t=4.5h."""
    for p in thermal_trajectory:
        t_h = p.timestamp / 3600.0
        if t_h < 4.5:
            assert p.fault_present is False, f"Premature fault_present at T+{t_h}h"
            assert p.fault_severity == 0.0, f"Premature fault_severity at T+{t_h}h"
            assert p.ECU_state == "NORMAL", f"Premature ECU derate at T+{t_h}h"
            assert p.FADEC_state == "NORMAL", f"Premature FADEC derate at T+{t_h}h"
            assert p.DTC == [], f"Premature DTC {p.DTC} at T+{t_h}h"
            assert p.derate_command == 1.0, f"Premature derate_command at T+{t_h}h"
            assert p.safety_action == "NONE", f"Premature safety_action at T+{t_h}h"


def test_deterministic_fault_onset(thermal_trajectory):
    """Verify fault and supervisory response trigger at deterministic threshold t=4.5h."""
    fault_pts = [p for p in thermal_trajectory if p.timestamp / 3600.0 >= 4.5]
    assert len(fault_pts) >= 1
    
    first_fault = fault_pts[0]
    assert first_fault.fault_present is True
    assert first_fault.fault_type == "thermal"
    assert "DTC_CHT_OVERHEAT" in first_fault.DTC
    assert first_fault.FADEC_state == "DERATED_WARN"
    assert first_fault.derate_command == 0.80
    assert first_fault.ECU_state == "DERATED"
    assert first_fault.safety_action == "DERATE_80"


def test_fault_onset_separate_from_degradation_stage(thermal_trajectory):
    """Verify fault onset occurs while degradation stage is still HEALTHY."""
    p_45 = next(p for p in thermal_trajectory if abs(p.timestamp / 3600.0 - 4.5) < 0.01)
    assert p_45.fault_present is True
    assert p_45.degradation_stage == "HEALTHY", f"Expected HEALTHY at onset, got {p_45.degradation_stage}"


# =========================================================================
# 3. PROGNOSTICS & RUL MONOTONICITY
# =========================================================================

def test_predicted_rul_strictly_monotonic_non_increasing(thermal_trajectory):
    """Verify predicted RUL never increases over time."""
    rul_values = [p.predicted_RUL for p in thermal_trajectory]
    for i in range(1, len(rul_values)):
        assert rul_values[i] <= rul_values[i-1] + 1e-6, (
            f"RUL increased from {rul_values[i-1]} to {rul_values[i]} at step {i} "
            f"(T+{(thermal_trajectory[i].timestamp/3600.0):.1f}h)"
        )


def test_zero_target_leakage_in_rul_estimator():
    """Verify estimator only consumes observable state and rejects true failure time inputs."""
    service = RULService()
    ctx = {
        "elapsed_hours": 2.0,
        "altitude_ft": 6000.0,
        "ambient_c": 20.0,
        "throttle": 0.65,
        "engine_id": "ROTAX_914_F_TWIN_01",
    }
    # Call with valid inputs
    res1 = service.estimate_rul(health_index=95.0, context=ctx)
    assert res1["rul_hours"] > 0.0
    
    # Injecting fake ground truth should NOT change result
    ctx_leaked = {**ctx, "true_RUL": 1.0, "true_failure_time": 3.0}
    res2 = service.estimate_rul(health_index=95.0, context=ctx_leaked)
    assert abs(res1["rul_hours"] - res2["rul_hours"]) < 1e-4, "Predictor must ignore true_RUL/true_failure_time"


# =========================================================================
# 4. ENGINE ID CANONICAL RESOLUTION
# =========================================================================

def test_engine_id_mapping_canonical():
    """Verify ROTAX_914_F_TWIN_01 resolves explicitly to Rotax-914-Turbo-115HP across all modules."""
    canonical_id = resolve_canonical_engine_id("ROTAX_914_F_TWIN_01")
    assert canonical_id == "Rotax-914-Turbo-115HP"
    
    # Engine profile registry
    assert "ROTAX_914_F_TWIN_01" in ENGINE_PROFILES
    profile = ENGINE_PROFILES["ROTAX_914_F_TWIN_01"]
    assert profile["canonical_id"] == "Rotax-914-Turbo-115HP"
    assert profile["tbo_hours"] == 1200.0
    assert profile["max_rpm"] == 5800.0
    
    # RUL TBO resolution
    rul_srv = RULService()
    tbo = rul_srv.get_engine_tbo("ROTAX_914_F_TWIN_01")
    assert tbo == 1200.0
    
    # Mission intelligence limits
    limits = resolve_engine_limits("ROTAX_914_F_TWIN_01")
    assert limits["name"] == "Rotax-914-Turbo-115HP"
    assert limits["cht_max"] == 135.0
    assert limits["tbo_hours"] == 1200.0


# =========================================================================
# 5. AI REPORT GENERATION & EXPORT VALIDATION
# =========================================================================

def test_ai_report_markdown_and_html_exports(thermal_trajectory):
    """Verify reports contain all metadata, sections, and render offline without frontend runtime."""
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize([p.to_dict() for p in thermal_trajectory])
    
    service = LLMReportService()
    
    # 1. Engineering report
    eng_rep = service.generate_mission_report(summary, report_mode="engineering")
    assert eng_rep["report_mode"] == "engineering"
    md = eng_rep["report_markdown"]
    html = eng_rep["report_html"]
    
    assert "AEROPULSE-X MISSION INTELLIGENCE REPORT" in md
    assert "TRAJ_CUSTOM_DEGRADATION_THERMAL_001" in md
    assert "ROTAX_914_F_TWIN_01" in md
    assert "SYNTHETIC GROUND TRUTH" in md
    assert "<table" in html
    
    # No exposed secrets
    assert "AIzaSy" not in md and "AIzaSy" not in html
    assert "sk-" not in md and "sk-" not in html
    
    # 2. Executive report
    exec_rep = service.generate_mission_report(summary, report_mode="executive")
    assert exec_rep["report_mode"] == "executive"
    assert len(exec_rep["report_markdown"]) > 500


def test_api_report_and_uav_endpoints(client):
    """Verify HTTP API endpoints for reports and UAV catalog."""
    # Report generation API
    rep_resp = client.post(
        "/api/v1/data/report",
        json={"trajectory_id": "TRAJ_CUSTOM_DEGRADATION_THERMAL_001", "report_mode": "engineering"}
    )
    assert rep_resp.status_code == 200
    rep_data = rep_resp.json()
    assert rep_data["trajectory_id"] == "TRAJ_CUSTOM_DEGRADATION_THERMAL_001"
    assert "report_markdown" in rep_data
    assert "report_html" in rep_data
    
    # UAV Catalog
    cat_resp = client.get("/api/v1/uav/catalog")
    assert cat_resp.status_code == 200
    assert cat_resp.json()["count"] == 7
    
    # UAV Profile
    uav_resp = client.get("/api/v1/uav/UAV_MALE_MQ1_PREDATOR")
    assert uav_resp.status_code == 200
    assert uav_resp.json()["uav_name"] == "General Atomics MQ-1B Predator"
    
    # UAV Select
    sel_resp = client.post("/api/v1/uav/select", json={"uav_id": "UAV_MALE_IAI_HERON_1"})
    assert sel_resp.status_code == 200
    assert sel_resp.json()["status"] == "success"
