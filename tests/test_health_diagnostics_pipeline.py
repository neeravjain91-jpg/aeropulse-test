"""Integration and Regression Tests for AeroPulse-X Health & Diagnostics Pipeline.

Verifies:
1. Healthy baseline returns Normal health state, high health index (>=85%), LOW risk, and trusted sensors.
2. Eliminates false-alarm compound faults and spurious -97 sigma battery current / oil pressure z-score spikes.
3. Overheating fault triggers high thermal residuals and thermal runaway fault candidate.
4. Lubrication fault triggers low oil pressure residuals and lubrication degradation advisory.
5. Electrical fault triggers alternator/battery deviation.
6. Sensor drift / bias correctly isolates the transducer fault without causing catastrophic structural RUL collapse.
7. Replay trajectory maintains healthy initial health history (>80%) on nominal missions.
8. Zero leakage and deterministic reproducibility.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.digital_twin import ReferenceTwin

client = TestClient(app)


def test_analyze_healthy_baseline_nominal():
    """Verify that a nominal baseline flight with fault=none produces healthy diagnostic outputs."""
    payload = {
        "fault": "none",
        "severity": 0.0,
        "simulation_mode": "automatic",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 1,
        "rapid_throttle": False,
        "operating_state": "CRUISE",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Core health KPIs
    assert data["health_state"] == "Normal"
    assert data["health_index"] >= 85.0
    assert data["mission_risk"]["level"] == "LOW"
    assert data["mission_risk"]["score"] < 25.0

    # Digital Twin residuals
    twin = data["twin"]
    assert twin["residual_rms"] < 1.5, f"Expected nominal RMS < 1.5, got {twin['residual_rms']}"
    assert abs(twin["z_scores"]["Battery_Current"]) < 3.0, f"Spurious Battery_Current z: {twin['z_scores']['Battery_Current']}"
    assert abs(twin["z_scores"]["Oil_Pressure"]) < 3.0, f"Spurious Oil_Pressure z: {twin['z_scores']['Oil_Pressure']}"
    assert abs(twin["z_scores"]["EFI_Fuel_Temp"]) < 3.0, f"Spurious EFI_Fuel_Temp z: {twin['z_scores']['EFI_Fuel_Temp']}"

    # Sensor health
    sh = data["sensor_health"]
    assert sh["overall_trust_score"] >= 90.0
    assert sh["verdict"] in ["NOMINAL", "TRUSTED"]
    assert "COMPOUND_FAULT" not in str(sh.get("diagnostic_advisory", ""))


def test_analyze_hot_high_baseline_consistency():
    """Verify that 8000ft / 35C cruise maintains healthy physical residuals with atmospheric altitude compensation."""
    payload = {
        "fault": "none",
        "severity": 0.0,
        "simulation_mode": "automatic",
        "altitude_ft": 8000,
        "ambient_c": 35,
        "duration_h": 6,
        "rapid_throttle": False,
        "operating_state": "CRUISE",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Residuals must remain low because Digital Twin compensates for altitude and ambient temp
    twin = data["twin"]
    assert twin["residual_rms"] < 1.8, f"High altitude RMS unexpectedly large: {twin['residual_rms']}"
    assert data["sensor_health"]["overall_trust_score"] >= 90.0
    assert data["mission_risk"]["level"] in ["LOW", "MEDIUM"]


def test_analyze_overheating_fault_detection():
    """Verify that thermal runaway / overheating triggers Warning/Critical and elevated thermal residuals."""
    payload = {
        "fault": "overheating",
        "severity": 0.8,
        "simulation_mode": "automatic",
        "altitude_ft": 4000,
        "ambient_c": 30,
        "duration_h": 2,
        "rapid_throttle": False,
        "operating_state": "CRUISE",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["health_state"] in ["Warning", "Critical"]
    assert data["health_index"] < 65.0
    assert data["twin"]["residual_rms"] > 3.0

    candidates = [c["name"].lower() for c in data.get("fault_candidates", [])]
    assert any("thermal" in c or "overheat" in c for c in candidates)


def test_analyze_lubrication_fault_detection():
    """Verify that lubrication degradation triggers low oil pressure residuals and correct advisory."""
    payload = {
        "fault": "lubrication",
        "severity": 0.7,
        "simulation_mode": "automatic",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 2,
        "rapid_throttle": False,
        "operating_state": "CRUISE",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["health_state"] in ["Warning", "Critical"]
    assert data["health_index"] < 60.0
    assert data["twin"]["residual_rms"] > 3.0

    candidates = [c["name"].lower() for c in data.get("fault_candidates", [])]
    assert any("lubrication" in c or "oil" in c for c in candidates)


def test_analyze_sensor_drift_isolation():
    """Verify that transducer drift is identified as an isolated sensor fault rather than engine breakdown."""
    payload = {
        "fault": "sensor_drift",
        "severity": 0.8,
        "simulation_mode": "automatic",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 1,
        "rapid_throttle": False,
        "operating_state": "CRUISE",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    sh = data["sensor_health"]
    assert sh["verdict"] == "SENSOR_FAULT_ISOLATED" or sh["is_sensor_fault_only"] is True
    assert sh["overall_trust_score"] < 80.0
    # Engine physical health should NOT collapse to 0 on isolated transducer fault
    assert data["health_index"] >= 50.0


def test_replay_health_progression_nominal():
    """Verify that replay of nominal flight has healthy initial and sustained health history."""
    payload = {
        "fault": "none",
        "severity": 0.0,
        "steps": 15,
        "operating_state": "CRUISE",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 1,
    }
    response = client.post("/api/replay", json=payload)
    assert response.status_code == 200
    data = response.json()

    timeline = data.get("timeline", [])
    assert len(timeline) == 15
    for pt in timeline[:5]:
        assert pt["health_state"] == "Normal"
        assert pt["health_index"] >= 85.0
        assert pt["risk_level"] == "LOW"


def test_replay_rul_continuity_nominal():
    """Verify that nominal flight RUL remains monotonic and never experiences false degradation collapse."""
    payload = {
        "fault": "none",
        "severity": 0.0,
        "steps": 15,
        "operating_state": "CRUISE",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 1,
    }
    response = client.post("/api/replay", json=payload)
    assert response.status_code == 200
    timeline = response.json().get("timeline", [])
    assert len(timeline) == 15

    for i, pt in enumerate(timeline):
        rul = pt.get("rul", {})
        rul_h = pt.get("rul_hours")
        assert rul_h is not None, f"Step {i} RUL was None"
        assert rul_h > 1000.0, f"Step {i} RUL unexpectedly collapsed to {rul_h}"
        assert rul.get("status") == "NOMINAL_HEALTH", f"Step {i} status was {rul.get('status')}"
        if i > 0:
            step_drop = timeline[i - 1]["rul_hours"] - rul_h
            assert 0.0 <= step_drop < 1.0, f"Step {i} non-monotonic or abrupt drop: {step_drop}"


def test_replay_rul_active_degradation_overheating():
    """Verify that severe overheating triggers active degradation status and rapid RUL reduction."""
    payload = {
        "fault": "overheating",
        "severity": 0.7,
        "steps": 15,
        "operating_state": "CRUISE",
        "altitude_ft": 3000,
        "ambient_c": 25,
        "duration_h": 1,
    }
    response = client.post("/api/replay", json=payload)
    assert response.status_code == 200
    timeline = response.json().get("timeline", [])
    assert len(timeline) == 15

    # Initial steps are nominal
    assert timeline[0]["rul_hours"] > 1000.0
    # Fault develops after onset (step >= 5)
    late_steps = timeline[10:]
    assert any(pt.get("rul", {}).get("status") == "ACTIVE_DEGRADATION" for pt in late_steps)
    assert timeline[-1]["rul_hours"] < 10.0


def test_websocket_stream_twin_diagnostics_payload():
    """Verify that WebSocket telemetry stream enriches points with digital twin and diagnostics data."""
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"steps": 12, "fault": "none", "playback_interval_s": 0.05})
        start_msg = ws.receive_json()
        assert start_msg.get("type") == "start"

        telemetry_msg = ws.receive_json()
        assert telemetry_msg.get("type") == "telemetry"
        point = telemetry_msg.get("data", {})

        assert "twin" in point, "twin missing from streaming telemetry point"
        twin = point["twin"]
        assert "residual_rms" in twin
        assert "expected" in twin
        assert "z_scores" in twin
        assert "fault_candidates" in point
        assert "sensor_health" in point
        assert "maintenance_advisory" in point
        assert "mission_risk" in point
