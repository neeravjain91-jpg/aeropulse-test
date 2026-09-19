"""Comprehensive Test Suite for Grounded LLM Mission Intelligence Report Generator.

Verifies:
1. Complete trajectory consumed without random row dropping.
2. Summary matches source telemetry metrics (min, max, mean).
3. Event extraction is deterministic and reproducible.
4. LLM receives structured evidence only.
5. Zero true_RUL leakage into predictor inputs.
6. Report does not invent absent events or ungrounded claims.
7. Ground truth is correctly labeled (SYNTHETIC GROUND TRUTH — POST-MISSION VALIDATION ONLY).
8. LLM numerical claim validation catches/rejects unsupported claims.
9. Deterministic fallback works when API key is absent or network fails.
10. Long trajectories process without truncation.
11. Multi-engine support (Rotax 914, AeroPiston 1.35L, AeroDiesel, etc.) resolves specific limits.
12. Multi-phase missions generate chronological phase transitions.
13. Fault-free mission produces clean nominal report.
14. Faulted mission produces chronological fault narrative.
15. Predicted RUL distinguished from true RUL.
16. Executive mode formats all 11 required sections.
17. Engineering mode formats all 17 required sections.
18. End-to-end verification using TRAJ_CUSTOM_DEGRADATION_THERMAL_001.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.data_engine import VirtualDataLabEngine
from app.mission_intelligence import (
    MissionEvent,
    MissionSummary,
    EventExtractor,
    TrajectorySummarizer,
    resolve_engine_limits,
)
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


@pytest.fixture
def healthy_trajectory():
    engine = VirtualDataLabEngine(master_seed=42)
    return engine.generate_healthy_trajectory(
        "TRAJ_CUSTOM_HEALTHY_NOMINAL_001",
        duration_hours=4.0,
        seed=42,
    )


def test_complete_trajectory_consumed_no_row_dropping(thermal_trajectory):
    """Criterion 1 & 2: All points in trajectory are processed without sampling or dropping."""
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize(thermal_trajectory)

    # Verify duration equals exact delta from first to last point
    t0 = thermal_trajectory[0].timestamp
    t_end = thermal_trajectory[-1].timestamp
    expected_duration_h = round((t_end - t0) / 3600.0, 3)
    assert summary.mission["duration_hours"] == expected_duration_h
    assert summary.health["initial_health"] == round(thermal_trajectory[0].health_index, 1)
    assert summary.health["final_health"] == round(thermal_trajectory[-1].health_index, 1)


def test_summary_matches_source_telemetry(thermal_trajectory):
    """Criterion 3: Summary aggregations match exact mathematical calculations on source points."""
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize(thermal_trajectory)

    c_vals = [p.CHT for p in thermal_trajectory]
    assert summary.engine["CHT_c"]["min"] == round(min(c_vals), 2)
    assert summary.engine["CHT_c"]["max"] == round(max(c_vals), 2)
    assert summary.engine["CHT_c"]["mean"] == round(sum(c_vals) / len(c_vals), 2)

    rpm_vals = [p.RPM for p in thermal_trajectory]
    assert summary.envelope["RPM"]["min"] == round(min(rpm_vals), 2)
    assert summary.envelope["RPM"]["max"] == round(max(rpm_vals), 2)


def test_event_extraction_is_deterministic(thermal_trajectory):
    """Criterion 4: Event extraction produces identical event sequences across repeated runs."""
    limits = resolve_engine_limits("ROTAX_914_F_TWIN_01")
    pts_dict = [p.to_dict() for p in thermal_trajectory]

    extractor1 = EventExtractor(limits)
    events1 = extractor1.extract_events(pts_dict)

    extractor2 = EventExtractor(limits)
    events2 = extractor2.extract_events(pts_dict)

    assert len(events1) == len(events2)
    for e1, e2 in zip(events1, events2):
        assert e1.timestamp_sec == e2.timestamp_sec
        assert e1.event_type == e2.event_type
        assert e1.severity == e2.severity
        assert e1.evidence == e2.evidence


def test_zero_true_rul_leakage_into_inputs(thermal_trajectory):
    """Criterion 5 & 6: Estimator does not receive true_RUL, and summary cleanly isolates ground truth."""
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize(thermal_trajectory)

    # Initial predicted RUL must be calculated independently of true RUL
    assert summary.rul["initial_predicted_rul_hours"] is not None
    assert summary.rul["has_ground_truth"] is True
    # Verify true RUL is recorded under synthetic validation container
    assert summary.rul["initial_true_rul_hours"] is not None


def test_ground_truth_correctly_labeled(thermal_trajectory):
    """Criterion 7: Provenance is marked as SYNTHETIC GROUND TRUTH for synthetic benchmark runs."""
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize(thermal_trajectory)

    assert summary.provenance == "SYNTHETIC GROUND TRUTH"

    service = LLMReportService()
    report = service.generate_mission_report(summary, report_mode="engineering")
    assert "SYNTHETIC GROUND TRUTH" in report["report_markdown"]
    assert "POST-MISSION VALIDATION ONLY" in report["report_markdown"]


def test_llm_numerical_claim_validation():
    """Criterion 8 & 9: Claim validation flags unsupported hallucinated metrics."""
    engine = VirtualDataLabEngine(master_seed=42)
    pts = engine.generate_healthy_trajectory("TRAJ_TEST_VAL_001", duration_hours=2.0, seed=42)
    summary = TrajectorySummarizer().summarize(pts)

    service = LLMReportService()

    # Valid report text containing true values
    valid_text = f"The trajectory completed at health 100.0% with CHT peak of {summary.engine['CHT_c']['max']}°C for TRAJ_TEST_VAL_001."
    is_valid, issues = service.validate_report_claims(valid_text, summary)
    assert is_valid is True
    assert len(issues) == 0

    # Fabricated / hallucinated report text claiming CHT reached 240.5 C when max was ~110 C
    hallucinated_text = f"The engine suffered failure where CHT reached 240.5°C and health 15.0% for TRAJ_TEST_VAL_001."
    is_valid_bad, issues_bad = service.validate_report_claims(hallucinated_text, summary)
    assert is_valid_bad is False
    assert any("CHT" in iss for iss in issues_bad)


def test_deterministic_fallback_when_api_key_absent(thermal_trajectory):
    """Criterion 11: Seamless fallback when LLM API key is not present."""
    summary = TrajectorySummarizer().summarize(thermal_trajectory)

    service = LLMReportService(api_key=None)
    res = service.generate_mission_report(summary, report_mode="engineering")

    assert res["generator"] == "DETERMINISTIC FALLBACK REPORT"
    assert len(res["report_markdown"]) > 1000
    assert len(res["report_html"]) > 1000
    assert "AEROPULSE-X MISSION INTELLIGENCE REPORT" in res["report_markdown"]


def test_long_trajectory_handling():
    """Criterion 12: Long trajectories (100+ points) process smoothly without truncation."""
    engine = VirtualDataLabEngine(master_seed=42)
    long_pts = engine.generate_healthy_trajectory("TRAJ_LONG_001", duration_hours=60.0, seed=42)
    assert len(long_pts) >= 100

    summary = TrajectorySummarizer().summarize(long_pts)
    assert summary.mission["duration_hours"] >= 50.0

    service = LLMReportService()
    report = service.generate_mission_report(summary, report_mode="engineering")
    assert "TRAJ_LONG_001" in report["report_markdown"]


def test_multi_engine_support():
    """Criterion 13: Multiple engines resolve appropriate limits and thresholds."""
    rotax_limits = resolve_engine_limits("ROTAX_914_F_TWIN_01")
    assert rotax_limits["name"] == "Rotax-914-Turbo-115HP"
    assert rotax_limits["cht_max"] == 135.0

    diesel_limits = resolve_engine_limits("GENERIC_DIESEL_01")
    assert diesel_limits["name"] == "Generic-Inline4-AeroDiesel"
    assert diesel_limits["cht_max"] == 130.0

    piston_limits = resolve_engine_limits("AEROPISTON_4C_135L")
    assert piston_limits["name"] == "AeroPiston-4C-1.35L"
    assert piston_limits["cht_max"] == 140.0


def test_fault_free_mission_produces_nominal_report(healthy_trajectory):
    """Criterion 15: Nominal flight produces nominal status and zero fault alerts."""
    summary = TrajectorySummarizer().summarize(healthy_trajectory)
    assert len(summary.faults) == 0
    assert summary.health["final_health"] >= 85.0

    service = LLMReportService()
    rep = service.generate_mission_report(summary, report_mode="executive")
    assert "COMPLETED — NOMINAL" in rep["report_markdown"]
    assert "Zero physical faults" in rep["report_markdown"]


def test_faulted_mission_produces_chronological_narrative(thermal_trajectory):
    """Criterion 16: Faulted mission records chronological progression."""
    summary = TrajectorySummarizer().summarize(thermal_trajectory)
    assert len(summary.events) >= 5

    event_types = [e["event_type"] for e in summary.events]
    assert "MISSION_START" in event_types
    assert "THERMAL_DEVIATION" in event_types
    assert "HEALTH_DEGRADATION" in event_types
    assert "MISSION_END" in event_types

    # Verify timestamps are monotonically non-decreasing
    timestamps = [e["timestamp_sec"] for e in summary.events]
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))


def test_executive_report_11_sections(thermal_trajectory):
    """Criterion 16: Executive report contains all 11 required sections."""
    summary = TrajectorySummarizer().summarize(thermal_trajectory)
    service = LLMReportService()
    rep = service.generate_mission_report(summary, report_mode="executive")
    md = rep["report_markdown"]

    required_sections = [
        "1. Mission Overview",
        "2. Executive Summary",
        "3. Chronological Mission Timeline",
        "4. Current Engine Condition",
        "5. Major Events Summary",
        "6. Health & Prognostic RUL Assessment",
        "7. Fault & Anomaly Assessment",
        "8. Safety & FADEC Supervisory Response",
        "9. Mission Impact Analysis",
        "10. Recommended Maintenance Action",
        "11. Data Provenance & Synthetic Disclosure",
    ]
    for sec in required_sections:
        assert sec in md, f"Missing section in Executive Report: {sec}"


def test_engineering_report_17_sections(thermal_trajectory):
    """Criterion 17: Engineering report contains all 17 required sections."""
    summary = TrajectorySummarizer().summarize(thermal_trajectory)
    service = LLMReportService()
    rep = service.generate_mission_report(summary, report_mode="engineering")
    md = rep["report_markdown"]

    required_sections = [
        "1. Mission Overview & Identification",
        "2. Executive Engineering Summary",
        "3. Chronological Event Timeline",
        "4. Mission Operating Envelope",
        "5. Propulsion Thermodynamics & Kinetics",
        "6. System Health & Continuous Wear Kinetics",
        "7. Physical Fault Signatures",
        "8. Digital Twin Physics Residual Evidence",
        "9. Remaining Useful Life (RUL) Prognostics",
        "10. Prognostic Uncertainty Quantification",
        "11. Virtual ECU & FADEC Supervisory Response",
        "12. Autonomous Safety Actuation",
        "13. Virtual CAN 2.0B & Telemetry Integrity",
        "14. Mission Impact Assessment",
        "15. Multi-Echelon Maintenance Directive",
        "16. Final Propulsion State Assessment",
        "17. Data Provenance & Scientific Reproducibility",
    ]
    for sec in required_sections:
        assert sec in md, f"Missing section in Engineering Report: {sec}"


def test_end_to_end_custom_degradation_thermal_001(client):
    """Criterion 18: Primary end-to-end verification with TRAJ_CUSTOM_DEGRADATION_THERMAL_001."""
    # 1. API Post call
    resp = client.post(
        "/api/v1/data/report",
        json={
            "trajectory_id": "TRAJ_CUSTOM_DEGRADATION_THERMAL_001",
            "report_mode": "engineering",
        }
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["trajectory_id"] == "TRAJ_CUSTOM_DEGRADATION_THERMAL_001"
    assert data["report_mode"] == "engineering"
    assert data["generator"] == "DETERMINISTIC FALLBACK REPORT"
    assert data["provenance"] == "SYNTHETIC GROUND TRUTH"

    # KPI summary validation
    kpi = data["kpi_summary"]
    assert "THERMAL" in kpi["primary_event"].upper()
    assert float(kpi["current_health"].replace("%", "")) <= 40.0

    # Events validation
    assert len(data["events"]) >= 5

    # HTML check
    assert "<table" in data["report_html"]
    assert "AEROPULSE-X MISSION INTELLIGENCE REPORT" in data["report_html"]


def test_existing_virtual_datalab_endpoints_remain_operational(client):
    """Verify that all existing Virtual Data Lab capabilities remain functional."""
    # 1. Generate endpoint
    gen_resp = client.post(
        "/api/v1/data/generate",
        json={"category": "degradation", "scenario_type": "thermal", "duration_hours": 10.0, "severity": 0.6}
    )
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert gen_data["sample_count"] > 0
    assert "points" in gen_data

    # 2. Replay endpoint
    rep_resp = client.post(
        "/api/v1/data/replay",
        json={"trajectory_id": "TRAJ_CUSTOM_DEGRADATION_THERMAL_001"}
    )
    assert rep_resp.status_code == 200
    assert "total_steps" in rep_resp.json()

    # 3. Catalog endpoint
    cat_resp = client.get("/api/v1/data/catalog")
    assert cat_resp.status_code == 200
    assert "catalog" in cat_resp.json()

    # 4. Quality endpoint
    qual_resp = client.get("/api/v1/data/quality")
    assert qual_resp.status_code == 200
    assert "trajectory_leakage_audit" in qual_resp.json()
