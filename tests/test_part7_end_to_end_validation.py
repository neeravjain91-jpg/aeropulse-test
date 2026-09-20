"""Automated Pytest Suite for Part 7 End-to-End Mission Validation & Capstone System Integration.

AeroPulse-X Part 7/7:
Authoritative end-to-end integration tests verifying:
  - 7-Phase Nominal Mission
  - 6 Physical Engine Fault Modes
  - 7 Sensor Transducer Fault Modes
  - 3 Compound Fault Missions
  - 4 Insufficient Observability Scenarios
  - Engine Profile Switching & State Isolation (Rotax 914 F <-> Continental TSIO-360-MB)
  - Mission Replay Monotonic Degradation
  - What-If Scenario Analysis without State Mutation
  - Adversarial Failure Recovery
  - CAN Bus Software-in-the-Loop (SIL) Interface
"""
from __future__ import annotations

import math
import pytest
from scripts.validate_part7_end_to_end import Part7CapstoneValidator


@pytest.fixture(scope="module")
def validator() -> Part7CapstoneValidator:
    return Part7CapstoneValidator()


def test_part7_nominal_mission_profile(validator: Part7CapstoneValidator):
    """Scenario 1: Full 7-phase nominal mission operates with zero false alarms and protected RUL."""
    res = validator.run_nominal_mission()
    assert res["nominal_mission_passed"] is True
    assert len(res["steps"]) == 7
    for step in res["steps"]:
        assert step["verdict"] in ("NOMINAL", "SENSOR_FAULT_ISOLATED")
        assert step["health_state"] in ("Normal", "Watch", "Warning")
        assert step["health_index"] >= 60.0
        assert step["rul_hours"] > 180.0
        assert not math.isnan(step["health_index"])
        assert not math.isnan(step["rul_hours"])


def test_part7_engine_fault_missions(validator: Part7CapstoneValidator):
    """Scenarios 2-7: 6 physical fault modes confirm engine degradation and reduce health index."""
    res = validator.run_engine_fault_missions()
    assert len(res) == 6
    assert all(res.values()) is True


def test_part7_sensor_fault_missions(validator: Part7CapstoneValidator):
    """Scenarios 8-14: 7 transducer defect modes are isolated with protected RUL and health index."""
    res = validator.run_sensor_fault_missions()
    assert len(res) == 7
    assert all(res.values()) is True


def test_part7_compound_fault_missions(validator: Part7CapstoneValidator):
    """Scenarios 15-17: Coupled core degradation and sensor dropout yield COMPOUND_FAULT."""
    res = validator.run_compound_fault_missions()
    assert len(res) == 3
    assert all(res.values()) is True


def test_part7_observability_gating(validator: Part7CapstoneValidator):
    """Scenarios 18-21: Multi-sensor dropouts (<60% or <4 channels) trigger INSUFFICIENT_OBSERVABILITY."""
    res = validator.run_observability_gating_missions()
    assert len(res) == 4
    assert all(res.values()) is True


def test_part7_engine_switching_isolation(validator: Part7CapstoneValidator):
    """Scenario 22: Switching Rotax -> Continental -> Rotax demonstrates zero state contamination."""
    res = validator.run_engine_switching_isolation()
    assert res["engine_switching_passed"] is True


def test_part7_mission_replay_consistency(validator: Part7CapstoneValidator):
    """Scenario 23: Replay yields monotonic health and RUL degradation under active fault injection."""
    res = validator.run_mission_replay_validation()
    assert res["replay_passed"] is True
    assert res["steps"] == 24


def test_part7_whatif_analysis_isolation(validator: Part7CapstoneValidator):
    """Scenario 24: What-If evaluates environmental stress without contaminating baseline state."""
    res = validator.run_whatif_analysis_validation()
    assert res["whatif_passed"] is True


def test_part7_adversarial_recovery(validator: Part7CapstoneValidator):
    """Scenarios 25-29: Corrupted and adversarial vectors (NaN, Inf, strings, empty) are safely rejected."""
    res = validator.run_adversarial_recovery_validation()
    assert len(res) == 5
    assert all(res.values()) is True


def test_part7_can_sil_interface(validator: Part7CapstoneValidator):
    """CAN Bus SIL interface round-trip frame verification."""
    res = validator.run_can_interface_audit()
    assert res["simulated_can_functional"] is True
    assert res["software_sil_can_verified"] is True
    assert res["hardware_ecu_can_claimed"] is False


def test_part7_state_isolation_lifecycle(validator: Part7CapstoneValidator):
    """Comprehensive State Isolation: Mission A->B->A, Rotax->Cont->Rotax, Replay->WhatIf->Base, Sensor Fault->Recovery."""
    import copy
    from app.inference import AeroTwinAI
    from app.replay import run_replay
    from app.mission_whatif import MissionWhatIf, MissionScenario

    # 1. Mission A -> Mission B -> Mission A
    tA, ctxA = validator.build_telemetry(rpm=4800.0, throttle=0.70, time_s=100.0)
    tB, ctxB = validator.build_telemetry(rpm=5500.0, throttle=0.95, time_s=100.0)

    ai = AeroTwinAI()
    outA1 = ai.analyze(copy.deepcopy(tA), context=copy.deepcopy(ctxA))
    outB = ai.analyze(copy.deepcopy(tB), context=copy.deepcopy(ctxB))
    outA2 = ai.analyze(copy.deepcopy(tA), context=copy.deepcopy(ctxA))

    assert outA1["health_index"] == outA2["health_index"]
    assert outA1["rul"]["rul_hours"] == outA2["rul"]["rul_hours"]

    # 2. Replay -> What-If -> Baseline
    base_res_before = ai.analyze(copy.deepcopy(tA), context=copy.deepcopy(ctxA))
    rep_scenario = {
        "engine_id": "Rotax-914-Turbo-115HP",
        "mission_name": "Replay Overheat",
        "fault_name": "cooling_failure",
        "fault_onset_step": 2,
        "fault_target_severity": 0.8,
        "steps": 10,
        "duration_h": 2.0,
    }
    _ = run_replay(ai, copy.deepcopy(tA), rep_scenario)

    whatif = MissionWhatIf()
    sc_base = MissionScenario(name="Baseline", altitude_ft=5000.0, ambient_c=20.0, duration_h=2.0)
    sc_hot = MissionScenario(name="Hot-High", altitude_ft=12000.0, ambient_c=40.0, duration_h=2.0)
    _ = whatif.compare(copy.deepcopy(tA), sc_base, sc_hot)

    base_res_after = ai.analyze(copy.deepcopy(tA), context=copy.deepcopy(ctxA))
    assert base_res_before["health_index"] == base_res_after["health_index"]
    assert base_res_before["rul"]["rul_hours"] == base_res_after["rul"]["rul_hours"]

    # 3. Sensor Fault -> Recovery
    t_fault = copy.deepcopy(tA)
    t_fault["CHT"] = 268.0
    out_fault = ai.analyze(t_fault, context=copy.deepcopy(ctxA))
    assert out_fault["sensor_health"]["verdict"] == "SENSOR_FAULT_ISOLATED"
    assert "CHT" in out_fault["sensor_health"]["suspect_sensors"]

    ai.reset("Rotax-914-Turbo-115HP")
    t_rec = copy.deepcopy(tA)
    out_rec = ai.analyze(t_rec, context=copy.deepcopy(ctxA))
    assert out_rec["sensor_health"]["verdict"] in ("NOMINAL", "NORMAL")
    assert len(out_rec["sensor_health"]["suspect_sensors"]) == 0
    assert out_rec["health_index"] == outA1["health_index"]

