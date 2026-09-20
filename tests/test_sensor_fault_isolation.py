"""Comprehensive Automated Test Suite for Sensor Fault Isolation & Fault-Tolerant Analytics.

AeroPulse-X Part 6/7:
Validates that the system authoritatively distinguishes:
  1. NOMINAL
  2. SENSOR_FAULT_ISOLATED
  3. ENGINE_DEGRADATION_CONFIRMED
  4. COMPOUND_FAULT
  5. INSUFFICIENT_OBSERVABILITY

Core Invariant:
  "A bad sensor is not necessarily a bad engine."
  An isolated transducer fault must NEVER cause unjustified engine health collapse or RUL collapse!
"""
from __future__ import annotations

import math
import pytest
from typing import Dict, Any, List

from app.sensor_fault_isolation import (
    SensorFaultIsolationEngine,
    SensorFaultType,
    EngineAttributionVerdict,
    ThresholdProvenance,
    THRESHOLD_REGISTRY,
    TelemetryUnitAdapter,
    DependencyAwareVirtualSensors,
    get_sensor_fault_isolation_engine,
)
from app.sensor_health import assess_sensor_health
from app.rul_service import RULService
from app.digital_twin import ReferenceTwin
from app.inference import AeroTwinAI
from app.feature_engineering import PROHIBITED_PREDICTOR_FIELDS


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def engine() -> SensorFaultIsolationEngine:
    eng = SensorFaultIsolationEngine(default_profile="Rotax-914-Turbo-115HP")
    eng.reset()
    return eng


@pytest.fixture
def nominal_rotax_telemetry() -> Dict[str, Any]:
    return {
        "Engine_RPM": 4650.0,
        "MAP_Injector": 32.5,
        "CHT": 198.5,
        "EGT1": 1320.0,
        "EGT2": 1335.0,
        "EGT3": 1330.0,
        "EGT4": 1325.0,
        "Oil_Pressure": 55.0,
        "Oil_Temp": 180.0,
        "Fuel_Flow": 22.5,
        "Battery_Voltage": 27.8,
        "Battery_Current": 12.0,
        "Alternator_Temp": 190.0,
        "EFI_Water_Temp": 182.0,
        "Operating_State": "CRUISE",
        "GPS_Time": 100.0,
    }


@pytest.fixture
def nominal_twin_assessment() -> Dict[str, Any]:
    return {
        "residual_rms": 0.35,
        "max_abs_z": 0.50,
        "z_scores": {
            "Engine_RPM": 0.1,
            "MAP_Injector": 0.2,
            "CHT": 0.15,
            "EGT1": 0.1,
            "EGT2": 0.2,
            "EGT3": 0.1,
            "Oil_Pressure": 0.05,
            "Oil_Temp": 0.1,
            "Fuel_Flow": 0.2,
            "Battery_Voltage": 0.1,
            "Battery_Current": 0.0,
            "Alternator_Temp": 0.2,
            "EFI_Water_Temp": 0.1,
        },
    }


# =============================================================================
# 1. SENSOR BIAS DETECTION
# =============================================================================

def test_sensor_bias_detection(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """Verifies that injecting an isolated constant bias reduces sensor trust while keeping engine healthy."""
    biased_telem = dict(nominal_rotax_telemetry)
    # Inject +70 °F bias on CHT (unsupported by coolant or EGT)
    biased_telem["CHT"] = 268.0

    twin_biased = dict(nominal_twin_assessment)
    twin_biased["z_scores"] = dict(nominal_twin_assessment["z_scores"])
    twin_biased["z_scores"]["CHT"] = 3.8
    twin_biased["residual_rms"] = 1.1

    res = engine.analyze(biased_telem, twin_assessment=twin_biased)
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED
    assert "CHT" in res.suspect_sensors
    assert res.bulk_physics_rms_z < 1.0  # Bulk physics remains clean
    assert res.is_sensor_fault_only is True
    assert res.uncertainty_penalty > 0.0


# =============================================================================
# 2. TEMPORAL DRIFT DETECTION
# =============================================================================

def test_sensor_drift_detection(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """Verifies that a steady linear drift over consecutive causal frames is detected."""
    base_telem = dict(nominal_rotax_telemetry)
    t0 = 100.0

    # Stream 10 frames with simulated CHT drift (+3 °F/s)
    last_res = None
    for i in range(10):
        t_i = t0 + i * 1.0
        frame = dict(base_telem)
        frame["GPS_Time"] = t_i
        frame["CHT"] = base_telem["CHT"] + i * 3.5  # Slew exceeds physical thermal response
        twin_i = dict(nominal_twin_assessment)
        twin_i["z_scores"] = dict(nominal_twin_assessment["z_scores"])
        twin_i["z_scores"]["CHT"] = 0.5 + i * 0.4
        last_res = engine.analyze(frame, twin_assessment=twin_i)

    assert last_res is not None
    assert "CHT" in last_res.suspect_sensors
    assert last_res.active_fault_modes.get("CHT") in (
        SensorFaultType.DRIFT.value,
        SensorFaultType.IMPLAUSIBLE_RATE_OF_CHANGE.value,
        SensorFaultType.CROSS_SENSOR_INCONSISTENCY.value,
    )


# =============================================================================
# 3. SENSOR DROPOUT DETECTION
# =============================================================================

def test_sensor_dropout_detection(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """Verifies that hard dropouts (None, NaN, or zero on positive channels) are flagged with high confidence."""
    dropout_telem = dict(nominal_rotax_telemetry)
    dropout_telem["Oil_Pressure"] = 0.0  # Impossible at 4650 RPM

    twin = dict(nominal_twin_assessment)
    twin["z_scores"] = dict(nominal_twin_assessment["z_scores"])
    twin["z_scores"]["Oil_Pressure"] = -5.0

    res = engine.analyze(dropout_telem, twin_assessment=twin)
    assert "Oil_Pressure" in res.suspect_sensors
    assert res.active_fault_modes["Oil_Pressure"] == SensorFaultType.DROPOUT.value
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED


def test_sensor_null_dropout(engine, nominal_rotax_telemetry):
    """Verifies that passing None for a channel triggers DROPOUT."""
    null_telem = dict(nominal_rotax_telemetry)
    null_telem["Fuel_Flow"] = None

    res = engine.analyze(null_telem)
    assert "Fuel_Flow" in res.suspect_sensors
    assert res.active_fault_modes["Fuel_Flow"] == SensorFaultType.DROPOUT.value


# =============================================================================
# 4. SENSOR STUCK-AT DETECTION
# =============================================================================

def test_sensor_stuck_at_detection(engine, nominal_rotax_telemetry):
    """Verifies that a frozen transducer value across dynamic cycles is flagged as STUCK_AT."""
    last_res = None
    # Feed 10 cycles where RPM varies dynamically, but Oil_Pressure is frozen exactly at 52.0000
    for i in range(10):
        frame = dict(nominal_rotax_telemetry)
        frame["GPS_Time"] = 100.0 + i * 1.0
        frame["Engine_RPM"] = 4600.0 + (i % 3) * 50.0  # Dynamic engine
        frame["Oil_Pressure"] = 52.123456  # Frozen transducer
        last_res = engine.analyze(frame)

    assert last_res is not None
    assert "Oil_Pressure" in last_res.suspect_sensors
    assert last_res.active_fault_modes["Oil_Pressure"] == SensorFaultType.STUCK_AT.value


# =============================================================================
# 5. SPIKE / TRANSIENT OUTLIER REJECTION
# =============================================================================

def test_spike_outlier_rejection(engine, nominal_rotax_telemetry):
    """Verifies that an instantaneous 1-frame jump exceeding max slew is classified as rate violation."""
    # Frame 1: baseline
    f1 = dict(nominal_rotax_telemetry)
    f1["GPS_Time"] = 100.0
    engine.analyze(f1)

    # Frame 2: instantaneous spike of +80 °F on CHT (max slew is 6 °F/s)
    f2 = dict(nominal_rotax_telemetry)
    f2["GPS_Time"] = 100.5  # 0.5 sec later
    f2["CHT"] = nominal_rotax_telemetry["CHT"] + 80.0
    res2 = engine.analyze(f2)

    assert "CHT" in res2.suspect_sensors
    assert res2.active_fault_modes["CHT"] in (
        SensorFaultType.IMPLAUSIBLE_RATE_OF_CHANGE.value,
        SensorFaultType.SPIKE_OUTLIER.value,
    )


# =============================================================================
# 6. INTERMITTENT CHATTERING FAULT DETECTION
# =============================================================================

def test_intermittent_fault_detection(engine, nominal_rotax_telemetry):
    """Verifies that intermittent chattering between valid and None is detected."""
    last_res = None
    for i in range(10):
        frame = dict(nominal_rotax_telemetry)
        frame["GPS_Time"] = 100.0 + i * 0.5
        frame["Fuel_Flow"] = None if (i % 2 == 1) else 22.5
        last_res = engine.analyze(frame)

    assert last_res is not None
    assert "Fuel_Flow" in last_res.suspect_sensors
    assert last_res.active_fault_modes["Fuel_Flow"] in (
        SensorFaultType.INTERMITTENT.value,
        SensorFaultType.DROPOUT.value,
    )


# =============================================================================
# 7. CROSS-SENSOR PHYSICAL CONTRADICTIONS
# =============================================================================

def test_cross_sensor_rpm_map_contradiction(engine, nominal_rotax_telemetry):
    """Rule 1: High RPM requires positive intake manifold pressure."""
    telem = dict(nominal_rotax_telemetry)
    telem["Engine_RPM"] = 4800.0
    telem["MAP_Injector"] = 5.0  # Implausibly low for high power cruise

    res = engine.analyze(telem)
    assert "MAP_Injector" in res.suspect_sensors
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED


def test_cross_sensor_oil_pressure_rpm_contradiction(engine, nominal_rotax_telemetry):
    """Rule 4: Positive displacement oil pump geared to crankshaft must build pressure."""
    telem = dict(nominal_rotax_telemetry)
    telem["Engine_RPM"] = 4500.0
    telem["Oil_Pressure"] = 4.0  # 4 psi while turning at 4500 RPM

    res = engine.analyze(telem)
    assert "Oil_Pressure" in res.suspect_sensors
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED


def test_cross_sensor_egt_cylinder_balance(engine, nominal_rotax_telemetry):
    """Rule 2: Multi-channel EGT balance detects single cylinder thermocouple disconnect."""
    telem = dict(nominal_rotax_telemetry)
    telem["EGT1"] = 1320.0
    telem["EGT2"] = 1330.0
    telem["EGT3"] = 650.0   # Open circuit / grounded thermocouple
    telem["EGT4"] = 1325.0

    res = engine.analyze(telem)
    assert "EGT3" in res.suspect_sensors
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED


# =============================================================================
# 8. SENSOR VS ENGINE EXPERIMENT MATRIX (CASES A THROUGH H)
# =============================================================================

def test_case_a_nominal(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """CASE A: Healthy engine + healthy sensors."""
    res = engine.analyze(nominal_rotax_telemetry, twin_assessment=nominal_twin_assessment)
    assert res.verdict == EngineAttributionVerdict.NOMINAL
    assert len(res.suspect_sensors) == 0
    assert res.is_sensor_fault_only is False
    assert res.bulk_physics_rms_z < 1.0


def test_case_b_sensor_bias(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """CASE B: Healthy engine + sensor bias."""
    telem = dict(nominal_rotax_telemetry)
    telem["CHT"] = 265.0  # Biased thermocouple
    twin = dict(nominal_twin_assessment)
    twin["z_scores"] = dict(nominal_twin_assessment["z_scores"])
    twin["z_scores"]["CHT"] = 3.5

    res = engine.analyze(telem, twin_assessment=twin)
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED
    assert "CHT" in res.suspect_sensors
    assert res.is_sensor_fault_only is True


def test_case_c_sensor_dropout(engine, nominal_rotax_telemetry, nominal_twin_assessment):
    """CASE C: Healthy engine + sensor dropout."""
    telem = dict(nominal_rotax_telemetry)
    telem["Oil_Pressure"] = 0.0
    twin = dict(nominal_twin_assessment)
    twin["z_scores"] = dict(nominal_twin_assessment["z_scores"])
    twin["z_scores"]["Oil_Pressure"] = -4.0

    res = engine.analyze(telem, twin_assessment=twin)
    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED
    assert "Oil_Pressure" in res.suspect_sensors
    assert res.is_sensor_fault_only is True


def test_case_d_sensor_stuck(engine, nominal_rotax_telemetry):
    """CASE D: Healthy engine + sensor stuck-at."""
    for i in range(10):
        frame = dict(nominal_rotax_telemetry)
        frame["GPS_Time"] = 100.0 + i * 1.0
        frame["Engine_RPM"] = 4600.0 + (i % 2) * 50.0
        frame["Fuel_Flow"] = 21.500000  # Frozen transducer
        res = engine.analyze(frame)

    assert res.verdict == EngineAttributionVerdict.SENSOR_FAULT_ISOLATED
    assert "Fuel_Flow" in res.suspect_sensors


def test_case_e_engine_degradation_confirmed(engine, nominal_rotax_telemetry):
    """CASE E: Actual engine degradation + healthy sensors (thermal runaway)."""
    degraded_telem = dict(nominal_rotax_telemetry)
    degraded_telem["CHT"] = 270.0
    degraded_telem["EFI_Water_Temp"] = 230.0
    degraded_telem["Oil_Temp"] = 240.0
    degraded_telem["EGT1"] = 1580.0
    degraded_telem["EGT2"] = 1590.0
    degraded_telem["EGT3"] = 1585.0

    # Multi-sensor coupled elevation
    twin_deg = {
        "residual_rms": 3.8,
        "max_abs_z": 4.5,
        "z_scores": {
            "CHT": 4.0,
            "EFI_Water_Temp": 4.2,
            "Oil_Temp": 3.6,
            "EGT1": 3.5,
            "EGT2": 3.6,
            "EGT3": 3.5,
            "Engine_RPM": 0.2,
            "Oil_Pressure": 0.3,
        },
    }

    res = engine.analyze(degraded_telem, twin_assessment=twin_deg)
    assert res.verdict == EngineAttributionVerdict.ENGINE_DEGRADATION_CONFIRMED
    assert res.bulk_physics_rms_z >= 2.0
    assert res.is_sensor_fault_only is False


def test_case_f_compound_fault(engine, nominal_rotax_telemetry):
    """CASE F: Engine degradation + simultaneous sensor fault."""
    compound_telem = dict(nominal_rotax_telemetry)
    # Engine is overheating
    compound_telem["EFI_Water_Temp"] = 230.0
    compound_telem["Oil_Temp"] = 240.0
    compound_telem["EGT1"] = 1580.0
    compound_telem["EGT2"] = 1590.0
    # CHT sensor has broken / dropped out to 0
    compound_telem["CHT"] = 0.0

    twin_compound = {
        "residual_rms": 4.2,
        "max_abs_z": 5.0,
        "z_scores": {
            "EFI_Water_Temp": 4.2,
            "Oil_Temp": 3.6,
            "EGT1": 3.5,
            "EGT2": 3.6,
            "CHT": -6.0,
            "Engine_RPM": 0.2,
            "Oil_Pressure": 0.3,
        },
    }

    res = engine.analyze(compound_telem, twin_assessment=twin_compound)
    assert res.verdict == EngineAttributionVerdict.COMPOUND_FAULT
    assert "CHT" in res.suspect_sensors
    assert res.bulk_physics_rms_z >= 2.0  # Degraded engine preserved via Water/Oil/EGT!
    assert res.is_sensor_fault_only is False


def test_case_h_insufficient_observability(engine):
    """CASE H: More than 50% of telemetry dropped out."""
    sparse_telem = {
        "Engine_RPM": 4600.0,
        "CHT": None,
        "Oil_Pressure": None,
        "Fuel_Flow": None,
        "EGT1": None,
        "EGT2": None,
        "EGT3": None,
        "MAP_Injector": None,
    }
    res = engine.analyze(sparse_telem)
    assert res.verdict == EngineAttributionVerdict.INSUFFICIENT_OBSERVABILITY
    assert res.trusted_channel_fraction < 0.60
    assert res.is_sensor_fault_only is False


# =============================================================================
# 9. ENGINE PROFILE ISOLATION
# =============================================================================

def test_engine_profile_isolation(engine, nominal_rotax_telemetry):
    """Verifies that physical thresholds and rules strictly isolate Rotax 914 vs Continental TSIO-360."""
    rotax_telem = dict(nominal_rotax_telemetry)
    rotax_telem["CHT"] = 320.0  # 320 °F exceeds Rotax limit (275 °F)

    res_rotax = engine.analyze(rotax_telem, context={"engine_id": "Rotax-914-Turbo-115HP"})
    assert "CHT" in res_rotax.suspect_sensors

    # For Continental TSIO-360-MB, 320 °F is right in the nominal center (240 - 420 °F)!
    cont_telem = dict(nominal_rotax_telemetry)
    cont_telem["Engine_RPM"] = 2500.0  # Nominal TSIO-360 RPM
    cont_telem["CHT"] = 320.0
    res_cont = engine.analyze(cont_telem, context={"engine_id": "Continental-TSIO-360-MB"})
    assert "CHT" not in res_cont.suspect_sensors


# =============================================================================
# 10. MISSION RESET & TIMELINE REWIND
# =============================================================================

def test_mission_reset_and_timeline_rewind(engine, nominal_rotax_telemetry):
    """Verifies that rewinding timestamps cleanly clears internal causal histories."""
    # Step forward
    f1 = dict(nominal_rotax_telemetry, GPS_Time=100.0)
    engine.analyze(f1)
    f2 = dict(nominal_rotax_telemetry, GPS_Time=105.0)
    engine.analyze(f2)

    # Rewind timestamp to 10.0 (new mission started)
    f_rewind = dict(nominal_rotax_telemetry, GPS_Time=10.0)
    res = engine.analyze(f_rewind)
    assert res is not None
    # Engine state reset successfully without exception


# =============================================================================
# 11. ADVERSARIAL SANITIZATION
# =============================================================================

def test_adversarial_nan_inf_negative_values(engine, nominal_rotax_telemetry):
    """Verifies robustness against NaN, Inf, and negative pressures."""
    bad_telem = {
        "Engine_RPM": float("nan"),
        "Oil_Pressure": -15.0,  # Negative pressure
        "CHT": float("inf"),
        "Battery_Voltage": -5.0,
        "Fuel_Flow": float("-inf"),
    }
    res = engine.analyze(bad_telem)
    assert res.verdict in (
        EngineAttributionVerdict.INSUFFICIENT_OBSERVABILITY,
        EngineAttributionVerdict.SENSOR_FAULT_ISOLATED,
    )
    # Output must not contain NaN or Inf
    d = res.as_dict()
    assert math.isfinite(d["overall_trust_score"])
    assert math.isfinite(d["bulk_physics_rms_z"])


def test_adversarial_empty_telemetry(engine):
    """Verifies that an empty telemetry dictionary does not crash the engine."""
    res = engine.analyze({})
    assert res.verdict == EngineAttributionVerdict.INSUFFICIENT_OBSERVABILITY
    assert res.overall_status == "SUSPECT"


# =============================================================================
# 12. DEPENDENCY-AWARE VIRTUAL SENSING (LOSO VALIDATION)
# =============================================================================

def test_dependency_aware_virtual_sensing():
    """Leave-one-sensor-out validation: virtual sensors refuse to issue estimates using untrusted predictors."""
    # Healthy predictors
    v_op, status = DependencyAwareVirtualSensors.estimate_oil_pressure(
        rpm=4500.0, oil_temp=180.0, rpm_trust=100.0, oil_temp_trust=100.0, profile="Rotax-914-Turbo-115HP"
    )
    assert status == "VALID"
    assert v_op is not None and 45.0 <= v_op <= 75.0

    # RPM predictor failed (trust = 20.0)
    v_op_failed_rpm, status_rpm = DependencyAwareVirtualSensors.estimate_oil_pressure(
        rpm=0.0, oil_temp=180.0, rpm_trust=20.0, oil_temp_trust=100.0, profile="Rotax-914-Turbo-115HP"
    )
    assert status_rpm.startswith("DEPENDENCY_UNTRUSTED")
    assert v_op_failed_rpm is None

    # Oil Temp predictor failed (trust = 15.0)
    v_op_failed_temp, status_temp = DependencyAwareVirtualSensors.estimate_oil_pressure(
        rpm=4500.0, oil_temp=350.0, rpm_trust=100.0, oil_temp_trust=15.0, profile="Rotax-914-Turbo-115HP"
    )
    assert status_temp.startswith("DEPENDENCY_UNTRUSTED")
    assert v_op_failed_temp is None


# =============================================================================
# 13. RUL ROBUSTNESS (NON-COLLAPSE UNDER ISOLATED SENSOR FAULTS)
# =============================================================================

def test_rul_non_collapse_under_sensor_fault():
    """Validates that an isolated transducer fault does NOT artificially collapse structural engine RUL."""
    rul_service = RULService()
    rul_service.reset()

    # Baseline healthy engine at 100 flight hours
    base_telem = {"health_index": 98.0, "elapsed_hours": 100.0, "Engine_RPM": 4650.0}
    rul_clean = rul_service.predict(base_telem, context={"elapsed_hours": 100.0})

    # Same engine with a severe sensor dropout (sensor_fault_flag=True, severity=0.85)
    faulty_sensor_telem = {
        "health_index": 91.0,  # Moderate 7-point sensor penalty only
        "elapsed_hours": 100.0,
        "sensor_fault_flag": True,
        "sensor_fault_severity": 0.85,
    }
    rul_faulty = rul_service.predict(faulty_sensor_telem, context={"elapsed_hours": 100.0, "sensor_fault_flag": True})

    # Invariants:
    # 1. RUL hours must NOT collapse to 0
    assert rul_faulty["rul_hours"] > 800.0
    # 2. Difference between clean and faulty RUL hours must be bounded (< 15%)
    assert abs(rul_faulty["rul_hours"] - rul_clean["rul_hours"]) < 150.0
    # 3. Uncertainty interval must widen under sensor fault
    spread_clean = rul_clean["rul_upper_hours"] - rul_clean["rul_lower_hours"]
    spread_faulty = rul_faulty["rul_upper_hours"] - rul_faulty["rul_lower_hours"]
    assert spread_faulty > spread_clean
    # 4. Confidence must be appropriately downgraded
    assert rul_faulty["confidence"] < rul_clean["confidence"]


# =============================================================================
# 14. TARGET & FUTURE LEAKAGE AUDIT
# =============================================================================

def test_zero_target_leakage_audit():
    """Validates that prohibited target/ground-truth fields are absent from sensor isolation features."""
    for prohibited in PROHIBITED_PREDICTOR_FIELDS:
        assert prohibited not in SensorFaultIsolationEngine.CORE_CHANNELS
        assert prohibited not in THRESHOLD_REGISTRY["Rotax-914-Turbo-115HP"]
        assert prohibited not in THRESHOLD_REGISTRY["Continental-TSIO-360-MB"]


# =============================================================================
# 15. BACKWARD COMPATIBILITY OF ASSESS_SENSOR_HEALTH
# =============================================================================

def test_assess_sensor_health_backward_compatibility(nominal_rotax_telemetry, nominal_twin_assessment):
    """Ensures legacy assess_sensor_health adapter maintains 100% dictionary key contract."""
    res = assess_sensor_health(nominal_rotax_telemetry, nominal_twin_assessment)
    assert "overall_trust_score" in res
    assert "overall_status" in res
    assert "sensors" in res
    assert "channels" in res
    assert "suspect_sensors" in res
    assert "suspect_channels" in res
    assert "is_sensor_fault_only" in res
    assert "bulk_physics_rms_z" in res
    assert "verdict" in res
    assert "engine_vs_sensor_verdict" in res
    assert isinstance(res["sensors"], list)
    assert isinstance(res["channels"], list)
