"""
Unit and regression test suite verifying Austin (2010) UAV Engineering Reference improvements.
"""
import pytest
import math
from app.engine_config import EngineConfig, ENGINE_PROFILES
from app.engine_model import ReducedOrderPistonEngine, EngineInputs
from app.vibration import VibrationPhysicsModel, VibrationSpectrum
from app.risk import mission_risk, calculate_system_availability
from app.advisory import fault_advisory, generate_echelon_advisory, POWERPLANT_HIERARCHY


def test_multi_cycle_engine_configs():
    """Verify engine configuration profiles across 4-stroke, 2-stroke, rotary, and diesel."""
    assert "AeroPiston-4C-1.35L" in ENGINE_PROFILES
    assert "Rotax-914-Turbo-115HP" in ENGINE_PROFILES
    assert "Generic-Inline4-AeroDiesel" in ENGINE_PROFILES
    assert "Generic-2Stroke-Twin-50HP" in ENGINE_PROFILES
    assert "Generic-Rotary-Wankel-40HP" in ENGINE_PROFILES

    cfg_4s = EngineConfig.rotax_914()
    cfg_2s = EngineConfig.two_stroke_twin()
    cfg_rot = EngineConfig.rotary_wankel()
    cfg_die = EngineConfig.inline4_diesel()

    assert cfg_4s.cycle == "4-stroke"
    assert cfg_2s.cycle == "2-stroke"
    assert cfg_rot.cycle == "rotary"
    assert cfg_die.cycle == "diesel"

    # Austin Ch 6.5.1: 2-stroke has higher BSFC than 4-stroke; Diesel has lowest BSFC
    assert cfg_2s.bsfc_nominal_kg_kwh > cfg_4s.bsfc_nominal_kg_kwh
    assert cfg_die.bsfc_nominal_kg_kwh < cfg_4s.bsfc_nominal_kg_kwh


def test_engine_performance_carpet_mapping():
    """Verify Austin Ch 19.2.3 carpet graph generation (Power & BSFC vs Throttle & RPM)."""
    engine = ReducedOrderPistonEngine(EngineConfig.default_135l())
    carpet = engine.generate_performance_carpet(
        rpm_steps=[2000.0, 3000.0, 4000.0],
        throttle_steps=[0.40, 0.80],
        altitude_ft=5000.0
    )

    assert len(carpet) == 6
    # Higher throttle must yield higher brake power
    p_low_th = [pt["brake_power_kw"] for pt in carpet if pt["throttle"] == 0.40 and pt["rpm"] == 3000.0][0]
    p_high_th = [pt["brake_power_kw"] for pt in carpet if pt["throttle"] == 0.80 and pt["rpm"] == 3000.0][0]
    assert p_high_th > p_low_th


def test_vibration_physics_harmonics():
    """Verify physics-correlated vibration harmonics and cylinder firing frequencies."""
    # 4-cylinder 4-stroke at 3000 RPM -> shaft = 50 Hz, firing = 50 * (4/2) = 100 Hz
    spec_4s = VibrationPhysicsModel.calculate_harmonics(rpm=3000.0, num_cylinders=4, cycle="4-stroke")
    assert spec_4s.shaft_order_1x_hz == 50.0
    assert spec_4s.firing_frequency_hz == 100.0
    assert spec_4s.condition == "NOMINAL_VIBRATION"

    # Misfire condition must trigger vibration anomaly
    spec_misfire = VibrationPhysicsModel.calculate_harmonics(rpm=3000.0, num_cylinders=4, cycle="4-stroke", misfire_fraction=0.35)
    assert spec_misfire.anomaly_flag is True
    assert spec_misfire.condition == "COMBUSTION_MISFIRE_IMBALANCE"


def test_austin_reliability_and_availability():
    """Verify Austin Ch 16 reliability synthesis and availability formula [10^5 - (N*T)]/10000."""
    # If N = 100 failures/100kh, MTTR = 4h -> downtime = 400h -> Availability = (100000 - 400)/1000 = 99.6%
    avail = calculate_system_availability(num_failures_100k=100.0, mttr_hours=4.0)
    assert avail == 99.6

    risk_res = mission_risk(
        analysis={"health_index": 85.0, "twin": {"residual_rms": 0.4}},
        scenario={"altitude_ft": 5000.0, "ambient_c": 25.0, "duration_h": 4.0}
    )

    assert "mtbf_hours" in risk_res
    assert "availability_pct" in risk_res
    assert "safety_tier" in risk_res
    assert risk_res["level"] in ["LOW", "MEDIUM", "HIGH"]


def test_powerplant_failure_hierarchy_and_advisory():
    """Verify fault mapping into Austin 7-subsystem hierarchy and multi-echelon maintenance actions."""
    assert len(POWERPLANT_HIERARCHY) == 7
    assert "FUEL_SYSTEM" in POWERPLANT_HIERARCHY
    assert "COMBUSTION_CORE" in POWERPLANT_HIERARCHY
    assert "LUBRICATION_SYSTEM" in POWERPLANT_HIERARCHY
    assert "COOLING_SYSTEM" in POWERPLANT_HIERARCHY
    assert "MECHANICAL_TRANSMISSION" in POWERPLANT_HIERARCHY
    assert "ELECTRICAL_GENERATION" in POWERPLANT_HIERARCHY
    assert "INSTRUMENTATION_SENSORS" in POWERPLANT_HIERARCHY

    twin_mock = {"z_scores": {"Oil_Pressure": -2.2, "Oil_Temp": 1.8}}
    findings = fault_advisory(telemetry={}, twin=twin_mock)
    actions = generate_echelon_advisory(findings)

    assert len(actions) >= 1
    lub_action = [a for a in actions if a.subsystem_tier == "LUBRICATION_SYSTEM"][0]
    assert lub_action.echelon == "I-Level"
    assert lub_action.dispatch_status == "NO_GO_MAINTENANCE_HOLD"
    assert "TO-UAV-ENG-LUB" in lub_action.technical_order
