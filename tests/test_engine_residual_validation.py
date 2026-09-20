"""Regression Test Suite for Engine-Specific Residual Validation (Part 4/7).

Validates:
1. Engine Identity (TSIO-360 vs Rotax 914 F specifications and canonical resolution)
2. Residual Engine Matching (domain isolation and bias reduction with matched engine profile)
3. Unit Consistency (all thermodynamic channels in native tested units: deg F, inHg, psi, L/h)
4. Residual Reproducibility (deterministic output across repeated evaluations)
5. Forbidden Feature Leakage (zero target, ground truth, or future label leakage)
6. Fault Direction Validation (empirical sign/magnitude agreement for 6 fault mechanisms)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.degradation_model import ContinuousDegradationModel, DegradationState
from app.engine_config import EngineConfig, ENGINE_PROFILES, resolve_canonical_engine_id
from app.engine_model import EngineInputs, ReducedOrderPistonEngine
from app.feature_engineering import (
    PROHIBITED_PREDICTOR_FIELDS,
    PhysicsResidualExtractor,
    assemble_experiment_dataset,
    audit_feature_names,
)


@pytest.fixture
def sample_aces_telemetry():
    """Synthetic slice representing ACES cruise telemetry."""
    return pd.DataFrame([
        {
            "Flight": "aces1am_2002_191",
            "GPS_Time": 1000.0 + i,
            "Engine_RPM": 4550.0 + 10.0 * np.sin(i),
            "MAP_Injector": 28.5 + 0.2 * np.cos(i),
            "CHT": 202.0 + 0.5 * np.sin(i * 0.1),
            "EGT1": 1285.0 + 2.0 * np.cos(i * 0.1),
            "EGT2": 1290.0 + 2.0 * np.sin(i * 0.1),
            "EGT3": 1288.0 + 2.0 * np.cos(i * 0.1),
            "Oil_Temp": 168.0 + 0.1 * i,
            "Oil_Pressure": 60.5 - 0.05 * i,
            "Fuel_Flow": 34.5 + 0.1 * np.sin(i),
            "Battery_Voltage": 27.6,
            "Battery_Current": 12.5,
            "Alternator_Temp": 185.0,
            "EFI_Fuel_Temp": 80.0,
            "EFI_Water_Temp": 85.0,
            "Ambient_Temp": 5.0,
            "Operating_State": "CRUISE",
            "Health_State": "Normal",
        }
        for i in range(30)
    ])


def test_engine_identity_and_specifications():
    """1. Verifies physical parameters and canonical resolution for TSIO-360 and Rotax 914."""
    # Continental TSIO-360-MB Profile
    tsio = EngineConfig.continental_tsio_360()
    assert tsio.name == "Continental-TSIO-360-MB"
    assert tsio.displacement_l == 5.89
    assert tsio.num_cylinders == 6
    assert tsio.base_power_kw == 156.6
    assert tsio.nominal_rpm == 2700.0
    assert tsio.layout == "opposed"
    assert tsio.compression_ratio == 7.5

    # Rotax 914 Profile
    rotax = EngineConfig.rotax_914()
    assert rotax.name == "Rotax-914-Turbo-115HP"
    assert rotax.displacement_l == 1.211
    assert rotax.num_cylinders == 4
    assert rotax.base_power_kw == 84.5
    assert rotax.nominal_rpm == 5500.0

    # Canonical alias resolution
    assert resolve_canonical_engine_id("ACES") == "Continental-TSIO-360-MB"
    assert resolve_canonical_engine_id("TSIO-360") == "Continental-TSIO-360-MB"
    assert resolve_canonical_engine_id("Continental") == "Continental-TSIO-360-MB"
    assert resolve_canonical_engine_id("ROTAX_914_F_TWIN_01") == "Rotax-914-Turbo-115HP"
    assert resolve_canonical_engine_id("rotax 914") == "Rotax-914-Turbo-115HP"


def test_residual_engine_matching(sample_aces_telemetry):
    """2. Verifies that TSIO-360 engine profile reduces oil pressure bias compared to generic model."""
    df = sample_aces_telemetry

    engine_gen = ReducedOrderPistonEngine(EngineConfig.default_135l())
    engine_tsio = ReducedOrderPistonEngine(EngineConfig.continental_tsio_360())

    row = df.iloc[0]
    inp = EngineInputs(rpm=float(row["Engine_RPM"]), throttle=0.65, altitude_ft=3000.0, ambient_c=float(row["Ambient_Temp"]))

    p_gen = engine_gen.predict(inp)
    p_tsio = engine_tsio.predict(inp)

    obs_op = float(row["Oil_Pressure"])  # ~60.5 psi
    bias_gen = obs_op - p_gen["Oil_Pressure"]
    bias_tsio = obs_op - p_tsio["Oil_Pressure"]

    # Generic model has severe bias (-10 psi) due to 1.35L 3000-RPM scaling on 4500+ RPM data
    assert abs(bias_gen) > 8.0, f"Generic bias unexpectedly small: {bias_gen}"
    # Engine-matched TSIO-360 reduces oil pressure bias significantly
    assert abs(bias_tsio) < abs(bias_gen), f"TSIO-360 did not reduce bias: tsio={bias_tsio}, gen={bias_gen}"


def test_unit_consistency_comprehensive(sample_aces_telemetry):
    """3. Verifies that all predicted and observed variables share matching physical units."""
    tsio = ReducedOrderPistonEngine(EngineConfig.continental_tsio_360())
    inp = EngineInputs(rpm=2600.0, throttle=0.65, altitude_ft=5000.0, ambient_c=15.0)
    pred = tsio.predict(inp)

    # Temperature units: deg F
    assert 150.0 < pred["CHT"] < 260.0, f"CHT out of plausible deg F range: {pred['CHT']}"
    assert 1100.0 < pred["EGT1"] < 1450.0, f"EGT1 out of plausible deg F range: {pred['EGT1']}"
    assert 130.0 < pred["Oil_Temp"] < 220.0, f"Oil_Temp out of plausible deg F range: {pred['Oil_Temp']}"

    # Pressure units: inHg and psi
    assert 15.0 < pred["MAP_Injector"] < 45.0, f"MAP out of plausible inHg range: {pred['MAP_Injector']}"
    assert 30.0 < pred["Oil_Pressure"] < 80.0, f"Oil_Pressure out of plausible psi range: {pred['Oil_Pressure']}"

    # Fuel flow: L/h
    assert 10.0 < pred["Fuel_Flow"] < 60.0, f"Fuel_Flow out of plausible L/h range: {pred['Fuel_Flow']}"


def test_residual_reproducibility(sample_aces_telemetry):
    """4. Verifies deterministic, byte-exact residual extraction across repeated evaluations."""
    extractor = PhysicsResidualExtractor(EngineConfig.continental_tsio_360()).fit_healthy_reference(sample_aces_telemetry)

    res1 = extractor.extract_residuals(sample_aces_telemetry)
    res2 = extractor.extract_residuals(sample_aces_telemetry)

    assert res1.shape == res2.shape
    assert list(res1.columns) == list(res2.columns)
    np.testing.assert_array_equal(res1.values, res2.values)


def test_forbidden_feature_leakage_in_residuals(sample_aces_telemetry):
    """5. Verifies zero prohibited target or ground-truth features in extracted residuals."""
    extractor = PhysicsResidualExtractor(EngineConfig.continental_tsio_360()).fit_healthy_reference(sample_aces_telemetry)
    res = extractor.extract_residuals(sample_aces_telemetry)

    is_clean, violations = audit_feature_names(res.columns.tolist())
    assert is_clean is True, f"Prohibited leakage features found in residuals: {violations}"

    for forbidden in PROHIBITED_PREDICTOR_FIELDS:
        assert forbidden not in res.columns, f"Prohibited field {forbidden} present in residuals!"


def test_fault_direction_validation_all_modes():
    """6. Verifies empirical sign and magnitude relationships for all 6 physical fault modes."""
    deg_model = ContinuousDegradationModel()
    nominal = {
        "Engine_RPM": 4800.0,
        "MAP_Injector": 28.5,
        "CHT": 195.0,
        "EGT1": 1280.0,
        "EGT2": 1280.0,
        "EGT3": 1280.0,
        "Oil_Temp": 165.0,
        "Oil_Pressure": 60.0,
        "Fuel_Flow": 32.0,
        "Battery_Voltage": 28.0,
        "Battery_Current": 15.0,
        "Alternator_Temp": 180.0,
        "Vibration": 1.15,
        "Efficiency": 0.32,
        "EFI_Water_Temp": 85.0,
    }
    sev = 0.50

    # 1. Overheating
    th = deg_model.apply(nominal, DegradationState(thermal=sev))
    assert th["CHT"] > nominal["CHT"], "Overheating must increase CHT"
    assert th["Oil_Temp"] > nominal["Oil_Temp"], "Overheating must increase Oil_Temp"
    assert th["EGT1"] > nominal["EGT1"], "Overheating must increase EGT"

    # 2. Lubrication
    lub = deg_model.apply(nominal, DegradationState(lubrication=sev))
    assert lub["Oil_Pressure"] < nominal["Oil_Pressure"], "Lubrication degradation must drop Oil_Pressure"
    assert lub["Oil_Temp"] > nominal["Oil_Temp"], "Lubrication degradation must elevate Oil_Temp"
    assert lub["Vibration"] > nominal["Vibration"], "Lubrication degradation must elevate Vibration"

    # 3. Misfire
    mis = deg_model.apply(nominal, DegradationState(misfire=sev))
    assert mis["EGT1"] < nominal["EGT1"], "Misfire must sharply drop EGT on affected cylinder"
    assert mis["Vibration"] > nominal["Vibration"], "Misfire must create torque ripple vibration"
    assert mis["Engine_RPM"] < nominal["Engine_RPM"], "Misfire must drop crankshaft RPM"

    # 4. Injector Abnormality
    inj = deg_model.apply(nominal, DegradationState(injector=sev))
    assert inj["Fuel_Flow"] < nominal["Fuel_Flow"], "Injector clogging must reduce fuel delivery"
    spread = max(inj["EGT1"], inj["EGT2"], inj["EGT3"]) - min(inj["EGT1"], inj["EGT2"], inj["EGT3"])
    assert spread > 20.0, "Injector fault must create EGT cylinder imbalance"

    # 5. Mechanical Degradation
    mech = deg_model.apply(nominal, DegradationState(mechanical=sev))
    assert mech["Vibration"] > nominal["Vibration"], "Mechanical wear must elevate vibration"
    assert mech["Engine_RPM"] < nominal["Engine_RPM"], "Mechanical drag must reduce RPM"

    # 6. Electrical Degradation
    elec = deg_model.apply(nominal, DegradationState(electrical=sev))
    assert elec["Battery_Voltage"] < nominal["Battery_Voltage"], "Electrical fault must cause bus voltage sag"
    assert elec["Alternator_Temp"] > nominal["Alternator_Temp"], "Electrical diode fault must overheat alternator"
