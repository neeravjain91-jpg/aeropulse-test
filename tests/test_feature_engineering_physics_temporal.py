"""Regression Test Suite for Physics & Causal Temporal Feature Engineering (Part 3/7).

Validates:
1. Temporal Causality (no future observation leakage)
2. Flight Boundary Isolation (zero cross-flight derivative bleeding)
3. Train-Only Residual Reference (strict anti-leakage parameter fitting)
4. Unit Consistency (physical sensor units match domain expectations)
5. Forbidden Features Exclusion (zero target or simulator ground-truth leakage)
6. Feature Reproducibility (deterministic output across repeated evaluations)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.feature_engineering import (
    PROHIBITED_PREDICTOR_FIELDS,
    PhysicsResidualExtractor,
    assemble_experiment_dataset,
    audit_feature_names,
    extract_cross_sensor_features,
    extract_temporal_features,
)


@pytest.fixture
def synthetic_flight_df():
    """Generates two synthetic flights with distinct telemetry trajectories."""
    rows = []
    # Flight A: 20 points
    for t in range(20):
        rows.append({
            "Flight": "FLIGHT_A",
            "GPS_Time": float(t),
            "Engine_RPM": 4500.0 + 10.0 * t,
            "MAP_Injector": 28.0 + 0.1 * t,
            "CHT": 190.0 + 0.5 * t,
            "EGT1": 1250.0 + 2.0 * t,
            "EGT2": 1260.0 + 2.0 * t,
            "EGT3": 1255.0 + 2.0 * t,
            "Oil_Temp": 160.0 + 0.2 * t,
            "Oil_Pressure": 60.0 - 0.1 * t,
            "Fuel_Flow": 32.0 + 0.2 * t,
            "Battery_Voltage": 27.6,
            "Battery_Current": 12.0,
            "Alternator_Temp": 180.0 + 0.1 * t,
            "EFI_Fuel_Temp": 80.0,
            "EFI_Water_Temp": 85.0,
            "Operating_State": "CRUISE",
            "Health_State": "Normal",
        })
    # Flight B: 20 points
    for t in range(20):
        rows.append({
            "Flight": "FLIGHT_B",
            "GPS_Time": float(t + 100),
            "Engine_RPM": 5200.0 - 15.0 * t,
            "MAP_Injector": 35.0 - 0.2 * t,
            "CHT": 210.0 - 0.4 * t,
            "EGT1": 1320.0 - 3.0 * t,
            "EGT2": 1330.0 - 3.0 * t,
            "EGT3": 1325.0 - 3.0 * t,
            "Oil_Temp": 175.0 - 0.3 * t,
            "Oil_Pressure": 65.0 + 0.1 * t,
            "Fuel_Flow": 45.0 - 0.5 * t,
            "Battery_Voltage": 27.8,
            "Battery_Current": 15.0,
            "Alternator_Temp": 195.0 - 0.2 * t,
            "EFI_Fuel_Temp": 85.0,
            "EFI_Water_Temp": 90.0,
            "Operating_State": "HIGH",
            "Health_State": "Normal",
        })
    return pd.DataFrame(rows)


def test_temporal_causality(synthetic_flight_df):
    """1. Verifies that temporal features at time t depend strictly on observations at tau <= t."""
    df_orig = synthetic_flight_df.copy()
    feats_orig = extract_temporal_features(df_orig, flight_col="Flight")

    # Mutate future samples for Flight A (indices 10 to 19)
    df_mutated = synthetic_flight_df.copy()
    df_mutated.loc[10:19, "Engine_RPM"] = df_mutated.loc[10:19, "Engine_RPM"] + 1000.0
    df_mutated.loc[10:19, "MAP_Injector"] = df_mutated.loc[10:19, "MAP_Injector"] * 2.0
    df_mutated.loc[10:19, "CHT"] = df_mutated.loc[10:19, "CHT"] + 50.0

    feats_mutated = extract_temporal_features(df_mutated, flight_col="Flight")

    # Points 0..9 in Flight A MUST be strictly identical despite future mutations
    for col in feats_orig.columns:
        np.testing.assert_array_almost_equal(
            feats_orig.loc[0:9, col].values,
            feats_mutated.loc[0:9, col].values,
            decimal=6,
            err_msg=f"Causality violation in feature {col}: past features altered by future mutation!",
        )


def test_flight_boundary_isolation(synthetic_flight_df):
    """2. Verifies that temporal differences and trends do not cross flight boundaries."""
    feats = extract_temporal_features(synthetic_flight_df, flight_col="Flight")

    # Index 19 is Flight A last sample; Index 20 is Flight B first sample
    assert synthetic_flight_df.loc[19, "Flight"] == "FLIGHT_A"
    assert synthetic_flight_df.loc[20, "Flight"] == "FLIGHT_B"

    # Flight B first sample MUST have zero diff from Flight A (no cross-flight bleed)
    assert feats.loc[20, "Engine_RPM_dt"] == 0.0, "Cross-flight leakage: RPM_dt is non-zero at flight start!"
    assert feats.loc[20, "MAP_Injector_dt"] == 0.0, "Cross-flight leakage: MAP_dt is non-zero at flight start!"
    assert feats.loc[20, "CHT_dt"] == 0.0, "Cross-flight leakage: CHT_dt is non-zero at flight start!"
    assert feats.loc[20, "Engine_RPM_trend_5s"] == 0.0, "Cross-flight leakage: trend_5s is non-zero at flight start!"


def test_train_only_residual_reference(synthetic_flight_df):
    """3. Verifies that healthy reference statistics are fitted strictly on train set with zero test leakage."""
    train_df = synthetic_flight_df.iloc[:25].copy()
    test_df = synthetic_flight_df.iloc[25:].copy()

    extractor = PhysicsResidualExtractor()
    # Unfitted extractor must raise RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted"):
        extractor.extract_residuals(test_df)

    extractor.fit_healthy_reference(train_df, health_state_col="Health_State", normal_label="Normal")
    assert extractor.is_fitted is True

    # Record fitted medians
    med_rpm_train = extractor.healthy_stats["_GLOBAL_"]["Engine_RPM"]["median"]

    # Mutate test data heavily
    test_df_mut = test_df.copy()
    test_df_mut["Engine_RPM"] = 99999.0

    # Extract residuals on mutated test data
    res = extractor.extract_residuals(test_df_mut)

    # Fitted parameters in extractor MUST remain completely unchanged
    assert extractor.healthy_stats["_GLOBAL_"]["Engine_RPM"]["median"] == med_rpm_train
    # Residual calculation on mutated test data should use train state median
    state_row25 = test_df_mut["Operating_State"].values[0]
    state_med_rpm = extractor.healthy_stats[state_row25]["Engine_RPM"]["median"]
    expected_res = 99999.0 - state_med_rpm
    np.testing.assert_almost_equal(res["Engine_RPM_res_healthy"].values[0], expected_res, decimal=4)


def test_unit_consistency(synthetic_flight_df):
    """4. Verifies physical unit consistency and valid ranges across sensor features."""
    df = synthetic_flight_df.copy()
    cross = extract_cross_sensor_features(df)

    # EGT spread must be non-negative (°F)
    assert (cross["EGT_spread"] >= 0.0).all()
    # EGT mean should lie within plausible bounds (800 - 1500 °F)
    assert ((cross["EGT_mean"] >= 800.0) & (cross["EGT_mean"] <= 1500.0)).all()
    # RPM / MAP ratio should be positive
    assert (cross["RPM_MAP_ratio"] > 0.0).all()
    # Fuel flow per RPM should be positive and small (< 0.1 L/h per RPM)
    assert ((cross["Fuel_Flow_per_RPM"] > 0.0) & (cross["Fuel_Flow_per_RPM"] < 0.1)).all()


def test_forbidden_features_exclusion(synthetic_flight_df):
    """5. Verifies zero prohibited target or ground-truth features in any experiment dataset."""
    extractor = PhysicsResidualExtractor().fit_healthy_reference(synthetic_flight_df)

    for exp_id in ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]:
        X = assemble_experiment_dataset(synthetic_flight_df, exp_id, residual_extractor=extractor)
        is_clean, violations = audit_feature_names(X.columns)
        assert is_clean is True, f"Experiment {exp_id} contains prohibited features: {violations}"

        # Explicitly assert absence of critical forbidden fields
        for forbidden in PROHIBITED_PREDICTOR_FIELDS:
            assert forbidden not in X.columns, f"Prohibited field {forbidden} found in {exp_id} feature set!"


def test_feature_reproducibility(synthetic_flight_df):
    """6. Verifies deterministic, byte-exact feature extraction across repeated evaluations."""
    extractor = PhysicsResidualExtractor().fit_healthy_reference(synthetic_flight_df)

    X1 = assemble_experiment_dataset(synthetic_flight_df, "E6", residual_extractor=extractor)
    X2 = assemble_experiment_dataset(synthetic_flight_df, "E6", residual_extractor=extractor)

    assert X1.shape == X2.shape
    assert list(X1.columns) == list(X2.columns)
    np.testing.assert_array_equal(X1.values, X2.values)
