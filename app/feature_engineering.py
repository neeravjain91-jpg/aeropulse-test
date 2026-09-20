"""Physics-Informed and Causal Temporal Feature Engineering Pipeline.

AeroPulse-X: Advanced MALE UAV Propulsion Diagnostics.
Adheres strictly to DATA_RESOURCE_SPEC.md and anti-leakage rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .digital_twin import PARAMS
from .engine_config import EngineConfig
from .engine_model import EngineInputs, ReducedOrderPistonEngine

# Prohibited target and simulator ground-truth fields (Anti-Leakage Quality Gate)
PROHIBITED_PREDICTOR_FIELDS: Set[str] = {
    "Health_State",
    "Degradation_Severity",
    "fault_severity",
    "target_severity",
    "true_RUL",
    "true_failure_time",
    "degradation_stage",
    "fault_present",
    "fault_type",
    "failure_mode",
    "sensor_fault_present",
    "sensor_fault_type",
    "predicted_RUL",  # downstream prognostic target
    "RUL_lower",
    "RUL_upper",
    "RUL_confidence",
    # Leakage exclusion fields from Part 1/2 audit
    "Robust_Anomaly_Score",
    "Robust_Max_Deviation",
    "Sensors_Above_2Sigma",
    "Sensors_Above_3Sigma",
}

# Baseline features (E0)
BASELINE_NUMERIC_FEATURES: List[str] = [
    "Engine_RPM", "EGT1", "EGT2", "EGT3", "CHT", "Fuel_Flow",
    "Oil_Temp", "Oil_Pressure", "Battery_Voltage", "Battery_Current",
    "Alternator_Temp", "EFI_Fuel_Temp", "EFI_Water_Temp", "MAP_Injector",
]
BASELINE_CATEGORICAL_FEATURES: List[str] = ["Operating_State"]
BASELINE_ALL_FEATURES: List[str] = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES


def audit_feature_names(feature_names: Sequence[str]) -> Tuple[bool, List[str]]:
    """Audits a feature list to ensure zero forbidden target or leakage columns are present."""
    violations = [f for f in feature_names if f in PROHIBITED_PREDICTOR_FIELDS or f.endswith("_rz")]
    return len(violations) == 0, violations


def extract_temporal_features(
    df: pd.DataFrame,
    flight_col: str = "Flight",
    time_col: str = "GPS_Time",
) -> pd.DataFrame:
    """Computes causal temporal derivatives and trends strictly partitioned by flight.

    Physical rationale:
    - 1 Hz continuous airborne telemetry.
    - Fast thermodynamic/electrical channels: dX/dt, d^2X/dt^2, 5-sample trends.
    - Thermal inertia channels (CHT, Oil_Temp, Alternator_Temp): 10-sample and 20-sample trends.
    - Strictly causal: only current and past samples within the SAME flight are used.
    - Zero cross-flight boundary bleeding.
    """
    temporal_dfs = []
    has_flight = flight_col in df.columns

    groups = df.groupby(flight_col, sort=False) if has_flight else [(None, df)]

    for _, group in groups:
        g_temp = pd.DataFrame(index=group.index)

        # 1. Fast mechanical/induction channels: RPM & MAP (dX/dt, d^2X/dt^2, 5s trend)
        for col in ["Engine_RPM", "MAP_Injector"]:
            if col in group.columns:
                s = group[col].astype(float)
                dt1 = s.diff().fillna(0.0)
                dt2 = dt1.diff().fillna(0.0)
                trend5 = (s - s.shift(5).bfill()).fillna(0.0)

                g_temp[f"{col}_dt"] = dt1
                g_temp[f"{col}_d2t"] = dt2
                g_temp[f"{col}_trend_5s"] = trend5

        # 2. Fast fuel & hydraulic channels: Fuel_Flow & Oil_Pressure
        for col in ["Fuel_Flow", "Oil_Pressure"]:
            if col in group.columns:
                s = group[col].astype(float)
                g_temp[f"{col}_dt"] = s.diff().fillna(0.0)
                g_temp[f"{col}_trend_5s"] = (s - s.shift(5).bfill()).fillna(0.0)
                if col == "Fuel_Flow":
                    g_temp[f"{col}_trend_10s"] = (s - s.shift(10).bfill()).fillna(0.0)

        # 3. Fast electrical channels: Battery_Voltage & Battery_Current
        for col in ["Battery_Voltage", "Battery_Current"]:
            if col in group.columns:
                s = group[col].astype(float)
                g_temp[f"{col}_dt"] = s.diff().fillna(0.0)
                g_temp[f"{col}_trend_5s"] = (s - s.shift(5).bfill()).fillna(0.0)

        # 4. Combustion exhaust thermocouple channels: EGT1, EGT2, EGT3 (3-5s thermal lag)
        for col in ["EGT1", "EGT2", "EGT3"]:
            if col in group.columns:
                s = group[col].astype(float)
                g_temp[f"{col}_dt"] = s.diff().fillna(0.0)
                g_temp[f"{col}_trend_5s"] = (s - s.shift(5).bfill()).fillna(0.0)
                g_temp[f"{col}_trend_10s"] = (s - s.shift(10).bfill()).fillna(0.0)

        # 5. Slow thermal mass channels: CHT, Oil_Temp, Alternator_Temp (10-60s thermal time constants)
        if "CHT" in group.columns:
            s = group["CHT"].astype(float)
            g_temp["CHT_dt"] = s.diff().fillna(0.0)
            g_temp["CHT_trend_10s"] = (s - s.shift(10).bfill()).fillna(0.0)
            g_temp["CHT_trend_20s"] = (s - s.shift(20).bfill()).fillna(0.0)

        for col in ["Oil_Temp", "Alternator_Temp"]:
            if col in group.columns:
                s = group[col].astype(float)
                g_temp[f"{col}_dt"] = s.diff().fillna(0.0)
                g_temp[f"{col}_trend_20s"] = (s - s.shift(20).bfill()).fillna(0.0)

        temporal_dfs.append(g_temp)

    result = pd.concat(temporal_dfs, axis=0).reindex(df.index)
    return result


def extract_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physically grounded cross-sensor interaction features.

    Physical Rationale:
    1. EGT Spread: max(EGT) - min(EGT). Measures cylinder combustion imbalance (injector clog, misfire).
    2. EGT Mean: Mean of active cylinder exhaust temperatures. Reflects global equivalence ratio (AFR).
    3. Per-Cylinder EGT Deviations: EGT_i - EGT_mean. Pinpoints localized rich/lean anomalies.
    4. RPM / MAP Relationship: RPM / (MAP + eps). Intake air charging efficiency vs engine speed.
    5. Fuel Flow / RPM Relationship: Fuel_Flow / (RPM + eps). Fuel mass delivery per revolution.
    6. Oil Pressure / RPM Relationship: Oil_Pressure / (RPM + eps). Positive displacement pump delivery; detects bearing wear/cavitation.
    7. Thermal Rate Coupling: CHT - Oil_Temp. Head convective cooling vs crankcase oil cooler heat rejection.
    8. EGT to CHT Ratio: EGT_mean / (CHT + eps). Combustion flame heat vs cylinder conduction; separates ignition timing from detonation.
    9. Electrical Bus Power: Battery_Voltage * Battery_Current. Alternator delivery vs battery drain.
    10. Alternator Thermal Stress: Alternator_Temp - Ambient_Temp (if available) or Alternator_Temp vs Voltage.
    """
    cross = pd.DataFrame(index=df.index)

    # 1. EGT Cylinder Relationships
    egt_cols = [c for c in ["EGT1", "EGT2", "EGT3"] if c in df.columns]
    if len(egt_cols) >= 2:
        egt_mat = df[egt_cols].astype(float)
        egt_max = egt_mat.max(axis=1)
        egt_min = egt_mat.min(axis=1)
        egt_mean = egt_mat.mean(axis=1)

        cross["EGT_spread"] = egt_max - egt_min
        cross["EGT_mean"] = egt_mean
        for c in egt_cols:
            cross[f"{c}_deviation"] = egt_mat[c] - egt_mean
        cross["EGT_max_abs_deviation"] = (egt_mat.sub(egt_mean, axis=0)).abs().max(axis=1)

    # 2. RPM / MAP Relationship (Engine breathing & manifold induction)
    if "Engine_RPM" in df.columns and "MAP_Injector" in df.columns:
        rpm = df["Engine_RPM"].astype(float)
        map_p = df["MAP_Injector"].astype(float).clip(lower=1.0)
        cross["RPM_MAP_ratio"] = rpm / map_p

    # 3. Fuel Flow per Crankshaft Revolution
    if "Fuel_Flow" in df.columns and "Engine_RPM" in df.columns:
        rpm = df["Engine_RPM"].astype(float).clip(lower=100.0)
        ff = df["Fuel_Flow"].astype(float)
        cross["Fuel_Flow_per_RPM"] = ff / rpm

    # 4. Oil Pressure per Crankshaft Revolution
    if "Oil_Pressure" in df.columns and "Engine_RPM" in df.columns:
        rpm = df["Engine_RPM"].astype(float).clip(lower=100.0)
        op = df["Oil_Pressure"].astype(float)
        cross["Oil_Pressure_per_RPM"] = op / rpm

    # 5. Thermal Decoupling: CHT vs Oil_Temp
    if "CHT" in df.columns and "Oil_Temp" in df.columns:
        cht = df["CHT"].astype(float)
        ot = df["Oil_Temp"].astype(float)
        cross["CHT_Oil_Temp_delta"] = cht - ot

    # 6. EGT to CHT Ratio (Combustion Flame vs Conduction)
    if "EGT_mean" in cross.columns and "CHT" in df.columns:
        cht = df["CHT"].astype(float).clip(lower=10.0)
        cross["EGT_to_CHT_ratio"] = cross["EGT_mean"] / cht
    elif "EGT1" in df.columns and "CHT" in df.columns:
        cht = df["CHT"].astype(float).clip(lower=100.0)
        cross["EGT_to_CHT_ratio"] = df["EGT1"].astype(float) / cht

    # 7. Electrical Power & Alternator Thermal Load
    if "Battery_Voltage" in df.columns and "Battery_Current" in df.columns:
        bv = df["Battery_Voltage"].astype(float)
        bi = df["Battery_Current"].astype(float)
        cross["Electrical_Power_est"] = bv * bi

    if "Alternator_Temp" in df.columns:
        at = df["Alternator_Temp"].astype(float)
        if "Ambient_Temp" in df.columns:
            amb = df["Ambient_Temp"].astype(float)
            cross["Alternator_Thermal_Stress"] = at - amb
        elif "Oil_Temp" in df.columns:
            ot = df["Oil_Temp"].astype(float)
            cross["Alternator_Oil_delta"] = at - ot

    return cross


class PhysicsResidualExtractor:
    """Physics-informed residual extraction engine.

    Strictly separates:
    1. Healthy-Reference Residuals: actual - healthy_reference (computed strictly from training data)
    2. Engine-Model Residuals: actual - expected_physics_model (thermodynamic Otto cycle model)
    3. Paired-Digital-Twin Residuals: 50% healthy-ref + 50% physics model (AeroTwin reference)
    """

    def __init__(self, engine_config: Optional[EngineConfig] = None):
        self.engine_config = engine_config
        self.engine_physics = ReducedOrderPistonEngine(config=engine_config)
        self.healthy_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.is_fitted: bool = False

    def fit_healthy_reference(
        self,
        train_df: pd.DataFrame,
        health_state_col: str = "Health_State",
        normal_label: str = "Normal",
        operating_state_col: str = "Operating_State",
    ) -> PhysicsResidualExtractor:
        """Fits healthy-state medians and standard deviations strictly from normal training flights."""
        if health_state_col in train_df.columns:
            healthy = train_df[train_df[health_state_col] == normal_label].copy()
            if len(healthy) == 0:
                healthy = train_df.copy()
        else:
            healthy = train_df.copy()

        num_cols = [c for c in BASELINE_NUMERIC_FEATURES if c in healthy.columns]

        stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        has_op = operating_state_col in healthy.columns

        if has_op:
            for state, group in healthy.groupby(operating_state_col):
                stats[str(state)] = {}
                for col in num_cols:
                    std = float(group[col].std())
                    stats[str(state)][col] = {
                        "median": float(group[col].median()),
                        "std": std if np.isfinite(std) and std > 1e-6 else 1.0,
                    }

        stats["_GLOBAL_"] = {}
        for col in num_cols:
            std = float(healthy[col].std())
            stats["_GLOBAL_"][col] = {
                "median": float(healthy[col].median()),
                "std": std if np.isfinite(std) and std > 1e-6 else 1.0,
            }

        self.healthy_stats = stats
        self.is_fitted = True
        return self

    def extract_residuals(
        self,
        df: pd.DataFrame,
        operating_state_col: str = "Operating_State",
    ) -> pd.DataFrame:
        """Computes healthy-reference, engine-model, and digital twin residuals."""
        if not self.is_fitted:
            raise RuntimeError("PhysicsResidualExtractor must be fitted on training data before extract_residuals()")

        residuals = pd.DataFrame(index=df.index)
        has_op = operating_state_col in df.columns

        # --- 1. Healthy-Reference Residuals ---
        num_cols = [c for c in BASELINE_NUMERIC_FEATURES if c in df.columns]
        for col in num_cols:
            global_med = self.healthy_stats["_GLOBAL_"][col]["median"]
            global_std = self.healthy_stats["_GLOBAL_"][col]["std"]

            if has_op:
                medians = df[operating_state_col].map(
                    lambda st: self.healthy_stats.get(str(st), self.healthy_stats["_GLOBAL_"]).get(col, {}).get("median", global_med)
                ).astype(float)
                stds = df[operating_state_col].map(
                    lambda st: self.healthy_stats.get(str(st), self.healthy_stats["_GLOBAL_"]).get(col, {}).get("std", global_std)
                ).astype(float)
            else:
                medians = pd.Series(global_med, index=df.index)
                stds = pd.Series(global_std, index=df.index)

            obs = df[col].astype(float)
            residuals[f"{col}_res_healthy"] = obs - medians
            residuals[f"{col}_z_healthy"] = (obs - medians) / stds.clip(lower=1e-6)

        # --- 2. Engine-Model Physics Residuals ---
        # Operating points for physics model: RPM, throttle estimate from MAP/operating state, ambient
        rpm = df["Engine_RPM"].astype(float) if "Engine_RPM" in df.columns else pd.Series(3000.0, index=df.index)
        # Operating-state throttle mapping
        throttle_map = {"CRUISE_LOW": 0.50, "CRUISE": 0.65, "HIGH": 0.85}
        if has_op:
            th_approx = df[operating_state_col].map(lambda st: throttle_map.get(str(st), 0.65)).astype(float)
        else:
            th_approx = pd.Series(0.65, index=df.index)

        ambient_c = df["Ambient_Temp"].astype(float) if "Ambient_Temp" in df.columns else pd.Series(25.0, index=df.index)

        # Precompute physics predictions for unique operating conditions to accelerate vector computation
        unique_ops = pd.DataFrame({
            "rpm_round": (rpm / 50.0).round() * 50.0,
            "throttle": th_approx,
            "ambient_round": (ambient_c / 5.0).round() * 5.0,
        }).drop_duplicates()

        phys_cache: Dict[Tuple[float, float, float], Dict[str, float]] = {}
        for _, row in unique_ops.iterrows():
            key = (row["rpm_round"], row["throttle"], row["ambient_round"])
            pred = self.engine_physics.predict(EngineInputs(
                rpm=key[0],
                throttle=key[1],
                altitude_ft=3000.0,
                ambient_c=key[2],
            ))
            phys_cache[key] = pred

        # Map predictions back
        keys = list(zip(
            (rpm / 50.0).round() * 50.0,
            th_approx,
            (ambient_c / 5.0).round() * 5.0,
        ))

        for target_ch, pred_ch in [
            ("MAP_Injector", "MAP_Injector"),
            ("CHT", "CHT"),
            ("EGT1", "EGT1"),
            ("Oil_Pressure", "Oil_Pressure"),
            ("Oil_Temp", "Oil_Temp"),
            ("Fuel_Flow", "Fuel_Flow"),
        ]:
            if target_ch in df.columns:
                expected_vals = pd.Series([phys_cache.get(k, {}).get(pred_ch, np.nan) for k in keys], index=df.index)
                res_phys = df[target_ch].astype(float) - expected_vals
                residuals[f"{target_ch}_res_physics"] = res_phys

                # --- 3. Paired-Digital-Twin Residual (50% healthy ref + 50% physics) ---
                if f"{target_ch}_res_healthy" in residuals.columns:
                    twin_res = 0.50 * residuals[f"{target_ch}_res_healthy"] + 0.50 * res_phys
                    residuals[f"{target_ch}_res_twin"] = twin_res

        # Overall Twin Residual RMS
        twin_cols = [c for c in residuals.columns if c.endswith("_res_twin")]
        if twin_cols:
            residuals["Twin_Residual_RMS"] = np.sqrt((residuals[twin_cols] ** 2).mean(axis=1))

        return residuals


def assemble_experiment_dataset(
    df: pd.DataFrame,
    experiment_id: str,
    residual_extractor: Optional[PhysicsResidualExtractor] = None,
    flight_col: str = "Flight",
) -> pd.DataFrame:
    """Assembles exact feature matrix for experiments E0 through E6.

    E0: Baseline (14 numerical + Operating_State)
    E1: Baseline + Temporal features
    E2: Baseline + Cross-sensor features
    E3: Baseline + Physics residuals
    E4: Baseline + Temporal + Cross-sensor
    E5: Baseline + Temporal + Residuals
    E6: Baseline + Temporal + Cross-sensor + Residuals
    """
    valid_exps = {"E0", "E1", "E2", "E3", "E4", "E5", "E6"}
    if experiment_id not in valid_exps:
        raise ValueError(f"Invalid experiment_id {experiment_id}. Must be one of {valid_exps}")

    # Base features
    base_cols = [c for c in BASELINE_ALL_FEATURES if c in df.columns]
    X = df[base_cols].copy()

    # Pre-extract if needed
    need_temporal = experiment_id in {"E1", "E4", "E5", "E6"}
    need_cross = experiment_id in {"E2", "E4", "E6"}
    need_residuals = experiment_id in {"E3", "E5", "E6"}

    if need_temporal:
        temp_df = extract_temporal_features(df, flight_col=flight_col)
        X = pd.concat([X, temp_df], axis=1)

    if need_cross:
        cross_df = extract_cross_sensor_features(df)
        X = pd.concat([X, cross_df], axis=1)

    if need_residuals:
        if residual_extractor is None:
            raise ValueError(f"Residual extractor is required for experiment {experiment_id}")
        res_df = residual_extractor.extract_residuals(df)
        X = pd.concat([X, res_df], axis=1)

    # Anti-leakage audit
    is_clean, viols = audit_feature_names(X.columns.tolist())
    if not is_clean:
        raise ValueError(f"Anti-leakage violation in assembled dataset: prohibited columns found: {viols}")

    return X
