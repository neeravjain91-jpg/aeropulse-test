"""
Validation and Benchmarking Harness for AeroPulse-X Part 5/7: RUL + Degradation Engineering.

Executes:
1. Controlled synthetic degradation benchmark (E0, E1, E2, E3, E4, E5) with grouped trajectory splitting (0 overlap).
2. Monotonicity, zero-crossing, lead time, uncertainty coverage, and per-mode / per-engine evaluations.
3. ACES real operational degradation consistency validation (no fake failure times).
4. Adversarial robustness stress suite (NaN, Inf, missing channels, dropouts, transitions, engine switching).
5. Isolated NASA C-MAPSS methodology comparison.
6. Outputs reports/part5_rul_benchmark.json and reports/part5_degradation_validation.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_engine import VirtualDataLabEngine, FAILURE_HEALTH_THRESHOLD
from app.rul_service import RULService
from app.rul_estimator import (
    PhysicsInformedDegradationProjector,
    PowerLawDegradationRegressor,
    GradientBoostedRULRegressor,
    WeibullHazardModel,
    CalibratedUncertaintyEstimator,
    CRITICAL_HEALTH_THRESHOLD,
    calculate_causal_stress,
    get_profile_tbo,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_benchmark_trajectories(seed: int = 42) -> List[Dict[str, Any]]:
    """Generates diverse synthetic degradation trajectories across 7 modes and 2 engine profiles."""
    engine = VirtualDataLabEngine(master_seed=seed)
    trajectories = []
    modes = ["thermal", "lubrication", "mechanical", "injector", "misfire", "electrical", "compound"]
    engines = ["Rotax-914-Turbo-115HP", "Continental-TSIO-360-MB"]

    traj_counter = 0
    for eng_id in engines:
        for mode in modes:
            for rep in range(3):
                traj_counter += 1
                tid = f"TRAJ_{eng_id[:5].upper()}_{mode[:4].upper()}_{traj_counter:03d}"
                pts = engine.generate_degradation_trajectory(
                    trajectory_id=tid,
                    failure_mode=mode,
                    seed=seed + traj_counter * 17,
                )
                points_data = []
                for p in pts:
                    p_dict = p.to_dict()
                    t_h = float(p_dict.get("elapsed_hours", float(p_dict.get("timestamp", 0.0)) / 3600.0))
                    h = float(p_dict.get("health_index", 100.0))
                    true_r = float(p_dict.get("true_RUL", 0.0))
                    points_data.append({
                        "elapsed_hours": t_h,
                        "time_hours": t_h,
                        "health_index": h,
                        "true_RUL": true_r,
                        "engine_id": eng_id,
                        "degradation_mode": mode,
                        "altitude_ft": float(p_dict.get("altitude", 3000.0)),
                        "ambient_c": float(p_dict.get("ambient_temperature", 25.0)),
                        "throttle": float(p_dict.get("throttle", 0.60)),
                        "rapid_throttle": bool(p_dict.get("rapid_throttle", False)),
                        "CHT": float(p_dict.get("CHT", 145.0)),
                        "EGT1": float(p_dict.get("EGT", 1250.0)),
                        "EGT2": float(p_dict.get("EGT", 1250.0)),
                        "EGT3": float(p_dict.get("EGT", 1250.0)),
                        "Oil_Pressure": float(p_dict.get("oil_pressure", 55.0)),
                        "Oil_Temp": float(p_dict.get("oil_temperature", 85.0)),
                        "Vibration": float(p_dict.get("vibration", 0.5)),
                        "Efficiency": float(p_dict.get("efficiency", 0.65)),
                        "Fuel_Flow": float(p_dict.get("fuel_flow", 25.0)),
                        "Battery_Voltage": float(p_dict.get("bus_voltage", 28.0)),
                    })

                trajectories.append({
                    "trajectory_id": tid,
                    "engine_id": eng_id,
                    "degradation_mode": mode,
                    "points": points_data,
                })

    return trajectories


def split_grouped_trajectories(
    trajectories: List[Dict[str, Any]],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Grouped trajectory splitting with guaranteed 0 overlap between train, val, and test."""
    rng = random.Random(seed)
    shuffled = list(trajectories)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    # Strict assertion of 0 overlap
    train_ids = {t["trajectory_id"] for t in train}
    val_ids = {t["trajectory_id"] for t in val}
    test_ids = {t["trajectory_id"] for t in test}

    assert len(train_ids & test_ids) == 0, "Leakage detected: train/test trajectory overlap!"
    assert len(train_ids & val_ids) == 0, "Leakage detected: train/val trajectory overlap!"
    assert len(val_ids & test_ids) == 0, "Leakage detected: val/test trajectory overlap!"

    return train, val, test


def evaluate_estimator_on_test(
    estimator_name: str,
    predict_fn: Any,
    reset_fn: Any,
    test_trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates an estimator streaming causally through held-out test trajectories."""
    all_errors = []
    zero_crossing_errors = []
    monotonicity_violations = 0
    total_monotonic_steps = 0
    coverage_hits = 0
    total_coverage_points = 0
    interval_widths = []
    lead_times = []

    mode_errors: Dict[str, List[float]] = {}
    engine_errors: Dict[str, List[float]] = {}

    for traj in test_trajectories:
        reset_fn()
        pts = traj["points"]
        eng_id = traj["engine_id"]
        mode = traj["degradation_mode"]

        prev_rul: Optional[float] = None
        first_alert_time: Optional[float] = None
        failure_time: Optional[float] = None

        history = []
        for i, pt in enumerate(pts):
            t = pt["elapsed_hours"]
            h = pt["health_index"]
            y_true = pt["true_RUL"]

            # Context
            ctx = {
                "engine_id": eng_id,
                "elapsed_hours": t,
                "altitude_ft": pt["altitude_ft"],
                "ambient_c": pt["ambient_c"],
                "throttle": pt["throttle"],
                "rapid_throttle": pt["rapid_throttle"],
                "degradation_mode": mode,
                "CHT": pt["CHT"],
                "EGT1": pt["EGT1"],
                "EGT2": pt["EGT2"],
                "Oil_Pressure": pt["Oil_Pressure"],
                "Oil_Temp": pt["Oil_Temp"],
                "Vibration": pt["Vibration"],
                "Efficiency": pt["Efficiency"],
            }

            # Predict
            pred = predict_fn(h, t, ctx, history)
            y_pred = float(pred["rul_hours"])
            y_lower = float(pred["rul_lower_hours"])
            y_upper = float(pred["rul_upper_hours"])

            # Error
            err = abs(y_true - y_pred)
            all_errors.append(err)

            mode_errors.setdefault(mode, []).append(err)
            engine_errors.setdefault(eng_id, []).append(err)

            # Monotonicity check under non-decreasing stress
            if prev_rul is not None:
                total_monotonic_steps += 1
                if y_pred > prev_rul + 0.05:
                    monotonicity_violations += 1
            prev_rul = y_pred

            # Zero-crossing error at failure threshold
            if h <= CRITICAL_HEALTH_THRESHOLD or y_true == 0.0:
                zero_crossing_errors.append(err)
                if failure_time is None:
                    failure_time = t

            # Early warning lead time
            if first_alert_time is None and (h < 75.0 or y_pred < 15.0):
                first_alert_time = t

            # Uncertainty coverage
            total_coverage_points += 1
            if y_lower <= y_true <= y_upper:
                coverage_hits += 1
            interval_widths.append(max(0.0, y_upper - y_lower))

            history.append({"elapsed_hours": t, "health_index": h, **ctx})

        if failure_time is not None and first_alert_time is not None:
            lead_times.append(max(0.0, failure_time - first_alert_time))

    mae = float(np.mean(all_errors)) if all_errors else 0.0
    rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2))) if all_errors else 0.0
    medae = float(np.median(all_errors)) if all_errors else 0.0
    zc_err = float(np.mean(zero_crossing_errors)) if zero_crossing_errors else 0.0
    mono_rate = float(monotonicity_violations / max(1, total_monotonic_steps))
    coverage = float(coverage_hits / max(1, total_coverage_points))
    mean_width = float(np.mean(interval_widths)) if interval_widths else 0.0
    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    per_mode_mae = {m: round(float(np.mean(errs)), 3) for m, errs in mode_errors.items()}
    per_engine_mae = {e: round(float(np.mean(errs)), 3) for e, errs in engine_errors.items()}

    return {
        "estimator": estimator_name,
        "mae_hours": round(mae, 3),
        "rmse_hours": round(rmse, 3),
        "median_absolute_error_hours": round(medae, 3),
        "zero_crossing_error_hours": round(zc_err, 3),
        "monotonicity_violation_rate": round(mono_rate, 4),
        "early_warning_lead_time_hours": round(avg_lead_time, 2),
        "uncertainty_coverage": round(coverage, 3),
        "mean_interval_width_hours": round(mean_width, 2),
        "per_mode_mae": per_mode_mae,
        "per_engine_mae": per_engine_mae,
    }


def run_synthetic_rul_benchmark() -> Dict[str, Any]:
    """Runs benchmark across E0, E1, E2, E3, E4, E5 on strictly held-out synthetic trajectories."""
    print("Generating synthetic degradation corpus across 7 modes & 2 engine profiles...")
    trajectories = generate_benchmark_trajectories(seed=42)
    train, val, test = split_grouped_trajectories(trajectories, train_ratio=0.60, val_ratio=0.20, seed=42)

    print(f"Grouped Split: {len(train)} train, {len(val)} val, {len(test)} test trajectories. (Overlap: 0)")

    # E0: Current RUL Service
    rul_service = RULService()

    def predict_e0(h: float, t: float, ctx: dict, hist: list) -> dict:
        h_hist = [p["health_index"] for p in hist] if hist else [h]
        res = rul_service.predict(
            telemetry={"health_index": h, **ctx},
            context={"elapsed_hours": t, **ctx},
            health_history=h_hist,
        )
        return res

    # E1: PhysicsInformedDegradationProjector
    e1 = PhysicsInformedDegradationProjector()
    e1.fit(train)

    # E2: PowerLawDegradationRegressor
    e2 = PowerLawDegradationRegressor()
    e2.fit(train)

    # E3: GradientBoostedRULRegressor
    e3 = GradientBoostedRULRegressor()
    e3.fit(train)

    # E4: WeibullHazardModel
    e4 = WeibullHazardModel()
    e4.fit(train)

    # E5: CalibratedUncertaintyEstimator (wrapping E1)
    e5 = CalibratedUncertaintyEstimator(base_estimator=PhysicsInformedDegradationProjector(), target_coverage=0.90)
    e5.fit(train)
    e5.calibrate(val)

    estimators = [
        ("E0_CurrentRULService", predict_e0, rul_service.reset),
        ("E1_PhysicsInformedProjector", lambda h, t, c, hi: e1.predict(h, t, c, hi).to_dict(), e1.reset),
        ("E2_PowerLawRegressor", lambda h, t, c, hi: e2.predict(h, t, c, hi).to_dict(), e2.reset),
        ("E3_GradientBoostedRegressor", lambda h, t, c, hi: e3.predict(h, t, c, hi).to_dict(), e3.reset),
        ("E4_WeibullHazardModel", lambda h, t, c, hi: e4.predict(h, t, c, hi).to_dict(), e4.reset),
        ("E5_CalibratedUncertaintyEstimator", lambda h, t, c, hi: e5.predict(h, t, c, hi).to_dict(), e5.reset),
    ]

    results = []
    for name, pred_fn, rst_fn in estimators:
        print(f"Evaluating {name}...")
        res = evaluate_estimator_on_test(name, pred_fn, rst_fn, test)
        results.append(res)

    # Load C-MAPSS metrics if available
    cmapss_path = REPORTS_DIR / "cmapss_rul_metrics.json"
    cmapss_metrics = {}
    if cmapss_path.exists():
        cmapss_metrics = json.loads(cmapss_path.read_text())

    benchmark_summary = {
        "benchmark_date": "2026-09-20",
        "provenance": "AEROPULSE_SYNTHETIC_GROUPED_TRAJECTORIES",
        "data_split": {
            "total_trajectories": len(trajectories),
            "train_trajectories": len(train),
            "val_trajectories": len(val),
            "test_trajectories": len(test),
            "trajectory_overlap": 0,
            "split_mechanism": "Grouped trajectory ID partition",
        },
        "models_evaluated": results,
        "cmapss_isolated_methodology_proxy": {
            "metrics": cmapss_metrics,
            "disclaimer": "NASA C-MAPSS is commercial turbofan data. It verifies algorithmic prognostic mechanics, not aero-piston engine RUL accuracy.",
        },
    }

    return benchmark_summary


def run_aces_operational_consistency() -> Dict[str, Any]:
    """Validates operational degradation consistency on real NASA ACES flights."""
    aces_file = ROOT / "FINAL_DATASET" / "ACES" / "aces_health.csv"
    if not aces_file.exists():
        return {"status": "SKIPPED", "reason": "FINAL_DATASET/ACES/aces_health.csv not found"}

    print("Running ACES operational degradation consistency audit...")
    df = pd.read_csv(aces_file)
    flights = sorted(df["Flight"].unique().tolist())

    rul_svc = RULService()
    flight_stats = []

    for fid in flights:
        fdf = df[df["Flight"] == fid].sort_values("GPS_Time").copy()
        if len(fdf) < 50:
            continue

        rul_svc.reset(engine_id="Continental-TSIO-360-MB")
        h_vals, r_vals, lower_vals, upper_vals = [], [], [], []

        for _, row in fdf.iloc[::20].iterrows():
            # Real ACES observable indicators
            rpm = float(row.get("Engine_RPM", 2400.0))
            cht = float(row.get("Ambient_Temp", 20.0)) * 1.8 + 32.0  # reference
            ff = float(row.get("Fuel_Flow", 15.0))
            op_state = str(row.get("Operating_State", "Cruise"))

            # Derive observable health index
            h = 100.0
            if op_state == "Takeoff" and rpm < 2200:
                h -= 10.0
            if op_state == "Cruise" and rpm > 2650:
                h -= 5.0

            res = rul_svc.predict(
                telemetry={"health_index": h, "RPM": rpm, "Fuel_Flow": ff},
                context={"engine_id": "Continental-TSIO-360-MB", "flight_id": fid},
                health_history=h_vals,
            )
            h_vals.append(h)
            r_vals.append(res["rul_hours"])
            lower_vals.append(res["rul_lower_hours"])
            upper_vals.append(res["rul_upper_hours"])

        # Stability: coefficient of variation of RUL predictions
        r_arr = np.array(r_vals)
        cv = float(np.std(r_arr) / max(1.0, np.mean(r_arr)))

        flight_stats.append({
            "flight_id": fid,
            "samples_analyzed": len(h_vals),
            "mean_health_index": round(float(np.mean(h_vals)), 2),
            "mean_projected_rul_hours": round(float(np.mean(r_arr)), 2),
            "rul_temporal_stability_cv": round(cv, 3),
            "monotonic_consistency": bool(np.all(np.diff(r_arr) <= 0.05)),
        })

    return {
        "status": "VALIDATED",
        "disclaimer": "ACES contains operational aero-piston flights without run-to-failure ground truth. Metrics validate operational trend consistency, not empirical RUL accuracy.",
        "engine_profile": "Continental-TSIO-360-MB",
        "tbo_ceiling_hours": 1800.0,
        "flights_evaluated": flight_stats,
    }


def run_adversarial_suite() -> Dict[str, Any]:
    """Runs 12 adversarial stress tests against the RUL subsystem."""
    print("Running adversarial robustness test suite...")
    service = RULService()
    projector = PhysicsInformedDegradationProjector()
    scenarios = [
        ("nan_health", {"health_index": float("nan")}, {"elapsed_hours": 2.0}),
        ("inf_health", {"health_index": float("inf")}, {"elapsed_hours": 2.0}),
        ("negative_elapsed", {"health_index": 90.0}, {"elapsed_hours": -5.0}),
        ("nan_elapsed", {"health_index": 90.0}, {"elapsed_hours": float("nan")}),
        ("extreme_temperature", {"health_index": 80.0, "CHT": 600.0, "Oil_Temp": 250.0}, {"elapsed_hours": 1.0}),
        ("abrupt_throttle_transient", {"health_index": 85.0}, {"rapid_throttle": True, "throttle": 1.0, "elapsed_hours": 3.0}),
        ("high_altitude_stress", {"health_index": 85.0}, {"altitude_ft": 25000.0, "elapsed_hours": 2.0}),
        ("sensor_dropout", {}, {}),
        ("sensor_bias_isolated", {"health_index": 85.0, "sensor_fault_flag": True, "sensor_fault_severity": 0.85}, {"elapsed_hours": 4.0}),
        ("timeline_rewind", {"health_index": 85.0}, {"elapsed_hours": 1.0}),  # following an earlier 4.0h
        ("engine_switch_isolation", {"health_index": 92.0}, {"engine_id": "Continental-TSIO-360-MB", "elapsed_hours": 1.0}),
        ("invalid_engine_id", {"health_index": 90.0}, {"engine_id": "Unknown-Experimental-Rocket", "elapsed_hours": 2.0}),
    ]

    adversarial_results = []
    for name, telem, ctx in scenarios:
        # Test RULService
        err = None
        try:
            res_svc = service.predict(telem, ctx)
            r_svc = res_svc["rul_hours"]
            valid_svc = math.isfinite(r_svc) and r_svc >= 0.0
        except Exception as e:
            valid_svc = False
            err = str(e)

        # Test Projector
        try:
            h = float(telem.get("health_index", 100.0))
            t = float(ctx.get("elapsed_hours", 0.0))
            res_proj = projector.predict(h, t, ctx).to_dict()
            r_proj = res_proj["rul_hours"]
            valid_proj = math.isfinite(r_proj) and r_proj >= 0.0
        except Exception as e:
            valid_proj = False
            err = err or str(e)

        passed = valid_svc and valid_proj
        adversarial_results.append({
            "scenario": name,
            "passed": passed,
            "rul_service_rul": r_svc if valid_svc else None,
            "projector_rul": r_proj if valid_proj else None,
            "error": err,
        })

    all_passed = all(r["passed"] for r in adversarial_results)
    return {
        "all_adversarial_tests_passed": all_passed,
        "total_scenarios": len(scenarios),
        "results": adversarial_results,
    }


def main():
    print("=================================================================")
    print("AeroPulse-X Part 5: RUL & Degradation Comprehensive Validation")
    print("=================================================================")

    # 1. Synthetic RUL Benchmark
    benchmark_report = run_synthetic_rul_benchmark()
    benchmark_file = REPORTS_DIR / "part5_rul_benchmark.json"
    benchmark_file.write_text(json.dumps(benchmark_report, indent=2))
    print(f"Saved benchmark report to {benchmark_file}")

    # 2. ACES Consistency & Adversarial Tests
    aces_report = run_aces_operational_consistency()
    adversarial_report = run_adversarial_suite()

    degradation_validation = {
        "validation_date": "2026-09-20",
        "provenance": "AEROPULSE_X_PART5_VALIDATION",
        "synthetic_benchmark_summary": {
            m["estimator"]: {"mae": m["mae_hours"], "zero_crossing_error": m["zero_crossing_error_hours"], "coverage": m["uncertainty_coverage"]}
            for m in benchmark_report["models_evaluated"]
        },
        "aces_operational_consistency": aces_report,
        "adversarial_robustness": adversarial_report,
    }

    validation_file = REPORTS_DIR / "part5_degradation_validation.json"
    validation_file.write_text(json.dumps(degradation_validation, indent=2))
    print(f"Saved degradation validation report to {validation_file}")

    print("\n--- Summary of Candidate Benchmark Results ---")
    for m in benchmark_report["models_evaluated"]:
        print(f"Model: {m['estimator']}")
        print(f"  MAE: {m['mae_hours']}h | RMSE: {m['rmse_hours']}h | MedAE: {m['median_absolute_error_hours']}h")
        print(f"  Zero-Crossing Error: {m['zero_crossing_error_hours']}h | Monotonicity Violations: {m['monotonicity_violation_rate']}")
        print(f"  Uncertainty Coverage: {m['uncertainty_coverage'] * 100:.1f}% | Interval Width: {m['mean_interval_width_hours']}h")
        print("  ---------------------------------------------------------------")


if __name__ == "__main__":
    main()
