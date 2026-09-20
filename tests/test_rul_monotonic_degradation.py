"""
Dedicated Scientific Regression Test Suite: Monotonic RUL Degradation & Zero Target Leakage.

Verifies:
1. Complete elimination of upward RUL jumps on TRAJ_CUSTOM_DEGRADATION_THERMAL_001.
2. Step monotonicity across all 35 degradation trajectories in VirtualDataLabEngine.
3. Zero target leakage: RULService receives only observable telemetry, health, and elapsed time.
4. Bounded upward revision rule under confirmed stress reductions (rho_stress > 1.05).
5. Monotonic decrement under constant or increasing operating stress.
6. Timeline pause (dt <= 1e-6) state preservation.
7. Multi-engine session isolation and independent state tracking.
8. Calibrated prediction interval ordering (0 <= lower <= pred <= upper).
"""
from __future__ import annotations

import pytest
import numpy as np

from app.data_engine import VirtualDataLabEngine
from app.rul_service import RULService


def test_custom_degradation_thermal_001_no_upward_jumps():
    """Verify that TRAJ_CUSTOM_DEGRADATION_THERMAL_001 has 0 upward jumps in predicted RUL."""
    engine = VirtualDataLabEngine(master_seed=42)
    pts = engine.generate_degradation_trajectory(
        "TRAJ_CUSTOM_DEGRADATION_THERMAL_001",
        failure_mode="thermal",
        seed=42,
    )

    assert len(pts) > 10
    upward_jumps = []
    prev_p = None

    for pt in pts:
        p = pt.predicted_RUL
        if prev_p is not None:
            diff = p - prev_p
            if diff > 0.001:
                upward_jumps.append((pt.timestamp / 3600.0, prev_p, p, diff))
        prev_p = p

    # Absolute verification: zero upward jumps
    assert len(upward_jumps) == 0, (
        f"Detected {len(upward_jumps)} upward jumps in TRAJ_CUSTOM_DEGRADATION_THERMAL_001: {upward_jumps}"
    )

    # Verification: final prediction reaches 0.0 near failure
    assert pts[-1].predicted_RUL <= 0.5
    assert pts[-1].true_RUL == 0.0


def test_all_35_degradation_trajectories_zero_upward_jumps():
    """Verify that all 35 degradation trajectories in VirtualDataLabEngine have 0 upward jumps."""
    engine = VirtualDataLabEngine(master_seed=42)
    corpus = engine.generate_master_corpus(
        num_healthy=0,
        num_degradation=35,
        num_sensor_faults=0,
        num_missions=0,
        num_can_faults=0,
        master_seed=42,
    )

    assert len(corpus) == 35

    total_transitions = 0
    upward_transitions = 0
    all_jumps = []
    errors = []

    for tid, pts in corpus.items():
        prev_p = None
        for pt in pts:
            p = pt.predicted_RUL
            if prev_p is not None:
                total_transitions += 1
                diff = p - prev_p
                if diff > 0.001:
                    upward_transitions += 1
                    all_jumps.append((tid, pt.timestamp / 3600.0, prev_p, p, diff))
            if pt.true_RUL is not None:
                errors.append(abs(p - pt.true_RUL))
            prev_p = p

    assert upward_transitions == 0, (
        f"Detected {upward_transitions} upward jumps across 35 trajectories: {all_jumps[:5]}"
    )
    # Monotonicity score across 35 trajectories must be 100.0%
    step_monotonicity = (1.0 - upward_transitions / total_transitions) * 100.0
    assert step_monotonicity == 100.0
    # Average Prognostic MAE must be < 3.0 hours
    assert np.mean(errors) < 3.0


def test_zero_target_leakage_isolation():
    """Verify RULService does not read true_RUL, true_failure_time, or ground-truth severity."""
    svc = RULService()

    # Telemetry without true_RUL, true_failure_time, or Degradation_Severity
    telemetry = {
        "health_index": 72.5,
        "Engine_RPM": 4500.0,
        "CHT": 130.0,
        "Oil_Pressure": 42.0,
    }
    context = {
        "elapsed_hours": 3.0,
        "altitude_ft": 4000.0,
        "ambient_c": 22.0,
        "throttle": 0.70,
    }

    res = svc.predict(telemetry, context=context)
    assert res["rul_hours"] > 0.0
    assert res["rul_lower_hours"] <= res["rul_hours"] <= res["rul_upper_hours"]
    assert "true_RUL" not in res
    assert "true_failure_time" not in res


def test_stress_reduction_bounded_upward_revision():
    """Verify bounded upward revision occurs only when operating stress drops significantly."""
    svc = RULService()

    # Step 1: High stress flight at 18,000 ft, 45 C, high throttle
    ctx_high = {
        "elapsed_hours": 2.0,
        "altitude_ft": 18000.0,
        "ambient_c": 45.0,
        "rapid_throttle": True,
        "engine_id": "AeroPiston-4C-1.35L",
    }
    res1 = svc.estimate_rul(health_index=80.0, context=ctx_high)
    rul1 = res1["rul_hours"]
    stress1 = res1["stress_multiplier"]

    # Step 2: Aircraft descends to 3,000 ft, 20 C, gentle throttle (stress drops significantly)
    ctx_low = {
        "elapsed_hours": 2.5,
        "altitude_ft": 3000.0,
        "ambient_c": 20.0,
        "rapid_throttle": False,
        "throttle": 0.50,
        "engine_id": "AeroPiston-4C-1.35L",
    }
    res2 = svc.estimate_rul(health_index=80.0, context=ctx_low)
    rul2 = res2["rul_hours"]
    stress2 = res2["stress_multiplier"]

    # Stress must have decreased significantly
    rho_stress = stress1 / stress2
    assert rho_stress > 1.05

    # Bounded upward revision rule: rul2 <= rul1 * min(1.20, rho_stress) - dt * stress2
    dt = 0.5
    max_physical_bound = rul1 * min(1.20, rho_stress) - dt * stress2
    assert rul2 <= max_physical_bound + 0.01


def test_stress_increase_accelerates_rul_consumption():
    """Verify that shifting to higher operating stress monotonically accelerates RUL loss."""
    svc = RULService()

    ctx_mild = {"elapsed_hours": 2.0, "altitude_ft": 3000.0, "ambient_c": 20.0, "engine_id": "Rotax-914-Turbo-115HP"}
    res_mild = svc.estimate_rul(health_index=90.0, context=ctx_mild)

    # In a separate test session with identical elapsed time but severe stress
    svc_stressed = RULService()
    ctx_severe = {"elapsed_hours": 2.0, "altitude_ft": 16000.0, "ambient_c": 42.0, "rapid_throttle": True, "engine_id": "Rotax-914-Turbo-115HP"}
    res_severe = svc_stressed.estimate_rul(health_index=90.0, context=ctx_severe)

    assert res_severe["stress_multiplier"] > res_mild["stress_multiplier"]
    assert res_severe["rul_hours"] < res_mild["rul_hours"]


def test_timeline_pause_preserves_state():
    """Verify that when dt <= 1e-6 under unchanged conditions, RUL is strictly preserved."""
    svc = RULService()
    ctx = {"elapsed_hours": 5.0, "altitude_ft": 5000.0, "ambient_c": 25.0, "engine_id": "AeroPiston-4C-1.35L"}

    res1 = svc.estimate_rul(health_index=85.0, context=ctx)
    # Re-evaluating at identical timestamp and conditions (e.g. simulator paused)
    res2 = svc.estimate_rul(health_index=85.0, context=ctx)

    assert res1["rul_hours"] == res2["rul_hours"]
    assert res1["status"] == res2["status"]


def test_multi_engine_session_isolation():
    """Verify that switching between different engines maintains independent state histories."""
    svc = RULService()

    # Engine A (Rotax: 1200h TBO)
    res_rotax = svc.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 1.0, "engine_id": "Rotax-914-Turbo-115HP"},
        engine_id="Rotax-914-Turbo-115HP",
    )

    # Engine B (AeroPiston: 2000h demonstrator fallback TBO)
    res_aero = svc.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 1.0, "engine_id": "AeroPiston-4C-1.35L"},
        engine_id="AeroPiston-4C-1.35L",
    )

    # Re-evaluate Engine A: must maintain its own state without AeroPiston contamination
    res_rotax_2 = svc.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 2.0, "engine_id": "Rotax-914-Turbo-115HP"},
        engine_id="Rotax-914-Turbo-115HP",
    )

    assert res_rotax["rul_hours"] < 1200.0
    assert res_aero["rul_hours"] > 1900.0
    assert res_rotax_2["rul_hours"] < res_rotax["rul_hours"]
    assert abs(res_rotax_2["rul_hours"] - (1200.0 - 2.0)) < 1.0


def test_prediction_interval_bounds_ordering():
    """Verify that 0 <= RUL_lower <= RUL_pred <= RUL_upper for all points in a degradation flight."""
    engine = VirtualDataLabEngine(master_seed=42)
    pts = engine.generate_degradation_trajectory("TRAJ_DEG_INTERVAL_TEST", failure_mode="lubrication", seed=42)

    for pt in pts:
        assert pt.predicted_RUL >= 0.0
        assert pt.RUL_lower >= 0.0
        assert pt.RUL_lower <= pt.predicted_RUL, (
            f"Interval violation: lower={pt.RUL_lower} > pred={pt.predicted_RUL}"
        )
        assert pt.predicted_RUL <= pt.RUL_upper, (
            f"Interval violation: pred={pt.predicted_RUL} > upper={pt.RUL_upper}"
        )
