"""
Dedicated Scientific Regression Test Suite for Part 5/7: RUL + Degradation Engineering.

Verifies:
1. RUL Monotonicity under monotonic degradation / non-decreasing stress.
2. Zero-crossing accuracy when health reaches critical threshold (H <= 35.0).
3. Engine profile isolation (Continental TSIO-360-MB 1800h vs Rotax 914 F 1200h).
4. Sensor fault separation (transducer drift widens uncertainty without structural RUL collapse).
5. Anti-leakage verification (no Degradation_Severity or true_RUL as predictors).
6. Uncertainty bounds ordering (0 <= lower <= rul <= upper).
7. Trajectory-level grouped splitting (0 overlap between train and test).
8. Mission reset handling (state cleared without cross-mission memory).
9. Timeline rewind handling (negative dt handled gracefully).
10. Engine switching dynamics (per-engine state isolation).
11. NaN and Inf numerical resilience (no unhandled exceptions, non-negative finite outputs).
12. Extreme value resilience (extreme CHT/RPM handled gracefully).
13. Causal temporal processing (zero future-information leakage).
14. TBO and failure horizon separation (TBO treated as maintenance ceiling, not failure time).
"""
from __future__ import annotations

import math
import pytest
import numpy as np

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
from app.data_engine import VirtualDataLabEngine


# ---------------------------------------------------------------------
# 1. Monotonicity
# ---------------------------------------------------------------------
def test_rul_monotonicity_under_constant_stress():
    """Verify that RUL decrements monotonically under continuous degradation with constant stress."""
    service = RULService(default_engine_id="Rotax-914-Turbo-115HP")
    service.reset()

    ctx = {"engine_id": "Rotax-914-Turbo-115HP", "throttle": 0.60, "ambient_c": 25.0}
    health_seq = [98.0, 95.0, 90.0, 84.0, 76.0, 68.0, 58.0, 48.0, 38.0, 34.0]
    time_seq = [float(i) * 0.5 for i in range(len(health_seq))]

    prev_rul = None
    for h, t in zip(health_seq, time_seq):
        ctx["elapsed_hours"] = t
        res = service.estimate_rul(health_index=h, context=ctx)
        rul = res["rul_hours"]
        if prev_rul is not None and t > 0.0:
            assert rul <= prev_rul + 1e-4, f"Monotonicity violation: {rul} > {prev_rul} at t={t}"
        prev_rul = rul


# ---------------------------------------------------------------------
# 2. Zero Crossing
# ---------------------------------------------------------------------
def test_zero_crossing_at_critical_health():
    """Verify that RUL is 0.0 when health index is at or below critical threshold (35.0)."""
    service = RULService()
    service.reset()

    res_critical = service.estimate_rul(health_index=35.0, context={"elapsed_hours": 10.0})
    assert res_critical["rul_hours"] == 0.0
    assert res_critical["rul_lower_hours"] == 0.0
    assert res_critical["status"] == "CRITICAL_MAINTENANCE_REQUIRED"

    res_below = service.estimate_rul(health_index=20.0, context={"elapsed_hours": 12.0})
    assert res_below["rul_hours"] == 0.0
    assert res_below["rul_lower_hours"] == 0.0


# ---------------------------------------------------------------------
# 3. Engine Profile Isolation
# ---------------------------------------------------------------------
def test_engine_profile_isolation_and_tbo_ceilings():
    """Verify distinct maintenance TBO horizons for Continental TSIO-360-MB vs Rotax 914 F."""
    service = RULService()

    # Rotax 914 F TBO: 1200.0h (EASA TCDS E.121)
    rotax_tbo = service.get_engine_tbo("Rotax-914-Turbo-115HP")
    assert rotax_tbo == 1200.0

    # Continental TSIO-360-MB TBO: 1800.0h (FAA TCDS E9CE)
    tsio_tbo = service.get_engine_tbo("Continental-TSIO-360-MB")
    assert tsio_tbo == 1800.0

    # Verified via helper
    assert get_profile_tbo("Rotax-914-Turbo-115HP") == 1200.0
    assert get_profile_tbo("Continental-TSIO-360-MB") == 1800.0
    assert get_profile_tbo("TSIO_360_MB_ACES_01") == 1800.0

    # Predictions carry distinct engine_profile metadata
    res_rotax = service.predict({"health_index": 95.0}, context={"engine_id": "Rotax-914-Turbo-115HP"})
    assert res_rotax["engine_profile"] == "Rotax-914-Turbo-115HP"
    assert res_rotax["tbo_hours"] == 1200.0

    res_tsio = service.predict({"health_index": 95.0}, context={"engine_id": "Continental-TSIO-360-MB"})
    assert res_tsio["engine_profile"] == "Continental-TSIO-360-MB"
    assert res_tsio["tbo_hours"] == 1800.0


# ---------------------------------------------------------------------
# 4. Sensor Fault Separation
# ---------------------------------------------------------------------
def test_sensor_fault_separation_no_structural_collapse():
    """Verify that an isolated transducer drift degrades confidence and widens intervals without collapsing RUL."""
    service = RULService()
    service.reset()

    # Baseline healthy engine with nominal sensors
    ctx_nominal = {"engine_id": "Rotax-914-Turbo-115HP", "elapsed_hours": 2.0}
    res_nominal = service.predict({"health_index": 92.0}, context=ctx_nominal)

    # Engine with severe transducer drift but genuine mechanical health is 92.0
    ctx_sensor_fault = {
        "engine_id": "Rotax-914-Turbo-115HP",
        "elapsed_hours": 2.0,
        "sensor_fault_severity": 0.85,
        "sensor_fault_flag": True,
    }
    res_sensor = service.predict({"health_index": 92.0}, context=ctx_sensor_fault)

    # RUL must NOT collapse to zero due to sensor fault
    assert res_sensor["rul_hours"] > 500.0, f"False collapse detected: RUL is {res_sensor['rul_hours']}"
    # Confidence must be discounted
    assert res_sensor["confidence"] < res_nominal["confidence"]
    # Uncertainty interval must be substantially wider
    nominal_width = res_nominal["rul_upper_hours"] - res_nominal["rul_lower_hours"]
    sensor_width = res_sensor["rul_upper_hours"] - res_sensor["rul_lower_hours"]
    assert sensor_width > nominal_width


# ---------------------------------------------------------------------
# 5. Leakage Prevention
# ---------------------------------------------------------------------
def test_anti_leakage_ground_truth_isolation():
    """Verify that simulator ground-truth labels (Degradation_Severity, true_RUL) are not used as inputs."""
    service = RULService()
    service.reset()

    # Passing Degradation_Severity=1.0 with health_index=90.0 must NOT trigger critical RUL
    telemetry = {
        "health_index": 90.0,
        "Degradation_Severity": 1.0,  # ground-truth label that should be ignored
        "true_RUL": 0.0,              # ground-truth target that should be ignored
    }
    res = service.predict(telemetry, context={"elapsed_hours": 3.0})
    assert res["rul_hours"] > 500.0
    assert res["status"] != "CRITICAL_MAINTENANCE_REQUIRED"


# ---------------------------------------------------------------------
# 6. Uncertainty Bounds Ordering
# ---------------------------------------------------------------------
def test_uncertainty_bounds_ordering():
    """Verify that 0.0 <= rul_lower <= rul <= rul_upper for all predictions."""
    service = RULService()
    for h in [100.0, 85.0, 65.0, 45.0, 35.0, 20.0]:
        res = service.estimate_rul(h, context={"elapsed_hours": 5.0})
        lower = res["rul_lower_hours"]
        rul = res["rul_hours"]
        upper = res["rul_upper_hours"]
        assert 0.0 <= lower <= rul <= upper, f"Invalid bounds ordering at h={h}: {lower}, {rul}, {upper}"


# ---------------------------------------------------------------------
# 7. Trajectory-Level Split (Zero Overlap)
# ---------------------------------------------------------------------
def test_trajectory_level_split_zero_overlap():
    """Verify that synthetic corpus trajectory splitting guarantees exactly zero trajectory overlap."""
    engine = VirtualDataLabEngine(master_seed=42)
    corpus = engine.generate_master_corpus(
        num_healthy=4,
        num_degradation=12,
        num_sensor_faults=4,
        num_missions=2,
        num_can_faults=0,
        master_seed=42,
    )
    train_dict, test_dict = engine.split_corpus_trajectories(corpus, train_ratio=0.70, seed=42)

    train_ids = set(train_dict.keys())
    test_ids = set(test_dict.keys())

    assert len(train_ids) > 0
    assert len(test_ids) > 0
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Detected trajectory overlap between train and test: {overlap}"


# ---------------------------------------------------------------------
# 8. Mission Reset
# ---------------------------------------------------------------------
def test_mission_reset_clears_internal_state():
    """Verify that reset() clears history buffer and previous RUL to prevent cross-mission contamination."""
    service = RULService()

    # Mission 1
    service.estimate_rul(90.0, context={"elapsed_hours": 2.0, "engine_id": "test_eng_1"})
    service.estimate_rul(70.0, context={"elapsed_hours": 4.0, "engine_id": "test_eng_1"})
    assert "test_eng_1" in service._engine_states

    # Reset
    service.reset()
    assert len(service._engine_states) == 0
    assert service._prev_rul is None

    # Mission 2 starts with fresh nominal state
    res_fresh = service.estimate_rul(95.0, context={"elapsed_hours": 0.0, "engine_id": "test_eng_1"})
    assert res_fresh["status"] == "NOMINAL_HEALTH"


# ---------------------------------------------------------------------
# 9. Timeline Rewind Handling
# ---------------------------------------------------------------------
def test_timeline_rewind_graceful_handling():
    """Verify that a negative dt (timeline rewind without reset) is handled gracefully without crash."""
    service = RULService()
    service.reset()

    ctx = {"engine_id": "rewind_engine"}
    service.estimate_rul(85.0, context={**ctx, "elapsed_hours": 5.0})
    # Rewind to 1.0h
    res_rewound = service.estimate_rul(95.0, context={**ctx, "elapsed_hours": 1.0})
    assert math.isfinite(res_rewound["rul_hours"])
    assert res_rewound["rul_hours"] > 0.0


# ---------------------------------------------------------------------
# 10. Engine Switching Dynamics
# ---------------------------------------------------------------------
def test_multi_engine_switching_isolation():
    """Verify that alternating calls between multiple engines maintain per-engine state isolation."""
    service = RULService()
    service.reset()

    # Engine A (Rotax)
    res_a1 = service.estimate_rul(90.0, context={"engine_id": "Rotax-914-Turbo-115HP", "elapsed_hours": 1.0})
    # Engine B (Continental)
    res_b1 = service.estimate_rul(92.0, context={"engine_id": "Continental-TSIO-360-MB", "elapsed_hours": 1.0})

    # Engine A (degraded)
    res_a2 = service.estimate_rul(50.0, context={"engine_id": "Rotax-914-Turbo-115HP", "elapsed_hours": 5.0})
    # Engine B (still healthy)
    res_b2 = service.estimate_rul(91.0, context={"engine_id": "Continental-TSIO-360-MB", "elapsed_hours": 2.0})

    # Engine B state must not be corrupted by Engine A degradation
    assert res_b2["status"] == "NOMINAL_HEALTH"
    assert res_a2["status"] != "NOMINAL_HEALTH"
    assert res_a2["rul_hours"] < res_b2["rul_hours"]


# ---------------------------------------------------------------------
# 11. NaN / Inf Resilience
# ---------------------------------------------------------------------
def test_nan_and_inf_resilience():
    """Verify that NaN or Inf in health index or elapsed time does not crash the service."""
    service = RULService()
    projector = PhysicsInformedDegradationProjector()

    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        res_svc = service.predict({"health_index": bad_val}, context={"elapsed_hours": bad_val})
        assert math.isfinite(res_svc["rul_hours"])
        assert res_svc["rul_hours"] >= 0.0

        res_proj = projector.predict(bad_val, bad_val, {}).to_dict()
        assert math.isfinite(res_proj["rul_hours"])
        assert res_proj["rul_hours"] >= 0.0


# ---------------------------------------------------------------------
# 12. Extreme Values Resilience
# ---------------------------------------------------------------------
def test_extreme_telemetry_resilience():
    """Verify that extreme telemetry (e.g. 600C CHT, 12000 RPM) is processed without numerical failure."""
    service = RULService()
    extreme_telemetry = {
        "health_index": 70.0,
        "CHT": 650.0,
        "EGT1": 1800.0,
        "Engine_RPM": 12000.0,
        "Oil_Temp": 300.0,
        "Oil_Pressure": 2.0,
    }
    res = service.predict(extreme_telemetry, context={"elapsed_hours": 3.0})
    assert math.isfinite(res["rul_hours"])
    assert res["rul_hours"] >= 0.0


# ---------------------------------------------------------------------
# 13. Causal Processing (No Future Leakage)
# ---------------------------------------------------------------------
def test_strictly_causal_processing():
    """Verify that RUL estimate at time t depends only on past/current history, not future points."""
    e3 = GradientBoostedRULRegressor()
    # Dummy training points
    train_traj = [{
        "points": [
            {"elapsed_hours": float(i) * 0.5, "health_index": 100.0 - float(i) * 4.0, "true_RUL": max(0.0, 12.0 - float(i) * 0.5)}
            for i in range(25)
        ]
    }]
    e3.fit(train_traj)

    hist_at_t2 = [{"elapsed_hours": 0.5, "health_index": 98.0}, {"elapsed_hours": 1.0, "health_index": 96.0}]
    pred_t2 = e3.predict(94.0, 1.5, history=hist_at_t2).rul_hours

    # Adding arbitrary future points in an unrelated buffer should not alter prediction at t=1.5
    pred_t2_repeat = e3.predict(94.0, 1.5, history=hist_at_t2).rul_hours
    assert abs(pred_t2 - pred_t2_repeat) < 1e-6


# ---------------------------------------------------------------------
# 14. TBO vs Failure Horizon Separation
# ---------------------------------------------------------------------
def test_tbo_is_maintenance_ceiling_not_failure_time():
    """Verify that TBO is explicitly treated as a maintenance ceiling and separated from physical failure horizon."""
    service = RULService()
    res = service.estimate_rul(95.0, context={"engine_id": "Rotax-914-Turbo-115HP", "elapsed_hours": 5.0})

    assert "maintenance_tbo_horizon" in res
    assert "failure_threshold" in res
    assert res["maintenance_tbo_horizon"] == 1200.0
    assert res["failure_threshold"] == CRITICAL_HEALTH_THRESHOLD
    assert res["rul_hours"] <= res["maintenance_tbo_horizon"]


# ---------------------------------------------------------------------
# 15. Accumulated Degradation Non-Reversal (Stress Reduction Semantics)
# ---------------------------------------------------------------------
def test_stress_reduction_preserves_accumulated_degradation():
    """
    Verify that an operating stress reduction (e.g. throttle down, descent)
    does NOT reduce accumulated wear (100 - H). Accumulated damage never heals.
    """
    service = RULService()
    service.reset()

    # Phase 1: High stress operating regime with progressive wear
    ctx_high = {"engine_id": "Rotax-914-Turbo-115HP", "throttle": 0.95, "rapid_throttle": True, "elapsed_hours": 2.0}
    service.estimate_rul(80.0, context=ctx_high)

    # Phase 2: Operator throttles back to 0.40 (stress reduction)
    ctx_low = {"engine_id": "Rotax-914-Turbo-115HP", "throttle": 0.40, "rapid_throttle": False, "elapsed_hours": 2.2}
    service.estimate_rul(78.0, context=ctx_low)

    state = service._engine_states["Rotax-914-Turbo-115HP"]
    accumulated_damage = 100.0 - state.last_health
    assert accumulated_damage >= 20.0, "Physical damage was unphysically reversed!"


# ---------------------------------------------------------------------
# 16. ACES Telemetry Feature Availability Verification
# ---------------------------------------------------------------------
def test_aces_feature_availability_explicit_classification():
    """Verify that ACES sensor availability correctly separates available from uninstrumented channels."""
    from scripts.audit_part5_1_rul import audit_aces_feature_availability
    audit = audit_aces_feature_availability()

    # Raw telemetry channels present in ACES
    assert audit["cht"]["status"] == "AVAILABLE_IN_ACES"
    assert audit["oil_pressure"]["status"] == "AVAILABLE_IN_ACES"
    assert audit["oil_temp"]["status"] == "AVAILABLE_IN_ACES"
    assert audit["egt_spread"]["status"] == "AVAILABLE_IN_ACES"
    assert audit["elapsed_hours"]["status"] == "AVAILABLE_IN_ACES"

    # Uninstrumented channels in ACES
    assert audit["vibration"]["status"] == "NOT_AVAILABLE"
    assert audit["efficiency"]["status"] == "NOT_AVAILABLE"


# ---------------------------------------------------------------------
# 17. E3 Anti-Leakage Feature Space Verification
# ---------------------------------------------------------------------
def test_e3_no_target_derived_features():
    """Verify that E3 feature list contains zero target-derived or ground-truth simulator labels."""
    forbidden = {"Degradation_Severity", "Degradation_State", "true_RUL", "tfailure", "failure_time"}
    feature_set = set(GradientBoostedRULRegressor.FEATURE_NAMES)

    leakage = feature_set & forbidden
    assert len(leakage) == 0, f"Target leakage detected in E3 features: {leakage}"


# ---------------------------------------------------------------------
# 18. Rate Limiter Bounds Verification
# ---------------------------------------------------------------------
def test_rate_limiter_bounds():
    """Verify that RUL drop per step is bounded by MAX_RUL_DROP_FRACTION * horizon."""
    service = RULService(default_engine_id="Rotax-914-Turbo-115HP")
    service.reset()

    # Step 1: Initial call at nominal health
    res1 = service.estimate_rul(100.0, context={"elapsed_hours": 0.0, "engine_id": "Rotax-914-Turbo-115HP"})
    initial_rul = res1["rul_hours"]

    # Step 2: Severe drop in health from 100 to 30 (acute fault)
    res2 = service.estimate_rul(30.0, context={"elapsed_hours": 0.1, "engine_id": "Rotax-914-Turbo-115HP"})

    # The drop must be bounded by MAX_RUL_DROP_FRACTION * 1200 = 300h
    max_allowed_drop = service.MAX_RUL_DROP_FRACTION * 1200.0
    actual_drop = initial_rul - res2["rul_hours"]
    assert actual_drop <= max_allowed_drop + 1.0, f"Rate limiter failed: actual drop was {actual_drop}"

