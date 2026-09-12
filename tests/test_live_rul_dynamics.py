import pytest
from app.rul_service import RULService
from app.replay import run_replay
from app.inference import AeroTwinAI
from app.config import DATA_SAMPLE_DIR
import pandas as pd


def test_rul_elapsed_time_monotonicity():
    """Verify that RUL monotonically decrements as operating flight hours elapse during nominal flight."""
    service = RULService()
    tbo = service.get_engine_tbo("Rotax-914-Turbo-115HP")
    assert tbo == 1200.0

    # Step 0: 0 hours elapsed
    rul_t0 = service.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 0.0, "engine_id": "Rotax-914-Turbo-115HP"},
        engine_id="Rotax-914-Turbo-115HP",
    )
    assert rul_t0["rul_hours"] == 1200.0

    # Step 1: 1.5 hours elapsed under nominal stress (1.0)
    rul_t1 = service.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 1.5, "engine_id": "Rotax-914-Turbo-115HP"},
        engine_id="Rotax-914-Turbo-115HP",
    )
    assert rul_t1["rul_hours"] == 1198.5

    # Step 2: 5.0 hours elapsed
    rul_t2 = service.estimate_rul(
        health_index=100.0,
        context={"elapsed_hours": 5.0, "engine_id": "Rotax-914-Turbo-115HP"},
        engine_id="Rotax-914-Turbo-115HP",
    )
    assert rul_t2["rul_hours"] == 1195.0
    assert rul_t0["rul_hours"] > rul_t1["rul_hours"] > rul_t2["rul_hours"]


def test_rul_stress_acceleration():
    """Verify that severe environmental / mission stress accelerates RUL consumption."""
    service = RULService()
    ctx_nominal = {"elapsed_hours": 4.0, "altitude_ft": 3000.0, "ambient_c": 20.0, "engine_id": "AeroPiston-4C-1.35L"}
    ctx_stressed = {"elapsed_hours": 4.0, "altitude_ft": 18000.0, "ambient_c": 45.0, "rapid_throttle": True, "engine_id": "AeroPiston-4C-1.35L"}

    rul_nom = service.estimate_rul(health_index=100.0, context=ctx_nominal)
    rul_stress = service.estimate_rul(health_index=100.0, context=ctx_stressed)

    # Stressed mission should have consumed more equivalent life
    assert rul_stress["stress_multiplier"] > rul_nom["stress_multiplier"]
    assert rul_stress["rul_hours"] < rul_nom["rul_hours"]


def test_rul_fault_degradation_trajectory():
    """Verify that progressive health degradation leads to rapid RUL reduction."""
    service = RULService()
    
    # Simulating a progressive health drop sequence
    history_healthy = [100.0, 99.8, 99.7, 99.5, 99.4, 99.2]
    history_degrading = [100.0, 95.0, 88.0, 78.0, 65.0, 52.0]

    rul_h = service.estimate_rul(health_index=99.2, health_history=history_healthy, engine_id="Rotax-914-Turbo-115HP")
    rul_d = service.estimate_rul(health_index=52.0, health_history=history_degrading, engine_id="Rotax-914-Turbo-115HP")

    assert rul_d["rul_hours"] < rul_h["rul_hours"]
    assert rul_d["status"] in ("ACTIVE_DEGRADATION", "WARNING_ELEVATED_WEAR")


def test_replay_timeline_rul_fields():
    """Verify that run_replay emits dynamic RUL and confidence intervals at each step."""
    ai = AeroTwinAI()
    path = DATA_SAMPLE_DIR / "aces_demo.csv"
    demo_df = pd.read_csv(path)
    base_row = demo_df.iloc[0].to_dict()

    scenario = {
        "fault": "overheating",
        "severity": 0.70,
        "altitude_ft": 8000,
        "ambient_c": 35,
        "duration_h": 4.0,
        "engine_id": "Rotax-914-Turbo-115HP",
    }

    result = run_replay(
        ai=ai,
        base=base_row,
        scenario=scenario,
        steps=20,
        step_minutes=3.0,
        fault_onset_ratio=0.35,
    )

    timeline = result["timeline"]
    assert len(timeline) == 20

    # Verify RUL properties across the timeline
    rul_start = timeline[0]["rul_hours"]
    rul_mid = timeline[7]["rul_hours"]
    rul_end = timeline[-1]["rul_hours"]

    assert rul_start is not None
    assert rul_end is not None
    assert rul_start > rul_end
    assert "rul" in timeline[0]
    assert timeline[0]["rul"]["rul_hours"] == rul_start
    assert timeline[-1]["rul_confidence"] is not None
