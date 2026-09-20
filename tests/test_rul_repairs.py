"""
Regression test suite verifying the RUL mathematical and architectural repairs.

Updated for RC-1 leakage fix: tests now use health_index (observable) instead
of Degradation_Severity or Degradation_State (ground-truth labels).
"""
import pytest
import numpy as np
from app.rul_service import RULService
from app.degradation import estimate_degradation_horizon

def test_rul_slope_continuity():
    """Verify that RUL does not have a 60x discontinuous jump at slope -0.20."""
    service = RULService()
    
    # Slopes around the boundary
    slope_a = -0.199
    slope_b = -0.200
    
    # RC-1 fix: use health_index (observable) instead of Degradation_Severity
    telemetry = {"health_index": 85.0}
    res_a = service.predict(telemetry, context={"degradation_slope": slope_a})
    res_b = service.predict(telemetry, context={"degradation_slope": slope_b})
    
    rul_a = res_a["rul_hours"]
    rul_b = res_b["rul_hours"]
    
    # The jump must be smooth and continuous (|rul_a - rul_b| < 5.0 hours)
    assert abs(rul_a - rul_b) < 5.0, f"Discontinuity detected: rul_a={rul_a}, rul_b={rul_b}"
    assert rul_a > rul_b, "Faster degradation must yield lower RUL"

def test_rul_health_scaling_critical_trigger():
    """Verify that health below threshold (35.0) triggers critical status."""
    service = RULService()
    
    # RC-1 fix: use health_index directly (28.75 = 100 - 0.95*75)
    telemetry = {"health_index": 28.75}
    res_severe = service.predict(telemetry)
    
    assert res_severe["rul_hours"] == 0.0
    assert res_severe["status"] == "CRITICAL_MAINTENANCE_REQUIRED"
    assert res_severe["confidence"] >= 0.90

def test_rul_sensor_fault_isolation():
    """Verify that sensor-only faults do not consume physical engine life."""
    service = RULService()
    
    # RC-1 fix: use health_index instead of Degradation_State
    # Sensor fault only → engine physically healthy → health_index=100
    # Mechanical wear at 0.85 → health_index = 100 - 0.85*75 = 36.25
    res_sensor = service.predict({"health_index": 100.0})
    res_mech = service.predict({"health_index": 36.25})
    
    # Sensor fault should leave baseline physical health at 100.0 (RUL nominal)
    assert res_sensor["health_index_for_rul"] == 100.0
    assert res_sensor["rul_hours"] == 2000.0
    # Mechanical wear fault should significantly reduce health and RUL
    assert res_mech["health_index_for_rul"] < 40.0
    assert res_mech["rul_hours"] < 100.0

def test_multi_engine_tbo_parameterization():
    """Verify that Rotax 914 (1200h documented TBO) and AeroPiston (2000h demonstrator fallback TBO) scale correctly."""
    service = RULService()
    
    assert service.get_engine_tbo("Rotax-914-Turbo-115HP") == 1200.0
    assert service.get_engine_tbo("AeroPiston-4C-1.35L") == 2000.0
    assert service.get_engine_tbo("Generic-Inline4-AeroDiesel") == 1500.0
    
    # RC-1 fix: use health_index instead of Degradation_Severity
    res_rotax = service.predict({"health_index": 100.0}, context={"engine_id": "Rotax-914-Turbo-115HP"})
    res_aeropiston = service.predict({"health_index": 100.0}, context={"engine_id": "AeroPiston-4C-1.35L"})
    
    assert res_rotax["rul_hours"] == 1200.0
    assert res_aeropiston["rul_hours"] == 2000.0

def test_degradation_horizon_dynamic_cap():
    """Verify that estimate_degradation_horizon respects configurable max_horizon_hours."""
    history = [100.0 - i * 0.16 for i in range(25)]
    
    res_default = estimate_degradation_horizon(history, step_minutes=60.0, critical_health_index=35.0, max_horizon_hours=2000.0)
    assert res_default["rul_hours"] is not None
    assert res_default["status"] == "DEGRADING"
    assert abs(res_default["rul_hours"] - (100.0 - 24*0.16 - 35.0)/0.16) < 1.0
