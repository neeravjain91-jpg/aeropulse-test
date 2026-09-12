from app.rul_service import RULService

def test_rul_rate_limiter():
    rul = RULService()
    
    # 1. Health=95, elapsed=1.0
    res1 = rul.estimate_rul(health_index=95.0, context={"elapsed_hours": 1.0})
    rul1 = res1["rul_hours"]
    assert rul1 > 0
    
    # 2. Health=25, elapsed=2.0 (triggers limiter)
    res2 = rul.estimate_rul(health_index=25.0, context={"elapsed_hours": 2.0})
    rul2 = res2["rul_hours"]
    assert rul2 > 0  # Did not jump to 0 in one step
    assert rul2 < rul1
    assert res2["status"] == "EMERGENCY_ACUTE_FAULT"
    
    # 3. Health=25, elapsed=3.0
    res3 = rul.estimate_rul(health_index=25.0, context={"elapsed_hours": 3.0})
    rul3 = res3["rul_hours"]
    assert rul3 < rul2
    
    # 4. Health=25, elapsed=4.0
    res4 = rul.estimate_rul(health_index=25.0, context={"elapsed_hours": 4.0})
    rul4 = res4["rul_hours"]
    assert rul4 < rul3 or rul4 == 0.0

def test_rul_rate_limiter_backwards_compat():
    rul = RULService()
    
    res1 = rul.estimate_rul(health_index=95.0, context={"elapsed_hours": 0.0})
    res2 = rul.estimate_rul(health_index=25.0, context={"elapsed_hours": 0.0})
    
    assert res2["rul_hours"] == 0.0  # Should instantly drop to 0 if elapsed_hours == 0
