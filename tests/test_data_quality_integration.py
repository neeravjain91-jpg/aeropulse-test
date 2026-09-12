from app.data_engine import VirtualDataLabEngine
from app.data_validator import DataQualityValidator

def test_data_quality_validator():
    engine = VirtualDataLabEngine(master_seed=42)
    # Generate a trajectory
    pts = engine.generate_degradation_trajectory(trajectory_id="TEST_TRAJ_001")
    
    # Run through validator
    report = DataQualityValidator.audit_points(pts)
    
    assert report["missing_value_count"] == 0
    assert report["nan_or_inf_count"] == 0
    assert report["timestamp_monotonicity_passed"] is True
    assert report["physical_bounds_passed"] is True
    assert report["rul_ground_truth_passed"] is True
