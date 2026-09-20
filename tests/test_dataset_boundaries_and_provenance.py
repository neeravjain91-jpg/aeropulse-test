"""Automated Test Suite for AeroPulse-X Data Resource Boundaries, Provenance & Anti-Leakage.

Verifies:
1. Dataset identity & catalog integrity
2. Engine identity & physical boundaries
3. Source-type separation (SYNTHETIC vs REAL vs PROXY vs REFERENCE)
4. Anti-concatenation rules (prohibition of blind multi-source merging)
5. Leakage prevention (target fields excluded from feature space)
6. Provenance metadata completeness
7. C-MAPSS exclusion from piston health training
8. CWRU exclusion from piston health labels
9. ALFA exclusion from engine health training
10. Marine dataset exclusion from active models
11. Propeller dataset exclusion from engine models
12. ACES operational compatibility
13. AeroPulse Synthetic ODE consistency
"""
import pytest
from pathlib import Path
import pandas as pd

from app.dataset_registry import DatasetRegistry, DatasetMetadata
from app.data_validator import DataQualityValidator
from app.config import DATA_SAMPLE_DIR, MODEL_DIR


@pytest.fixture
def registry():
    return DatasetRegistry()


def test_dataset_identity_and_catalog_completeness(registry):
    """Verify all 7 authoritative data resources are registered with valid identities."""
    core_summary = registry.get_summary()
    assert core_summary["total_datasets"] == 5
    assert core_summary["primary_datasets"] == 1
    assert core_summary["operational_context_datasets"] == 1
    assert core_summary["cross_domain_proxies"] == 3

    full_summary = registry.get_summary(include_rejected=True)
    assert full_summary["total_datasets"] >= 7
    assert full_summary["rejected_references"] >= 2

    # Verify primary active resources
    aces = registry.get("REAL_ACES")
    assert aces is not None
    assert aces.decision == "ACTIVE"
    assert aces.dataset_id == "NASA_ACES"

    synth = registry.get("AERO_PULSE_SYNTHETIC")
    assert synth is not None
    assert synth.decision == "ACTIVE"
    assert synth.target_engine_relevance == "PRIMARY_TARGET_ROTAX_914"


def test_engine_identity_and_physical_boundaries(registry):
    """Verify explicit engine identities and physics assumptions."""
    aces = registry.get("NASA_ACES")
    assert "Altus II" in aces.source
    assert aces.domain == "GENERAL_AVIATION_PISTON_TELEMETRY"

    synth = registry.get("AERO_PULSE_SYNTHETIC")
    assert "Rotax 914" in synth.source
    assert synth.domain == "AERO_PISTON_4_STROKE_TURBO"

    cmapss = registry.get("NASA_CMAPSS")
    assert cmapss.domain == "COMMERCIAL_AERO_TURBOFAN"
    assert "Turbofan" in cmapss.name


def test_source_type_separation(registry):
    """Verify clean source-type classification (SYNTHETIC, REAL, PROXY, REFERENCE)."""
    assert registry.get("AERO_PULSE_SYNTHETIC").real_or_synthetic == "SYNTHETIC"
    assert registry.get("NASA_ACES").real_or_synthetic == "REAL_OPERATIONAL"
    assert registry.get("NASA_CMAPSS").real_or_synthetic == "CROSS_DOMAIN_PROXY"
    assert registry.get("CWRU_BEARING").real_or_synthetic == "CROSS_DOMAIN_PROXY"
    assert registry.get("ALFA_UAV").real_or_synthetic == "CROSS_DOMAIN_PROXY"
    assert registry.get("REFERENCE_MARINE").real_or_synthetic == "RESEARCH_REFERENCE"
    assert registry.get("REFERENCE_PROPELLER").real_or_synthetic == "RESEARCH_REFERENCE"


def test_anti_concatenation_validation_rules():
    """Verify DataQualityValidator catches prohibited cross-domain combinations."""
    # Prohibited pair: ACES + C-MAPSS
    ok, msg = DataQualityValidator.validate_anti_concatenation(["NASA_ACES", "NASA_CMAPSS"])
    assert not ok
    assert "Anti-concatenation violation" in msg

    # Prohibited pair: Synthetic + Marine
    ok, msg = DataQualityValidator.validate_anti_concatenation(["AERO_PULSE_SYNTHETIC", "REFERENCE_MARINE"])
    assert not ok
    assert "Anti-concatenation violation" in msg

    # Permitted single source
    ok, msg = DataQualityValidator.validate_anti_concatenation(["NASA_ACES"])
    assert ok


def test_feature_leakage_prevention():
    """Verify ground-truth target fields are detected if present in feature vectors."""
    clean_features = ["Engine_RPM", "EGT1", "CHT", "Oil_Pressure", "Fuel_Flow", "MAP_Injector"]
    is_clean, violations = DataQualityValidator.audit_feature_leakage(clean_features)
    assert is_clean
    assert len(violations) == 0

    leaky_features = ["Engine_RPM", "Degradation_Severity", "Health_State", "true_RUL"]
    is_clean, violations = DataQualityValidator.audit_feature_leakage(leaky_features)
    assert not is_clean
    assert "Degradation_Severity" in violations
    assert "Health_State" in violations
    assert "true_RUL" in violations


def test_provenance_metadata_completeness(registry):
    """Verify every dataset has explicit limitations and scientific disclaimers."""
    for ds_dict in registry.list_all():
        ds = DatasetMetadata(**ds_dict)
        assert len(ds.limitations) > 0
        assert ds.license_status is not None
        assert ds.generation_method is not None


def test_cmapss_exclusion_from_piston_health_training():
    """Verify C-MAPSS is NOT used to train aces_health.joblib or piston classifiers."""
    manifest_path = MODEL_DIR / "model_manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        # Verify aces model training features do not contain turbofan channels
        aces_info = manifest.get("aces_health", {})
        if "feature_columns" in aces_info:
            cols = aces_info["feature_columns"]
            assert "T24" not in cols
            assert "P30" not in cols
            assert "Nf" not in cols


def test_cwru_exclusion_from_piston_health_labels():
    """Verify CWRU bearing fault labels do not contaminate piston engine health state."""
    cwru_demo_path = DATA_SAMPLE_DIR / "cwru_demo.csv"
    if cwru_demo_path.exists():
        df_cwru = pd.read_csv(cwru_demo_path)
        # Verify CWRU columns are statistical features, not ACES engine columns
        assert "EGT1" not in df_cwru.columns
        assert "CHT" not in df_cwru.columns


def test_alfa_exclusion_from_engine_health_training():
    """Verify ALFA fixed-wing flight data is isolated from engine health models."""
    aces_demo_path = DATA_SAMPLE_DIR / "aces_demo.csv"
    assert aces_demo_path.exists()
    df_aces = pd.read_csv(aces_demo_path)
    # ACES has piston engine telemetry, not Pixhawk PWM
    assert "Engine_RPM" in df_aces.columns
    assert "mavros-rc-out" not in df_aces.columns


def test_marine_dataset_exclusion(registry):
    """Verify Marine Engine dataset is marked REJECTED and barred from production models."""
    marine = registry.get("REFERENCE_MARINE")
    assert marine is not None
    assert marine.decision == "REJECTED"
    assert "physically incompatible" in marine.limitations[0].lower()


def test_propeller_dataset_exclusion(registry):
    """Verify Propeller blade damage dataset is marked REJECTED for engine health."""
    prop = registry.get("REFERENCE_PROPELLER")
    assert prop is not None
    assert prop.decision == "REJECTED"
    assert "lacks reciprocating" in prop.limitations[0].lower()


def test_aces_operational_compatibility():
    """Verify NASA ACES demo rows have valid physical ranges for aero-piston engines."""
    aces_demo_path = DATA_SAMPLE_DIR / "aces_demo.csv"
    df = pd.read_csv(aces_demo_path)
    assert len(df) > 0
    assert (df["Engine_RPM"] >= 500).all()
    assert (df["Engine_RPM"] <= 6000).all()
    assert (df["Oil_Pressure"] >= 0).all()


def test_aeropulse_synthetic_ode_consistency():
    """Verify AeroPulse synthetic point generator produces physically valid telemetry."""
    from app.data_schema import CanonicalTelemetryPoint
    pt = CanonicalTelemetryPoint(
        timestamp=100.0,
        trajectory_id="TRAJ_TEST_001",
        RPM=3000.0,
        throttle=0.65,
        MAP=980.0,
        ambient_temperature=20.0,
        altitude=5000.0,
        CHT=145.0,
        coolant_temperature=85.0,
        EGT=650.0,
        oil_pressure=4.5,
        oil_temperature=90.0,
        fuel_flow=28.0,
        vibration=1.15,
        bus_voltage=28.0,
        health_index=95.0,
        degradation_severity=0.05,
    )
    ok, viols = pt.validate_physical_bounds()
    assert ok, f"Violations: {viols}"
