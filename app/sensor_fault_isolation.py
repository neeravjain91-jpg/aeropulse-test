"""Sensor Fault Isolation & Fault-Tolerant Analytics Engine for AeroPulse-X.

Authoritative sensor health and fault isolation layer distinguishing:
  1. NOMINAL
  2. SENSOR_FAULT_ISOLATED
  3. ENGINE_DEGRADATION_CONFIRMED
  4. COMPOUND_FAULT
  5. INSUFFICIENT_OBSERVABILITY

Core Principle:
  "A bad sensor is not necessarily a bad engine."
  Avoids BOTH failure modes:
    - FALSE CATASTROPHE: bad sensor -> fake engine failure -> RUL collapse
    - FALSE REASSURANCE: bad sensor -> ignored evidence -> incorrectly declared healthy

Key Architectural Pillars:
  - Strict Unit Contract: explicit canonical units (RPM, deg_F, psi, inHg, V, A, g, L/h).
  - Threshold Forensic Provenance: A (externally sourced), B (empirically derived), C (model-derived), D (heuristic).
  - Engine Profile Isolation: Rotax 914 F vs Continental TSIO-360-MB.
  - Operating-Phase Conditioning: dynamic slew widening during throttle transitions.
  - 8-Class Sensor Fault Taxonomy: DROPOUT, STUCK_AT, BIAS, DRIFT, SPIKE_OUTLIER, INTERMITTENT, IMPLAUSIBLE_RATE_OF_CHANGE, CROSS_SENSOR_INCONSISTENCY.
  - Observable SensorTrustScore [0, 100] (zero target leakage).
  - Bulk Physics Residual Safety: low-trust channels excluded from bulk engine residual aggregation.
  - Dependency-Aware Analytic Redundancy: virtual sensors refuse to validate signals using untrusted inputs.
  - Leave-One-Sensor-Out (LOSO) verification.
"""
from __future__ import annotations

import copy
import enum
import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =============================================================================
# 1. FAULT TAXONOMY & ENUMS
# =============================================================================

class SensorFaultType(str, enum.Enum):
    """Authoritative 8-class taxonomy of transducer fault modes."""
    NOMINAL = "NOMINAL"
    DROPOUT = "DROPOUT"
    STUCK_AT = "STUCK_AT"
    BIAS = "BIAS"
    DRIFT = "DRIFT"
    SPIKE_OUTLIER = "SPIKE_OUTLIER"
    INTERMITTENT = "INTERMITTENT"
    IMPLAUSIBLE_RATE_OF_CHANGE = "IMPLAUSIBLE_RATE_OF_CHANGE"
    CROSS_SENSOR_INCONSISTENCY = "CROSS_SENSOR_INCONSISTENCY"


class EngineAttributionVerdict(str, enum.Enum):
    """High-level system attribution verdict."""
    NOMINAL = "NOMINAL"
    SENSOR_FAULT_ISOLATED = "SENSOR_FAULT_ISOLATED"
    ENGINE_DEGRADATION_CONFIRMED = "ENGINE_DEGRADATION_CONFIRMED"
    COMPOUND_FAULT = "COMPOUND_FAULT"
    INSUFFICIENT_OBSERVABILITY = "INSUFFICIENT_OBSERVABILITY"


class ThresholdProvenance(str, enum.Enum):
    """Classification of threshold origin for forensic traceability."""
    A_EXTERNAL = "A_EXTERNAL"          # FAA / EASA TCDS, manufacturer POH/manuals
    B_EMPIRICAL = "B_EMPIRICAL"        # Validated empirical flight envelope (e.g. ACES)
    C_MODEL_DERIVED = "C_MODEL_DERIVED"# ReducedOrderPistonEngine thermodynamic cycle
    D_HEURISTIC = "D_HEURISTIC"        # Parameterized engineering safety heuristic


# =============================================================================
# 2. CANONICAL UNIT CONTRACT & TELEMETRY ADAPTER
# =============================================================================

# Standard Canonical Units:
#   RPM: RPM
#   Temperatures: deg_F (for unified ACES / SIL reference matching healthy_reference.json)
#   Pressures: psi (Oil_Pressure), inHg (MAP_Injector)
#   Fuel Flow: L/h (synthetic) or lb/h (ACES raw) - scaled to standard rate
#   Electrical: V (bus voltage), A (current)
#   Vibration: g (RMS tri-axial)

class TelemetryUnitAdapter:
    """
    Explicit unit conversion adapter.
    Enforces a strict unit contract: conversions occur ONLY at the boundary.
    No ambiguous heuristic guessing inside core analytics.
    """

    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return c * 9.0 / 5.0 + 32.0

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32.0) * 5.0 / 9.0

    @staticmethod
    def kpa_to_inhg(kpa: float) -> float:
        return kpa * 0.2952998

    @staticmethod
    def inhg_to_kpa(inhg: float) -> float:
        return inhg / 0.2952998

    @staticmethod
    def bar_to_psi(bar: float) -> float:
        return bar * 14.50377

    @classmethod
    def adapt_to_canonical(cls, raw_telemetry: Dict[str, Any], is_synthetic_metric: bool = False) -> Dict[str, float]:
        """
        Adapts raw telemetry dictionary to canonical internal representation.
        If is_synthetic_metric is True, converts synthetic °C to canonical °F and kPa to inHg.
        """
        adapted: Dict[str, float] = {}

        for k, v in raw_telemetry.items():
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
            except (TypeError, ValueError):
                continue

            # Explicit conversion if telemetry is flagged as synthetic metric
            if is_synthetic_metric:
                if k in ("CHT", "coolant_temperature", "EFI_Water_Temp", "oil_temperature", "Oil_Temp", "ambient_temperature"):
                    # Synthetic °C -> canonical °F
                    adapted[k] = round(cls.celsius_to_fahrenheit(fv), 2)
                    continue
                elif k in ("manifold_pressure", "ambient_pressure"):
                    # Synthetic kPa -> canonical inHg
                    adapted[k] = round(cls.kpa_to_inhg(fv), 2)
                    continue

            # Standard direct mapping
            adapted[k] = fv

        return adapted


# =============================================================================
# 3. THRESHOLD FORENSIC REGISTRY & ENGINE PROFILES
# =============================================================================

@dataclass(frozen=True)
class ThresholdRecord:
    parameter: str
    engine_profile: str
    nominal_min: float
    nominal_max: float
    critical_min: float
    critical_max: float
    max_slew_per_sec: float
    unit: str
    provenance: ThresholdProvenance
    justification: str


THRESHOLD_REGISTRY: Dict[str, Dict[str, ThresholdRecord]] = {
    "Rotax-914-Turbo-115HP": {
        "Engine_RPM": ThresholdRecord(
            parameter="Engine_RPM",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=1400.0,
            nominal_max=5800.0,
            critical_min=800.0,
            critical_max=6000.0,
            max_slew_per_sec=1200.0,
            unit="RPM",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="EASA TCDS E.121 & Rotax 914 Operators Manual Sec 2.1 (Idle 1400, Max 5800 RPM)",
        ),
        "MAP_Injector": ThresholdRecord(
            parameter="MAP_Injector",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=15.0,
            nominal_max=39.9,
            critical_min=10.0,
            critical_max=45.0,
            max_slew_per_sec=18.0,
            unit="inHg",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 Turbocharger TCU Boost Limit: 39.9 inHg (1350 hPa) max continuous",
        ),
        "CHT": ThresholdRecord(
            parameter="CHT",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=140.0,
            nominal_max=250.0,
            critical_min=32.0,
            critical_max=275.0,  # 135 °C = 275 °F
            max_slew_per_sec=6.0,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 OM Sec 2.3: Cylinder head max operating temp 135 °C (275 °F)",
        ),
        "Oil_Pressure": ThresholdRecord(
            parameter="Oil_Pressure",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=29.0,   # 2.0 bar
            nominal_max=72.5,   # 5.0 bar
            critical_min=11.6,  # 0.8 bar
            critical_max=101.5, # 7.0 bar (cold start limit)
            max_slew_per_sec=25.0,
            unit="psi",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 OM Sec 2.4: Oil pressure nominal 2.0-5.0 bar, min 0.8 bar, max 7.0 bar",
        ),
        "Oil_Temp": ThresholdRecord(
            parameter="Oil_Temp",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=120.0,
            nominal_max=230.0,  # 110 °C
            critical_min=32.0,
            critical_max=266.0,  # 130 °C
            max_slew_per_sec=1.5,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 OM Sec 2.4: Normal 90-110 °C (194-230 °F), Max 130 °C (266 °F)",
        ),
        "EFI_Water_Temp": ThresholdRecord(
            parameter="EFI_Water_Temp",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=140.0,
            nominal_max=220.0,
            critical_min=32.0,
            critical_max=240.0,  # 115 °C
            max_slew_per_sec=3.0,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 OM Coolant max limit 115 °C (239 °F) with 50/50 water-glycol",
        ),
        "Fuel_Flow": ThresholdRecord(
            parameter="Fuel_Flow",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=5.0,
            nominal_max=38.0,
            critical_min=0.0,
            critical_max=50.0,
            max_slew_per_sec=12.0,
            unit="L/h",
            provenance=ThresholdProvenance.C_MODEL_DERIVED,
            justification="BSFC 0.33 kg/kWh at 84.5 kW brake power = 38.7 L/h takeoff maximum",
        ),
        "EGT1": ThresholdRecord(
            parameter="EGT1",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=1100.0,
            nominal_max=1550.0,
            critical_min=400.0,
            critical_max=1650.0, # 900 °C = 1652 °F
            max_slew_per_sec=40.0,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="Rotax 914 OM Sec 2.5: EGT normal max 880 °C (1616 °F), max limit 900 °C (1652 °F)",
        ),
        "Battery_Voltage": ThresholdRecord(
            parameter="Battery_Voltage",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=24.0,
            nominal_max=29.5,
            critical_min=18.0,
            critical_max=33.0,
            max_slew_per_sec=4.0,
            unit="V",
            provenance=ThresholdProvenance.B_EMPIRICAL,
            justification="28V DC nominal avionics electrical bus with lead-acid/LiFePO4 regulation",
        ),
        "Battery_Current": ThresholdRecord(
            parameter="Battery_Current",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=-10.0,
            nominal_max=45.0,
            critical_min=-50.0,
            critical_max=80.0,
            max_slew_per_sec=30.0,
            unit="A",
            provenance=ThresholdProvenance.B_EMPIRICAL,
            justification="Alternator 28V 40A charging bus with battery transient discharge buffer",
        ),
        "Alternator_Temp": ThresholdRecord(
            parameter="Alternator_Temp",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=80.0,
            nominal_max=230.0,
            critical_min=32.0,
            critical_max=275.0,
            max_slew_per_sec=2.5,
            unit="deg_F",
            provenance=ThresholdProvenance.D_HEURISTIC,
            justification="Engineering thermal limit for alternator winding insulation (Class H 180 °C)",
        ),
        "Vibration": ThresholdRecord(
            parameter="Vibration",
            engine_profile="Rotax-914-Turbo-115HP",
            nominal_min=0.2,
            nominal_max=4.0,
            critical_min=0.01,
            critical_max=15.0,
            max_slew_per_sec=10.0,
            unit="g",
            provenance=ThresholdProvenance.C_MODEL_DERIVED,
            justification="Synthetic SIL 3-axis casing accelerometer vibration model (synthetic only)",
        ),
    },
    "Continental-TSIO-360-MB": {
        "Engine_RPM": ThresholdRecord(
            parameter="Engine_RPM",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=700.0,
            nominal_max=2700.0,
            critical_min=500.0,
            critical_max=2900.0,
            max_slew_per_sec=600.0,
            unit="RPM",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="FAA TCDS E9CE: Continental TSIO-360-MB Rated 210 HP at 2700 RPM, Idle 700 RPM",
        ),
        "MAP_Injector": ThresholdRecord(
            parameter="MAP_Injector",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=15.0,
            nominal_max=38.0,
            critical_min=10.0,
            critical_max=42.0,
            max_slew_per_sec=15.0,
            unit="inHg",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="FAA TCDS E9CE: Full throttle rated manifold pressure 38.0 inHg sea level",
        ),
        "CHT": ThresholdRecord(
            parameter="CHT",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=240.0,
            nominal_max=420.0,
            critical_min=32.0,
            critical_max=460.0,  # 238 °C per FAA TCDS
            max_slew_per_sec=8.0,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="FAA TCDS E9CE: Max permissible cylinder head temperature 460 °F (238 °C)",
        ),
        "Oil_Pressure": ThresholdRecord(
            parameter="Oil_Pressure",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=30.0,
            nominal_max=60.0,
            critical_min=10.0,   # 10 psi minimum at idle
            critical_max=100.0,  # Cold start limit
            max_slew_per_sec=20.0,
            unit="psi",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="FAA TCDS E9CE: Oil pressure nominal 30-60 psi, minimum idle 10 psi",
        ),
        "Oil_Temp": ThresholdRecord(
            parameter="Oil_Temp",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=140.0,
            nominal_max=220.0,
            critical_min=32.0,
            critical_max=240.0,  # 116 °C max
            max_slew_per_sec=1.5,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="FAA TCDS E9CE: Maximum permissible oil temperature 240 °F (116 °C)",
        ),
        "Fuel_Flow": ThresholdRecord(
            parameter="Fuel_Flow",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=8.0,
            nominal_max=75.0,
            critical_min=0.0,
            critical_max=90.0,
            max_slew_per_sec=20.0,
            unit="L/h",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="POH / TCM Maintenance Manual: Takeoff fuel flow 19.0-21.5 GPH (72-81 L/h)",
        ),
        "EGT1": ThresholdRecord(
            parameter="EGT1",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=1200.0,
            nominal_max=1550.0,
            critical_min=400.0,
            critical_max=1650.0,
            max_slew_per_sec=45.0,
            unit="deg_F",
            provenance=ThresholdProvenance.A_EXTERNAL,
            justification="TCM TSIO-360 Operating Spec: Max continuous turbine inlet/EGT 1650 °F",
        ),
        "Battery_Voltage": ThresholdRecord(
            parameter="Battery_Voltage",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=24.0,
            nominal_max=29.5,
            critical_min=18.0,
            critical_max=33.0,
            max_slew_per_sec=4.0,
            unit="V",
            provenance=ThresholdProvenance.B_EMPIRICAL,
            justification="28V DC Aircraft Electrical Bus Standard",
        ),
        "Battery_Current": ThresholdRecord(
            parameter="Battery_Current",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=-10.0,
            nominal_max=50.0,
            critical_min=-60.0,
            critical_max=90.0,
            max_slew_per_sec=35.0,
            unit="A",
            provenance=ThresholdProvenance.B_EMPIRICAL,
            justification="28V 60A engine driven alternator bus",
        ),
        "Alternator_Temp": ThresholdRecord(
            parameter="Alternator_Temp",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=80.0,
            nominal_max=240.0,
            critical_min=32.0,
            critical_max=280.0,
            max_slew_per_sec=2.5,
            unit="deg_F",
            provenance=ThresholdProvenance.D_HEURISTIC,
            justification="Air-cooled alternator casing thermal ceiling",
        ),
        "Vibration": ThresholdRecord(
            parameter="Vibration",
            engine_profile="Continental-TSIO-360-MB",
            nominal_min=0.2,
            nominal_max=4.5,
            critical_min=0.01,
            critical_max=15.0,
            max_slew_per_sec=10.0,
            unit="g",
            provenance=ThresholdProvenance.C_MODEL_DERIVED,
            justification="Reduced-order 6-cylinder opposed vibration baseline (synthetic only)",
        ),
    },
}

# Alias mapping
THRESHOLD_REGISTRY["Rotax_914"] = THRESHOLD_REGISTRY["Rotax-914-Turbo-115HP"]
THRESHOLD_REGISTRY["Continental_TSIO_360"] = THRESHOLD_REGISTRY["Continental-TSIO-360-MB"]


# =============================================================================
# 4. DATA STRUCTURES & PER-CHANNEL ASSESSMENTS
# =============================================================================

@dataclass
class ChannelAssessment:
    """Individual assessment for a single telemetry channel."""
    name: str
    trust_score: float                  # [0.0, 100.0]
    status: str                         # "TRUSTED", "CHECK", "SUSPECT", "FAILED"
    fault_type: SensorFaultType
    fault_severity: float               # [0.0, 1.0]
    reason: str
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    residual: Optional[float] = None
    z_score: Optional[float] = None
    rate_of_change: Optional[float] = None
    is_stuck: bool = False
    dropout_count: int = 0
    provenance_classification: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trust_score": round(self.trust_score, 1),
            "status": self.status,
            "fault_type": self.fault_type.value,
            "fault_severity": round(self.fault_severity, 3),
            "reason": self.reason,
            "observed_value": round(self.observed_value, 2) if self.observed_value is not None else None,
            "expected_value": round(self.expected_value, 2) if self.expected_value is not None else None,
            "residual": round(self.residual, 2) if self.residual is not None else None,
            "z_score": round(self.z_score, 2) if self.z_score is not None else None,
            "rate_of_change": round(self.rate_of_change, 3) if self.rate_of_change is not None else None,
            "is_stuck": self.is_stuck,
            "provenance_classification": self.provenance_classification,
        }


@dataclass
class CrossSensorRuleResult:
    """Evaluation result of an explicit physical cross-sensor rule."""
    rule_name: str
    primary_channel: str
    corroborating_channels: List[str]
    passed: bool
    severity: float                     # [0.0, 1.0]
    physical_rationale: str
    operating_state: str
    provenance: ThresholdProvenance
    failure_interpretation: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "primary_channel": self.primary_channel,
            "corroborating_channels": self.corroborating_channels,
            "passed": self.passed,
            "severity": round(self.severity, 2),
            "physical_rationale": self.physical_rationale,
            "operating_state": self.operating_state,
            "provenance": self.provenance.value,
            "failure_interpretation": self.failure_interpretation,
        }


@dataclass
class SensorFaultIsolationResult:
    """Comprehensive, authoritative output of the Sensor Fault Isolation Engine."""
    overall_trust_score: float          # [0.0, 100.0]
    overall_status: str                 # "TRUSTED", "CHECK", "SUSPECT"
    verdict: EngineAttributionVerdict   # NOMINAL, SENSOR_FAULT_ISOLATED, etc.
    engine_vs_sensor_verdict: str       # String representation for downstream compatibility
    suspect_sensors: List[str]
    suspect_channels: List[str]
    is_sensor_fault_only: bool          # Backward-compatibility flag
    bulk_physics_rms_z: float           # RMS z-score strictly across TRUSTED channels
    clean_rms_z: float                  # Alias for bulk_physics_rms_z
    all_channels_rms_z: float           # Raw unmitigated RMS z-score including faulty sensors
    trusted_channel_count: int
    total_monitored_channels: int
    trusted_channel_fraction: float
    excluded_channel_count: int
    engine_profile: str
    operating_phase: str
    sensors: List[Dict[str, Any]]       # Backward-compatible sensor list
    channels: List[Dict[str, Any]]      # Backward-compatible channel list
    active_fault_modes: Dict[str, str]  # Map of channel -> fault_type
    virtual_sensor_estimates: Dict[str, float]
    cross_sensor_rule_evaluations: List[Dict[str, Any]]
    uncertainty_penalty: float          # Recommended prognostic spread expansion factor
    confidence_multiplier: float        # Recommended prognostic confidence multiplier
    diagnostic_advisory: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overall_trust_score": round(self.overall_trust_score, 1),
            "overall_status": self.overall_status,
            "verdict": self.verdict.value,
            "engine_vs_sensor_verdict": self.engine_vs_sensor_verdict,
            "suspect_sensors": self.suspect_sensors,
            "suspect_channels": self.suspect_channels,
            "is_sensor_fault_only": self.is_sensor_fault_only,
            "bulk_physics_rms_z": round(self.bulk_physics_rms_z, 3),
            "clean_rms_z": round(self.clean_rms_z, 3),
            "all_channels_rms_z": round(self.all_channels_rms_z, 3),
            "trusted_channel_count": self.trusted_channel_count,
            "total_monitored_channels": self.total_monitored_channels,
            "trusted_channel_fraction": round(self.trusted_channel_fraction, 3),
            "excluded_channel_count": self.excluded_channel_count,
            "engine_profile": self.engine_profile,
            "operating_phase": self.operating_phase,
            "sensors": self.sensors,
            "channels": self.channels,
            "active_fault_modes": self.active_fault_modes,
            "virtual_sensor_estimates": {k: round(v, 2) for k, v in self.virtual_sensor_estimates.items()},
            "cross_sensor_rule_evaluations": self.cross_sensor_rule_evaluations,
            "uncertainty_penalty": round(self.uncertainty_penalty, 3),
            "confidence_multiplier": round(self.confidence_multiplier, 3),
            "diagnostic_advisory": self.diagnostic_advisory,
        }


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Converts a value to float safely, returning default if null, non-numeric, or non-finite."""
    if val is None:
        return default
    try:
        fv = float(val)
        if math.isfinite(fv):
            return fv
        return default
    except (TypeError, ValueError):
        return default


# =============================================================================
# 5. TEMPORAL CAUSAL ROLLING BUFFER
# =============================================================================

class CausalChannelHistory:
    """Strictly causal temporal history for a single channel. Zero future leakage."""

    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self.values: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)

    def push(self, val: Optional[float], ts: float) -> None:
        self.values.append(val)
        self.timestamps.append(ts)

    def clear(self) -> None:
        self.values.clear()
        self.timestamps.clear()

    @property
    def valid_numeric_values(self) -> List[float]:
        nums: List[float] = []
        for v in self.values:
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        nums.append(fv)
                except (TypeError, ValueError):
                    pass
        return nums

    def compute_rate_of_change(self) -> Optional[float]:
        """Calculates instantaneous dX/dt using last two valid causal samples."""
        if len(self.values) < 2:
            return None
        try:
            v_curr = float(self.values[-1])
            v_prev = float(self.values[-2])
            if not (math.isfinite(v_curr) and math.isfinite(v_prev)):
                return None
        except (TypeError, ValueError):
            return None

        try:
            t_curr = float(self.timestamps[-1])
            t_prev = float(self.timestamps[-2])
            dt = t_curr - t_prev
            if dt <= 1e-4 or dt > 30.0:
                return None
        except (TypeError, ValueError):
            return None
        return (v_curr - v_prev) / dt

    def is_frozen_stuck(self, min_samples: int = 6, variance_eps: float = 1e-6) -> bool:
        """Checks if the sensor output has zero variance across last min_samples."""
        nums = self.valid_numeric_values
        if len(nums) < min_samples:
            return False
        tail = nums[-min_samples:]
        mean = sum(tail) / len(tail)
        var = sum((x - mean) ** 2 for x in tail) / len(tail)
        return var < variance_eps

    def compute_causal_drift_slope(self, min_samples: int = 8) -> Optional[float]:
        """Calculates linear drift slope dX/dt via causal least squares."""
        nums = self.valid_numeric_values
        if len(nums) < min_samples:
            return None
        tail_vals = nums[-min_samples:]
        tail_ts = list(self.timestamps)[-len(tail_vals):]
        t0 = tail_ts[0]
        rel_ts = [t - t0 for t in tail_ts]
        n = len(tail_vals)
        sum_t = sum(rel_ts)
        sum_v = sum(tail_vals)
        sum_tv = sum(t * v for t, v in zip(rel_ts, tail_vals))
        sum_t2 = sum(t * t for t in rel_ts)
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-6:
            return 0.0
        slope = (n * sum_tv - sum_t * sum_v) / denom
        return slope

    def is_intermittent_chattering(self, min_samples: int = 8) -> bool:
        """Detects high-frequency alternating between valid and dropout/None."""
        if len(self.values) < min_samples:
            return False
        tail = list(self.values)[-min_samples:]
        transitions = 0
        def _is_missing(val: Any) -> bool:
            if val is None:
                return True
            try:
                fv = float(val)
                return not math.isfinite(fv)
            except (TypeError, ValueError):
                return True

        for i in range(1, len(tail)):
            prev_missing = _is_missing(tail[i - 1])
            curr_missing = _is_missing(tail[i])
            if prev_missing != curr_missing:
                transitions += 1
        return transitions >= 3


# =============================================================================
# 6. ANALYTIC REDUNDANCY & VIRTUAL SENSORS (DEPENDENCY-AWARE)
# =============================================================================

class DependencyAwareVirtualSensors:
    """
    Computes analytic redundancy estimates from first-principles and empirical physics.
    Crucial Safety Invariant:
      If a predictor sensor is UNTRUSTED (trust < 60), the virtual sensor refuses
      to declare another sensor healthy using corrupted inputs!
    """

    @classmethod
    def estimate_oil_pressure(
        cls,
        rpm: float,
        oil_temp: float,
        rpm_trust: float,
        oil_temp_trust: float,
        profile: str = "Rotax-914-Turbo-115HP",
    ) -> Tuple[Optional[float], str]:
        """
        Positive-displacement oil pump characteristic:
        P_oil = P_relief * tanh(RPM / RPM_knee) * (1.0 - 0.0015 * (T_oil - 180.0))
        """
        if rpm_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: Engine_RPM is untrusted"
        if oil_temp_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: Oil_Temp is untrusted"

        if "continental" in profile.lower() or "360" in profile.lower():
            # Continental TSIO-360
            p_relief = 55.0
            rpm_knee = 1800.0
            t_ref = 175.0
        else:
            # Rotax 914
            p_relief = 65.0
            rpm_knee = 2200.0
            t_ref = 190.0

        rpm_factor = math.tanh(max(0.0, rpm) / rpm_knee)
        temp_factor = max(0.6, min(1.3, 1.0 - 0.0018 * (oil_temp - t_ref)))
        est_p = p_relief * rpm_factor * temp_factor
        return round(est_p, 1), "VALID"

    @classmethod
    def estimate_fuel_flow(
        cls,
        rpm: float,
        map_inhg: float,
        rpm_trust: float,
        map_trust: float,
        profile: str = "Rotax-914-Turbo-115HP",
    ) -> Tuple[Optional[float], str]:
        """
        Brake power proportional fuel delivery:
        Fuel_Flow ~ k * (RPM / RPM_nom) * (MAP / MAP_nom)
        """
        if rpm_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: Engine_RPM is untrusted"
        if map_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: MAP_Injector is untrusted"

        if "continental" in profile.lower() or "360" in profile.lower():
            # Continental TSIO-360: up to 75 L/h at rated power (2700 RPM, 38 inHg)
            ff_max = 75.0
            rpm_ratio = max(0.0, rpm / 2700.0)
            map_ratio = max(0.0, map_inhg / 38.0)
        else:
            # Rotax 914: up to 38 L/h at takeoff (5800 RPM, 39 inHg)
            ff_max = 38.0
            rpm_ratio = max(0.0, rpm / 5800.0)
            map_ratio = max(0.0, map_inhg / 39.0)

        est_ff = max(2.0, ff_max * (0.15 + 0.85 * rpm_ratio * map_ratio))
        return round(est_ff, 1), "VALID"

    @classmethod
    def estimate_cht(
        cls,
        rpm: float,
        map_inhg: float,
        rpm_trust: float,
        map_trust: float,
        profile: str = "Rotax-914-Turbo-115HP",
    ) -> Tuple[Optional[float], str]:
        """
        Thermodynamic heat rejection estimate for CHT:
        T_head ~ T_base + Delta_T * (Power_fraction)
        """
        if rpm_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: Engine_RPM is untrusted"
        if map_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: MAP_Injector is untrusted"

        if "continental" in profile.lower() or "360" in profile.lower():
            # Air-cooled Continental heads run 300 - 420 °F
            t_base = 250.0
            delta_t = 160.0
            load = (rpm / 2700.0) * (map_inhg / 38.0)
        else:
            # Liquid-cooled Rotax heads run 160 - 240 °F
            t_base = 160.0
            delta_t = 85.0
            load = (rpm / 5800.0) * (map_inhg / 39.0)

        est_cht = t_base + delta_t * max(0.1, min(1.2, load))
        return round(est_cht, 1), "VALID"

    @classmethod
    def estimate_battery_voltage(
        cls,
        current: float,
        current_trust: float,
    ) -> Tuple[Optional[float], str]:
        """
        Regulated 28V DC bus estimate:
        V_bus ~ 27.8 - R_bus * I_draw
        """
        if current_trust < 60.0:
            return None, "DEPENDENCY_UNTRUSTED: Battery_Current is untrusted"
        r_bus = 0.015  # 15 mOhm internal resistance
        est_v = 28.0 - r_bus * current
        return round(max(24.0, min(30.0, est_v)), 2), "VALID"


# =============================================================================
# 7. SENSOR FAULT ISOLATION ENGINE
# =============================================================================

class SensorFaultIsolationEngine:
    """
    Master Sensor Fault Isolation & Fault-Tolerant Analytics Engine.
    Executes multi-stage forensic validation to distinguish:
      1. NOMINAL
      2. SENSOR_FAULT_ISOLATED
      3. ENGINE_DEGRADATION_CONFIRMED
      4. COMPOUND_FAULT
      5. INSUFFICIENT_OBSERVABILITY
    """

    CORE_CHANNELS = [
        "Engine_RPM",
        "MAP_Injector",
        "CHT",
        "EGT1",
        "EGT2",
        "EGT3",
        "Oil_Pressure",
        "Oil_Temp",
        "Fuel_Flow",
        "Battery_Voltage",
        "Battery_Current",
        "Alternator_Temp",
        "EFI_Water_Temp",
    ]

    def __init__(self, default_profile: str = "Rotax-914-Turbo-115HP"):
        self.default_profile = default_profile
        self._histories: Dict[str, Dict[str, CausalChannelHistory]] = {}
        self._prev_ts: Dict[str, float] = {}

    def _get_history(self, engine_id: str, channel: str) -> CausalChannelHistory:
        if engine_id not in self._histories:
            self._histories[engine_id] = {}
        if channel not in self._histories[engine_id]:
            self._histories[engine_id][channel] = CausalChannelHistory(maxlen=25)
        return self._histories[engine_id][channel]

    def reset(self, engine_id: Optional[str] = None) -> None:
        """Resets causal state to prevent cross-mission / cross-engine leakage."""
        if engine_id:
            self._histories.pop(engine_id, None)
            self._prev_ts.pop(engine_id, None)
        else:
            self._histories.clear()
            self._prev_ts.clear()

    # -------------------------------------------------------------------------
    # Core Analysis Pipeline
    # -------------------------------------------------------------------------

    def analyze(
        self,
        telemetry: Dict[str, Any],
        twin_assessment: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SensorFaultIsolationResult:
        """
        Executes end-to-end sensor health and fault isolation analysis.
        Strict zero-leakage contract: uses ONLY observable signals.
        """
        context = context or {}
        twin = twin_assessment or {}
        z_dict = twin.get("z_scores", {})
        raw_rms_z = float(twin.get("residual_rms", 0.0))

        # 1. Resolve engine profile and operating phase
        engine_id = str(context.get("engine_id") or telemetry.get("engine_id") or self.default_profile)
        profile_key = "Continental-TSIO-360-MB" if ("continental" in engine_id.lower() or "360" in engine_id.lower()) else "Rotax-914-Turbo-115HP"
        operating_phase = str(telemetry.get("Operating_State") or context.get("mission_phase") or "CRUISE").upper()
        rapid_throttle = bool(context.get("rapid_throttle", False) or telemetry.get("rapid_throttle", False))

        # 2. Timestamp and timeline tracking
        has_explicit_time = ("GPS_Time" in telemetry or "timestamp" in telemetry)
        sim_time_s = float(telemetry.get("GPS_Time", telemetry.get("timestamp", 0.0)))
        prev_t = self._prev_ts.get(engine_id, sim_time_s)
        dt = sim_time_s - prev_t
        if dt < -1e-4 or (has_explicit_time and dt > 60.0):
            # Timeline rewound or mission discontinuity: reset history to avoid leakage
            self.reset(engine_id)
        self._prev_ts[engine_id] = sim_time_s

        # 3. Retrieve engine-specific threshold registry
        spec_dict = THRESHOLD_REGISTRY.get(profile_key, THRESHOLD_REGISTRY["Rotax-914-Turbo-115HP"])

        # 4. Phase 1 & 2: Individual Channel Assessment (Range, Slew, Temporal, Dropout, Stuck)
        assessments: Dict[str, ChannelAssessment] = {}

        # Channels to assess: all channels provided in telemetry that match spec_dict or CORE_CHANNELS
        channels_to_evaluate = [k for k in telemetry.keys() if k in spec_dict or k in self.CORE_CHANNELS]
        if not channels_to_evaluate:
            channels_to_evaluate = list(self.CORE_CHANNELS)

        for name in channels_to_evaluate:
            val = telemetry.get(name)
            hist = self._get_history(engine_id, name)
            hist.push(val, sim_time_s)

            spec = spec_dict.get(name)
            z_val = abs(float(z_dict.get(name, 0.0)))

            # Case 1: Missing / Dropout / NaN / Inf / Non-numeric
            fval = _safe_float(val)
            if fval is None:
                assessments[name] = ChannelAssessment(
                    name=name,
                    trust_score=5.0,
                    status="SUSPECT",
                    fault_type=SensorFaultType.DROPOUT,
                    fault_severity=0.95,
                    reason=f"{name} transducer signal dropped out (null/NaN/Inf/non-numeric)",
                    observed_value=None,
                    z_score=z_val,
                    provenance_classification=spec.provenance.value if spec else "D_HEURISTIC",
                )
                continue

            # Vibration accelerometer special check (when present in telemetry)
            if name == "Vibration":
                rpm_z = abs(float(z_dict.get("Engine_RPM", 0.0)))
                op_z = abs(float(z_dict.get("Oil_Pressure", 0.0)))
                if fval < 0.05 or fval > 15.0:
                    assessments[name] = ChannelAssessment(
                        name=name,
                        trust_score=10.0,
                        status="SUSPECT",
                        fault_type=SensorFaultType.DROPOUT,
                        fault_severity=0.90,
                        reason="Vibration accelerometer disconnected or dropout",
                        observed_value=fval,
                        z_score=z_val,
                        provenance_classification=spec.provenance.value if spec else "C_MODEL_DERIVED",
                    )
                    continue
                elif z_val > 2.8 and rpm_z < 1.2 and op_z < 1.2:
                    assessments[name] = ChannelAssessment(
                        name=name,
                        trust_score=30.0,
                        status="SUSPECT",
                        fault_type=SensorFaultType.CROSS_SENSOR_INCONSISTENCY,
                        fault_severity=0.70,
                        reason="High accelerometer vibration spike without mechanical RPM or lubrication anomaly",
                        observed_value=fval,
                        z_score=z_val,
                        provenance_classification=spec.provenance.value if spec else "C_MODEL_DERIVED",
                    )
                    continue

            # Case 2: Implausible Zero / Negative / Out of Physical Bounds
            if spec is not None:
                if fval < spec.critical_min or fval > spec.critical_max:
                    assessments[name] = ChannelAssessment(
                        name=name,
                        trust_score=10.0,
                        status="SUSPECT",
                        fault_type=SensorFaultType.DROPOUT if fval <= 0.0 and spec.critical_min > 0.0 else SensorFaultType.SPIKE_OUTLIER,
                        fault_severity=0.90,
                        reason=f"{name} value {fval} outside physical bounds [{spec.critical_min}, {spec.critical_max}] {spec.unit}",
                        observed_value=fval,
                        z_score=z_val,
                        provenance_classification=spec.provenance.value,
                    )
                    continue

            # Case 3: Slew Rate / Implausible Rate-of-Change
            roc = hist.compute_rate_of_change()
            roc_viol = False
            if roc is not None and spec is not None:
                # Widen slew limit during rapid throttle transition
                slew_limit = spec.max_slew_per_sec * (2.5 if rapid_throttle else 1.0)
                if abs(roc) > slew_limit:
                    roc_viol = True

            # Case 4: Frozen / Stuck-at Value
            # Evaluated after collecting all channel histories below
            is_stuck = False

            # Case 5: Intermittent Chattering
            is_intermittent = hist.is_intermittent_chattering(min_samples=8)

            # Case 6: Causal Drift
            drift_slope = hist.compute_causal_drift_slope(min_samples=8)

            # Determine preliminary score
            score = 100.0
            f_type = SensorFaultType.NOMINAL
            f_sev = 0.0
            reason = "Channel telemetry within nominal envelope"

            if is_intermittent:
                score = 15.0
                f_type = SensorFaultType.INTERMITTENT
                f_sev = 0.85
                reason = f"{name} intermittent transducer chattering detected across consecutive cycles"
            elif roc_viol:
                score = 30.0
                f_type = SensorFaultType.IMPLAUSIBLE_RATE_OF_CHANGE
                f_sev = 0.70
                reason = f"{name} rate-of-change ({round(roc, 1)} {spec.unit}/s) exceeded physical dynamic limit ({round(slew_limit, 1)})"
            elif drift_slope is not None and abs(drift_slope) > (spec.max_slew_per_sec * 0.15 if spec else 5.0):
                score = 45.0
                f_type = SensorFaultType.DRIFT
                f_sev = 0.55
                reason = f"{name} temporal drift slope ({round(drift_slope, 2)}/s) indicates transducer calibration loss"

            assessments[name] = ChannelAssessment(
                name=name,
                trust_score=score,
                status="TRUSTED" if score >= 80 else ("CHECK" if score >= 55 else "SUSPECT"),
                fault_type=f_type,
                fault_severity=f_sev,
                reason=reason,
                observed_value=fval,
                z_score=z_val,
                rate_of_change=roc,
                is_stuck=is_stuck,
                provenance_classification=spec.provenance.value if spec else "D_HEURISTIC",
            )

        # Context-Aware Stuck-At Detection:
        # Avoids falsely flagging steady-state simulation channels as stuck.
        # An isolated channel is STUCK_AT if:
        #   a) It has sample variance < 1e-6 over >= 8 samples
        #   b) AND (rpm is dynamic or only 1-2 channels are flat)
        rpm_hist = self._get_history(engine_id, "Engine_RPM")
        rpm_nums = rpm_hist.valid_numeric_values
        rpm_dynamic = False
        if len(rpm_nums) >= 6:
            rpm_mean = sum(rpm_nums[-6:]) / 6.0
            rpm_var = sum((x - rpm_mean) ** 2 for x in rpm_nums[-6:]) / 6.0
            rpm_dynamic = (rpm_var > 10.0)

        flat_channels = [
            ch for ch in channels_to_evaluate
            if self._get_history(engine_id, ch).is_frozen_stuck(min_samples=8, variance_eps=1e-6)
        ]

        for ch in flat_channels:
            if ch in assessments and assessments[ch].fault_type == SensorFaultType.NOMINAL:
                # If RPM is dynamic, coupled channels (Fuel_Flow, MAP, Oil_Pressure, RPM) must vary!
                is_isolated = (1 <= len(flat_channels) <= 3)
                is_coupled_to_dynamic_rpm = (rpm_dynamic and ch in ("Fuel_Flow", "MAP_Injector", "Oil_Pressure", "Engine_RPM"))
                if is_isolated or is_coupled_to_dynamic_rpm:
                    assessments[ch].is_stuck = True
                    assessments[ch].trust_score = min(assessments[ch].trust_score, 25.0)
                    assessments[ch].status = "SUSPECT"
                    assessments[ch].fault_type = SensorFaultType.STUCK_AT
                    assessments[ch].fault_severity = 0.75
                    assessments[ch].reason = f"{ch} frozen output detected (sample variance < 1e-6 over 8 cycles)"

        # ---------------------------------------------------------------------
        # 5. Phase 3: Analytic Redundancy (Virtual Sensors)
        # -------------------------------------------------------------------------
        virtual_estimates: Dict[str, float] = {}

        rpm_val = _safe_float(telemetry.get("Engine_RPM"), 0.0)
        rpm_trust = assessments.get("Engine_RPM", ChannelAssessment("Engine_RPM", 100.0, "TRUSTED", SensorFaultType.NOMINAL, 0.0, "")).trust_score

        oil_t_val = _safe_float(telemetry.get("Oil_Temp"), 180.0)
        oil_t_trust = assessments.get("Oil_Temp", ChannelAssessment("Oil_Temp", 100.0, "TRUSTED", SensorFaultType.NOMINAL, 0.0, "")).trust_score

        map_val = _safe_float(telemetry.get("MAP_Injector"), 30.0)
        map_trust = assessments.get("MAP_Injector", ChannelAssessment("MAP_Injector", 100.0, "TRUSTED", SensorFaultType.NOMINAL, 0.0, "")).trust_score

        curr_val = _safe_float(telemetry.get("Battery_Current"), 0.0)
        curr_trust = assessments.get("Battery_Current", ChannelAssessment("Battery_Current", 100.0, "TRUSTED", SensorFaultType.NOMINAL, 0.0, "")).trust_score

        # 1. Virtual Oil Pressure
        v_op, op_status = DependencyAwareVirtualSensors.estimate_oil_pressure(
            rpm=rpm_val or 0.0,
            oil_temp=oil_t_val or 180.0,
            rpm_trust=rpm_trust,
            oil_temp_trust=oil_t_trust,
            profile=profile_key,
        )
        if v_op is not None:
            virtual_estimates["Oil_Pressure"] = v_op
            # Compare actual vs virtual
            if "Oil_Pressure" in assessments and assessments["Oil_Pressure"].trust_score > 30.0:
                act_op = assessments["Oil_Pressure"].observed_value
                if act_op is not None:
                    resid = abs(act_op - v_op)
                    assessments["Oil_Pressure"].expected_value = v_op
                    assessments["Oil_Pressure"].residual = resid
                    # If discrepancy is large but predictors are trusted, penalize Oil_Pressure
                    if resid > 30.0 and act_op < 15.0 and (rpm_val or 0.0) > 2000.0:
                        assessments["Oil_Pressure"].trust_score = 15.0
                        assessments["Oil_Pressure"].status = "SUSPECT"
                        assessments["Oil_Pressure"].fault_type = SensorFaultType.DROPOUT
                        assessments["Oil_Pressure"].fault_severity = 0.85
                        assessments["Oil_Pressure"].reason = f"Oil Pressure ({act_op} psi) collapsed while positive displacement virtual model predicts {v_op} psi"

        # 2. Virtual Fuel Flow
        v_ff, ff_status = DependencyAwareVirtualSensors.estimate_fuel_flow(
            rpm=rpm_val or 0.0,
            map_inhg=map_val or 30.0,
            rpm_trust=rpm_trust,
            map_trust=map_trust,
            profile=profile_key,
        )
        if v_ff is not None:
            virtual_estimates["Fuel_Flow"] = v_ff
            if "Fuel_Flow" in assessments and assessments["Fuel_Flow"].trust_score > 30.0:
                act_ff = assessments["Fuel_Flow"].observed_value
                if act_ff is not None:
                    resid = abs(act_ff - v_ff)
                    assessments["Fuel_Flow"].expected_value = v_ff
                    assessments["Fuel_Flow"].residual = resid
                    if resid > 25.0 and act_ff < 2.0 and (rpm_val or 0.0) > 2500.0:
                        assessments["Fuel_Flow"].trust_score = 15.0
                        assessments["Fuel_Flow"].status = "SUSPECT"
                        assessments["Fuel_Flow"].fault_type = SensorFaultType.DROPOUT
                        assessments["Fuel_Flow"].fault_severity = 0.85
                        assessments["Fuel_Flow"].reason = f"Fuel flow transducer dropped out (<2 L/h) while engine operating at {rpm_val} RPM"

        # 3. Virtual CHT
        v_cht, cht_status = DependencyAwareVirtualSensors.estimate_cht(
            rpm=rpm_val or 0.0,
            map_inhg=map_val or 30.0,
            rpm_trust=rpm_trust,
            map_trust=map_trust,
            profile=profile_key,
        )
        if v_cht is not None:
            virtual_estimates["CHT"] = v_cht
            if "CHT" in assessments and assessments["CHT"].trust_score > 30.0:
                act_cht = assessments["CHT"].observed_value
                if act_cht is not None:
                    resid = abs(act_cht - v_cht)
                    assessments["CHT"].expected_value = v_cht
                    assessments["CHT"].residual = resid

        # 4. Virtual Bus Voltage
        v_v, v_status = DependencyAwareVirtualSensors.estimate_battery_voltage(
            current=curr_val or 0.0,
            current_trust=curr_trust,
        )
        if v_v is not None:
            virtual_estimates["Battery_Voltage"] = v_v
            if "Battery_Voltage" in assessments and assessments["Battery_Voltage"].trust_score > 30.0:
                act_v = assessments["Battery_Voltage"].observed_value
                if act_v is not None:
                    resid = abs(act_v - v_v)
                    assessments["Battery_Voltage"].expected_value = v_v
                    assessments["Battery_Voltage"].residual = resid

        # ---------------------------------------------------------------------
        # 6. Phase 4: Explicit Cross-Sensor Physical Consistency Rules
        # -------------------------------------------------------------------------
        rule_results: List[CrossSensorRuleResult] = []

        # Rule 1: RPM <-> MAP Coupling (in cruise/high load)
        if rpm_val is not None and map_val is not None:
            f_rpm = rpm_val
            f_map = map_val
            if operating_phase in ("CRUISE", "HIGH_ALTITUDE", "TAKEOFF", "CLIMB") and f_rpm > 3500.0:
                # Engine running at high RPM requires positive MAP
                passed = f_map >= 18.0
                sev = 0.85 if not passed else 0.0
                rule_results.append(CrossSensorRuleResult(
                    rule_name="RPM_MAP_COUPLING",
                    primary_channel="MAP_Injector",
                    corroborating_channels=["Engine_RPM"],
                    passed=passed,
                    severity=sev,
                    physical_rationale="High crankshaft speed under flight load requires intake manifold pressure >= 18 inHg",
                    operating_state=operating_phase,
                    provenance=ThresholdProvenance.C_MODEL_DERIVED,
                    failure_interpretation="MAP transducer leak, line disconnection, or sensor dropout",
                ))
                if not passed and "MAP_Injector" in assessments:
                    assessments["MAP_Injector"].trust_score = min(assessments["MAP_Injector"].trust_score, 20.0)
                    assessments["MAP_Injector"].status = "SUSPECT"
                    assessments["MAP_Injector"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                    assessments["MAP_Injector"].reason = f"MAP ({f_map} inHg) implausibly low for high engine speed ({f_rpm} RPM)"

        # Rule 2: Multi-Channel EGT Balance
        egt_keys = [f"EGT{i}" for i in range(1, 5) if f"EGT{i}" in telemetry]
        egt_vals = [_safe_float(telemetry[k]) for k in egt_keys if _safe_float(telemetry[k]) is not None]
        if len(egt_vals) >= 3:
            mean_egt = sum(egt_vals) / len(egt_vals)
            max_spread = max(egt_vals) - min(egt_vals)
            # Check individual cylinder outlier
            for k in egt_keys:
                v_k = _safe_float(telemetry.get(k))
                if v_k is not None:
                    peer_vals = [_safe_float(telemetry[x]) for x in egt_keys if x != k and _safe_float(telemetry.get(x)) is not None]
                    if peer_vals:
                        peer_mean = sum(peer_vals) / len(peer_vals)
                        dev = abs(v_k - peer_mean)
                        # An isolated thermocouple drop > 400 °F while peers are tight (< 80 °F) is a thermocouple fault
                        peer_spread = max(peer_vals) - min(peer_vals)
                        if dev > 350.0 and peer_spread < 100.0:
                            if k in assessments:
                                assessments[k].trust_score = min(assessments[k].trust_score, 20.0)
                                assessments[k].status = "SUSPECT"
                                assessments[k].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                                assessments[k].reason = f"Single cylinder {k} ({v_k} °F) deviates {round(dev, 1)} °F from peer cylinders ({round(peer_mean, 1)} °F)"

        # Rule 3: CHT Thermal Corroboration
        # A real engine overheating event elevates CHT, EGT, and Coolant/Oil temperatures together.
        # An isolated CHT jump with flat coolant and oil temps is a thermocouple fault.
        if "CHT" in telemetry:
            cht_f = _safe_float(telemetry["CHT"])
            cht_z = float(z_dict.get("CHT", 0.0))
            water_z = float(z_dict.get("EFI_Water_Temp", 0.0))
            oil_z = float(z_dict.get("Oil_Temp", 0.0))
            avg_egt_z = sum(abs(float(z_dict.get(k, 0.0))) for k in egt_keys) / max(1, len(egt_keys))

            if abs(cht_z) > 3.0 and abs(water_z) < 1.2 and abs(oil_z) < 1.2 and avg_egt_z < 1.5:
                passed = False
                rule_results.append(CrossSensorRuleResult(
                    rule_name="CHT_THERMAL_CORROBORATION",
                    primary_channel="CHT",
                    corroborating_channels=["EFI_Water_Temp", "Oil_Temp", "EGT"],
                    passed=False,
                    severity=0.80,
                    physical_rationale="True combustion heat rejection transfers to coolant and oil circuits within thermal time constant",
                    operating_state=operating_phase,
                    provenance=ThresholdProvenance.C_MODEL_DERIVED,
                    failure_interpretation="Isolated CHT thermocouple detachment or junction drift without bulk engine heat rejection",
                ))
                if "CHT" in assessments:
                    assessments["CHT"].trust_score = min(assessments["CHT"].trust_score, 25.0)
                    assessments["CHT"].status = "SUSPECT"
                    assessments["CHT"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                    assessments["CHT"].reason = "CHT excursion uncorroborated by coolant, oil, or exhaust gas temperatures"
            else:
                rule_results.append(CrossSensorRuleResult(
                    rule_name="CHT_THERMAL_CORROBORATION",
                    primary_channel="CHT",
                    corroborating_channels=["EFI_Water_Temp", "Oil_Temp", "EGT"],
                    passed=True,
                    severity=0.0,
                    physical_rationale="CHT aligns with coupled thermal circuits",
                    operating_state=operating_phase,
                    provenance=ThresholdProvenance.C_MODEL_DERIVED,
                    failure_interpretation="Nominal thermal coupling",
                ))

        # Rule 4: Oil Pressure vs RPM Mechanical Drive
        if rpm_val is not None and "Oil_Pressure" in telemetry:
            f_rpm = rpm_val
            f_op = _safe_float(telemetry.get("Oil_Pressure"), 0.0) or 0.0
            if f_rpm > 2000.0 and f_op < 10.0:
                rule_results.append(CrossSensorRuleResult(
                    rule_name="OIL_PRESSURE_RPM_PUMP",
                    primary_channel="Oil_Pressure",
                    corroborating_channels=["Engine_RPM"],
                    passed=False,
                    severity=0.90,
                    physical_rationale="Positive displacement oil pump geared to crankshaft builds pressure above idle",
                    operating_state=operating_phase,
                    provenance=ThresholdProvenance.A_EXTERNAL,
                    failure_interpretation="Oil pressure transducer diaphragm failure or signal wire severance",
                ))
                if "Oil_Pressure" in assessments:
                    assessments["Oil_Pressure"].trust_score = min(assessments["Oil_Pressure"].trust_score, 15.0)
                    assessments["Oil_Pressure"].status = "SUSPECT"
                    assessments["Oil_Pressure"].fault_type = SensorFaultType.DROPOUT
                    assessments["Oil_Pressure"].reason = f"Oil Pressure ({f_op} psi) collapsed while engine turning at {f_rpm} RPM"

        # Rule 5: DC Bus Voltage vs Battery Current
        if "Battery_Voltage" in telemetry and "Battery_Current" in telemetry:
            f_v = _safe_float(telemetry.get("Battery_Voltage"), 28.0) or 28.0
            f_i = _safe_float(telemetry.get("Battery_Current"), 0.0) or 0.0
            # If voltage drops significantly (<23.0V) without substantial discharge current (f_i < -5A), voltage transducer is suspect
            if f_v < 23.0 and f_i >= 0.0:
                if "Battery_Voltage" in assessments:
                    assessments["Battery_Voltage"].trust_score = min(assessments["Battery_Voltage"].trust_score, 30.0)
                    assessments["Battery_Voltage"].status = "SUSPECT"
                    assessments["Battery_Voltage"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                    assessments["Battery_Voltage"].reason = f"Battery voltage dropped to {f_v} V with zero discharge current ({f_i} A)"

        # Statistical Twin Z-Score Multi-Channel Corroboration (when twin is provided)
        water_z = abs(float(z_dict.get("EFI_Water_Temp", 0.0)))
        oil_t_z = abs(float(z_dict.get("Oil_Temp", 0.0)))
        oil_p_z = abs(float(z_dict.get("Oil_Pressure", 0.0)))
        rpm_z = abs(float(z_dict.get("Engine_RPM", 0.0)))
        map_z = abs(float(z_dict.get("MAP_Injector", 0.0)))
        egt_zs = [abs(float(z_dict.get(k, 0.0))) for k in egt_keys if k in z_dict]
        avg_egt_abs = sum(egt_zs) / max(1, len(egt_zs))

        # EGT peer z-scores
        for k in egt_keys:
            if k in z_dict and k in assessments:
                k_z = abs(float(z_dict[k]))
                others_z = [abs(float(z_dict[x])) for x in egt_keys if x != k and x in z_dict]
                if k_z > 3.5 and (not others_z or max(others_z) < 1.5):
                    assessments[k].trust_score = min(assessments[k].trust_score, 35.0)
                    assessments[k].status = "SUSPECT"
                    if assessments[k].fault_type == SensorFaultType.NOMINAL:
                        assessments[k].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                    assessments[k].reason = "Single-channel EGT deviation inconsistent with peer channels"

        # CHT uncorroborated z-score
        cht_z = abs(float(z_dict.get("CHT", 0.0)))
        if cht_z > 2.5 and avg_egt_abs < 1.2 and water_z < 1.2:
            if "CHT" in assessments:
                assessments["CHT"].trust_score = min(assessments["CHT"].trust_score, 30.0)
                assessments["CHT"].status = "SUSPECT"
                if assessments["CHT"].fault_type == SensorFaultType.NOMINAL:
                    assessments["CHT"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                assessments["CHT"].reason = "CHT spike without corroboration in coolant or exhaust gas temps"

        # EFI_Water_Temp uncorroborated z-score
        if water_z > 2.5 and avg_egt_abs < 1.2 and oil_t_z < 1.2:
            if "EFI_Water_Temp" in assessments:
                assessments["EFI_Water_Temp"].trust_score = min(assessments["EFI_Water_Temp"].trust_score, 25.0)
                assessments["EFI_Water_Temp"].status = "SUSPECT"
                if assessments["EFI_Water_Temp"].fault_type == SensorFaultType.NOMINAL:
                    assessments["EFI_Water_Temp"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                assessments["EFI_Water_Temp"].reason = "Water-temperature excursion is not supported by other thermal channels"

        # Oil Pressure uncorroborated z-score
        if oil_p_z > 3.5 and oil_t_z < 0.8:
            if "Oil_Pressure" in assessments:
                assessments["Oil_Pressure"].trust_score = min(assessments["Oil_Pressure"].trust_score, 50.0)
                assessments["Oil_Pressure"].status = "SUSPECT"
                if assessments["Oil_Pressure"].fault_type == SensorFaultType.NOMINAL:
                    assessments["Oil_Pressure"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                assessments["Oil_Pressure"].reason = "Large oil-pressure deviation with little thermal corroboration"

        # Fuel Flow uncorroborated z-score
        ff_z = abs(float(z_dict.get("Fuel_Flow", 0.0)))
        if ff_z > 2.8 and map_z < 1.2 and avg_egt_abs < 1.2:
            if "Fuel_Flow" in assessments:
                assessments["Fuel_Flow"].trust_score = min(assessments["Fuel_Flow"].trust_score, 30.0)
                assessments["Fuel_Flow"].status = "SUSPECT"
                if assessments["Fuel_Flow"].fault_type == SensorFaultType.NOMINAL:
                    assessments["Fuel_Flow"].fault_type = SensorFaultType.CROSS_SENSOR_INCONSISTENCY
                assessments["Fuel_Flow"].reason = "Fuel flow transducer shift without manifold pressure or EGT response"

        # ---------------------------------------------------------------------
        # 7. Phase 5: Bulk Physics Residual Safety & Observability Gate
        # -------------------------------------------------------------------------
        assessment_dicts = [a.as_dict() for a in assessments.values()]
        suspects = [a.name for a in assessments.values() if a.status in ("SUSPECT", "FAILED")]
        trusted_channels = [a.name for a in assessments.values() if a.status == "TRUSTED"]

        total_monitored = len(assessments)
        trusted_count = len(trusted_channels)
        trusted_fraction = trusted_count / max(1, total_monitored)
        excluded_count = total_monitored - trusted_count

        # Compute Bulk Physics RMS z-score STRICTLY over TRUSTED channels
        trusted_z_squares = [
            (float(z_dict.get(ch, 0.0)) ** 2)
            for ch in trusted_channels
            if ch in z_dict
        ]
        if trusted_z_squares:
            bulk_rms_z = math.sqrt(sum(trusted_z_squares) / len(trusted_z_squares))
        else:
            bulk_rms_z = 0.0

        # Compute overall trust score
        if assessments:
            overall_trust = min(a.trust_score for a in assessments.values())
        else:
            overall_trust = 100.0

        overall_status = "TRUSTED" if overall_trust >= 80.0 else ("CHECK" if overall_trust >= 55.0 else "SUSPECT")

        # ---------------------------------------------------------------------
        # 8. Phase 6: Authoritative Attribution Logic (Cases A - H)
        # -------------------------------------------------------------------------
        advisories: List[str] = []
        active_fault_modes = {a.name: a.fault_type.value for a in assessments.values() if a.fault_type != SensorFaultType.NOMINAL}

        # Thresholds:
        # Minimum trusted fraction required for high-integrity engine determination: 0.60
        # Critical minimum channels required: at least 4
        if trusted_fraction < 0.60 or trusted_count < 4:
            verdict = EngineAttributionVerdict.INSUFFICIENT_OBSERVABILITY
            advisories.append(f"INSUFFICIENT_OBSERVABILITY: Only {trusted_count}/{total_monitored} channels trusted ({round(trusted_fraction*100, 1)}%). Cannot guarantee engine health.")
            uncertainty_penalty = 0.70  # +70% uncertainty spread
            confidence_multiplier = 0.35
            is_sensor_fault_only = False
        elif len(suspects) == 0:
            # All sensors healthy
            if bulk_rms_z >= 2.0:
                verdict = EngineAttributionVerdict.ENGINE_DEGRADATION_CONFIRMED
                advisories.append(f"ENGINE_DEGRADATION_CONFIRMED: Bulk physics residual RMS ({round(bulk_rms_z, 2)}) elevated across trusted sensor channels.")
                uncertainty_penalty = 0.10
                confidence_multiplier = 0.95
                is_sensor_fault_only = False
            else:
                verdict = EngineAttributionVerdict.NOMINAL
                advisories.append("NOMINAL: Telemetry aligned with thermodynamic digital twin expectations.")
                uncertainty_penalty = 0.0
                confidence_multiplier = 1.0
                is_sensor_fault_only = False
        else:
            # One or more sensors suspect
            if bulk_rms_z < 2.0:
                # Bulk engine physics normal on remaining trusted channels!
                verdict = EngineAttributionVerdict.SENSOR_FAULT_ISOLATED
                advisories.append(f"SENSOR_FAULT_ISOLATED: Isolated transducer fault on [{', '.join(suspects)}]. Bulk engine physics nominal (RMS z={round(bulk_rms_z, 2)}).")
                uncertainty_penalty = 0.40  # Widen uncertainty interval without RUL collapse
                confidence_multiplier = 0.70
                is_sensor_fault_only = True
            else:
                # Both bulk engine physics degraded AND sensors faulty!
                verdict = EngineAttributionVerdict.COMPOUND_FAULT
                advisories.append(f"COMPOUND_FAULT: Bulk engine degradation confirmed (RMS z={round(bulk_rms_z, 2)}) AND isolated sensor faults on [{', '.join(suspects)}].")
                uncertainty_penalty = 0.50
                confidence_multiplier = 0.60
                is_sensor_fault_only = False

        return SensorFaultIsolationResult(
            overall_trust_score=overall_trust,
            overall_status=overall_status,
            verdict=verdict,
            engine_vs_sensor_verdict=verdict.value,
            suspect_sensors=suspects,
            suspect_channels=suspects,
            is_sensor_fault_only=is_sensor_fault_only,
            bulk_physics_rms_z=bulk_rms_z,
            clean_rms_z=bulk_rms_z,
            all_channels_rms_z=raw_rms_z,
            trusted_channel_count=trusted_count,
            total_monitored_channels=total_monitored,
            trusted_channel_fraction=trusted_fraction,
            excluded_channel_count=excluded_count,
            engine_profile=profile_key,
            operating_phase=operating_phase,
            sensors=assessment_dicts,
            channels=assessment_dicts,
            active_fault_modes=active_fault_modes,
            virtual_sensor_estimates=virtual_estimates,
            cross_sensor_rule_evaluations=[r.as_dict() for r in rule_results],
            uncertainty_penalty=uncertainty_penalty,
            confidence_multiplier=confidence_multiplier,
            diagnostic_advisory=advisories,
        )


# Singleton engine instance
_GLOBAL_ISOLATION_ENGINE = SensorFaultIsolationEngine()


def get_sensor_fault_isolation_engine() -> SensorFaultIsolationEngine:
    """Returns the singleton SensorFaultIsolationEngine."""
    return _GLOBAL_ISOLATION_ENGINE
