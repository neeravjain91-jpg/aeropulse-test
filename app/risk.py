"""Reliability, Availability & Mission Risk Analytics Engine for AeroPulse-X.
Implements Austin (2010) Chapter 16 (Design for Reliability, pp. 205-216).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    return max(min_val, min(max_val, float(value)))


@dataclass
class ReliabilityMetrics:
    """System Reliability and Availability Metrics (Austin Ch 16)."""
    mtbf_hours: float
    failure_rate_per_100k_hours: float
    availability_pct: float
    catastrophic_risk_tier: str  # "Catastrophic (10^-9)", "Class A (10^-5)", "Class B (10^-3)"
    mission_risk_score: float
    risk_level: str


def calculate_system_availability(num_failures_100k: float, mttr_hours: float = 4.0) -> float:
    """
    Austin Ch 16 Availability Formula:
    Availability % = [10^5 - (N * T)] / 10000
    Where N = failures per 100,000h, T = Mean Time To Repair (MTTR) hours.
    """
    downtime_hours = float(num_failures_100k) * float(mttr_hours)
    avail = (100000.0 - downtime_hours) / 1000.0
    return round(max(0.0, min(100.0, avail)), 2)


def mission_risk(analysis: dict, scenario: Optional[dict] = None) -> dict:
    """
    Comprehensive Mission-Risk, Reliability & Availability Analysis.
    Ref: Reg Austin (2010), Chapter 16 (Design for Reliability).
    """
    scenario = scenario or {}

    # 1. Health & Digital Twin Residual Components
    health = float(analysis.get("health_index", 100.0))
    health_component = _clamp(100.0 - health)
    
    twin_rms = float(analysis.get("twin", {}).get("residual_rms", 0.0)) if isinstance(analysis.get("twin"), dict) else 0.0
    residual_component = _clamp(twin_rms * 12.0)

    # 2. Environmental & Mission Profile Stress
    altitude = max(0.0, float(scenario.get("altitude_ft", 3000.0)))
    ambient = float(scenario.get("ambient_c", 25.0))
    duration = max(0.0, float(scenario.get("duration_h", 4.0)))
    rapid = bool(scenario.get("rapid_throttle", False))

    altitude_stress = _clamp(max(0.0, altitude - 5000.0) / 120.0)
    thermal_stress = _clamp(max(0.0, ambient - 30.0) * 4.0)
    endurance_stress = _clamp(max(0.0, duration - 4.0) * 8.0)
    throttle_stress = 25.0 if rapid else 0.0

    mission_stress = _clamp(
        0.35 * altitude_stress
        + 0.30 * thermal_stress
        + 0.25 * endurance_stress
        + 0.10 * throttle_stress
    )

    # 3. Fault Severity & Diagnostic Classification
    severity_map = {"low": 15.0, "medium": 50.0, "high": 85.0, "critical": 95.0}
    fault_candidates = analysis.get("fault_candidates", [])
    if fault_candidates:
        fault_component = max([severity_map.get(str(item.get("severity", "low")).lower(), 15.0) for item in fault_candidates])
    else:
        fault_component = 0.0

    # 4. Sensor Trust & Uncertainty Discrimination
    sensor_trust = float(analysis.get("sensor_health", {}).get("overall_trust_score", 100.0))
    sensor_uncertainty = _clamp(100.0 - sensor_trust)

    # 5. Composite Mission Risk Index (0 - 100)
    score = _clamp(
        0.38 * health_component
        + 0.20 * residual_component
        + 0.20 * mission_stress
        + 0.16 * fault_component
        + 0.06 * sensor_uncertainty
    )

    if score >= 70.0:
        level = "HIGH"
        safety_tier = "Catastrophic Potential / Immediate Action Required"
    elif score >= 40.0:
        level = "MEDIUM"
        safety_tier = "Class A (Severe Degradation / Restricted Flight Envelope)"
    else:
        level = "LOW"
        safety_tier = "Class B (Nominal Operating Range / Minor Wear)"

    # 6. Reliability Synthesis (Austin Ch 16: MTBF and Failure Rates)
    # Baseline UAV powertrain failure rate: ~111 defects per 100,000h under nominal health
    # Degrades exponentially as health declines
    degraded_factor = 1.0 + (health_component / 10.0)**1.5
    est_failures_100k = min(5000.0, round(111.0 * degraded_factor, 1))
    est_mtbf_hours = round(100000.0 / max(1.0, est_failures_100k), 1)
    availability_pct = calculate_system_availability(est_failures_100k, mttr_hours=4.0)

    return {
        "score": round(score, 1),
        "level": level,
        "safety_tier": safety_tier,
        "mission_reliability_index": round(100.0 - score, 1),
        "mtbf_hours": est_mtbf_hours,
        "estimated_failures_per_100k_hours": est_failures_100k,
        "availability_pct": availability_pct,
        "components": {
            "engine_condition": round(health_component, 1),
            "digital_twin_deviation": round(residual_component, 1),
            "mission_stress": round(mission_stress, 1),
            "fault_evidence": round(fault_component, 1),
            "sensor_uncertainty": round(sensor_uncertainty, 1),
        },
        "stress_breakdown": {
            "altitude": round(altitude_stress, 1),
            "temperature": round(thermal_stress, 1),
            "endurance": round(endurance_stress, 1),
            "rapid_throttle": round(throttle_stress, 1),
        },
        "provenance": "Austin (2010) Ch 16 Reliability Synthesis & System Availability Formulation",
        "note": "Decision-support reliability & risk analytics; not a certified airworthiness determination.",
    }
