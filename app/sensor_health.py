"""Sensor Health Subsystem for AeroPulse-X.

Provides unified sensor health assessment by delegating to the authoritative
SensorFaultIsolationEngine while preserving 100% backward compatibility for all
existing callers and test suites.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sensor_fault_isolation import (
    SensorFaultIsolationEngine,
    SensorFaultType,
    EngineAttributionVerdict,
    get_sensor_fault_isolation_engine,
)


@dataclass(frozen=True)
class SensorAssessment:
    name: str
    trust_score: float
    status: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "trust_score": round(self.trust_score, 1),
            "status": self.status,
            "reason": self.reason,
        }


def _status(score: float) -> str:
    if score >= 80:
        return "TRUSTED"
    if score >= 55:
        return "CHECK"
    return "SUSPECT"


def assess_sensor_health(telemetry: dict, twin: dict, engine_profile: Optional[str] = None) -> dict:
    """
    Evaluates sensor health and integrity.
    Delegates to SensorFaultIsolationEngine with zero target leakage.
    Returns backward-compatible dictionary enriched with authoritative attribution metadata.
    """
    engine = get_sensor_fault_isolation_engine()
    context = {"engine_id": engine_profile} if engine_profile else {}
    result = engine.analyze(telemetry, twin_assessment=twin, context=context)
    return result.as_dict()
