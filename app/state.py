from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class EngineStateRecord:
    timestamp: float = field(default_factory=time.time)
    data_source: str = "SIMULATED"  # LIVE, SIMULATED, REPLAY, STATIC
    operating_state: str = "CRUISE"
    telemetry: Dict[str, float] = field(default_factory=dict)
    physics_expected: Dict[str, float] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)
    z_scores: Dict[str, float] = field(default_factory=dict)
    residual_slopes: Dict[str, float] = field(default_factory=dict)
    sensor_trust_score: float = 100.0
    suspect_sensors: List[str] = field(default_factory=list)
    ml_health_state: str = "Normal"
    ml_probabilities: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    is_unknown_anomaly: bool = False
    degradation_severity: float = 0.0
    rul_hours: float = 2200.0
    rul_confidence_interval: tuple[float, float] = (1980.0, 2420.0)
    rul_provenance: str = "SIMULATION-DERIVED"
    mission_risk_index: float = 0.0
    advisory_message: str = "System nominal."
    reason_codes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "data_source": self.data_source,
            "operating_state": self.operating_state,
            "telemetry": self.telemetry,
            "physics_expected": self.physics_expected,
            "residuals": self.residuals,
            "z_scores": self.z_scores,
            "residual_slopes": self.residual_slopes,
            "sensor_trust_score": round(self.sensor_trust_score, 1),
            "suspect_sensors": self.suspect_sensors,
            "ml_health_state": self.ml_health_state,
            "ml_probabilities": self.ml_probabilities,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_unknown_anomaly": self.is_unknown_anomaly,
            "degradation_severity": round(self.degradation_severity, 3),
            "rul_hours": round(self.rul_hours, 1),
            "rul_confidence_interval": [round(self.rul_confidence_interval[0], 1), round(self.rul_confidence_interval[1], 1)],
            "rul_provenance": self.rul_provenance,
            "mission_risk_index": round(self.mission_risk_index, 1),
            "advisory_message": self.advisory_message,
            "reason_codes": self.reason_codes,
        }
