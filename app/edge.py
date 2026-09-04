from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .can_bus import CANBusInterface, CANFrame
from .sensor_health import assess_sensor_health
from .digital_twin import ReferenceTwin
from .rul_service import RULService


@dataclass
class EdgeHealthSummary:
    timestamp_s: float
    health_state: str
    anomaly_detected: bool
    sensor_trust_score: float
    suspect_sensors: List[str]
    local_safety_action: str
    edge_latency_ms: float


class UAVEdgeNode:
    def __init__(self, twin_engine: Optional[ReferenceTwin] = None):
        self.twin = twin_engine or ReferenceTwin()
        self.can_interface = CANBusInterface()
        self.last_state = 'Normal'
        self.consecutive_anomalies: int = 0

    def process_telemetry(self, telemetry: Dict[str, Any]) -> EdgeHealthSummary:
        start_time = time.perf_counter()
        sanitized = dict(telemetry)
        state = str(sanitized.get('Operating_State', 'CRUISE'))
        expected_vals = self.twin.expected(state)

        max_z = 0.0
        z_scores = {}
        for param, exp in expected_vals.items():
            obs = float(sanitized.get(param, exp))
            std = max(abs(exp) * 0.04, 1e-4)
            z = (obs - exp) / std
            z_scores[param] = z
            if abs(z) > max_z:
                max_z = abs(z)

        twin_stub = {'z_scores': z_scores}
        sh = assess_sensor_health(sanitized, twin_stub)
        trust_score = float(sh.get('overall_trust_score', 100.0))
        suspects = sh.get('suspect_sensors', [])

        anomaly = max_z >= 3.0 or trust_score < 50.0
        if anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1)

        if max_z >= 4.5 or self.consecutive_anomalies >= 5:
            health_state = 'Critical'
            safety_action = 'RESTRICT_THROTTLE_INITIATE_RTL'
        elif max_z >= 2.5 or self.consecutive_anomalies >= 2:
            health_state = 'Warning'
            safety_action = 'MONITOR_THERMAL_AVOID_MAX_THRUST'
        elif max_z >= 1.5:
            health_state = 'Watch'
            safety_action = 'NOMINAL_ENVELOPE'
        else:
            health_state = 'Normal'
            safety_action = 'GO_MISSION_CLEARED'

        self.last_state = health_state
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return EdgeHealthSummary(
            timestamp_s=round(time.time(), 3),
            health_state=health_state,
            anomaly_detected=anomaly,
            sensor_trust_score=round(trust_score, 1),
            suspect_sensors=suspects,
            local_safety_action=safety_action,
            edge_latency_ms=round(latency_ms, 3),
        )


class GCSAnalyticsServer:
    def __init__(self):
        self.rul_service = RULService()
        self.twin = ReferenceTwin()

    def process_gcs_packet(self, telemetry: Dict[str, Any], edge_summary: EdgeHealthSummary, context: Optional[dict] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        twin_result = self.twin.compare(telemetry, context=context)
        sh = assess_sensor_health(telemetry, twin_result)
        rul_res = self.rul_service.predict(telemetry, context=context)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            'edge_summary': edge_summary,
            'digital_twin': twin_result,
            'sensor_health': sh,
            'rul': rul_res,
            'gcs_latency_ms': round(latency_ms, 3),
        }


def benchmark_edge_performance(sample_telemetry: dict, iterations: int = 500) -> dict[str, float]:
    edge = UAVEdgeNode()
    gcs = GCSAnalyticsServer()

    edge_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        summary = edge.process_telemetry(sample_telemetry)
        edge_times.append((time.perf_counter() - t0) * 1000.0)

    gcs_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gcs.process_gcs_packet(sample_telemetry, summary)
        gcs_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        'edge_mean_latency_ms': round(sum(edge_times)/len(edge_times), 3),
        'edge_p99_latency_ms': round(sorted(edge_times)[int(0.99 * len(edge_times))], 3),
        'gcs_mean_latency_ms': round(sum(gcs_times)/len(gcs_times), 3),
        'gcs_p99_latency_ms': round(sorted(gcs_times)[int(0.99 * len(gcs_times))], 3),
        'benchmark_platform': 'CPU Software Benchmark (Desktop / Host CPU)',
    }
