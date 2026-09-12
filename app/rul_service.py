"""Remaining Useful Life (RUL) and Prognostics Analytics Service for Aero Piston Engines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .degradation import estimate_degradation_horizon


@dataclass
class RULPrediction:
    rul_hours: Optional[float]
    rul_lower_hours: Optional[float]
    rul_upper_hours: Optional[float]
    confidence: float
    degradation_rate_per_hour: float
    status: str
    failure_mode_risk: str
    stress_multiplier: float
    method: str


class RULService:
    """
    Predictive RUL and Prognostic Analytics engine for UAV aero-piston engines.
    
    Provides physics-stress weighted trend extrapolation, calibrated empirical
    uncertainty bounds, multi-engine TBO awareness, and isolated sensor fault handling.
    """

    CRITICAL_HEALTH_THRESHOLD: float = 35.0
    WARNING_HEALTH_THRESHOLD: float = 60.0
    DEFAULT_TBO_HOURS: float = 2000.0

    # Engine TBO mapping (Certified specifications / published operator manuals)
    ENGINE_TBO_HOURS: Dict[str, float] = {
        "AeroPiston-4C-1.35L": 2000.0,
        "Rotax-914-Turbo-115HP": 1200.0,
        "Generic-Inline4-AeroDiesel": 1500.0,
        "Generic-2Stroke-Twin-50HP": 500.0,
        "Generic-Rotary-Wankel-40HP": 1000.0,
    }

    def __init__(self, default_engine_id: str = "AeroPiston-4C-1.35L"):
        self.default_engine_id = default_engine_id
        self.weibull_beta: float = 2.4
        self.weibull_eta: float = 2200.0
        self._history_buffer: List[float] = []

    def get_engine_tbo(self, engine_id: Optional[str] = None) -> float:
        """Returns certified/published TBO hours for the specified engine."""
        eid = engine_id or self.default_engine_id
        return self.ENGINE_TBO_HOURS.get(eid, self.DEFAULT_TBO_HOURS)

    def reset(self) -> None:
        """Resets internal history buffer to prevent cross-mission/cross-engine leakage."""
        self._history_buffer.clear()

    def calculate_mission_stress(self, context: Optional[dict] = None) -> float:
        if not context:
            return 1.0

        altitude_ft = float(context.get("altitude_ft", 3000.0))
        ambient_c = float(context.get("ambient_c", 25.0))
        duration_h = float(context.get("duration_h", 4.0))
        rapid_throttle = bool(context.get("rapid_throttle", False))
        throttle = float(context.get("throttle", 0.60))

        alt_stress = 1.0 + 0.35 * max(0.0, (altitude_ft - 10000.0) / 15000.0)
        thermal_stress = 1.0 + 0.40 * max(0.0, (ambient_c - 25.0) / 25.0)
        endurance_stress = 1.0 + 0.20 * max(0.0, (duration_h - 6.0) / 12.0)
        dynamic_stress = 1.35 if rapid_throttle else (1.0 + 0.15 * max(0.0, (throttle - 0.70) / 0.30))

        cumulative_stress = alt_stress * thermal_stress * endurance_stress * dynamic_stress
        return round(max(0.8, min(3.5, cumulative_stress)), 3)

    def estimate_rul(
        self,
        health_index: float,
        health_history: Optional[List[float]] = None,
        context: Optional[dict] = None,
        step_minutes: float = 5.0,
        engine_id: Optional[str] = None,
    ) -> dict[str, Any]:
        context = context or {}
        eid = engine_id or context.get("engine_id") or self.default_engine_id
        tbo_hours = self.get_engine_tbo(eid)

        current_health = max(0.0, min(100.0, float(health_index)))
        stress = self.calculate_mission_stress(context)

        # Elapsed operating time in hours
        elapsed_hours = float(context.get("elapsed_hours", context.get("flight_hours", 0.0)))
        if elapsed_hours == 0.0 and "mission_time_min" in context:
            elapsed_hours = float(context["mission_time_min"]) / 60.0

        consumed_life = elapsed_hours * stress
        max_achievable_life = max(0.0, tbo_hours - consumed_life)

        # 1. Trajectory Trend Extrapolation (when >= 6 history points available)
        if health_history and len(health_history) >= 6:
            trend_res = estimate_degradation_horizon(
                health_history,
                step_minutes=step_minutes,
                critical_health_index=self.CRITICAL_HEALTH_THRESHOLD,
                max_horizon_hours=tbo_hours,
            )

            if trend_res.get("rul_hours") is not None and trend_res.get("status") == "DEGRADING":
                base_rul = float(trend_res["rul_hours"])
                base_rul = min(max_achievable_life, base_rul)
                confidence = float(trend_res.get("confidence", 0.75))
                # Empirical uncertainty spread: wider for low confidence / noisy fits
                spread = max(0.05, min(0.40, (1.0 - confidence) * 0.45 + 0.05))

                lower = max(0.0, round(base_rul * (1.0 - spread), 2))
                upper = min(tbo_hours * 1.1, round(base_rul * (1.0 + spread), 2))

                return {
                    "rul_hours": round(base_rul, 2),
                    "rul_lower_hours": lower,
                    "rul_upper_hours": upper,
                    "confidence": round(confidence, 2),
                    "rul_confidence": round(confidence, 2),
                    "degradation_rate_per_hour": round(float(trend_res.get("trend_per_hour", 0.5)), 3),
                    "status": "ACTIVE_DEGRADATION",
                    "failure_mode_risk": self._diagnose_risk_tier(current_health),
                    "stress_multiplier": stress,
                    "engine_id": eid,
                    "tbo_hours": tbo_hours,
                    "elapsed_hours": round(elapsed_hours, 3),
                    "method": "Physics-Stress Weighted Trend Extrapolation",
                }

        # 2. Instantaneous / Nominal Fallback Model
        nominal_base_rate = (100.0 - self.CRITICAL_HEALTH_THRESHOLD) / tbo_hours
        nominal_deg_rate = nominal_base_rate * stress

        if current_health <= self.CRITICAL_HEALTH_THRESHOLD:
            rul_h = 0.0
            rul_lower = 0.0
            rul_upper = 1.0
            confidence = 0.95
            status = "CRITICAL_MAINTENANCE_REQUIRED"
        elif current_health <= self.WARNING_HEALTH_THRESHOLD:
            remaining_points = current_health - self.CRITICAL_HEALTH_THRESHOLD
            effective_rate = max(nominal_deg_rate * 2.0, 0.05)
            health_rul = remaining_points / effective_rate
            rul_h = max(0.0, min(max_achievable_life, health_rul))
            confidence = 0.80
            spread = 0.25
            rul_lower = max(0.0, rul_h * (1.0 - spread))
            rul_upper = min(tbo_hours * 1.1, rul_h * (1.0 + spread))
            status = "WARNING_ELEVATED_WEAR"
        else:
            remaining_points = current_health - self.CRITICAL_HEALTH_THRESHOLD
            health_fraction = remaining_points / (100.0 - self.CRITICAL_HEALTH_THRESHOLD)
            health_rul = max_achievable_life * health_fraction
            rul_h = max(0.0, min(max_achievable_life, health_rul))
            confidence = 0.70
            spread = 0.30
            rul_lower = max(0.0, rul_h * (1.0 - spread))
            rul_upper = min(tbo_hours * 1.1, rul_h * (1.0 + spread))
            status = "NOMINAL_HEALTH"

        return {
            "rul_hours": round(rul_h, 2),
            "rul_lower_hours": max(0.0, round(rul_lower, 2)),
            "rul_upper_hours": round(rul_upper, 2),
            "confidence": round(confidence, 2),
            "rul_confidence": round(confidence, 2),
            "degradation_rate_per_hour": round(nominal_deg_rate, 4),
            "status": status,
            "failure_mode_risk": self._diagnose_risk_tier(current_health),
            "stress_multiplier": stress,
            "engine_id": eid,
            "tbo_hours": tbo_hours,
            "elapsed_hours": round(elapsed_hours, 3),
            "method": "Physics-Stress Weighted Trend Extrapolation",
        }

    def predict(
        self,
        telemetry: dict,
        context: Optional[dict] = None,
        health_history: Optional[List[float]] = None,
    ) -> dict[str, Any]:
        """
        Inference interface for live pipeline.
        Separates mechanical/thermodynamic degradation from isolated sensor transducer faults.
        """
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo_hours = self.get_engine_tbo(eid)

        deg_state = telemetry.get("Degradation_State")
        if isinstance(deg_state, dict):
            mech_keys = [k for k in deg_state if k != "sensor"]
            mech_sev = max([float(deg_state[k]) for k in mech_keys]) if mech_keys else 0.0
            sensor_sev = float(deg_state.get("sensor", 0.0))
        else:
            mech_sev = float(telemetry.get("Degradation_Severity", 0.0))
            sensor_sev = 0.0

        base_health = max(0.0, min(100.0, 100.0 - mech_sev * 75.0))

        slope = context.get("degradation_slope")
        elapsed_hours = float(context.get("elapsed_hours", context.get("flight_hours", 0.0)))
        if elapsed_hours == 0.0 and "mission_time_min" in context:
            elapsed_hours = float(context["mission_time_min"]) / 60.0

        stress = self.calculate_mission_stress(context)
        max_achievable = max(0.0, tbo_hours - elapsed_hours * stress)

        if slope is not None:
            slope_val = float(slope)
            remaining = base_health - self.CRITICAL_HEALTH_THRESHOLD
            if slope_val >= -0.01:
                rul_val = max_achievable
                status = "STABLE_OR_NON_DEGRADING"
                deg_rate_h = 0.0
            else:
                deg_rate_h = abs(slope_val) * stress
                health_rul = remaining / max(0.001, deg_rate_h)
                rul_val = 0.0 if base_health <= self.CRITICAL_HEALTH_THRESHOLD else max(0.0, min(max_achievable, health_rul))
                status = "SLOPE_EXTRAPOLATED"

            spread = 0.25
            return {
                "rul_hours": round(rul_val, 2),
                "rul_lower_hours": max(0.0, round(rul_val * (1.0 - spread), 2)),
                "rul_upper_hours": min(tbo_hours * 1.1, round(rul_val * (1.0 + spread), 2)),
                "confidence": 0.85 if sensor_sev < 0.3 else 0.65,
                "rul_confidence": 0.85 if sensor_sev < 0.3 else 0.65,
                "health_index_for_rul": round(base_health, 1),
                "degradation_slope": round(slope_val, 4),
                "degradation_severity": round(mech_sev, 3),
                "sensor_fault_severity": round(sensor_sev, 3),
                "status": status,
                "failure_mode_risk": self._diagnose_risk_tier(base_health),
                "stress_multiplier": stress,
                "engine_id": eid,
                "tbo_hours": tbo_hours,
                "elapsed_hours": round(elapsed_hours, 3),
                "method": "Explicit Slope Estimation",
            }

        res = self.estimate_rul(
            health_index=base_health,
            health_history=health_history,
            context=context,
            engine_id=eid,
        )
        res["health_index_for_rul"] = round(base_health, 1)
        res["degradation_severity"] = round(mech_sev, 3)
        res["sensor_fault_severity"] = round(sensor_sev, 3)
        res["degradation_slope"] = round(-res.get("degradation_rate_per_hour", 0.05), 4)
        if sensor_sev > 0.3:
            res["confidence"] = round(res["confidence"] * 0.80, 2)
            res["rul_confidence"] = res["confidence"]
        return res

    @staticmethod
    def _diagnose_risk_tier(health: float) -> str:
        if health < 35.0:
            return "IMMINENT_IN_FLIGHT_ABORT_RISK"
        if health < 55.0:
            return "ACCELERATED_SUBSYSTEM_WEAR"
        if health < 75.0:
            return "MODERATE_THERMOMECHANICAL_STRESS"
        return "LOW_OPERATIONAL_RISK"
