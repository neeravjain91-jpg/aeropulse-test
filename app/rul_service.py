"""Remaining Useful Life (RUL) and Prognostics Analytics Service for Aero Piston Engines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .degradation import estimate_degradation_horizon


@dataclass
class EngineRULState:
    """State tracking container for an individual engine session."""
    engine_id: str
    last_rul: float
    last_elapsed_hours: float
    last_health: float
    last_stress: float
    health_ewma: float
    rate_ewma: Optional[float] = None
    health_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)


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
    # Maximum RUL drop fraction per evaluation step (prevents 1199→0 collapse)
    MAX_RUL_DROP_FRACTION: float = 0.25

    # Engine TBO mapping (Documented specifications / published operator manuals)
    # TBO is a maintenance service-life ceiling, not a physical failure time.
    ENGINE_TBO_HOURS: Dict[str, float] = {
        "AeroPiston-4C-1.35L": 2000.0,
        "Rotax-914-Turbo-115HP": 1200.0,
        "ROTAX_914_F_TWIN_01": 1200.0,
        "Continental-TSIO-360-MB": 1800.0,
        "TSIO_360_MB_ACES_01": 1800.0,
        "Generic-Inline4-AeroDiesel": 1500.0,
        "Generic-2Stroke-Twin-50HP": 500.0,
        "Generic-Rotary-Wankel-40HP": 1000.0,
    }

    def __init__(self, default_engine_id: str = "AeroPiston-4C-1.35L"):
        self.default_engine_id = default_engine_id
        self.weibull_beta: float = 2.4
        self.weibull_eta: float = 2200.0
        self._history_buffer: List[float] = []
        self._prev_rul: Optional[float] = None  # rate-limiter state
        self._engine_states: Dict[str, EngineRULState] = {}

    def get_engine_tbo(self, engine_id: Optional[str] = None) -> float:
        """Returns documented engine-specific TBO hours for the specified engine."""
        eid = engine_id or self.default_engine_id
        if eid in self.ENGINE_TBO_HOURS:
            return self.ENGINE_TBO_HOURS[eid]
        eid_lower = eid.lower()
        if "continental" in eid_lower or "tsio" in eid_lower or "360" in eid_lower:
            return self.ENGINE_TBO_HOURS["Continental-TSIO-360-MB"]
        if "rotax" in eid_lower or "914" in eid_lower:
            return self.ENGINE_TBO_HOURS["Rotax-914-Turbo-115HP"]
        return self.DEFAULT_TBO_HOURS

    def reset(self, engine_id: Optional[str] = None) -> None:
        """Resets internal history buffer and engine states to prevent cross-mission/cross-engine leakage."""
        if engine_id:
            self._engine_states.pop(engine_id, None)
        else:
            self._engine_states.clear()
            self._history_buffer.clear()
            self._prev_rul = None

    def calculate_mission_stress(self, context: Optional[dict] = None) -> float:
        if not context:
            return 1.0

        try:
            altitude_ft = float(context.get("altitude_ft", 3000.0))
            if not math.isfinite(altitude_ft):
                altitude_ft = 3000.0
        except (TypeError, ValueError):
            altitude_ft = 3000.0

        try:
            ambient_c = float(context.get("ambient_c", 25.0))
            if not math.isfinite(ambient_c):
                ambient_c = 25.0
        except (TypeError, ValueError):
            ambient_c = 25.0

        try:
            duration_h = float(context.get("duration_h", 4.0))
            if not math.isfinite(duration_h):
                duration_h = 4.0
        except (TypeError, ValueError):
            duration_h = 4.0

        rapid_throttle = bool(context.get("rapid_throttle", False))
        try:
            throttle = float(context.get("throttle", 0.60))
            if not math.isfinite(throttle):
                throttle = 0.60
        except (TypeError, ValueError):
            throttle = 0.60

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

        try:
            raw_health = float(health_index)
            if not math.isfinite(raw_health):
                raw_health = 100.0
        except (TypeError, ValueError):
            raw_health = 100.0
        current_health = max(0.0, min(100.0, raw_health))
        stress = self.calculate_mission_stress(context)

        # Elapsed operating time in hours
        try:
            elapsed_hours = float(context.get("elapsed_hours", context.get("flight_hours", 0.0)))
            if not math.isfinite(elapsed_hours) or elapsed_hours < 0:
                elapsed_hours = 0.0
        except (TypeError, ValueError):
            elapsed_hours = 0.0
        if elapsed_hours == 0.0 and "mission_time_min" in context:
            try:
                elapsed_hours = max(0.0, float(context["mission_time_min"]) / 60.0)
            except (TypeError, ValueError):
                elapsed_hours = 0.0
        elif elapsed_hours == 0.0 and "mission_hours" in context:
            try:
                elapsed_hours = max(0.0, float(context["mission_hours"]))
            except (TypeError, ValueError):
                elapsed_hours = 0.0
        elif elapsed_hours == 0.0 and "timestamp" in context:
            try:
                elapsed_hours = max(0.0, float(context["timestamp"]) / 3600.0)
            except (TypeError, ValueError):
                elapsed_hours = 0.0

        # Mission / Profile Horizon Ceiling if specified
        # TBO is a maintenance ceiling/horizon, not an inherent physical failure time.
        mission_horizon = context.get("mission_horizon_hours", context.get("max_horizon_hours"))
        horizon_ceiling = float(mission_horizon) if mission_horizon is not None else tbo_hours

        consumed_life = elapsed_hours * stress
        max_achievable_life = max(0.0, horizon_ceiling - consumed_life)

        # Retrieve previous engine state if present (strict per-engine isolation)
        state = self._engine_states.get(eid)
        prev_rul = state.last_rul if state is not None else None
        prev_t = state.last_elapsed_hours if state is not None else 0.0
        prev_stress = state.last_stress if state is not None else stress
        dt = elapsed_hours - prev_t if state is not None else elapsed_hours

        # 1. Trajectory Trend Extrapolation
        effective_history = health_history or (state.health_history if state else None)
        trend_res = None
        if effective_history and len(effective_history) >= 6:
            trend_res = estimate_degradation_horizon(
                effective_history,
                step_minutes=step_minutes,
                critical_health_index=self.CRITICAL_HEALTH_THRESHOLD,
                max_horizon_hours=horizon_ceiling,
            )

        if trend_res is not None and trend_res.get("rul_hours") is not None and trend_res.get("status") == "DEGRADING":
            candidate_rul = min(max_achievable_life, float(trend_res["rul_hours"]))
            confidence = float(trend_res.get("confidence", 0.75))
            spread = max(0.05, min(0.40, (1.0 - confidence) * 0.45 + 0.05))
            candidate_status = "ACTIVE_DEGRADATION"
            deg_rate = round(float(trend_res.get("trend_per_hour", 0.5)), 3)
            method = "Physics-Stress Weighted Trend Extrapolation"
        elif current_health <= self.CRITICAL_HEALTH_THRESHOLD:
            candidate_rul = 0.0
            candidate_status = "CRITICAL_MAINTENANCE_REQUIRED"
            confidence = 0.95
            spread = 0.10
            deg_rate = round(((100.0 - self.CRITICAL_HEALTH_THRESHOLD) / tbo_hours) * stress, 4)
            method = "Physics-Stress Weighted Trend Extrapolation"
        elif current_health <= self.WARNING_HEALTH_THRESHOLD:
            nominal_base_rate = (100.0 - self.CRITICAL_HEALTH_THRESHOLD) / tbo_hours
            nominal_deg_rate = nominal_base_rate * stress
            remaining_points = current_health - self.CRITICAL_HEALTH_THRESHOLD
            effective_rate = max(nominal_deg_rate * 2.0, 0.05)
            health_rul = remaining_points / effective_rate
            candidate_rul = max(0.0, min(max_achievable_life, health_rul))
            confidence = 0.80
            spread = 0.25
            candidate_status = "WARNING_ELEVATED_WEAR"
            deg_rate = round(nominal_deg_rate, 4)
            method = "Physics-Stress Weighted Trend Extrapolation"
        else:
            remaining_points = current_health - self.CRITICAL_HEALTH_THRESHOLD
            health_fraction = remaining_points / (100.0 - self.CRITICAL_HEALTH_THRESHOLD)
            health_rul = max_achievable_life * health_fraction
            candidate_rul = max(0.0, min(max_achievable_life, health_rul))
            confidence = 0.70
            spread = 0.30
            candidate_status = "NOMINAL_HEALTH"
            deg_rate = round(((100.0 - self.CRITICAL_HEALTH_THRESHOLD) / tbo_hours) * stress, 4)
            method = "Physics-Stress Weighted Trend Extrapolation"

        # ── Temporal Continuity & Bounded Revision Enforcement ──
        if elapsed_hours == 0.0 or prev_rul is None:
            # Starting point of mission / unsequenced call
            rul_h = candidate_rul
            status = candidate_status
            if elapsed_hours == 0.0:
                self._engine_states[eid] = EngineRULState(
                    engine_id=eid,
                    last_rul=rul_h,
                    last_elapsed_hours=0.0,
                    last_health=current_health,
                    last_stress=stress,
                    health_ewma=current_health,
                    health_history=[current_health],
                    time_history=[0.0],
                )
                self._prev_rul = rul_h
            else:
                self._engine_states[eid] = EngineRULState(
                    engine_id=eid,
                    last_rul=rul_h,
                    last_elapsed_hours=elapsed_hours,
                    last_health=current_health,
                    last_stress=stress,
                    health_ewma=current_health,
                    health_history=[current_health],
                    time_history=[elapsed_hours],
                )
                self._prev_rul = rul_h
        elif dt < -1e-5:
            # Timeline rewound (new mission run without explicit reset)
            rul_h = candidate_rul
            status = candidate_status
            self._engine_states[eid] = EngineRULState(
                engine_id=eid,
                last_rul=rul_h,
                last_elapsed_hours=elapsed_hours,
                last_health=current_health,
                last_stress=stress,
                health_ewma=current_health,
                health_history=[current_health],
                time_history=[elapsed_hours],
            )
            self._prev_rul = rul_h
        elif abs(dt) <= 1e-6:
            # Same timestamp
            if abs(stress - prev_stress) > 1e-3:
                # Operating stress / condition changed (scenario evaluation or flight condition shift)
                rul_h = candidate_rul
                status = candidate_status
            else:
                # Paused timeline: keep prior RUL unchanged
                rul_h = prev_rul
                status = candidate_status
        else:
            # dt > 0: continuous time step
            dt_consumed = max(0.0, dt) * stress
            rho_stress = prev_stress / max(0.01, stress)

            if candidate_rul <= prev_rul - dt_consumed:
                # Normal or accelerated decline towards candidate target
                # Rate-of-change limiter: prevent single-step collapse
                max_drop = self.MAX_RUL_DROP_FRACTION * horizon_ceiling
                if (prev_rul - candidate_rul) > max_drop:
                    rul_h = max(0.0, prev_rul - max_drop)
                    if candidate_status == "CRITICAL_MAINTENANCE_REQUIRED":
                        status = "EMERGENCY_ACUTE_FAULT"
                    else:
                        status = candidate_status
                else:
                    rul_h = candidate_rul
                    status = candidate_status
            else:
                # Candidate is higher than prev_rul - dt_consumed
                # Upward movement detected! Check if supported by genuine operating stress reduction:
                if rho_stress > 1.05:
                    # Operating stress reduced (e.g. throttled down, cooler ambient, lower altitude)
                    # Allow bounded upward revision according to physical revision rule:
                    max_allowed_rul = prev_rul * min(1.20, rho_stress) - dt_consumed
                    rul_h = max(0.0, min(candidate_rul, max_allowed_rul))
                    status = candidate_status
                else:
                    # No stress reduction: confirmed monotonic degradation regime
                    # Strictly enforce no unexplained upward jumps:
                    rul_h = max(0.0, prev_rul - dt_consumed)
                    status = candidate_status

            # Update engine state
            curr_h_ewma = 0.40 * current_health + 0.60 * (state.health_ewma if state else current_health)
            h_hist = (state.health_history if state else []) + [current_health]
            t_hist = (state.time_history if state else []) + [elapsed_hours]
            self._engine_states[eid] = EngineRULState(
                engine_id=eid,
                last_rul=rul_h,
                last_elapsed_hours=elapsed_hours,
                last_health=current_health,
                last_stress=stress,
                health_ewma=curr_h_ewma,
                health_history=h_hist[-30:],
                time_history=t_hist[-30:],
            )
            self._prev_rul = rul_h

        rul_lower = max(0.0, round(rul_h * (1.0 - spread), 2))
        rul_upper = min(tbo_hours * 1.1, round(rul_h * (1.0 + spread), 2))

        deg_mode = str(context.get("degradation_mode") or context.get("fault_mode") or context.get("failure_mode") or "unspecified")
        traj_ver = str(context.get("trajectory_version", "v2.0-physics"))
        prov = str(context.get("provenance", "AEROPULSE_SYNTHETIC"))

        return {
            "rul_hours": round(rul_h, 2),
            "rul_lower_hours": rul_lower,
            "rul_upper_hours": rul_upper,
            "confidence": round(confidence, 2),
            "rul_confidence": round(confidence, 2),
            "degradation_rate_per_hour": deg_rate,
            "status": status,
            "failure_mode_risk": self._diagnose_risk_tier(current_health),
            "stress_multiplier": stress,
            "engine_id": eid,
            "engine_profile": eid,
            "tbo_hours": tbo_hours,
            "maintenance_tbo_horizon": tbo_hours,
            "failure_threshold": self.CRITICAL_HEALTH_THRESHOLD,
            "degradation_mode": deg_mode,
            "trajectory_version": traj_ver,
            "provenance": prov,
            "elapsed_hours": round(elapsed_hours, 3),
            "method": method,
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
        Zero target leakage: uses only observable health/telemetry.
        """
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo_hours = self.get_engine_tbo(eid)

        sensor_sev = 0.0
        try:
            sensor_sev = float(telemetry.get("sensor_fault_severity", context.get("sensor_fault_severity", 0.0)))
            if not math.isfinite(sensor_sev):
                sensor_sev = 0.0
        except (TypeError, ValueError):
            sensor_sev = 0.0

        if telemetry.get("sensor_fault_flag", False) or context.get("sensor_fault_flag", False):
            sensor_sev = max(sensor_sev, 0.75)

        mech_sev = 0.0
        if "health_index" in telemetry:
            try:
                base_health = float(telemetry["health_index"])
                if not math.isfinite(base_health):
                    base_health = 100.0
            except (TypeError, ValueError):
                base_health = 100.0
        elif "health_index" in context:
            try:
                base_health = float(context["health_index"])
                if not math.isfinite(base_health):
                    base_health = 100.0
            except (TypeError, ValueError):
                base_health = 100.0
        else:
            # RC-1 leakage fix: Do NOT derive base_health from
            # Degradation_State or Degradation_Severity (ground-truth labels).
            base_health = 100.0

        slope = context.get("degradation_slope")
        elapsed_hours = 0.0
        try:
            elapsed_hours = float(context.get("elapsed_hours", context.get("flight_hours", 0.0)))
            if not math.isfinite(elapsed_hours) or elapsed_hours < 0:
                elapsed_hours = 0.0
        except (TypeError, ValueError):
            elapsed_hours = 0.0
        if elapsed_hours == 0.0 and "mission_time_min" in context:
            try:
                elapsed_hours = max(0.0, float(context["mission_time_min"]) / 60.0)
            except (TypeError, ValueError):
                elapsed_hours = 0.0

        stress = self.calculate_mission_stress(context)
        max_achievable = max(0.0, tbo_hours - elapsed_hours * stress)

        deg_mode = str(context.get("degradation_mode") or context.get("fault_mode") or context.get("failure_mode") or "unspecified")
        traj_ver = str(context.get("trajectory_version", "v2.0-physics"))
        prov = str(context.get("provenance", "AEROPULSE_SYNTHETIC"))

        if slope is not None:
            try:
                slope_val = float(slope)
                if not math.isfinite(slope_val):
                    slope_val = -0.05
            except (TypeError, ValueError):
                slope_val = -0.05

            remaining = base_health - self.CRITICAL_HEALTH_THRESHOLD
            if base_health <= self.CRITICAL_HEALTH_THRESHOLD:
                rul_val = 0.0
                status = "CRITICAL_MAINTENANCE_REQUIRED"
                deg_rate_h = 0.0
            elif slope_val >= -0.01:
                rul_val = max_achievable
                status = "STABLE_OR_NON_DEGRADING"
                deg_rate_h = 0.0
            else:
                deg_rate_h = abs(slope_val) * stress
                health_rul = remaining / max(0.001, deg_rate_h)
                rul_val = max(0.0, min(max_achievable, health_rul))
                status = "SLOPE_EXTRAPOLATED"

            spread = 0.25 if sensor_sev < 0.3 else 0.45
            conf = 0.85 if sensor_sev < 0.3 else 0.60
            return {
                "rul_hours": round(rul_val, 2),
                "rul_lower_hours": max(0.0, round(rul_val * (1.0 - spread), 2)),
                "rul_upper_hours": min(tbo_hours * 1.1, round(rul_val * (1.0 + spread), 2)),
                "confidence": conf,
                "rul_confidence": conf,
                "health_index_for_rul": round(base_health, 1),
                "degradation_slope": round(slope_val, 4),
                "degradation_severity": round(mech_sev, 3),
                "sensor_fault_severity": round(sensor_sev, 3),
                "status": status,
                "failure_mode_risk": self._diagnose_risk_tier(base_health),
                "stress_multiplier": stress,
                "engine_id": eid,
                "engine_profile": eid,
                "tbo_hours": tbo_hours,
                "maintenance_tbo_horizon": tbo_hours,
                "failure_threshold": self.CRITICAL_HEALTH_THRESHOLD,
                "degradation_mode": deg_mode,
                "trajectory_version": traj_ver,
                "provenance": prov,
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
            # Sensor degradation reduces confidence and widens uncertainty interval
            # without triggering catastrophic engine structural RUL collapse
            res["confidence"] = round(res["confidence"] * 0.70, 2)
            res["rul_confidence"] = res["confidence"]
            cur_rul = float(res["rul_hours"])
            res["rul_lower_hours"] = max(0.0, round(cur_rul * 0.60, 2))
            res["rul_upper_hours"] = min(tbo_hours * 1.1, round(cur_rul * 1.40, 2))
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
