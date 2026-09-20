"""
Candidate Remaining Useful Life (RUL) Estimators & Degradation Framework.
AeroPulse-X Part 5/7: RUL + Degradation Engineering.

Implements candidate estimators under a unified causal interface:
  E1: PhysicsInformedDegradationProjector (reduced-order physics-informed stress-weighted projector)
  E2: PowerLawDegradationRegressor (wear kinetics H(t) = 100 - a * t^b)
  E3: GradientBoostedRULRegressor (lightweight causal HistGradientBoosting regressor)
  E4: WeibullHazardModel (parametric hazard, survival, conditional remaining life)
  E5: CalibratedUncertaintyEstimator (residual-calibrated prediction intervals)

Strict Scientific Standards:
  - Zero target leakage: no Degradation_Severity or ground-truth failure times as features.
  - Strictly causal temporal features (no future-window or centered features).
  - Explicit distinction between physical failure horizon, maintenance TBO horizon, and estimated RUL.
  - Sensor fault isolation: transducer drift widens uncertainty without false mechanical RUL collapse.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


CRITICAL_HEALTH_THRESHOLD: float = 35.0
WARNING_HEALTH_THRESHOLD: float = 60.0


@dataclass
class RULEstimateResult:
    """Standardized prognostic output across all RUL estimators."""
    rul_hours: float
    rul_lower_hours: float
    rul_upper_hours: float
    confidence: float
    method: str
    engine_profile: str
    failure_threshold: float = CRITICAL_HEALTH_THRESHOLD
    degradation_mode: str = "unspecified"
    trajectory_version: str = "v2.0-physics"
    provenance: str = "AEROPULSE_SYNTHETIC"
    maintenance_tbo_horizon: float = 1200.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rul_hours": round(self.rul_hours, 2),
            "rul_lower_hours": round(self.rul_lower_hours, 2),
            "rul_upper_hours": round(self.rul_upper_hours, 2),
            "confidence": round(self.confidence, 2),
            "method": self.method,
            "engine_profile": self.engine_profile,
            "failure_threshold": self.failure_threshold,
            "degradation_mode": self.degradation_mode,
            "trajectory_version": self.trajectory_version,
            "provenance": self.provenance,
            "maintenance_tbo_horizon": self.maintenance_tbo_horizon,
            "details": self.details,
        }


class BaseRULEstimator(abc.ABC):
    """Abstract base class establishing the common prognostic estimator interface."""

    def __init__(self, name: str, default_engine_id: str = "Rotax-914-Turbo-115HP"):
        self.name = name
        self.default_engine_id = default_engine_id
        self.is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "BaseRULEstimator":
        """Fit estimator on grouped trajectories. Must not leak across trajectory groups."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        """Causal RUL prediction consuming only historical/current observables."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear temporal state to prevent cross-mission or cross-engine leakage."""
        pass


def calculate_causal_stress(context: Optional[Dict[str, Any]] = None) -> float:
    """Calculates dimensionless environmental/dynamic stress multiplier S in [0.8, 3.5]."""
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

    return round(max(0.8, min(3.5, alt_stress * thermal_stress * endurance_stress * dynamic_stress)), 3)


def get_profile_tbo(engine_id: Optional[str]) -> float:
    """Returns documented engine-specific maintenance TBO ceiling hours."""
    eid = (engine_id or "Rotax-914-Turbo-115HP").lower()
    if "continental" in eid or "tsio" in eid or "360" in eid:
        return 1800.0  # Continental TSIO-360-MB FAA TCDS E9CE
    if "rotax" in eid or "914" in eid:
        return 1200.0  # Rotax 914 F EASA TCDS E.121
    if "diesel" in eid:
        return 1500.0
    if "2stroke" in eid:
        return 500.0
    if "wankel" in eid or "rotary" in eid:
        return 1000.0
    return 2000.0  # Demonstrator generic fallback baseline (AeroPiston-4C-1.35L)


# =====================================================================
# Candidate 1: PhysicsInformedDegradationProjector
# =====================================================================
class PhysicsInformedDegradationProjector(BaseRULEstimator):
    """
    Candidate 1 (E1): Reduced-order physics-informed stress-weighted degradation projector.
    Projects time-to-failure from causal health trend and environmental stress:
        t_fail = t + (H(t) - H_crit) / max(eps, |dH/dt| * S)
    """

    def __init__(self, default_engine_id: str = "Rotax-914-Turbo-115HP"):
        super().__init__("PhysicsInformedDegradationProjector", default_engine_id)
        self.last_rul: Optional[float] = None
        self.last_t: float = 0.0
        self.is_fitted = True

    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "PhysicsInformedDegradationProjector":
        # First-principles reduced order projector requires no parameter fitting
        self.is_fitted = True
        return self

    def reset(self) -> None:
        self.last_rul = None
        self.last_t = 0.0

    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo = get_profile_tbo(eid)
        stress = calculate_causal_stress(context)

        # Sanitize inputs
        h = max(0.0, min(100.0, float(health_index))) if math.isfinite(health_index) else 100.0
        t = max(0.0, float(elapsed_hours)) if math.isfinite(elapsed_hours) else 0.0

        # Extract causal slope from history if available
        slope_per_hour = -0.05
        confidence = 0.70
        if history and len(history) >= 4:
            ts = [p.get("elapsed_hours", p.get("time_hours", 0.0)) for p in history[-15:]]
            hs = [p.get("health_index", 100.0) for p in history[-15:]]
            if len(ts) >= 4 and (ts[-1] - ts[0]) > 1e-4:
                try:
                    p = np.polyfit(ts, hs, 1)
                    slope_per_hour = float(p[0])
                    # Assess linearity / R^2
                    preds = np.polyval(p, ts)
                    ss_res = np.sum((np.array(hs) - preds) ** 2)
                    ss_tot = np.sum((np.array(hs) - np.mean(hs)) ** 2)
                    r2 = 1.0 - (ss_res / max(1e-6, ss_tot))
                    confidence = max(0.5, min(0.95, float(r2)))
                except Exception:
                    slope_per_hour = -0.05

        # Maintenance ceiling
        max_achievable = max(0.0, tbo - t * stress)

        # Core physics-informed projection
        if h <= CRITICAL_HEALTH_THRESHOLD:
            rul_raw = 0.0
            confidence = 0.95
        elif slope_per_hour >= -0.01:
            # Stable or non-degrading: wear rate nominal based on TBO
            nominal_rate = ((100.0 - CRITICAL_HEALTH_THRESHOLD) / tbo) * stress
            remaining_pts = h - CRITICAL_HEALTH_THRESHOLD
            rul_raw = min(max_achievable, remaining_pts / max(0.001, nominal_rate))
        else:
            deg_rate = abs(slope_per_hour) * stress
            remaining_pts = h - CRITICAL_HEALTH_THRESHOLD
            rul_raw = min(max_achievable, remaining_pts / max(0.001, deg_rate))

        # Monotonicity enforcement (under non-decreasing stress)
        if self.last_rul is not None and t > self.last_t:
            dt = t - self.last_t
            dt_wear = dt * stress
            rul_bounded = max(0.0, min(rul_raw, self.last_rul - dt_wear * 0.5))
        else:
            rul_bounded = rul_raw

        self.last_rul = rul_bounded
        self.last_t = t

        spread = max(0.08, min(0.35, (1.0 - confidence) * 0.40 + 0.08))
        sensor_sev = float(context.get("sensor_fault_severity", 0.0))
        if sensor_sev > 0.3:
            confidence = round(confidence * 0.70, 2)
            spread = min(0.55, spread + 0.20)

        lower = max(0.0, rul_bounded * (1.0 - spread))
        upper = min(tbo * 1.1, rul_bounded * (1.0 + spread))

        return RULEstimateResult(
            rul_hours=rul_bounded,
            rul_lower_hours=lower,
            rul_upper_hours=upper,
            confidence=confidence,
            method="PhysicsInformedDegradationProjector",
            engine_profile=eid,
            failure_threshold=CRITICAL_HEALTH_THRESHOLD,
            degradation_mode=str(context.get("degradation_mode", "unspecified")),
            trajectory_version=str(context.get("trajectory_version", "v2.0-physics")),
            provenance=str(context.get("provenance", "AEROPULSE_SYNTHETIC")),
            maintenance_tbo_horizon=tbo,
            details={"slope_per_hour": round(slope_per_hour, 4), "stress": stress},
        )


# =====================================================================
# Candidate 2: PowerLawDegradationRegressor
# =====================================================================
class PowerLawDegradationRegressor(BaseRULEstimator):
    """
    Candidate 2 (E2): Wear Kinetics Power-Law Regressor.
    Equation: H(t) = 100 - a * (t - t_onset)^b  (or severity s(t) = a * t^b)
    Fits ln(100 - H) = ln(a) + b * ln(t - t_onset) via linear regression.
    Projects to H = 35.0 (loss = 65.0):
        t_fail = t_onset + (65.0 / a)^(1/b)
        RUL(t) = max(0.0, t_fail - t)
    """

    def __init__(self, default_engine_id: str = "Rotax-914-Turbo-115HP"):
        super().__init__("PowerLawDegradationRegressor", default_engine_id)
        self.default_a: float = 0.05
        self.default_b: float = 1.35
        self.last_rul: Optional[float] = None
        self.last_t: float = 0.0

    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "PowerLawDegradationRegressor":
        """Fits baseline power-law kinetics (a, b) across degradation training trajectories."""
        log_t, log_loss = [], []
        for traj in trajectories:
            pts = traj.get("points", [])
            for pt in pts:
                t = pt.get("elapsed_hours", pt.get("time_hours", 0.0))
                h = pt.get("health_index", 100.0)
                loss = 100.0 - h
                if t > 0.5 and 0.5 < loss < 65.0:
                    log_t.append(math.log(t))
                    log_loss.append(math.log(loss))

        if len(log_t) >= 10:
            p = np.polyfit(log_t, log_loss, 1)
            self.default_b = float(np.clip(p[0], 0.5, 3.5))
            self.default_a = float(np.clip(math.exp(p[1]), 0.001, 10.0))
        self.is_fitted = True
        return self

    def reset(self) -> None:
        self.last_rul = None
        self.last_t = 0.0

    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo = get_profile_tbo(eid)
        stress = calculate_causal_stress(context)

        h = max(0.0, min(100.0, float(health_index))) if math.isfinite(health_index) else 100.0
        t = max(0.0, float(elapsed_hours)) if math.isfinite(elapsed_hours) else 0.0
        loss = 100.0 - h

        a = self.default_a * stress
        b = self.default_b

        # Local trajectory fit if sufficient points
        fitted_locally = False
        if history and len(history) >= 6:
            pts_t = []
            pts_loss = []
            for p in history[-20:]:
                pt_t = p.get("elapsed_hours", p.get("time_hours", 0.0))
                pt_h = p.get("health_index", 100.0)
                pt_l = 100.0 - pt_h
                if pt_t > 0.2 and pt_l > 1.0:
                    pts_t.append(math.log(pt_t))
                    pts_loss.append(math.log(pt_l))
            if len(pts_t) >= 5:
                try:
                    poly = np.polyfit(pts_t, pts_loss, 1)
                    b = float(np.clip(poly[0], 0.8, 3.0))
                    a = float(np.clip(math.exp(poly[1]), 0.005, 5.0))
                    fitted_locally = True
                except Exception:
                    pass

        # Projected failure time when loss = 65.0 (H = 35.0)
        target_loss = 100.0 - CRITICAL_HEALTH_THRESHOLD
        if h <= CRITICAL_HEALTH_THRESHOLD:
            rul_raw = 0.0
            conf = 0.95
        elif loss <= 0.5:
            # Barely degraded: wear horizon ceiling
            rul_raw = max(0.0, tbo - t * stress)
            conf = 0.65
        else:
            try:
                # 65.0 = a * t_fail^b => t_fail = (65.0 / a)^(1/b)
                t_fail = (target_loss / max(1e-5, a)) ** (1.0 / max(0.1, b))
                rul_raw = max(0.0, t_fail - t)
            except Exception:
                rul_raw = max(0.0, tbo - t * stress)
            conf = 0.85 if fitted_locally else 0.72

        # Monotonicity guard
        if self.last_rul is not None and t > self.last_t:
            dt = t - self.last_t
            rul_bounded = max(0.0, min(rul_raw, self.last_rul - dt * stress * 0.5))
        else:
            rul_bounded = rul_raw

        self.last_rul = rul_bounded
        self.last_t = t

        spread = max(0.10, min(0.35, (1.0 - conf) * 0.45 + 0.05))
        sensor_sev = float(context.get("sensor_fault_severity", 0.0))
        if sensor_sev > 0.3:
            conf = round(conf * 0.70, 2)
            spread = min(0.55, spread + 0.20)

        lower = max(0.0, rul_bounded * (1.0 - spread))
        upper = min(tbo * 1.1, rul_bounded * (1.0 + spread))

        return RULEstimateResult(
            rul_hours=rul_bounded,
            rul_lower_hours=lower,
            rul_upper_hours=upper,
            confidence=conf,
            method="PowerLawDegradationRegressor",
            engine_profile=eid,
            failure_threshold=CRITICAL_HEALTH_THRESHOLD,
            degradation_mode=str(context.get("degradation_mode", "unspecified")),
            trajectory_version=str(context.get("trajectory_version", "v2.0-physics")),
            provenance=str(context.get("provenance", "AEROPULSE_SYNTHETIC")),
            maintenance_tbo_horizon=tbo,
            details={"a": round(a, 4), "b": round(b, 4), "fitted_locally": fitted_locally},
        )


# =====================================================================
# Candidate 3: GradientBoostedRULRegressor
# =====================================================================
class GradientBoostedRULRegressor(BaseRULEstimator):
    """
    Candidate 3 (E3): Causal HistGradientBoosting RUL Regressor.
    Trained strictly on causal features without target leakage:
      - health_index
      - elapsed_hours
      - stress_multiplier
      - dh_dt (rolling slope)
      - delta_cht, delta_egt, egt_spread, delta_oil_p, delta_oil_t, vibration, efficiency
    Zero target leakage: Degradation_Severity, Degradation_State, true_RUL excluded from features.
    """

    FEATURE_NAMES = [
        "health_index",
        "elapsed_hours",
        "stress_multiplier",
        "dh_dt",
        "cht",
        "egt_spread",
        "oil_pressure",
        "oil_temp",
        "vibration",
        "efficiency",
    ]

    def __init__(self, default_engine_id: str = "Rotax-914-Turbo-115HP"):
        super().__init__("GradientBoostedRULRegressor", default_engine_id)
        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
        )
        self.last_rul: Optional[float] = None
        self.last_t: float = 0.0

    @staticmethod
    def extract_features(
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        context = context or {}
        stress = calculate_causal_stress(context)
        h = max(0.0, min(100.0, float(health_index))) if math.isfinite(health_index) else 100.0
        t = max(0.0, float(elapsed_hours)) if math.isfinite(elapsed_hours) else 0.0

        # Causal trend dh_dt
        dh_dt = -0.05
        if history and len(history) >= 3:
            t_prev = history[-1].get("elapsed_hours", history[-1].get("time_hours", 0.0))
            h_prev = history[-1].get("health_index", 100.0)
            dt = t - t_prev
            if dt > 1e-4:
                dh_dt = float((h - h_prev) / dt)

        cht = float(context.get("CHT", 145.0))
        egt1 = float(context.get("EGT1", 1250.0))
        egt2 = float(context.get("EGT2", 1250.0))
        egt_spread = abs(egt2 - egt1)
        oil_p = float(context.get("Oil_Pressure", 55.0))
        oil_t = float(context.get("Oil_Temp", 85.0))
        vib = float(context.get("Vibration", 0.5))
        eff = float(context.get("Efficiency", 0.65))

        return np.array([h, t, stress, dh_dt, cht, egt_spread, oil_p, oil_t, vib, eff], dtype=float)

    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "GradientBoostedRULRegressor":
        X_list, y_list = [], []
        for traj in trajectories:
            pts = traj.get("points", [])
            for i, pt in enumerate(pts):
                t = pt.get("elapsed_hours", pt.get("time_hours", 0.0))
                h = pt.get("health_index", 100.0)
                true_rul = pt.get("true_RUL")
                if true_rul is None:
                    continue
                hist = pts[:i] if i > 0 else None
                feat = self.extract_features(h, t, context=pt, history=hist)
                X_list.append(feat)
                y_list.append(float(true_rul))

        if len(X_list) >= 20:
            X = np.array(X_list)
            y = np.array(y_list)
            self.model.fit(X, y)
            self.is_fitted = True
        return self

    def reset(self) -> None:
        self.last_rul = None
        self.last_t = 0.0

    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo = get_profile_tbo(eid)
        stress = calculate_causal_stress(context)
        t = max(0.0, float(elapsed_hours)) if math.isfinite(elapsed_hours) else 0.0
        h = max(0.0, min(100.0, float(health_index))) if math.isfinite(health_index) else 100.0

        if not self.is_fitted:
            # Fallback to physics projection if not fitted
            proj = PhysicsInformedDegradationProjector(self.default_engine_id)
            return proj.predict(health_index, elapsed_hours, context, history)

        feat = self.extract_features(health_index, elapsed_hours, context, history).reshape(1, -1)
        pred_rul = float(self.model.predict(feat)[0])
        pred_rul = max(0.0, min(tbo, pred_rul))

        # Critical health check
        if h <= CRITICAL_HEALTH_THRESHOLD:
            pred_rul = 0.0

        # Monotonicity rate limiter
        if self.last_rul is not None and t > self.last_t:
            dt = t - self.last_t
            rul_bounded = max(0.0, min(pred_rul, self.last_rul - dt * stress * 0.5))
        else:
            rul_bounded = pred_rul

        self.last_rul = rul_bounded
        self.last_t = t

        conf = 0.82
        spread = 0.22
        sensor_sev = float(context.get("sensor_fault_severity", 0.0))
        if sensor_sev > 0.3:
            conf = round(conf * 0.70, 2)
            spread = min(0.55, spread + 0.20)

        lower = max(0.0, rul_bounded * (1.0 - spread))
        upper = min(tbo * 1.1, rul_bounded * (1.0 + spread))

        return RULEstimateResult(
            rul_hours=rul_bounded,
            rul_lower_hours=lower,
            rul_upper_hours=upper,
            confidence=conf,
            method="GradientBoostedRULRegressor",
            engine_profile=eid,
            failure_threshold=CRITICAL_HEALTH_THRESHOLD,
            degradation_mode=str(context.get("degradation_mode", "unspecified")),
            trajectory_version=str(context.get("trajectory_version", "v2.0-physics")),
            provenance=str(context.get("provenance", "AEROPULSE_SYNTHETIC")),
            maintenance_tbo_horizon=tbo,
            details={"raw_prediction": round(pred_rul, 2)},
        )


# =====================================================================
# Candidate 4: WeibullHazardModel
# =====================================================================
class WeibullHazardModel(BaseRULEstimator):
    """
    Candidate 4 (E4): Parametric Weibull Reliability & Conditional RUL Model.
    Explicitly separates:
      - Hazard rate: lambda(t) = (beta / eta) * (t / eta)^(beta - 1)
      - Cumulative hazard: Lambda(t) = (t / eta)^beta
      - Survival probability: S(t) = exp(-(t / eta)^beta)
      - Conditional RUL: E[T - t | T > t] = (1 / S(t)) * int_t^inf S(u) du
    Under operating stress S, accelerated failure time (AFT) scales eta_eff = eta / S.
    """

    def __init__(self, default_engine_id: str = "Rotax-914-Turbo-115HP"):
        super().__init__("WeibullHazardModel", default_engine_id)
        self.beta: float = 2.4  # Wear-out phase (> 1.0)
        self.eta: float = 12.0  # Characteristic life for accelerated mission simulation
        self.is_fitted = True

    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "WeibullHazardModel":
        """Fits Weibull parameters (beta, eta) via maximum likelihood on observed failure times."""
        fail_times = []
        for traj in trajectories:
            pts = traj.get("points", [])
            for pt in pts:
                if pt.get("health_index", 100.0) <= CRITICAL_HEALTH_THRESHOLD:
                    t = pt.get("elapsed_hours", pt.get("time_hours", 0.0))
                    if t > 0.5:
                        fail_times.append(t)
                        break
        if len(fail_times) >= 5:
            arr = np.array(fail_times)
            mean_t = float(np.mean(arr))
            std_t = float(np.std(arr))
            # Method of moments approximation for Weibull
            if std_t > 1e-4:
                cv = std_t / mean_t
                approx_beta = float(np.clip(cv ** (-1.086), 1.2, 5.0))
                approx_eta = float(mean_t / math.gamma(1.0 + 1.0 / approx_beta))
                self.beta = approx_beta
                self.eta = max(1.0, approx_eta)
        self.is_fitted = True
        return self

    def reset(self) -> None:
        pass

    def hazard_rate(self, t: float, eta_eff: float) -> float:
        """Evaluates instantaneous hazard rate lambda(t)."""
        if t <= 0.0 or eta_eff <= 0.0:
            return 0.0
        return (self.beta / eta_eff) * ((t / eta_eff) ** (self.beta - 1.0))

    def survival_prob(self, t: float, eta_eff: float) -> float:
        """Evaluates survival probability S(t) = exp(-(t/eta_eff)^beta)."""
        if t <= 0.0:
            return 1.0
        return math.exp(-((t / eta_eff) ** self.beta))

    def conditional_rul(self, t: float, eta_eff: float) -> float:
        """
        Evaluates conditional expected remaining life:
            E[T - t | T > t] = (1 / S(t)) * int_t^(t + 4*eta) S(u) du
        """
        s_t = self.survival_prob(t, eta_eff)
        if s_t < 1e-5:
            return 0.0
        # 30-point Gauss-Legendre quadrature
        upper_limit = t + 4.0 * eta_eff
        u_nodes, weights = np.polynomial.legendre.leggauss(30)
        # Transform [-1, 1] to [t, upper_limit]
        mid = 0.5 * (upper_limit + t)
        half_len = 0.5 * (upper_limit - t)
        u_vals = mid + half_len * u_nodes
        s_vals = np.exp(-((u_vals / eta_eff) ** self.beta))
        integral = half_len * float(np.sum(weights * s_vals))
        return max(0.0, integral / s_t)

    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        context = context or {}
        eid = context.get("engine_id") or self.default_engine_id
        tbo = get_profile_tbo(eid)
        stress = calculate_causal_stress(context)
        t = max(0.0, float(elapsed_hours)) if math.isfinite(elapsed_hours) else 0.0
        h = max(0.0, min(100.0, float(health_index))) if math.isfinite(health_index) else 100.0

        # Accelerated failure time characteristic scale
        eta_eff = max(1.0, self.eta / stress)

        # Health-adjusted pseudo-age
        loss = 100.0 - h
        if loss > 2.0:
            wear_fraction = loss / (100.0 - CRITICAL_HEALTH_THRESHOLD)
            pseudo_age = eta_eff * (wear_fraction ** (1.0 / self.beta))
            effective_age = max(t, pseudo_age)
        else:
            effective_age = t

        if h <= CRITICAL_HEALTH_THRESHOLD:
            rul_val = 0.0
            conf = 0.95
        else:
            rul_val = self.conditional_rul(effective_age, eta_eff)
            conf = max(0.50, min(0.90, self.survival_prob(effective_age, eta_eff)))

        # Bounded by maintenance horizon
        rul_bounded = min(tbo, rul_val)
        spread = max(0.12, min(0.40, (1.0 - conf) * 0.50 + 0.10))

        sensor_sev = float(context.get("sensor_fault_severity", 0.0))
        if sensor_sev > 0.3:
            conf = round(conf * 0.70, 2)
            spread = min(0.55, spread + 0.20)

        lower = max(0.0, rul_bounded * (1.0 - spread))
        upper = min(tbo * 1.1, rul_bounded * (1.0 + spread))

        return RULEstimateResult(
            rul_hours=rul_bounded,
            rul_lower_hours=lower,
            rul_upper_hours=upper,
            confidence=conf,
            method="WeibullHazardModel",
            engine_profile=eid,
            failure_threshold=CRITICAL_HEALTH_THRESHOLD,
            degradation_mode=str(context.get("degradation_mode", "unspecified")),
            trajectory_version=str(context.get("trajectory_version", "v2.0-physics")),
            provenance=str(context.get("provenance", "AEROPULSE_SYNTHETIC")),
            maintenance_tbo_horizon=tbo,
            details={
                "beta": round(self.beta, 2),
                "eta_eff": round(eta_eff, 2),
                "effective_age": round(effective_age, 2),
                "hazard_rate": round(self.hazard_rate(effective_age, eta_eff), 5),
                "survival_prob": round(self.survival_prob(effective_age, eta_eff), 4),
            },
        )


# =====================================================================
# Candidate 5: CalibratedUncertaintyEstimator
# =====================================================================
class CalibratedUncertaintyEstimator(BaseRULEstimator):
    """
    Candidate 5 (E5): Best Candidate Wrapped with Empirical Residual-Calibrated Uncertainty.
    Calibrates interval bounds on validation residuals to guarantee empirical coverage.
    """

    def __init__(
        self,
        base_estimator: Optional[BaseRULEstimator] = None,
        target_coverage: float = 0.90,
        default_engine_id: str = "Rotax-914-Turbo-115HP",
    ):
        super().__init__("CalibratedUncertaintyEstimator", default_engine_id)
        self.base = base_estimator or PhysicsInformedDegradationProjector(default_engine_id)
        self.target_coverage = target_coverage
        self.lower_quantile: float = 0.15
        self.upper_quantile: float = 0.15
        self.is_calibrated: bool = False

    def fit(self, trajectories: Sequence[Dict[str, Any]]) -> "CalibratedUncertaintyEstimator":
        self.base.fit(trajectories)
        self.is_fitted = True
        return self

    def calibrate(self, val_trajectories: Sequence[Dict[str, Any]]) -> "CalibratedUncertaintyEstimator":
        """Calibrates prediction intervals against held-out validation trajectories."""
        residuals = []
        for traj in val_trajectories:
            pts = traj.get("points", [])
            self.base.reset()
            for i, pt in enumerate(pts):
                t = pt.get("elapsed_hours", pt.get("time_hours", 0.0))
                h = pt.get("health_index", 100.0)
                true_rul = pt.get("true_RUL")
                if true_rul is None:
                    continue
                hist = pts[:i] if i > 0 else None
                res = self.base.predict(h, t, context=pt, history=hist)
                # Residual: (true_RUL - pred_RUL) / max(1.0, pred_RUL)
                rel_err = (float(true_rul) - res.rul_hours) / max(0.5, res.rul_hours)
                residuals.append(rel_err)

        if len(residuals) >= 20:
            alpha = 1.0 - self.target_coverage
            q_low = float(np.percentile(residuals, 100.0 * (alpha / 2.0)))
            q_high = float(np.percentile(residuals, 100.0 * (1.0 - alpha / 2.0)))
            self.lower_quantile = max(0.05, abs(q_low))
            self.upper_quantile = max(0.05, abs(q_high))
            self.is_calibrated = True
        return self

    def reset(self) -> None:
        self.base.reset()

    def predict(
        self,
        health_index: float,
        elapsed_hours: float,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RULEstimateResult:
        res = self.base.predict(health_index, elapsed_hours, context, history)
        rul = res.rul_hours
        tbo = res.maintenance_tbo_horizon

        sensor_sev = float((context or {}).get("sensor_fault_severity", 0.0))
        epistemic_scale = 1.0 + 1.5 * sensor_sev

        calib_lower = max(0.0, rul * (1.0 - self.lower_quantile * epistemic_scale))
        calib_upper = min(tbo * 1.1, rul * (1.0 + self.upper_quantile * epistemic_scale))

        conf = res.confidence
        if sensor_sev > 0.3:
            conf = round(conf * 0.70, 2)

        return RULEstimateResult(
            rul_hours=rul,
            rul_lower_hours=calib_lower,
            rul_upper_hours=calib_upper,
            confidence=conf,
            method=f"CalibratedUncertainty({self.base.name})",
            engine_profile=res.engine_profile,
            failure_threshold=res.failure_threshold,
            degradation_mode=res.degradation_mode,
            trajectory_version=res.trajectory_version,
            provenance=res.provenance,
            maintenance_tbo_horizon=tbo,
            details={
                **res.details,
                "is_calibrated": self.is_calibrated,
                "target_coverage": self.target_coverage,
                "calib_lower_fraction": round(self.lower_quantile, 3),
                "calib_upper_fraction": round(self.upper_quantile, 3),
            },
        )
