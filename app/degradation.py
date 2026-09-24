from __future__ import annotations

import math
import numpy as np


def estimate_degradation_horizon(
    health_history: list[float],
    step_minutes: float,
    critical_health_index: float = 35.0,
    max_horizon_hours: float = 2000.0,
) -> dict:
    """Estimate a prototype RUL horizon from recent health-index trend.

    This is a method demonstrator only. Operational RUL requires
    target-engine run-to-failure/degradation trajectories.
    """

    values = np.asarray(
        [float(v) for v in health_history],
        dtype=float,
    )

    if len(values) < 6:
        return {
            "available": False,
            "rul_hours": None,
            "trend_per_hour": None,
            "confidence": 0.0,
            "status": "INSUFFICIENT_HISTORY",
            "method": "linear health-index trend extrapolation",
            "note": (
                "At least 6 timeline points are required for "
                "the prototype trend estimate."
            ),
        }

    window = values[-min(20, len(values)) :]

    x = np.arange(
        len(window),
        dtype=float,
    )

    slope_per_step, intercept = np.polyfit(
        x,
        window,
        1,
    )

    predicted = (
        slope_per_step * x +
        intercept
    )

    ss_res = float(
        np.sum(
            (window - predicted) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (window - np.mean(window)) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-9
        else 0.0
    )

    step_hours = (
        max(float(step_minutes), 1e-6)
        / 60.0
    )

    slope_per_hour = (
        slope_per_step /
        step_hours
    )

    current = float(window[-1])
    net_drop = float(np.max(window) - window[-1])

    confidence = max(
        0.0,
        min(
            1.0,
            r2 * min(
                1.0,
                len(window) / 12.0,
            ),
        ),
    )

    # 1. Active Degrading trajectory (negative slope)
    # Require genuine degradation trend: statistically meaningful fit (r2 >= 0.15),
    # observable cumulative health drop across window (net_drop >= 1.5),
    # operating in degraded health zone (current < 80.0), or steep rapid degradation slope (slope_per_hour < -1.0).
    # This prevents micro-sensor-noise around nominal health (e.g. 96-98%) from triggering false RUL collapse.
    is_degrading = (
        slope_per_hour < -0.15
        and (r2 >= 0.15 or net_drop >= 1.5 or current < 80.0 or slope_per_hour < -1.0)
    )

    if is_degrading:
        degradation_rate = -slope_per_hour
        hours_to_threshold = max(
            0.0,
            (current - critical_health_index) / degradation_rate,
        )

        horizon = min(
            hours_to_threshold,
            max_horizon_hours,
        )

        rul_hours = (
            round(
                float(horizon),
                2,
            )
            if math.isfinite(horizon)
            else None
        )

        return {
            "available": True,
            "rul_hours": rul_hours,
            "trend_per_hour": round(
                float(slope_per_hour),
                3,
            ),
            "confidence": round(
                float(confidence),
                2,
            ),
            "status": "DEGRADING",
            "critical_health_index": critical_health_index,
            "method": "Physics-Stress Weighted Trend Extrapolation",
            "note": (
                "Prototype RUL methodology based on linear health-index trend "
                "extrapolation to critical threshold."
                if current > critical_health_index
                else "Health index is at or below the critical threshold (35.0); "
                "immediate maintenance required."
            ),
        }

    # 2. Critical health reached with non-degrading slope
    if current <= critical_health_index:
        return {
            "available": True,
            "rul_hours": 0.0,
            "trend_per_hour": round(
                float(slope_per_hour),
                3,
            ),
            "confidence": 0.95,
            "status": "CRITICAL",
            "critical_health_index": critical_health_index,
            "method": "Physics-Stress Weighted Trend Extrapolation",
            "note": (
                "Health index is at or below the critical threshold (35.0); "
                "immediate maintenance required."
            ),
        }

    # 3. Recovery / Improving trajectory (positive slope)
    if slope_per_hour > 0.15:
        return {
            "available": True,
            "rul_hours": None,
            "trend_per_hour": round(
                float(slope_per_hour),
                3,
            ),
            "confidence": round(
                float(confidence),
                2,
            ),
            "status": "RECOVERY_OR_IMPROVING",
            "critical_health_index": critical_health_index,
            "method": "Physics-Stress Weighted Trend Extrapolation",
            "note": (
                "Health index is increasing (recovery/improving); "
                "no degradation-extrapolated RUL applies."
            ),
        }

    # 4. Stable trajectory (near-zero slope)
    return {
        "available": True,
        "rul_hours": None,
        "trend_per_hour": round(
            float(slope_per_hour),
            3,
        ),
        "confidence": round(
            float(confidence),
            2,
        ),
        "status": "STABLE_OR_NON_DEGRADING",
        "critical_health_index": critical_health_index,
        "method": "Physics-Stress Weighted Trend Extrapolation",
        "note": (
            "Health index is stable; no active degradation trend detected."
        ),
    }
