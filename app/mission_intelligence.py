"""AeroPulse-X Mission Intelligence & Deterministic Trajectory Summarizer.

Provides deterministic validation, feature/trend extraction, multi-engine threshold
resolution, chronological event extraction, and multi-subsystem mission summarization.

Pipeline:
COMPLETE TRAJECTORY -> DETERMINISTIC VALIDATION -> FEATURE EXTRACTION ->
EVENT EXTRACTION -> MISSION SUMMARY -> (LLM or DETERMINISTIC FALLBACK)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class MissionEvent:
    """Chronological mission event extracted deterministically from telemetry evidence."""
    timestamp_sec: float
    timestamp_hours: float
    phase: str
    event_type: str  # e.g., MISSION_START, TAKEOFF, CLIMB, CRUISE, DESCENT, LANDING, MISSION_END,
                     # THROTTLE_TRANSITION, THERMAL_DEVIATION, VIBRATION_ANOMALY, LUBRICATION_ABNORMALITY,
                     # INJECTOR_ISSUE, MISFIRE, MECHANICAL_DEGRADATION, SENSOR_DRIFT,
                     # HEALTH_DEGRADATION, RUL_DROP, DERATE_WARNING, CRITICAL_MAINTENANCE, SAFETY_ACTION
    severity: str    # INFO, WARN, CRITICAL
    evidence: str    # Exact metric, e.g. "CHT reached 135.2°C (limit 135.0°C)"
    result: str      # e.g. "FADEC commanded DERATE_80"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_engine_limits(engine_id: str) -> Dict[str, Any]:
    """Dynamically resolves operational limits and thresholds for a specific engine profile.
    
    Prevents mixing parameters across Rotax, AeroPiston, AeroDiesel, 2-stroke, or Wankel.
    """
    eid = (engine_id or "").lower()

    if "rotax" in eid or "914" in eid:
        return {
            "name": "Rotax-914-Turbo-115HP",
            "engine_type": "turbocharged_spark_ignition",
            "cycle": "4-stroke",
            "nominal_rpm": 5500.0,
            "max_rpm": 5800.0,
            "cht_warn": 130.0,
            "cht_max": 135.0,
            "oil_press_min": 25.0,
            "oil_press_nominal": 45.0,
            "oil_temp_max": 130.0,
            "egt_max": 880.0,
            "vibration_warn": 1.8,
            "vibration_crit": 2.5,
            "bus_voltage_min": 24.0,
            "bus_voltage_nominal": 28.0,
            "tbo_hours": 1200.0,
        }
    elif "diesel" in eid or "inline4" in eid:
        return {
            "name": "Generic-Inline4-AeroDiesel",
            "engine_type": "compression_ignition",
            "cycle": "diesel",
            "nominal_rpm": 2800.0,
            "max_rpm": 3880.0,
            "cht_warn": 125.0,
            "cht_max": 130.0,
            "oil_press_min": 35.0,
            "oil_press_nominal": 55.0,
            "oil_temp_max": 120.0,
            "egt_max": 750.0,
            "vibration_warn": 2.2,
            "vibration_crit": 3.0,
            "bus_voltage_min": 24.0,
            "bus_voltage_nominal": 28.0,
            "tbo_hours": 1500.0,
        }
    elif "2stroke" in eid or "twin-50hp" in eid:
        return {
            "name": "Generic-2Stroke-Twin-50HP",
            "engine_type": "spark_ignition",
            "cycle": "2-stroke",
            "nominal_rpm": 6500.0,
            "max_rpm": 7200.0,
            "cht_warn": 150.0,
            "cht_max": 160.0,
            "oil_press_min": 0.0,
            "oil_press_nominal": 0.0,
            "oil_temp_max": 140.0,
            "egt_max": 700.0,
            "vibration_warn": 2.5,
            "vibration_crit": 3.5,
            "bus_voltage_min": 12.0,
            "bus_voltage_nominal": 14.0,
            "tbo_hours": 500.0,
        }
    elif "wankel" in eid or "rotary" in eid:
        return {
            "name": "Generic-Rotary-Wankel-40HP",
            "engine_type": "rotary_wankel",
            "cycle": "rotary",
            "nominal_rpm": 6000.0,
            "max_rpm": 7500.0,
            "cht_warn": 140.0,
            "cht_max": 145.0,
            "oil_press_min": 20.0,
            "oil_press_nominal": 40.0,
            "oil_temp_max": 130.0,
            "egt_max": 800.0,
            "vibration_warn": 1.5,
            "vibration_crit": 2.2,
            "bus_voltage_min": 24.0,
            "bus_voltage_nominal": 28.0,
            "tbo_hours": 800.0,
        }
    else:
        # Default: AeroPiston-4C-1.35L
        return {
            "name": "AeroPiston-4C-1.35L",
            "engine_type": "spark_ignition",
            "cycle": "4-stroke",
            "nominal_rpm": 3000.0,
            "max_rpm": 5800.0,
            "cht_warn": 135.0,
            "cht_max": 140.0,
            "oil_press_min": 30.0,
            "oil_press_nominal": 50.0,
            "oil_temp_max": 135.0,
            "egt_max": 850.0,
            "vibration_warn": 2.0,
            "vibration_crit": 2.8,
            "bus_voltage_min": 24.0,
            "bus_voltage_nominal": 28.0,
            "tbo_hours": 2000.0,
        }


@dataclass
class MissionSummary:
    """Comprehensive, strongly-typed mission intelligence summary container."""
    mission: Dict[str, Any]
    envelope: Dict[str, Any]
    engine: Dict[str, Any]
    health: Dict[str, Any]
    faults: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    rul: Dict[str, Any]
    safety: Dict[str, Any]
    telemetry: Dict[str, Any]
    events: List[Dict[str, Any]]
    provenance: str
    engine_limits: Dict[str, Any]
    kpi_summary: Dict[str, Any]
    schema_version: str = "2.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventExtractor:
    """Deterministically extracts chronological mission events from canonical trajectory points."""

    def __init__(self, engine_limits: Dict[str, Any]):
        self.limits = engine_limits

    def extract_events(self, points: List[Dict[str, Any]]) -> List[MissionEvent]:
        if not points:
            return []

        events: List[MissionEvent] = []

        # 1. Mission Start
        first_p = points[0]
        t0_sec = float(first_p.get("timestamp", 0.0))
        t0_h = t0_sec / 3600.0
        p0_phase = str(first_p.get("mission_phase", "STARTUP"))
        events.append(
            MissionEvent(
                timestamp_sec=t0_sec,
                timestamp_hours=round(t0_h, 3),
                phase=p0_phase,
                event_type="MISSION_START",
                severity="INFO",
                evidence=f"Mission initialized in {p0_phase} phase; engine speed {first_p.get('RPM', 0.0):.0f} RPM",
                result="Avionics and propulsion telemetry monitoring active",
            )
        )
        if p0_phase.upper() in ("CLIMB", "TAKEOFF"):
            events.append(
                MissionEvent(
                    timestamp_sec=t0_sec,
                    timestamp_hours=round(t0_h, 3),
                    phase=p0_phase,
                    event_type=p0_phase.upper(),
                    severity="INFO",
                    evidence=f"Initial {p0_phase.lower()} active; climbing toward cruise altitude at {first_p.get('vertical_speed', 0.0):.0f} fpm",
                    result="Propulsion and flight controls set to climb profile",
                )
            )

        prev_phase = p0_phase
        prev_throttle = float(first_p.get("throttle", 0.0))
        prev_stage = str(first_p.get("degradation_stage", "HEALTHY"))
        prev_fault_present = bool(first_p.get("fault_present", False))
        prev_fault_type = str(first_p.get("fault_type", "none"))
        prev_sensor_fault = bool(first_p.get("sensor_fault_present", False))
        prev_fadec = str(first_p.get("FADEC_state", "NOMINAL"))
        prev_safety = str(first_p.get("safety_action", "NONE"))
        seen_dtcs = set(first_p.get("DTC", []) or [])

        # Threshold crossing trackers
        h_85_crossed = False
        h_60_crossed = False
        h_35_crossed = False
        cht_warn_crossed = False
        cht_crit_crossed = False
        oil_low_crossed = False
        vib_warn_crossed = False

        for i, p in enumerate(points[1:], start=1):
            t_sec = float(p.get("timestamp", 0.0))
            t_h = t_sec / 3600.0
            phase = str(p.get("mission_phase", prev_phase))
            h_idx = float(p.get("health_index", 100.0))
            cht = float(p.get("CHT", 100.0))
            oil_p = float(p.get("oil_pressure", 45.0))
            vib = float(p.get("vibration", 1.0))
            rpm = float(p.get("RPM", 0.0))
            throttle = float(p.get("throttle", 0.0))
            stage = str(p.get("degradation_stage", prev_stage))
            fault_pres = bool(p.get("fault_present", False))
            fault_tp = str(p.get("fault_type", "none"))
            sens_pres = bool(p.get("sensor_fault_present", False))
            fadec = str(p.get("FADEC_state", "NOMINAL"))
            safety = str(p.get("safety_action", "NONE"))
            pred_rul = p.get("predicted_RUL")
            dtcs = set(p.get("DTC", []) or [])

            # Phase transition
            if phase != prev_phase:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type=phase.upper() if phase.upper() in ("TAKEOFF", "CLIMB", "CRUISE", "DESCENT", "LANDING") else "PHASE_TRANSITION",
                        severity="INFO",
                        evidence=f"Mission phase transitioned from {prev_phase} to {phase} at altitude {p.get('altitude', 0.0):.0f} ft",
                        result=f"Flight controls and engine profile adjusted for {phase}",
                    )
                )
                prev_phase = phase

            # Large throttle step change
            if abs(throttle - prev_throttle) >= 0.25:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="THROTTLE_TRANSITION",
                        severity="INFO",
                        evidence=f"Throttle adjusted from {prev_throttle*100:.0f}% to {throttle*100:.0f}% (RPM: {rpm:.0f})",
                        result="Power plant load envelope modulated",
                    )
                )
                prev_throttle = throttle

            # Thermal deviations against engine-specific limits
            cht_warn_lim = self.limits["cht_warn"]
            cht_crit_lim = self.limits["cht_max"]
            if cht >= cht_crit_lim and not cht_crit_crossed:
                cht_crit_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="THERMAL_DEVIATION",
                        severity="CRITICAL",
                        evidence=f"CHT reached critical limit: {cht:.1f}°C (Engine Max Limit: {cht_crit_lim:.1f}°C)",
                        result="Thermal runaway threshold breached; supervisory cooling/derate triggered",
                    )
                )
            elif cht >= cht_warn_lim and not cht_warn_crossed:
                cht_warn_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="THERMAL_DEVIATION",
                        severity="WARN",
                        evidence=f"Elevated cylinder head temperature: CHT = {cht:.1f}°C (Warning Limit: {cht_warn_lim:.1f}°C)",
                        result="Digital Twin registers positive thermal residual",
                    )
                )

            # Lubrication breakdown
            oil_min_lim = self.limits["oil_press_min"]
            if oil_min_lim > 0.0 and oil_p < oil_min_lim and not oil_low_crossed:
                oil_low_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="LUBRICATION_ABNORMALITY",
                        severity="CRITICAL",
                        evidence=f"Oil pressure dropped to {oil_p:.1f} psi (Minimum Safe Limit: {oil_min_lim:.1f} psi)",
                        result="Lubrication starvation detected; bearing wear acceleration risk",
                    )
                )

            # Vibration anomaly
            vib_warn_lim = self.limits["vibration_warn"]
            if vib >= vib_warn_lim and not vib_warn_crossed:
                vib_warn_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="VIBRATION_ANOMALY",
                        severity="WARN" if vib < self.limits["vibration_crit"] else "CRITICAL",
                        evidence=f"Tri-axial engine vibration reached {vib:.2f} g (Baseline Warning: {vib_warn_lim:.2f} g)",
                        result="Harmonic structural excitation / imbalance noted",
                    )
                )

            # Sensor fault onset
            if sens_pres and not prev_sensor_fault:
                sens_type = p.get("sensor_fault_type", "transducer_drift")
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="SENSOR_DRIFT",
                        severity="WARN",
                        evidence=f"Transducer anomaly detected: {sens_type} on instrumentation channel (Trust: {p.get('sensor_trust', 0.0):.1f}%)",
                        result="Cross-channel sensor fusion isolated faulty transducer channel",
                    )
                )
                prev_sensor_fault = True

            # Physical fault onset
            if fault_pres and (not prev_fault_present or fault_tp != prev_fault_type):
                sev = float(p.get("fault_severity", 0.0))
                ev_type = "MECHANICAL_DEGRADATION"
                if "thermal" in fault_tp.lower() or "overheat" in fault_tp.lower():
                    ev_type = "THERMAL_DEVIATION"
                elif "lubricat" in fault_tp.lower() or "oil" in fault_tp.lower():
                    ev_type = "LUBRICATION_ABNORMALITY"
                elif "inject" in fault_tp.lower():
                    ev_type = "INJECTOR_ISSUE"
                elif "misfire" in fault_tp.lower():
                    ev_type = "MISFIRE"

                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type=ev_type,
                        severity="WARN" if sev < 0.6 else "CRITICAL",
                        evidence=f"Active propulsion fault manifested: {fault_tp.upper()} (Severity: {sev*100:.0f}%)",
                        result="Digital Twin physical model divergence registered",
                    )
                )
                prev_fault_present = True
                prev_fault_type = fault_tp

            # Degradation stage progression
            if stage != prev_stage:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="HEALTH_DEGRADATION",
                        severity="INFO" if stage == "HEALTHY" else ("WARN" if stage in ("EARLY", "MODERATE") else "CRITICAL"),
                        evidence=f"Degradation kinetics transitioned from {prev_stage} to {stage} (Health: {h_idx:.1f}%)",
                        result=f"System entered {stage} wear progression state",
                    )
                )
                prev_stage = stage

            # Health threshold crossings (85%, 60%, 35%)
            if h_idx <= 85.0 and not h_85_crossed:
                h_85_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="HEALTH_DEGRADATION",
                        severity="WARN",
                        evidence=f"Health Index crossed below early warning threshold: H = {h_idx:.1f}% (Threshold: 85.0%)",
                        result="Predictive prognostics active; degradation confirmed",
                    )
                )

            if h_idx <= 60.0 and not h_60_crossed:
                h_60_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="HEALTH_DEGRADATION",
                        severity="WARN",
                        evidence=f"Health Index crossed below moderate threshold: H = {h_idx:.1f}% (Threshold: 60.0%)",
                        result="RUL decrement rate accelerating; mission planning notified",
                    )
                )

            if h_idx <= 35.0 and not h_35_crossed:
                h_35_crossed = True
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="CRITICAL_MAINTENANCE",
                        severity="CRITICAL",
                        evidence=f"Health Index reached terminal failure limit: H = {h_idx:.1f}% (Threshold: 35.0%)",
                        result="End of useful life reached; mandatory immediate overhaul required",
                    )
                )

            # Significant RUL drop
            if pred_rul is not None and i > 1:
                prev_pred = points[i-1].get("predicted_RUL")
                if prev_pred is not None:
                    delta_r = prev_pred - pred_rul
                    if delta_r >= 4.0:  # Rapid drop > 4h
                        events.append(
                            MissionEvent(
                                timestamp_sec=t_sec,
                                timestamp_hours=round(t_h, 3),
                                phase=phase,
                                event_type="RUL_DROP",
                                severity="WARN",
                                evidence=f"Predicted RUL dropped sharply from {prev_pred:.1f} h to {pred_rul:.1f} h (Δ = -{delta_r:.1f} h)",
                                result="Prognostic horizon updated based on accelerated degradation evidence",
                            )
                        )

            # FADEC state change
            if fadec != prev_fadec:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="DERATE_WARNING" if "DERATE" in fadec else ("SAFETY_ACTION" if "EMERGENCY" in fadec else "FADEC_TRANSITION"),
                        severity="WARN" if "WARN" in fadec else ("CRITICAL" if "CRITICAL" in fadec or "EMERGENCY" in fadec else "INFO"),
                        evidence=f"FADEC supervisory control shifted: {prev_fadec} -> {fadec} (Throttle command: {throttle*100:.0f}%)",
                        result=f"Governor control laws enforce {fadec} operational envelope",
                    )
                )
                prev_fadec = fadec

            # Safety Action execution
            if safety != "NONE" and safety != prev_safety:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="SAFETY_ACTION",
                        severity="CRITICAL" if "EMERGENCY" in safety else "WARN",
                        evidence=f"Autonomous flight computer executed safety action: {safety}",
                        result=f"Action commanded: {safety} to protect airframe and propulsion integrity",
                    )
                )
                prev_safety = safety

            # New DTCs
            new_dtcs = dtcs - seen_dtcs
            if new_dtcs:
                events.append(
                    MissionEvent(
                        timestamp_sec=t_sec,
                        timestamp_hours=round(t_h, 3),
                        phase=phase,
                        event_type="CRITICAL_MAINTENANCE",
                        severity="WARN",
                        evidence=f"Diagnostic Trouble Code logged on CAN bus: {', '.join(sorted(new_dtcs))}",
                        result="Fault recorded in non-volatile diagnostic memory",
                    )
                )
                seen_dtcs.update(new_dtcs)

        # Mission End
        last_p = points[-1]
        t_end_sec = float(last_p.get("timestamp", 0.0))
        t_end_h = t_end_sec / 3600.0
        last_phase = str(last_p.get("mission_phase", prev_phase))
        events.append(
            MissionEvent(
                timestamp_sec=t_end_sec,
                timestamp_hours=round(t_end_h, 3),
                phase=last_phase,
                event_type="MISSION_END",
                severity="INFO" if float(last_p.get("health_index", 100.0)) > 35.0 else "CRITICAL",
                evidence=f"Mission terminated at T+{t_end_h:.2f} h (Final Health: {last_p.get('health_index', 0.0):.1f}%, RUL: {last_p.get('predicted_RUL', 0.0) if last_p.get('predicted_RUL') is not None else 0.0:.1f} h)",
                result=f"Final engine state: {last_p.get('ECU_state', 'UNKNOWN')}, Safety action: {last_p.get('safety_action', 'NONE')}",
            )
        )

        return events


class TrajectorySummarizer:
    """Consumes complete multivariate trajectory and generates deterministic statistics and summaries."""

    def __init__(self):
        pass

    def summarize(self, raw_points: List[Union[Dict[str, Any], Any]]) -> MissionSummary:
        if not raw_points:
            raise ValueError("Trajectory points list cannot be empty.")

        # Convert to dicts if dataclass instances
        points: List[Dict[str, Any]] = [
            p.to_dict() if hasattr(p, "to_dict") else dict(p)
            for p in raw_points
        ]

        first_pt = points[0]
        last_pt = points[-1]

        trajectory_id = str(first_pt.get("trajectory_id", "TRAJ_UNKNOWN"))
        engine_id = str(first_pt.get("engine_id", "ROTAX_914_F_TWIN_01"))
        mission_id = str(first_pt.get("mission_id", "MSN_UNKNOWN"))

        # Resolve engine limits dynamically
        limits = resolve_engine_limits(engine_id)

        # 1. Mission Duration & Phases
        t_start_sec = float(first_pt.get("timestamp", 0.0))
        t_end_sec = float(last_pt.get("timestamp", 0.0))
        duration_sec = max(0.0, t_end_sec - t_start_sec)
        duration_hours = round(duration_sec / 3600.0, 3)

        phases_present: List[str] = []
        phase_durations: Dict[str, float] = {}
        curr_phase = None
        curr_p_start = t_start_sec

        for p in points:
            ph = str(p.get("mission_phase", "CRUISE"))
            if ph not in phases_present:
                phases_present.append(ph)
            if curr_phase is None:
                curr_phase = ph
                curr_p_start = float(p.get("timestamp", 0.0))
            elif ph != curr_phase:
                p_t = float(p.get("timestamp", 0.0))
                phase_durations[curr_phase] = phase_durations.get(curr_phase, 0.0) + (p_t - curr_p_start)
                curr_phase = ph
                curr_p_start = p_t
        if curr_phase:
            phase_durations[curr_phase] = phase_durations.get(curr_phase, 0.0) + (t_end_sec - curr_p_start)
        # Convert phase durations to hours
        phase_durations_h = {k: round(v / 3600.0, 3) for k, v in phase_durations.items()}

        mission_info = {
            "mission_id": mission_id,
            "trajectory_id": trajectory_id,
            "engine_id": engine_id,
            "engine_profile": limits["name"],
            "duration_sec": duration_sec,
            "duration_hours": duration_hours,
            "phases": phases_present,
            "phase_durations_hours": phase_durations_h,
            "initial_state": {
                "phase": first_pt.get("mission_phase", "STARTUP"),
                "altitude_ft": float(first_pt.get("altitude", 0.0)),
                "RPM": float(first_pt.get("RPM", 0.0)),
                "health_index": float(first_pt.get("health_index", 100.0)),
                "predicted_RUL": first_pt.get("predicted_RUL"),
            },
            "final_state": {
                "phase": last_pt.get("mission_phase", "LANDING"),
                "altitude_ft": float(last_pt.get("altitude", 0.0)),
                "RPM": float(last_pt.get("RPM", 0.0)),
                "health_index": float(last_pt.get("health_index", 0.0)),
                "predicted_RUL": last_pt.get("predicted_RUL"),
                "ECU_state": last_pt.get("ECU_state", "ACTIVE_RUN"),
                "FADEC_state": last_pt.get("FADEC_state", "NOMINAL"),
                "safety_action": last_pt.get("safety_action", "NONE"),
            }
        }

        # 2. Operating Envelope Statistics
        def calc_stats(key: str, default: float = 0.0) -> Dict[str, float]:
            vals = [float(p.get(key, default)) for p in points if p.get(key) is not None]
            if not vals:
                return {"min": default, "max": default, "mean": default}
            return {
                "min": round(float(min(vals)), 2),
                "max": round(float(max(vals)), 2),
                "mean": round(float(sum(vals) / len(vals)), 2),
            }

        envelope_stats = {
            "altitude_ft": calc_stats("altitude", 5000.0),
            "RPM": calc_stats("RPM", 4500.0),
            "throttle": calc_stats("throttle", 0.65),
            "engine_load": calc_stats("engine_load", 0.65),
            "ambient_temperature_c": calc_stats("ambient_temperature", 15.0),
            "ambient_pressure_kpa": calc_stats("ambient_pressure", 101.3),
            "ground_speed_kts": calc_stats("ground_speed", 85.0),
            "MAP_inHg": calc_stats("MAP", 28.5),
        }

        # 3. Engine Parameters
        engine_stats = {
            "brake_power_kw": calc_stats("brake_power", 55.0),
            "torque_nm": calc_stats("torque", 115.0),
            "fuel_flow_l_h": calc_stats("fuel_flow", 22.0),
            "AFR_or_lambda": calc_stats("AFR_or_lambda", 14.7),
            "CHT_c": calc_stats("CHT", 110.0),
            "coolant_temperature_c": calc_stats("coolant_temperature", 85.0),
            "EGT_c": calc_stats("EGT", 780.0),
            "oil_pressure_psi": calc_stats("oil_pressure", 45.0),
            "oil_temperature_c": calc_stats("oil_temperature", 92.0),
            "vibration_g": calc_stats("vibration", 1.1),
            "bus_voltage_v": calc_stats("bus_voltage", 28.0),
            "current_a": calc_stats("current", 18.0),
            "battery_SOC_pct": calc_stats("battery_SOC", 98.0),
        }

        # 4. System Health & Degradation Assessment
        health_vals = [float(p.get("health_index", 100.0)) for p in points]
        init_h = float(first_pt.get("health_index", 100.0))
        min_h = float(min(health_vals))
        final_h = float(last_pt.get("health_index", 100.0))

        # Degradation onset
        deg_onset_sec = None
        for p in points:
            if float(p.get("degradation_severity", 0.0)) > 0.0 or str(p.get("degradation_stage", "HEALTHY")) != "HEALTHY" or float(p.get("health_index", 100.0)) < 99.0:
                deg_onset_sec = float(p.get("timestamp", 0.0))
                break

        deg_rate_h = 0.0
        if duration_hours > 0:
            deg_rate_h = round((init_h - final_h) / duration_hours, 2)

        health_stats = {
            "initial_health": round(init_h, 1),
            "minimum_health": round(min_h, 1),
            "final_health": round(final_h, 1),
            "health_change": round(final_h - init_h, 1),
            "degradation_onset_sec": deg_onset_sec,
            "degradation_onset_hours": round(deg_onset_sec / 3600.0, 3) if deg_onset_sec is not None else None,
            "degradation_rate_pct_per_hour": deg_rate_h,
            "stages_encountered": list(dict.fromkeys(str(p.get("degradation_stage", "HEALTHY")) for p in points)),
        }

        # 5. Faults & Failure Modes
        faults_detected = []
        fault_types_seen = set()
        for p in points:
            if p.get("fault_present"):
                ft = str(p.get("fault_type", "unspecified"))
                if ft not in fault_types_seen and ft != "none":
                    fault_types_seen.add(ft)
                    faults_detected.append({
                        "fault_type": ft,
                        "failure_mode": str(p.get("failure_mode", ft)),
                        "onset_timestamp_sec": float(p.get("timestamp", 0.0)),
                        "onset_timestamp_hours": round(float(p.get("timestamp", 0.0)) / 3600.0, 3),
                        "peak_severity": round(max(float(pt.get("fault_severity", 0.0)) for pt in points if pt.get("fault_type") == ft), 2),
                        "affected_subsystem": "COOLING_SYSTEM" if "thermal" in ft else ("LUBRICATION_SYSTEM" if "oil" in ft or "lub" in ft else ("COMBUSTION_CORE" if "misfire" in ft or "injector" in ft else "MECHANICAL_TRANSMISSION")),
                        "evidence": f"Peak severity {max(float(pt.get('fault_severity', 0.0)) for pt in points if pt.get('fault_type') == ft)*100:.0f}% observed during mission",
                    })

        # 6. Anomalies & Physics Residuals
        anomalies_detected = []
        # Check CHT elevation above nominal baseline (110.0 C)
        max_cht = engine_stats["CHT_c"]["max"]
        if max_cht > limits["cht_warn"]:
            anomalies_detected.append({
                "anomaly_type": "THERMAL_RESIDUAL_HIGH",
                "affected_signals": ["CHT", "EGT"],
                "peak_deviation": f"+{max_cht - 110.0:.1f}°C above nominal baseline",
                "evidence": f"CHT peak {max_cht:.1f}°C exceeds threshold {limits['cht_warn']:.1f}°C",
            })
        if engine_stats["oil_pressure_psi"]["min"] < limits["oil_press_min"] and limits["oil_press_min"] > 0:
            anomalies_detected.append({
                "anomaly_type": "OIL_PRESSURE_DEFICIT",
                "affected_signals": ["oil_pressure"],
                "peak_deviation": f"{engine_stats['oil_pressure_psi']['min']:.1f} psi",
                "evidence": f"Oil pressure dropped below safe minimum {limits['oil_press_min']:.1f} psi",
            })
        if engine_stats["vibration_g"]["max"] > limits["vibration_warn"]:
            anomalies_detected.append({
                "anomaly_type": "VIBRATION_EXCEEDANCE",
                "affected_signals": ["vibration"],
                "peak_deviation": f"{engine_stats['vibration_g']['max']:.2f} g",
                "evidence": f"Vibration exceeded warning threshold {limits['vibration_warn']:.2f} g",
            })

        # 7. RUL & Prognostics
        pred_rul_vals = [float(p["predicted_RUL"]) for p in points if p.get("predicted_RUL") is not None]
        true_rul_vals = [float(p["true_RUL"]) for p in points if p.get("true_RUL") is not None]
        ci_lower_vals = [float(p["RUL_lower"]) for p in points if p.get("RUL_lower") is not None]
        ci_upper_vals = [float(p["RUL_upper"]) for p in points if p.get("RUL_upper") is not None]
        conf_vals = [float(p["RUL_confidence"]) for p in points if p.get("RUL_confidence") is not None]

        init_pred_rul = pred_rul_vals[0] if pred_rul_vals else None
        final_pred_rul = pred_rul_vals[-1] if pred_rul_vals else None
        init_true_rul = true_rul_vals[0] if true_rul_vals else None
        final_true_rul = true_rul_vals[-1] if true_rul_vals else None

        pred_err = None
        if final_pred_rul is not None and final_true_rul is not None:
            pred_err = round(abs(final_pred_rul - final_true_rul), 2)

        rul_stats = {
            "initial_predicted_rul_hours": round(init_pred_rul, 2) if init_pred_rul is not None else None,
            "final_predicted_rul_hours": round(final_pred_rul, 2) if final_pred_rul is not None else None,
            "predicted_rul_change_hours": round(final_pred_rul - init_pred_rul, 2) if (init_pred_rul is not None and final_pred_rul is not None) else None,
            "min_ci_lower_hours": round(min(ci_lower_vals), 2) if ci_lower_vals else None,
            "max_ci_upper_hours": round(max(ci_upper_vals), 2) if ci_upper_vals else None,
            "mean_confidence_pct": round(sum(conf_vals)/len(conf_vals), 1) if conf_vals else None,
            "final_confidence_pct": round(conf_vals[-1], 1) if conf_vals else None,
            "initial_true_rul_hours": round(init_true_rul, 2) if init_true_rul is not None else None,
            "final_true_rul_hours": round(final_true_rul, 2) if final_true_rul is not None else None,
            "prediction_error_hours": pred_err,
            "has_ground_truth": len(true_rul_vals) > 0,
        }

        # 8. Safety & Flight Computer Actions
        ecu_states = list(dict.fromkeys(str(p.get("ECU_state", "ACTIVE_RUN")) for p in points))
        fadec_states = list(dict.fromkeys(str(p.get("FADEC_state", "NOMINAL")) for p in points))
        safety_actions = list(dict.fromkeys(str(p.get("safety_action", "NONE")) for p in points))
        dtcs_set = set()
        for p in points:
            for c in (p.get("DTC") or []):
                dtcs_set.add(str(c))

        derate_vals = [float(p.get("derate_command", 1.0)) for p in points if p.get("derate_command") is not None]
        min_derate = min(derate_vals) if derate_vals else 1.0

        safety_stats = {
            "ecu_states": ecu_states,
            "fadec_states": fadec_states,
            "safety_actions_executed": safety_actions,
            "final_safety_action": last_pt.get("safety_action", "NONE"),
            "dtcs_encountered": sorted(list(dtcs_set)),
            "min_derate_clamp": round(min_derate, 2),
            "derate_active": min_derate < 0.99,
        }

        # 9. Telemetry & CAN Bus Integrity
        pkt_loss_vals = [float(p.get("CAN_packet_loss", 0.0)) for p in points]
        crc_statuses = [str(p.get("CAN_CRC_status", "VALID")) for p in points]
        lat_vals = [float(p.get("CAN_latency", 0.045)) for p in points]
        watchdogs = list(dict.fromkeys(str(p.get("watchdog_state", "HEALTHY")) for p in points))
        deadlines_missed = sum(1 for p in points if bool(p.get("deadline_missed", False)))

        valid_crc_count = sum(1 for s in crc_statuses if s == "VALID")
        crc_pct = round((valid_crc_count / len(crc_statuses)) * 100.0, 1) if crc_statuses else 100.0

        telemetry_stats = {
            "can_packet_loss_mean": round(sum(pkt_loss_vals) / len(pkt_loss_vals), 4) if pkt_loss_vals else 0.0,
            "can_packet_loss_max": round(max(pkt_loss_vals), 4) if pkt_loss_vals else 0.0,
            "can_crc_valid_pct": crc_pct,
            "can_latency_mean_ms": round(sum(lat_vals) / len(lat_vals), 4) if lat_vals else 0.045,
            "can_latency_max_ms": round(max(lat_vals), 4) if lat_vals else 0.045,
            "watchdog_states": watchdogs,
            "deadline_misses_count": deadlines_missed,
            "flight_computer_state": last_pt.get("flight_computer_state", "OPERATIONAL"),
        }

        # 10. Extract Chronological Events
        extractor = EventExtractor(engine_limits=limits)
        events = extractor.extract_events(points)
        events_dicts = [e.to_dict() for e in events]

        # 11. Determine Data Provenance
        provenance = "SIMULATED TELEMETRY"
        if len(true_rul_vals) > 0 or "TRAJ_" in trajectory_id:
            provenance = "SYNTHETIC GROUND TRUTH"
        elif "MEASURED" in trajectory_id:
            provenance = "MEASURED TELEMETRY"

        # 12. Top-Level KPI Summary Determination
        final_safety = last_pt.get("safety_action", "NONE")
        if final_h <= 35.0 or "EMERGENCY" in final_safety:
            mission_status = "ABORTED — EMERGENCY RTL"
            engine_status = "CRITICAL FAILURE / SHUTDOWN"
            risk = "CRITICAL (RISK LEVEL 5)"
            mission_impact = "Mission aborted early due to engine life depletion. Aircraft returned to base."
            rec_action = f"Mandatory Depot-Level Overhaul (D-Level) for {limits['name']} power plant."
        elif final_h <= 60.0 or "DERATE" in final_safety:
            mission_status = "COMPLETED WITH RESTRICTION"
            engine_status = "DEGRADED — DERATED"
            risk = "HIGH (RISK LEVEL 3)"
            mission_impact = "Flight completed under restricted FADEC throttle ceiling clamp."
            rec_action = "Execute Flight-Line (O-Level) and Field Workshop (I-Level) inspection prior to next flight."
        elif len(faults_detected) > 0 or final_h < 85.0:
            mission_status = "COMPLETED — CAUTION"
            engine_status = "MINOR DEGRADATION"
            risk = "MEDIUM (RISK LEVEL 2)"
            mission_impact = "Nominal trajectory completed; minor degradation detected by digital twin."
            rec_action = "Perform pre-flight BITE check and sensor calibration before next sortie."
        else:
            mission_status = "COMPLETED — NOMINAL"
            engine_status = "NOMINAL HEALTH"
            risk = "LOW (RISK LEVEL 1)"
            mission_impact = "All mission objectives accomplished within normal operating envelope."
            rec_action = "Routine post-flight turn-around servicing."

        primary_event = "Nominal Multi-Phase Flight"
        if faults_detected:
            primary_event = f"{faults_detected[0]['fault_type'].upper()} Degradation"
        elif final_h < 85.0:
            primary_event = "Progressive Mechanical Wear"

        kpi_summary = {
            "mission_status": mission_status,
            "engine_status": engine_status,
            "primary_event": primary_event,
            "current_health": f"{final_h:.1f}%",
            "current_rul": f"{final_pred_rul:.1f} h" if final_pred_rul is not None else "N/A",
            "rul_confidence": f"{rul_stats['final_confidence_pct']:.1f}%" if rul_stats['final_confidence_pct'] is not None else "90%",
            "risk": risk,
            "safety_action": final_safety,
            "mission_impact": mission_impact,
            "recommended_action": rec_action,
        }

        return MissionSummary(
            mission=mission_info,
            envelope=envelope_stats,
            engine=engine_stats,
            health=health_stats,
            faults=faults_detected,
            anomalies=anomalies_detected,
            rul=rul_stats,
            safety=safety_stats,
            telemetry=telemetry_stats,
            events=events_dicts,
            provenance=provenance,
            engine_limits=limits,
            kpi_summary=kpi_summary,
        )
