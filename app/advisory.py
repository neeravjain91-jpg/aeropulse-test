"""Defence-Grade Autonomous Maintenance Advisory System for MALE UAV Propulsion.
Structured around Austin (2010) Power-plant Failure Hierarchy & Maintenance Echelons.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# Reg Austin (2010) Power-plant Decomposition Hierarchy (Ch 5.2.1, p. 78; Ch 16.5, p. 214)
POWERPLANT_HIERARCHY = {
    "FUEL_SYSTEM": ["Fuel Tank", "Fuel Pump", "Injectors/Carburettor", "Fuel Lines", "Fuel Filter"],
    "COMBUSTION_CORE": ["Cylinder Head", "Pistons", "Valves", "Spark Ignition", "Combustion Chamber"],
    "LUBRICATION_SYSTEM": ["Oil Sump", "Oil Pump", "Oil Cooler/Radiator", "Oil Filters", "Journal Bearings"],
    "COOLING_SYSTEM": ["Cooling Jacket", "Radiator", "Cowl Flaps", "Coolant Pump", "De-humidifier"],
    "MECHANICAL_TRANSMISSION": ["Crankshaft", "Connecting Rods", "Reduction Gearbox", "Shafts", "Propeller Hub"],
    "ELECTRICAL_GENERATION": ["Alternator", "Power Conditioning Unit", "Battery Assembly", "Wiring Loom"],
    "INSTRUMENTATION_SENSORS": ["CHT Thermocouples", "EGT Probes", "MAP Transducers", "Oil Pressure Sensor", "Vibration Accelerometer"],
}


@dataclass
class MaintenanceAction:
    subsystem_tier: str       # From POWERPLANT_HIERARCHY
    subsystem_component: str
    detected_condition: str
    severity: str             # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    echelon: str              # "O-Level" (Flight-Line), "I-Level" (Field Workshop), "D-Level" (Depot Overhaul)
    dispatch_status: str      # "GO_MISSION_READY", "CAUTION_RESTRICTED_ENVELOPE", "NO_GO_MAINTENANCE_HOLD"
    recommended_action: str
    technical_order: str
    urgency: str
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def fault_advisory(telemetry: dict, twin: dict, sensor_health: dict | None = None) -> list[tuple[str, str, list[str]]]:
    """
    Evaluates Digital Twin residuals, cross-channel statistics, and sensor integrity
    to isolate failure modes and candidate root causes mapped to Austin's hierarchy.
    """
    z = twin.get("z_scores", {}) if isinstance(twin, dict) else {}
    findings = []

    # 1. Sensor Instrumentation Fault (Priority isolation)
    if sensor_health and sensor_health.get("suspected_sensor_fault"):
        suspects = sensor_health.get("suspect_channels", [])
        findings.append(
            (
                "Sensor Drift / Instrumentation Bias",
                "medium",
                ["cross-channel discrepancy detected", f"suspect channels: {', '.join(suspects)}"],
            )
        )

    # 2. Lubrication Subsystem Degradation
    if z.get("Oil_Pressure", 0.0) < -1.5 and z.get("Oil_Temp", 0.0) > 1.0:
        findings.append(
            (
                "Lubrication System Degradation / Oil Starvation",
                "high",
                ["low oil pressure residual (<-1.5 sigma)", "elevated oil temperature residual (>+1.0 sigma)"],
            )
        )

    # 3. Severe Overheating & Thermal Runaway
    egt_max_z = max(z.get("EGT1", 0.0), z.get("EGT2", 0.0), z.get("EGT3", 0.0))
    water_z = z.get("EFI_Water_Temp", 0.0)
    cht_z = z.get("CHT", 0.0)

    if max(egt_max_z, water_z, cht_z) > 3.0:
        findings.append(
            (
                "Thermal Runaway / Cooling System Breakdown",
                "high",
                ["temperatures exceeding 3-sigma healthy reference", f"Max CHT z: {cht_z:.2f}, EGT z: {egt_max_z:.2f}"],
            )
        )
    elif max(egt_max_z, water_z, cht_z) > 1.8:
        findings.append(
            (
                "Elevated Thermal Stress / Cooling Degradation",
                "medium",
                ["moderate temperature elevation above healthy reference"],
            )
        )

    # 4. Combustion Instability & Cylinder Misfire
    def _safe_float(val, default: float) -> float:
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    egt1 = _safe_float(telemetry.get("EGT1"), 1200.0)
    egt2 = _safe_float(telemetry.get("EGT2"), 1200.0)
    egt3 = _safe_float(telemetry.get("EGT3"), 1200.0)
    egt_spread = max(egt1, egt2, egt3) - min(egt1, egt2, egt3)

    if egt_spread > 120.0:
        findings.append(
            (
                "Combustion Imbalance / Single-Cylinder Misfire",
                "high" if egt_spread > 220.0 else "medium",
                [f"EGT delta across cylinders: {egt_spread:.1f}F", "asymmetric exhaust gas thermal release"],
            )
        )

    # 5. Fuel Injection & MAP Abnormality
    if abs(z.get("MAP_Injector", 0.0)) > 2.0 and abs(z.get("Fuel_Flow", 0.0)) > 1.0:
        findings.append(
            (
                "Fuel Injection Rail / Injector Clogging",
                "medium",
                ["manifold injector pressure residual mismatch", "fuel flow delivery anomaly"],
            )
        )

    # 6. Electrical Generation & Alternator Thermal Fault
    if "Battery_Voltage" in z and "Alternator_Temp" in z:
        if z.get("Battery_Voltage", 0.0) < -1.8 or z.get("Alternator_Temp", 0.0) > 2.2:
            findings.append(
                (
                    "Electrical Power / Alternator Overheat",
                    "medium",
                    ["anomalous bus voltage drop or excessive alternator winding temperature"],
                )
            )

    # 7. Mechanical Wear & High Vibration
    if z.get("Vibration", 0.0) > 2.0:
        findings.append(
            (
                "Mechanical Wear / High Vibration Harmonics",
                "high",
                ["elevated engine vibration exceeding 2-sigma baseline"],
            )
        )

    return findings


def generate_echelon_advisory(findings: list[tuple[str, str, list[str]]]) -> List[MaintenanceAction]:
    """Translates diagnostic findings into structured Multi-Echelon Maintenance Actions."""
    actions = []
    
    for cond, sev, evidence in findings:
        s_upper = sev.upper()
        if "Sensor" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="INSTRUMENTATION_SENSORS",
                    subsystem_component="Transducer Harness / Sensor Interface",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="O-Level",
                    dispatch_status="CAUTION_RESTRICTED_ENVELOPE",
                    recommended_action="Execute pre-flight BITE sensor calibration; inspect thermocouple connectors.",
                    technical_order="TO-UAV-ENG-SENS-04",
                    urgency="NEXT_SERVICE_WINDOW",
                    evidence=evidence,
                )
            )
        elif "Lubrication" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="LUBRICATION_SYSTEM",
                    subsystem_component="Oil Sump / Oil Pump",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="I-Level",
                    dispatch_status="NO_GO_MAINTENANCE_HOLD",
                    recommended_action="Drain and inspect oil filter for ferrous debris; perform oil pump pressure relief test.",
                    technical_order="TO-UAV-ENG-LUB-02",
                    urgency="IMMEDIATE_PRE_FLIGHT",
                    evidence=evidence,
                )
            )
        elif "Thermal" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="COOLING_SYSTEM",
                    subsystem_component="Radiator / Cowl Cooling Passages",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="O-Level" if s_upper == "MEDIUM" else "I-Level",
                    dispatch_status="CAUTION_RESTRICTED_ENVELOPE" if s_upper == "MEDIUM" else "NO_GO_MAINTENANCE_HOLD",
                    recommended_action="Inspect radiator matrix for debris; verify cowl flap servo range of motion.",
                    technical_order="TO-UAV-ENG-COOL-01",
                    urgency="PRIORITY_24H" if s_upper == "MEDIUM" else "IMMEDIATE_PRE_FLIGHT",
                    evidence=evidence,
                )
            )
        elif "Combustion" in cond or "Misfire" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="COMBUSTION_CORE",
                    subsystem_component="Spark Plugs / Ignition Harness",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="O-Level",
                    dispatch_status="NO_GO_MAINTENANCE_HOLD" if s_upper == "HIGH" else "CAUTION_RESTRICTED_ENVELOPE",
                    recommended_action="Perform ignition drop check; inspect and gap spark plugs; check ignition coils.",
                    technical_order="TO-UAV-ENG-IGN-03",
                    urgency="IMMEDIATE_PRE_FLIGHT",
                    evidence=evidence,
                )
            )
        elif "Fuel" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="FUEL_SYSTEM",
                    subsystem_component="Fuel Injectors / Pressure Regulator",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="I-Level",
                    dispatch_status="CAUTION_RESTRICTED_ENVELOPE",
                    recommended_action="Ultrasonically clean fuel injectors; replace in-line fuel filter element.",
                    technical_order="TO-UAV-ENG-FUEL-01",
                    urgency="PRIORITY_24H",
                    evidence=evidence,
                )
            )
        elif "Electrical" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="ELECTRICAL_GENERATION",
                    subsystem_component="Alternator / Voltage Regulator",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="O-Level",
                    dispatch_status="CAUTION_RESTRICTED_ENVELOPE",
                    recommended_action="Inspect alternator drive belt tension; check generator bus output at 3000 RPM.",
                    technical_order="TO-UAV-ENG-ELEC-05",
                    urgency="NEXT_SERVICE_WINDOW",
                    evidence=evidence,
                )
            )
        elif "Mechanical" in cond:
            actions.append(
                MaintenanceAction(
                    subsystem_tier="MECHANICAL_TRANSMISSION",
                    subsystem_component="Crankshaft Bearings / Reduction Drive",
                    detected_condition=cond,
                    severity=s_upper,
                    echelon="D-Level",
                    dispatch_status="NO_GO_MAINTENANCE_HOLD",
                    recommended_action="Perform cylinder borescope inspection; dynamic propeller balancing check.",
                    technical_order="TO-UAV-ENG-MECH-09",
                    urgency="IMMEDIATE_PRE_FLIGHT",
                    evidence=evidence,
                )
            )

    return actions


def detailed_maintenance_action(finding: Any) -> MaintenanceAction:
    """Formats individual diagnostic finding or list of findings into structured MaintenanceAction."""
    if isinstance(finding, list):
        if not finding:
            return MaintenanceAction(
                subsystem_tier="POWERPLANT_GENERAL",
                subsystem_component="General Engine",
                detected_condition="Nominal Operation",
                severity="LOW",
                echelon="O-Level",
                dispatch_status="GO_MISSION_READY",
                recommended_action="Standard scheduled phase inspection.",
                technical_order="TO-UAV-ENG-GEN-01",
                urgency="ROUTINE",
                evidence=["All parameters nominal"],
            )
        item = finding[0]
    else:
        item = finding

    if isinstance(item, dict):
        cond = item.get("condition", "Unknown")
        sev = item.get("severity", "LOW")
        ev = item.get("evidence", [])
    elif isinstance(item, (tuple, list)) and len(item) >= 3:
        cond, sev, ev = item[0], item[1], item[2]
    else:
        cond, sev, ev = str(item), "LOW", []

    actions = generate_echelon_advisory([(cond, sev, ev)])
    if actions:
        return actions[0]

    return MaintenanceAction(
        subsystem_tier="POWERPLANT_GENERAL",
        subsystem_component="General Engine",
        detected_condition=cond,
        severity=str(sev).upper(),
        echelon="O-Level",
        dispatch_status="GO_MISSION_READY",
        recommended_action="Standard scheduled phase inspection.",
        technical_order="TO-UAV-ENG-GEN-01",
        urgency="ROUTINE",
        evidence=ev if isinstance(ev, list) else [str(ev)],
    )


def maintenance_advice(findings: list[tuple[str, str, list[str]]], sensor_health: dict | None = None) -> list[str]:
    """Generates plain-language operational maintenance action recommendations."""
    if not findings:
        return ["All monitored engine parameters nominal. Clear for standard mission profile."]
    
    advice = []
    actions = generate_echelon_advisory(findings)
    for a in actions:
        advice.append(f"[{a.echelon} | {a.dispatch_status}] {a.detected_condition}: {a.recommended_action} (Ref: {a.technical_order})")
    return advice
