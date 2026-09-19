"""What-If Mission Scenario and Comparative RUL Impact Engine."""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

from .engine_model import EngineInputs, ReducedOrderPistonEngine
from .mission_whatif import MissionScenario
from .rul_service import RULService
from .simulator import mission_adjust
from .uav_platform_config import UAVPlatformProfile, get_uav_profile


class MissionWhatIfRUL:
    """
    Simulates and compares the degradation impact and Remaining Useful Life (RUL)
    of alternative mission flight profiles against a baseline flight plan.
    """

    def __init__(self, degradation_weights: Optional[dict[str, float]] = None):
        self.weights = degradation_weights or {
            "injector": 0.05,
            "lubrication": 0.04,
            "thermal": 0.03,
            "mechanical": 0.02,
            "electrical": 0.01,
            "sensor": 0.02,
        }
        self.engine = ReducedOrderPistonEngine()
        self.rul_service = RULService()

    def run(
        self,
        telemetry: dict,
        scenario: MissionScenario,
        uav_profile: Optional[UAVPlatformProfile | str] = None,
    ) -> dict[str, Any]:
        """Runs a single scenario simulation with degradation kinetics and RUL."""
        profile = get_uav_profile(uav_profile) if isinstance(uav_profile, str) else uav_profile
        sim_telemetry = mission_adjust(
            telemetry,
            altitude_ft=scenario.altitude_ft,
            ambient_c=scenario.ambient_c,
            duration_h=scenario.duration_h,
            rapid_throttle=scenario.rapid_throttle,
            degradation_rates=self.weights,
        )
        deg_sev = float(sim_telemetry.get("Degradation_Severity", 0.0))
        health_index = max(0.0, min(100.0, 100.0 - deg_sev * 75.0))
        rul_res = self.rul_service.predict(sim_telemetry, scenario.to_dict())
        cht = float(sim_telemetry.get("CHT", 145.0))
        fuel_flow = float(sim_telemetry.get("Fuel_Flow", 30.0))
        if profile:
            mtom_factor = 0.65 + 0.35 * (profile.max_takeoff_mass_kg / 1000.0)
            fuel_flow *= mtom_factor
        total_fuel = round(fuel_flow * scenario.duration_h, 2)

        return {
            "scenario": scenario.name,
            "telemetry": sim_telemetry,
            "cht": round(cht, 1),
            "fuel_flow": round(fuel_flow, 2),
            "total_fuel_burn_l": total_fuel,
            "degradation_severity": round(deg_sev, 3),
            "health_index": round(health_index, 1),
            "rul": rul_res,
            "uav_platform": profile.to_dict() if profile else None,
        }

    def evaluate_scenario(
        self,
        base_telemetry: dict,
        scenario: MissionScenario,
        uav_profile: Optional[UAVPlatformProfile | str] = None,
    ) -> dict[str, Any]:
        """Evaluates engine response and projected health/RUL under a single mission scenario."""
        profile = get_uav_profile(uav_profile) if isinstance(uav_profile, str) else uav_profile
        sim_telemetry = mission_adjust(
            base_telemetry,
            altitude_ft=scenario.altitude_ft,
            ambient_c=scenario.ambient_c,
            duration_h=scenario.duration_h,
            rapid_throttle=scenario.rapid_throttle,
        )

        stress = self.rul_service.calculate_mission_stress(scenario.to_dict())

        eng_inputs = EngineInputs(
            rpm=float(sim_telemetry.get("Engine_RPM", 3000.0)),
            throttle=0.75 if scenario.rapid_throttle else 0.60,
            altitude_ft=scenario.altitude_ft,
            ambient_c=scenario.ambient_c,
        )
        phys_state = self.engine.estimate_state(eng_inputs)

        cht = float(sim_telemetry.get("CHT", 220.0))
        oil_temp = float(sim_telemetry.get("Oil_Temp", 90.0))
        egt_avg = (
            float(sim_telemetry.get("EGT1", 1200.0))
            + float(sim_telemetry.get("EGT2", 1200.0))
            + float(sim_telemetry.get("EGT3", 1200.0))
        ) / 3.0

        thermal_severity = max(0.0, (cht - 210.0) / 100.0) + max(0.0, (oil_temp - 95.0) / 30.0)
        base_health_loss_per_h = (0.045 + 0.03 * thermal_severity) * stress
        cumulative_mission_health_loss = base_health_loss_per_h * scenario.duration_h
        projected_health_end_of_mission = max(0.0, 100.0 - cumulative_mission_health_loss)

        rul_res = self.rul_service.estimate_rul(
            health_index=projected_health_end_of_mission,
            context=scenario.to_dict(),
        )

        fuel_rate_l_h = float(sim_telemetry.get("Fuel_Flow", 20.0))
        if profile:
            mtom_factor = 0.65 + 0.35 * (profile.max_takeoff_mass_kg / 1000.0)
            fuel_rate_l_h *= mtom_factor

        total_mission_fuel_l = fuel_rate_l_h * scenario.duration_h

        platform_warnings: list[str] = []
        if profile:
            if scenario.altitude_ft > profile.service_ceiling_ft:
                platform_warnings.append(
                    f"Scenario altitude ({int(scenario.altitude_ft)} ft) exceeds {profile.uav_name} service ceiling ({int(profile.service_ceiling_ft)} ft)."
                )
            if scenario.duration_h > profile.endurance_hours:
                platform_warnings.append(
                    f"Scenario duration ({scenario.duration_h:.1f}h) exceeds {profile.uav_name} maximum endurance ({profile.endurance_hours:.1f}h)."
                )
            if profile.fuel_capacity_l and total_mission_fuel_l > profile.fuel_capacity_l:
                platform_warnings.append(
                    f"Scenario fuel burn ({total_mission_fuel_l:.1f} L) exceeds {profile.uav_name} fuel capacity ({profile.fuel_capacity_l:.1f} L)."
                )

        return {
            "scenario": scenario.to_dict(),
            "stress_multiplier": stress,
            "thermal_severity_index": round(thermal_severity, 3),
            "projected_final_health_index": round(projected_health_end_of_mission, 2),
            "mission_health_loss": round(cumulative_mission_health_loss, 2),
            "degradation_rate_per_hour": round(base_health_loss_per_h, 3),
            "rul_hours": rul_res["rul_hours"],
            "rul_lower_hours": rul_res["rul_lower_hours"],
            "rul_upper_hours": rul_res["rul_upper_hours"],
            "rul_confidence": rul_res["confidence"],
            "fuel_flow_l_h": round(fuel_rate_l_h, 2),
            "total_fuel_burn_l": round(total_mission_fuel_l, 1),
            "thermal_telemetry": {
                "CHT": round(cht, 1),
                "Oil_Temp": round(oil_temp, 1),
                "EGT_Avg": round(egt_avg, 1),
            },
            "indicated_power_kw": phys_state.indicated_power_kw,
            "brake_power_kw": phys_state.brake_power_kw,
            "volumetric_efficiency": phys_state.volumetric_efficiency,
            "uav_platform": profile.to_dict() if profile else None,
            "platform_warnings": platform_warnings,
        }

    def compare(
        self,
        telemetry: dict,
        baseline: MissionScenario,
        alternative: MissionScenario,
        uav_profile: Optional[UAVPlatformProfile | str] = None,
    ) -> dict[str, Any]:
        """Compares baseline vs alternative mission scenarios with optional UAV platform constraints."""
        profile = get_uav_profile(uav_profile) if isinstance(uav_profile, str) else uav_profile
        base_res = self.run(telemetry, baseline, profile)
        alt_res = self.run(telemetry, alternative, profile)

        rul_delta = round(alt_res["rul"]["rul_hours"] - base_res["rul"]["rul_hours"], 2)
        health_delta = round(alt_res["health_index"] - base_res["health_index"], 2)

        impact = {
            "rul_hours": rul_delta,
            "health_index": health_delta,
            "degradation_severity": round(alt_res["degradation_severity"] - base_res["degradation_severity"], 3),
            "fuel_flow": round(alt_res["fuel_flow"] - base_res["fuel_flow"], 2),
            "cht": round(alt_res["cht"] - base_res["cht"], 1),
        }

        base_eval = self.evaluate_scenario(telemetry, baseline, profile)
        alt_eval = self.evaluate_scenario(telemetry, alternative, profile)
        fuel_diff = round(alt_eval["total_fuel_burn_l"] - base_eval["total_fuel_burn_l"], 1)
        fuel_pct = round(100.0 * fuel_diff / max(1.0, base_eval["total_fuel_burn_l"]), 2)

        return {
            "baseline": base_res,
            "alternative": alt_res,
            "impact": impact,
            "uav_platform": profile.to_dict() if profile else None,
            "platform_warnings": alt_eval.get("platform_warnings", []),
            "comparison": {
                "rul_delta_hours": rul_delta,
                "fuel_delta_liters": fuel_diff,
                "fuel_delta_percent": fuel_pct,
                "stress_multiplier_delta": round(alt_eval["stress_multiplier"] - base_eval["stress_multiplier"], 3),
                "risk_assessment": "ELEVATED_THERMOMECHANICAL_PENALTY" if rul_delta < 0 else "NOMINAL_OPTIMAL",
                "recommendation": f"Alternative profile results in RUL delta of {rul_delta}h (Fuel delta: {fuel_diff} L).",
            },
        }
