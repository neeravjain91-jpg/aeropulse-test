"""Configurable Engine Parameter Schema for AeroPulse-X Propulsion Digital Twin.
Enhanced with Austin (2010) UAV Engineering Reference Principles.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .engine_parameters import EngineParameterRegistry, get_default_parameter_registry


@dataclass
class EngineConfig:
    name: str = "AeroPiston-4C-1.35L"
    engine_type: str = "spark_ignition"
    cycle: str = "4-stroke"  # "4-stroke", "2-stroke", "rotary", "diesel"
    layout: str = "opposed"  # "opposed", "inline", "rotary", "vee"
    displacement_l: float = 1.352
    bore_mm: float = 84.0
    stroke_mm: float = 61.0
    num_cylinders: int = 4
    compression_ratio: float = 9.0
    gamma: float = 1.33
    fuel_type: str = "avgas_100ll"  # "avgas_100ll", "mogas", "jet_a1", "diesel"
    fuel_lhv_mj_kg: float = 43.5
    afr_stoich: float = 14.7
    base_power_kw: float = 84.5
    nominal_rpm: float = 3000.0
    max_rpm: float = 5800.0
    idle_rpm: float = 1400.0
    base_friction_kw: float = 6.5
    friction_rpm_exp: float = 1.8
    thermal_capacity_j_k: float = 35000.0
    cooling_area_m2: float = 0.85
    cooling_coeff_w_m2k: float = 120.0
    cooling_architecture: str = "air_liquid_hybrid"  # "air_cooled", "liquid_cooled", "air_liquid_hybrid"
    oil_volume_l: float = 3.5
    oil_viscosity_cst: float = 14.0
    lubrication_architecture: str = "dry_sump"  # "dry_sump", "wet_sump", "pre_mix"
    turbo_critical_alt_ft: float = 15000.0
    volumetric_efficiency_base: float = 0.88
    fuel_density_kg_l: float = 0.72
    mass_power_ratio_kg_kw: float = 0.82  # Austin Ch 6 Fig 6.6
    bsfc_nominal_kg_kwh: float = 0.34     # Austin Ch 6.5.1 (0.3 - 0.4 kg/kWh for 4-stroke)
    torque_ripple_factor: float = 1.0     # Baseline 4-cylinder 4-stroke
    tbo_hours: float = 2000.0

    @property
    def displacement_liters(self) -> float:
        return self.displacement_l

    @property
    def fuel_lower_heating_value_mj_kg(self) -> float:
        return self.fuel_lhv_mj_kg

    @property
    def air_fuel_ratio_stoich(self) -> float:
        return self.afr_stoich

    @property
    def rated_power_kw(self) -> float:
        return self.base_power_kw

    def get_registry(self) -> EngineParameterRegistry:
        """Returns the formal parameter provenance registry for this engine configuration."""
        return get_default_parameter_registry()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "engine_type": self.engine_type,
            "cycle": self.cycle,
            "layout": self.layout,
            "displacement_l": self.displacement_l,
            "bore_mm": self.bore_mm,
            "stroke_mm": self.stroke_mm,
            "num_cylinders": self.num_cylinders,
            "compression_ratio": self.compression_ratio,
            "fuel_type": self.fuel_type,
            "fuel_lhv_mj_kg": self.fuel_lhv_mj_kg,
            "base_power_kw": self.base_power_kw,
            "nominal_rpm": self.nominal_rpm,
            "max_rpm": self.max_rpm,
            "idle_rpm": self.idle_rpm,
            "gamma": self.gamma,
            "afr_stoich": self.afr_stoich,
            "base_friction_kw": self.base_friction_kw,
            "cooling_area_m2": self.cooling_area_m2,
            "cooling_coeff_w_m2k": self.cooling_coeff_w_m2k,
            "cooling_architecture": self.cooling_architecture,
            "oil_volume_l": self.oil_volume_l,
            "oil_viscosity_cst": self.oil_viscosity_cst,
            "lubrication_architecture": self.lubrication_architecture,
            "turbo_critical_alt_ft": self.turbo_critical_alt_ft,
            "volumetric_efficiency_base": self.volumetric_efficiency_base,
            "fuel_density_kg_l": self.fuel_density_kg_l,
            "mass_power_ratio_kg_kw": self.mass_power_ratio_kg_kw,
            "bsfc_nominal_kg_kwh": self.bsfc_nominal_kg_kwh,
            "torque_ripple_factor": self.torque_ripple_factor,
            "tbo_hours": self.tbo_hours,
        }

    @classmethod
    def default_135l(cls) -> EngineConfig:
        return cls()

    @classmethod
    def rotax_914(cls) -> EngineConfig:
        return cls(
            name="Rotax-914-Turbo-115HP",
            engine_type="turbocharged_spark_ignition",
            cycle="4-stroke",
            layout="opposed",
            displacement_l=1.211,
            bore_mm=79.5,
            stroke_mm=61.0,
            num_cylinders=4,
            compression_ratio=9.0,
            base_power_kw=84.5,
            nominal_rpm=5500.0,
            max_rpm=5800.0,
            idle_rpm=1400.0,
            cooling_architecture="air_liquid_hybrid",
            lubrication_architecture="dry_sump",
            turbo_critical_alt_ft=16000.0,
            mass_power_ratio_kg_kw=0.88,
            bsfc_nominal_kg_kwh=0.33,
            tbo_hours=1200.0,
        )

    @classmethod
    def inline4_diesel(cls) -> EngineConfig:
        return cls(
            name="Generic-Inline4-AeroDiesel",
            engine_type="compression_ignition",
            cycle="diesel",
            layout="inline",
            displacement_l=1.991,
            bore_mm=83.0,
            stroke_mm=92.0,
            num_cylinders=4,
            compression_ratio=18.0,
            fuel_type="jet_a1",
            fuel_lhv_mj_kg=42.8,
            afr_stoich=14.5,
            base_power_kw=114.0,
            nominal_rpm=2800.0,
            max_rpm=3880.0,
            idle_rpm=900.0,
            cooling_architecture="liquid_cooled",
            lubrication_architecture="wet_sump",
            turbo_critical_alt_ft=18000.0,
            mass_power_ratio_kg_kw=1.18,
            bsfc_nominal_kg_kwh=0.25,
            tbo_hours=1500.0,
        )

    @classmethod
    def two_stroke_twin(cls) -> EngineConfig:
        return cls(
            name="Generic-2Stroke-Twin-50HP",
            engine_type="spark_ignition",
            cycle="2-stroke",
            layout="opposed",
            displacement_l=0.550,
            bore_mm=66.0,
            stroke_mm=40.0,
            num_cylinders=2,
            compression_ratio=8.2,
            base_power_kw=37.0,
            nominal_rpm=6500.0,
            max_rpm=7200.0,
            idle_rpm=1800.0,
            cooling_architecture="air_cooled",
            lubrication_architecture="pre_mix",
            turbo_critical_alt_ft=0.0,
            mass_power_ratio_kg_kw=0.52,
            bsfc_nominal_kg_kwh=0.48,
            torque_ripple_factor=0.65,
            tbo_hours=500.0,
        )

    @classmethod
    def rotary_wankel(cls) -> EngineConfig:
        return cls(
            name="Generic-Rotary-Wankel-40HP",
            engine_type="rotary_wankel",
            cycle="rotary",
            layout="rotary",
            displacement_l=0.294,
            bore_mm=0.0,
            stroke_mm=0.0,
            num_cylinders=1,
            compression_ratio=9.2,
            base_power_kw=29.8,
            nominal_rpm=6000.0,
            max_rpm=7500.0,
            idle_rpm=2000.0,
            cooling_architecture="liquid_cooled",
            lubrication_architecture="total_loss_metered",
            turbo_critical_alt_ft=0.0,
            mass_power_ratio_kg_kw=0.48,
            bsfc_nominal_kg_kwh=0.37,
            torque_ripple_factor=0.35,
            tbo_hours=1000.0,
        )

    @classmethod
    def custom(cls, **kwargs) -> EngineConfig:
        return cls(**kwargs)


ENGINE_PROFILES: Dict[str, Dict[str, Any]] = {
    "AeroPiston-4C-1.35L": {
        "id": "AeroPiston-4C-1.35L",
        "name": "AeroPiston 4C 1.35L (Opposed-4)",
        "cycle": "4-stroke",
        "layout": "opposed",
        "cylinders": 4,
        "displacement_l": 1.352,
        "bore_mm": 84.0,
        "stroke_mm": 61.0,
        "nominal_rpm": 3000.0,
        "max_rpm": 5800.0,
        "idle_rpm": 1400.0,
        "compression_ratio": 9.0,
        "valvetrain": "ohv_pushrod",
        "firing_order": [1, 3, 2, 4],
        "tbo_hours": 2000.0,
        "bsfc_nominal": 0.34,
        "provenance": "Published generic aero-piston digital twin specifications / Austin Ch 6",
    },
    "Rotax-914-Turbo-115HP": {
        "id": "Rotax-914-Turbo-115HP",
        "name": "Rotax 914 Turbo 115HP (Opposed-4)",
        "cycle": "4-stroke",
        "layout": "opposed",
        "cylinders": 4,
        "displacement_l": 1.211,
        "bore_mm": 79.5,
        "stroke_mm": 61.0,
        "nominal_rpm": 5500.0,
        "max_rpm": 5800.0,
        "idle_rpm": 1400.0,
        "compression_ratio": 9.0,
        "valvetrain": "ohv_pushrod",
        "firing_order": [1, 3, 2, 4],
        "tbo_hours": 1200.0,
        "bsfc_nominal": 0.33,
        "provenance": "Rotax 914 F/UL Operator Manual / EASA TCDS E.121 / Austin Ch 6.5.1",
    },
    "Generic-Inline4-AeroDiesel": {
        "id": "Generic-Inline4-AeroDiesel",
        "name": "Generic Inline-4 2.0L (Aero-Diesel)",
        "cycle": "diesel",
        "layout": "inline",
        "cylinders": 4,
        "displacement_l": 1.991,
        "bore_mm": 83.0,
        "stroke_mm": 92.0,
        "nominal_rpm": 2800.0,
        "max_rpm": 3880.0,
        "idle_rpm": 900.0,
        "compression_ratio": 18.0,
        "valvetrain": "dohc",
        "firing_order": [1, 3, 4, 2],
        "tbo_hours": 1500.0,
        "bsfc_nominal": 0.25,
        "provenance": "Literature-informed generic inline-4 aero-diesel reduced-order proxy / Austin Ch 6.5.1",
    },
    "Generic-2Stroke-Twin-50HP": {
        "id": "Generic-2Stroke-Twin-50HP",
        "name": "Generic 2-Stroke Opposed-Twin 50HP",
        "cycle": "2-stroke",
        "layout": "opposed",
        "cylinders": 2,
        "displacement_l": 0.550,
        "nominal_rpm": 6500.0,
        "max_rpm": 7200.0,
        "idle_rpm": 1800.0,
        "compression_ratio": 8.2,
        "tbo_hours": 500.0,
        "bsfc_nominal": 0.48,
        "provenance": "Austin Ch 6.5.1 (Two-stroke UAV power-plants)",
    },
    "Generic-Rotary-Wankel-40HP": {
        "id": "Generic-Rotary-Wankel-40HP",
        "name": "Generic Wankel Rotary 40HP",
        "cycle": "rotary",
        "layout": "rotary",
        "cylinders": 1,
        "displacement_l": 0.294,
        "nominal_rpm": 6000.0,
        "max_rpm": 7500.0,
        "idle_rpm": 2000.0,
        "compression_ratio": 9.2,
        "tbo_hours": 1000.0,
        "bsfc_nominal": 0.37,
        "provenance": "Austin Ch 6.5.1 / Ch 27 (Wankel rotary UAV power-plants)",
    },
    "ROTAX_914_F_TWIN_01": {
        "id": "ROTAX_914_F_TWIN_01",
        "canonical_id": "Rotax-914-Turbo-115HP",
        "name": "Rotax 914 Turbo 115HP (Opposed-4 Twin Serial 01)",
        "cycle": "4-stroke",
        "layout": "opposed",
        "cylinders": 4,
        "displacement_l": 1.211,
        "bore_mm": 79.5,
        "stroke_mm": 61.0,
        "nominal_rpm": 5500.0,
        "max_rpm": 5800.0,
        "idle_rpm": 1400.0,
        "compression_ratio": 9.0,
        "valvetrain": "ohv_pushrod",
        "firing_order": [1, 3, 2, 4],
        "tbo_hours": 1200.0,
        "bsfc_nominal": 0.33,
        "provenance": "Rotax 914 F/UL Operator Manual / EASA TCDS E.121 / Austin Ch 6.5.1",
    },
}


def resolve_canonical_engine_id(engine_id: Optional[str]) -> str:
    """Resolves arbitrary engine identifiers, serial strings, or aliases to canonical profile ID."""
    if not engine_id:
        return "Rotax-914-Turbo-115HP"
    eid = str(engine_id).strip()
    if eid in ENGINE_PROFILES and "canonical_id" not in ENGINE_PROFILES[eid]:
        return eid
    if eid in ENGINE_PROFILES and "canonical_id" in ENGINE_PROFILES[eid]:
        return ENGINE_PROFILES[eid]["canonical_id"]
    eid_lower = eid.lower()
    if "rotax" in eid_lower or "914" in eid_lower:
        return "Rotax-914-Turbo-115HP"
    if "diesel" in eid_lower or "inline4" in eid_lower:
        return "Generic-Inline4-AeroDiesel"
    if "2stroke" in eid_lower or "twin-50hp" in eid_lower:
        return "Generic-2Stroke-Twin-50HP"
    if "rotary" in eid_lower or "wankel" in eid_lower:
        return "Generic-Rotary-Wankel-40HP"
    if "aeropiston" in eid_lower or "1.35" in eid_lower:
        return "AeroPiston-4C-1.35L"
    return "Rotax-914-Turbo-115HP"


def default_engine_config() -> EngineConfig:
    return EngineConfig.default_135l()
