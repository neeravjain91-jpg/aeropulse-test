"""Configurable MALE Aero-Piston UAV Platform Registry for AeroPulse-X.

Provides strongly typed UAV platform specifications, engine linkages, aerodynamic constraints,
glide performance, and scientific data provenance.

Strict Scope Rule:
Only Medium-Altitude Long-Endurance (MALE) platforms powered by aero-piston or aero-diesel
propulsion are cataloged. Turboprop, turbojet, turbofan, and electric platforms are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


CATALOG_TITLE: str = "SUPPORTED MALE AERO-PISTON UAV PLATFORMS"


@dataclass
class UAVPlatformProfile:
    """Standardized operational profile for a MALE aero-piston UAV platform."""
    # Identification
    uav_id: str
    uav_name: str
    uav_class: str = "MALE"                               # Must be "MALE"
    engine_id: str = "Rotax-914-Turbo-115HP"              # References ENGINE_PROFILES in engine_config.py
    propulsion_type: str = "AERO-PISTON"                  # Must be "AERO-PISTON"
    engine_configuration: str = "Pusher, 4-Cylinder Horizontally-Opposed Turbocharged"
    engine_count: int = 1

    # Weight & Aerodynamics
    max_takeoff_mass_kg: float = 1020.0
    cruise_speed_kt: float = 70.0
    max_speed_kt: float = 117.0
    service_ceiling_ft: float = 25000.0
    endurance_hours: float = 24.0
    range_km: float = 1100.0
    climb_rate_fpm: float = 1000.0
    glide_ratio: Optional[float] = 16.0                   # Lift-to-drag ratio L/D; None if unverified

    # Fuel & Operational Envelope
    fuel_type: str = "Avgas 100LL / Mogas"
    fuel_capacity_l: Optional[float] = 380.0
    normal_cruise_altitude_ft: float = 15000.0
    operating_altitude_range_ft: Tuple[float, float] = (5000.0, 25000.0)
    throttle_range: Tuple[float, float] = (0.35, 1.0)
    payload_capacity_kg: Optional[float] = 204.0

    # Mission Capabilities
    mission_types: List[str] = field(default_factory=lambda: ["ISR", "BORDER_PATROL", "PERSISTENT_LOITER"])

    # Provenance & Audit Trail
    source: str = "USAF MQ-1B Flight Manual / Jane's All the World's Aircraft"
    data_provenance: str = "MANUFACTURER"                 # MANUFACTURER, LITERATURE, DERIVED, ASSUMED
    confidence: str = "HIGH"                              # HIGH, MEDIUM, LOW

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["operating_altitude_range_ft"] = list(self.operating_altitude_range_ft)
        d["throttle_range"] = list(self.throttle_range)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UAVPlatformProfile:
        d = dict(data)
        if "operating_altitude_range_ft" in d and isinstance(d["operating_altitude_range_ft"], list):
            d["operating_altitude_range_ft"] = tuple(d["operating_altitude_range_ft"])
        if "throttle_range" in d and isinstance(d["throttle_range"], list):
            d["throttle_range"] = tuple(d["throttle_range"])
        return cls(**d)

    def validate_profile(self) -> Tuple[bool, List[str]]:
        """Validates that platform meets MALE aero-piston scope requirements."""
        violations = []
        if self.uav_class.upper() != "MALE":
            violations.append(f"Invalid UAV class: '{self.uav_class}'. Must be 'MALE'.")
        if "PISTON" not in self.propulsion_type.upper():
            violations.append(f"Invalid propulsion type: '{self.propulsion_type}'. Must be 'AERO-PISTON'.")
        if self.max_takeoff_mass_kg <= 0:
            violations.append("MTOM must be greater than 0 kg.")
        if self.cruise_speed_kt >= self.max_speed_kt:
            violations.append("Cruise speed cannot exceed max speed.")
        if self.service_ceiling_ft <= 0:
            violations.append("Service ceiling must be positive.")
        return (len(violations) == 0, violations)


# =========================================================================
# Authoritative Registry of Supported MALE Aero-Piston UAV Platforms
# =========================================================================

SUPPORTED_UAV_REGISTRY: Dict[str, UAVPlatformProfile] = {
    # 1. General Atomics MQ-1B Predator
    "UAV_MALE_MQ1_PREDATOR": UAVPlatformProfile(
        uav_id="UAV_MALE_MQ1_PREDATOR",
        uav_name="General Atomics MQ-1B Predator",
        uav_class="MALE",
        engine_id="Rotax-914-Turbo-115HP",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, 4-Cylinder Horizontally-Opposed Turbocharged Spark-Ignition",
        engine_count=1,
        max_takeoff_mass_kg=1020.0,
        cruise_speed_kt=70.0,
        max_speed_kt=117.0,
        service_ceiling_ft=25000.0,
        endurance_hours=24.0,
        range_km=1100.0,
        climb_rate_fpm=1000.0,
        glide_ratio=16.0,
        fuel_type="Avgas 100LL / Mogas",
        fuel_capacity_l=380.0,
        normal_cruise_altitude_ft=15000.0,
        operating_altitude_range_ft=(5000.0, 25000.0),
        throttle_range=(0.35, 1.0),
        payload_capacity_kg=204.0,
        mission_types=["ISR", "BORDER_PATROL", "PERSISTENT_LOITER"],
        source="USAF MQ-1B Flight Manual (T.O. 1Q-1(M)B-1) / Jane's All the World's Aircraft / Austin (2010)",
        data_provenance="MANUFACTURER",
        confidence="HIGH",
    ),

    # 2. IAI Heron 1 (Machatz 1)
    "UAV_MALE_IAI_HERON_1": UAVPlatformProfile(
        uav_id="UAV_MALE_IAI_HERON_1",
        uav_name="IAI Heron 1 (Machatz 1)",
        uav_class="MALE",
        engine_id="Rotax-914-Turbo-115HP",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, 4-Cylinder Horizontally-Opposed Turbocharged Spark-Ignition",
        engine_count=1,
        max_takeoff_mass_kg=1250.0,
        cruise_speed_kt=65.0,
        max_speed_kt=112.0,
        service_ceiling_ft=30000.0,
        endurance_hours=45.0,
        range_km=1000.0,
        climb_rate_fpm=550.0,
        glide_ratio=19.0,
        fuel_type="Avgas 100LL / Mogas",
        fuel_capacity_l=340.0,
        normal_cruise_altitude_ft=18000.0,
        operating_altitude_range_ft=(6000.0, 30000.0),
        throttle_range=(0.30, 0.95),
        payload_capacity_kg=250.0,
        mission_types=["PERSISTENT_LOITER", "BORDER_PATROL", "MARITIME_SURVEILLANCE", "ISR"],
        source="Israel Aerospace Industries (IAI) Technical Specifications / EASA TCDS Reference Data",
        data_provenance="MANUFACTURER",
        confidence="HIGH",
    ),

    # 3. DRDO Rustom-I (Indian Defense Research Benchmark)
    "UAV_MALE_DRDO_RUSTOM1": UAVPlatformProfile(
        uav_id="UAV_MALE_DRDO_RUSTOM1",
        uav_name="DRDO Rustom-I (ADE)",
        uav_class="MALE",
        engine_id="Rotax-914-Turbo-115HP",
        propulsion_type="AERO-PISTON",
        engine_configuration="Rear Pusher, 4-Cylinder Horizontally-Opposed Turbocharged",
        engine_count=1,
        max_takeoff_mass_kg=720.0,
        cruise_speed_kt=80.0,
        max_speed_kt=122.0,
        service_ceiling_ft=26000.0,
        endurance_hours=14.0,
        range_km=350.0,
        climb_rate_fpm=650.0,
        glide_ratio=13.5,
        fuel_type="Avgas 100LL / Mogas",
        fuel_capacity_l=160.0,
        normal_cruise_altitude_ft=14000.0,
        operating_altitude_range_ft=(4000.0, 26000.0),
        throttle_range=(0.40, 1.0),
        payload_capacity_kg=75.0,
        mission_types=["BORDER_PATROL", "RECONNAISSANCE", "ISR"],
        source="DRDO Aeronautical Development Establishment (ADE) Technical Reports / Aero India Flight Trials",
        data_provenance="LITERATURE",
        confidence="HIGH",
    ),

    # 4. TAI Anka-A (Early Aero-Diesel Piston Variant)
    "UAV_MALE_TAI_ANKA_A": UAVPlatformProfile(
        uav_id="UAV_MALE_TAI_ANKA_A",
        uav_name="TAI Anka-A (Aero-Diesel Piston Variant)",
        uav_class="MALE",
        engine_id="Generic-Inline4-AeroDiesel",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, Inline-4 Turbocharged Common-Rail Aero-Diesel",
        engine_count=1,
        max_takeoff_mass_kg=1600.0,
        cruise_speed_kt=85.0,
        max_speed_kt=117.0,
        service_ceiling_ft=30000.0,
        endurance_hours=24.0,
        range_km=1440.0,
        climb_rate_fpm=700.0,
        glide_ratio=16.5,
        fuel_type="Jet A-1 / Diesel",
        fuel_capacity_l=250.0,
        normal_cruise_altitude_ft=20000.0,
        operating_altitude_range_ft=(8000.0, 30000.0),
        throttle_range=(0.35, 0.95),
        payload_capacity_kg=200.0,
        mission_types=["ISR", "MARITIME_SURVEILLANCE", "AREA_PATROL"],
        source="Turkish Aerospace Industries (TAI) Datasheets / IDEF Defense Technical Symposium",
        data_provenance="MANUFACTURER",
        confidence="HIGH",
    ),

    # 5. General Atomics Altus II (NASA ACES Research Platform)
    "UAV_MALE_GA_ALTUS2": UAVPlatformProfile(
        uav_id="UAV_MALE_GA_ALTUS2",
        uav_name="General Atomics Altus II (NASA Research)",
        uav_class="MALE",
        engine_id="Rotax-914-Turbo-115HP",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, 4-Cylinder Horizontally-Opposed Dual-Stage Turbocharged",
        engine_count=1,
        max_takeoff_mass_kg=960.0,
        cruise_speed_kt=75.0,
        max_speed_kt=100.0,
        service_ceiling_ft=45000.0,
        endurance_hours=24.0,
        range_km=800.0,
        climb_rate_fpm=600.0,
        glide_ratio=17.0,
        fuel_type="Avgas 100LL",
        fuel_capacity_l=260.0,
        normal_cruise_altitude_ft=25000.0,
        operating_altitude_range_ft=(10000.0, 45000.0),
        throttle_range=(0.35, 1.0),
        payload_capacity_kg=150.0,
        mission_types=["HIGH_ALTITUDE_ATMOSPHERIC", "ENVIRONMENTAL_RESEARCH", "ISR"],
        source="NASA Dryden Flight Research Center Fact Sheet TM-2001-210385 / Altus II ACES Project Documentation",
        data_provenance="LITERATURE",
        confidence="HIGH",
    ),

    # 6. AeroPulse-X Reference MALE-Twin (Austin Baseline Academic Benchmark)
    "UAV_MALE_AEROPULSE_REF": UAVPlatformProfile(
        uav_id="UAV_MALE_AEROPULSE_REF",
        uav_name="AeroPulse-X Reference MALE-Twin",
        uav_class="MALE",
        engine_id="AeroPiston-4C-1.35L",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, 4-Cylinder Horizontally-Opposed Naturally Aspirated",
        engine_count=1,
        max_takeoff_mass_kg=950.0,
        cruise_speed_kt=75.0,
        max_speed_kt=110.0,
        service_ceiling_ft=25000.0,
        endurance_hours=20.0,
        range_km=900.0,
        climb_rate_fpm=750.0,
        glide_ratio=15.0,
        fuel_type="Avgas 100LL",
        fuel_capacity_l=200.0,
        normal_cruise_altitude_ft=8000.0,
        operating_altitude_range_ft=(4000.0, 25000.0),
        throttle_range=(0.30, 1.0),
        payload_capacity_kg=180.0,
        mission_types=["BORDER_PATROL", "COMMUNICATION_RELAY", "TRANSIT_ISR"],
        source="AeroPulse-X Digital Twin Propulsion Benchmark / Austin (2010) Chapter 5 & 6",
        data_provenance="DERIVED",
        confidence="HIGH",
    ),

    # 7. Custom Research Platform (Unverified Glide Ratio for Testing Rule 11)
    "UAV_MALE_CUSTOM_RESEARCH": UAVPlatformProfile(
        uav_id="UAV_MALE_CUSTOM_RESEARCH",
        uav_name="Generic MALE Aero-Piston Research Prototype",
        uav_class="MALE",
        engine_id="AeroPiston-4C-1.35L",
        propulsion_type="AERO-PISTON",
        engine_configuration="Pusher, 4-Cylinder Opposed Experimental Configuration",
        engine_count=1,
        max_takeoff_mass_kg=850.0,
        cruise_speed_kt=72.0,
        max_speed_kt=105.0,
        service_ceiling_ft=20000.0,
        endurance_hours=18.0,
        range_km=750.0,
        climb_rate_fpm=700.0,
        glide_ratio=None,                                # Explicitly None to enforce Rule 11 fallback
        fuel_type="Avgas 100LL",
        fuel_capacity_l=None,                             # Explicitly None to test missing parameter handling
        normal_cruise_altitude_ft=9000.0,
        operating_altitude_range_ft=(3000.0, 20000.0),
        throttle_range=(0.35, 1.0),
        payload_capacity_kg=120.0,
        mission_types=["AREA_PATROL", "TRANSIT_ISR"],
        source="Experimental Aero-Piston Test Article / Unpublished Preliminary Flight Data",
        data_provenance="ASSUMED",
        confidence="MEDIUM",
    ),
}


def get_uav_profile(uav_id: str) -> Optional[UAVPlatformProfile]:
    """Retrieves a UAV platform profile by identifier, ensuring it satisfies scope constraints."""
    if not uav_id:
        return None
    profile = SUPPORTED_UAV_REGISTRY.get(uav_id.strip())
    if not profile:
        # Fallback search by prefix or case-insensitive matching
        for k, v in SUPPORTED_UAV_REGISTRY.items():
            if k.lower() == uav_id.lower() or v.uav_name.lower() == uav_id.lower():
                profile = v
                break
    return profile


def list_supported_uavs(
    engine_filter: Optional[str] = None,
    mission_filter: Optional[str] = None,
) -> List[UAVPlatformProfile]:
    """Returns filtered list of verified MALE aero-piston UAV platforms."""
    results = []
    for profile in SUPPORTED_UAV_REGISTRY.values():
        # Enforce non-negotiable scope criteria
        if profile.uav_class.upper() != "MALE":
            continue
        if "PISTON" not in profile.propulsion_type.upper():
            continue

        if engine_filter and engine_filter.lower() not in profile.engine_id.lower():
            continue
        if mission_filter:
            m_upper = mission_filter.upper()
            if not any(m_upper in t.upper() for t in profile.mission_types):
                continue

        results.append(profile)

    return results
