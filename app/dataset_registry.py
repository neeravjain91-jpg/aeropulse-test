"""Dataset Registry for AeroPulse-X Virtual Data Lab & Scientific Provenance.

Provides auditable metadata, provenance, domain classifications, and scientific boundaries
for all primary synthetic and cross-domain proxy datasets.
Synchronized with docs/DATA_RESOURCE_SPEC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetMetadata:
    dataset_id: str
    name: str
    purpose: str
    source: str
    domain: str
    real_or_synthetic: str                        # "SYNTHETIC", "REAL_OPERATIONAL", "CROSS_DOMAIN_PROXY", "RESEARCH_REFERENCE"
    ground_truth_available: bool
    ground_truth_description: str
    target_engine_relevance: str                  # "PRIMARY_TARGET_ROTAX_914", "OPERATIONAL_ENVELOPE_CONTEXT", "CROSS_DOMAIN_TURBOFAN", "CROSS_DOMAIN_BEARING", "CROSS_DOMAIN_UAV_AVIONICS", "INCOMPATIBLE_MARINE", "INCOMPATIBLE_PROPELLER"
    license_status: str
    generation_method: str
    version: str
    number_of_trajectories: int
    number_of_samples: int
    features: List[str]
    limitations: List[str]
    decision: str = "ACTIVE"                      # "ACTIVE", "SELECTIVE_PROXY", "METHODOLOGY_ONLY", "NAVIGATION_ONLY", "REJECTED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetRegistry:
    """Master catalog and registry for all datasets utilized or audited within AeroPulse-X."""

    def __init__(self):
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._rejected_datasets: Dict[str, DatasetMetadata] = {}
        self._aliases: Dict[str, str] = {}
        self._initialize_core_registry()

    def _initialize_core_registry(self) -> None:
        # 1. AeroPulse Synthetic Master Engine Corpus
        synth_meta = DatasetMetadata(
            dataset_id="AERO_PULSE_SYNTHETIC",
            name="AeroPulse-X Synthetic Aero-Piston Degradation & SIL Corpus",
            purpose="Primary physics-informed aero-piston/RUL demonstrator.",
            source="AeroPulse-X Physics-Grounded Simulation Engine (Rotax 914 F ODE Wear Models)",
            domain="AERO_PISTON_4_STROKE_TURBO",
            real_or_synthetic="SYNTHETIC",
            ground_truth_available=True,
            ground_truth_description="Exact mathematical RUL (y_true = max(0, t_failure - t)) derived from continuous degradation ODEs to H_failure = 35.0.",
            target_engine_relevance="PRIMARY_TARGET_ROTAX_914",
            license_status="Open Source / SIH 2026 Academic Demonstrator",
            generation_method="Reduced-Order Otto Cycle Thermodynamics, Arrhenius Wear Kinetics & Virtual Hardware SIL",
            version="2.0.0",
            number_of_trajectories=1500,
            number_of_samples=180000,
            features=[
                "timestamp", "trajectory_id", "RPM", "throttle", "MAP", "ambient_temperature",
                "altitude", "CHT", "coolant_temperature", "EGT", "oil_pressure", "oil_temperature",
                "fuel_flow", "vibration", "bus_voltage", "health_index", "true_RUL", "sensor_trust",
                "FADEC_state", "CAN_CRC_status"
            ],
            limitations=[
                "Purely synthetic simulation; does not replace physical test-cell dynamometer wear measurements.",
                "Quasi-steady-state thermodynamic model with lumped thermal capacitance approximations."
            ],
            decision="ACTIVE",
        )
        self.register(synth_meta)

        # 2. NASA ACES (Altus II Operational Flight Telemetry)
        aces_meta = DatasetMetadata(
            dataset_id="NASA_ACES",
            name="NASA ACES — Altus II Operational Flight Telemetry",
            purpose="NASA ACES — Altus II operational/mechanical flight telemetry used for operational-envelope/contextual real-flight baseline.",
            source="NASA Dryden Flight Research Center (Altus II UAV Flight Logs)",
            domain="GENERAL_AVIATION_PISTON_TELEMETRY",
            real_or_synthetic="REAL_OPERATIONAL",
            ground_truth_available=False,
            ground_truth_description="Recorded during healthy operational flight missions without run-to-failure degradation. Contains NO run-to-failure RUL ground truth.",
            target_engine_relevance="OPERATIONAL_ENVELOPE_CONTEXT",
            license_status="NASA Open Data / Public Domain",
            generation_method="Airborne Transducer Array & In-Flight Flight Recorder Data Logging",
            version="1.0.0",
            number_of_trajectories=85,
            number_of_samples=42500,
            features=[
                "Engine_RPM", "Manifold_Pressure", "Fuel_Flow", "CHT", "EGT", "Oil_Temp", "Oil_Pressure",
                "Altitude", "Ambient_Temp", "Airspeed"
            ],
            limitations=[
                "Contains NO run-to-failure RUL ground truth.",
                "Not target-engine Rotax 914 data; utilized strictly for operational-envelope and contextual real-flight validation."
            ],
            decision="ACTIVE",
        )
        self.register(aces_meta)
        self.register_alias("REAL_ACES", "NASA_ACES")

        # 3. NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
        cmapss_meta = DatasetMetadata(
            dataset_id="NASA_CMAPSS",
            name="NASA C-MAPSS Turbofan Degradation Benchmark (FD001-FD004 / N-CMAPSS)",
            purpose="Turbofan cross-domain RUL/prognostics proxy.",
            source="NASA Prognostics Center of Excellence (PCoE)",
            domain="COMMERCIAL_AERO_TURBOFAN",
            real_or_synthetic="CROSS_DOMAIN_PROXY",
            ground_truth_available=True,
            ground_truth_description="Simulated cycle-based run-to-failure ground truth for high-pressure compressor and fan degradation.",
            target_engine_relevance="CROSS_DOMAIN_TURBOFAN",
            license_status="NASA Open Data / Public Domain",
            generation_method="Physics-Based Thermodynamic Simulation of 90,000 lb Thrust Turbofan Engine (MAPSS)",
            version="1.0.0",
            number_of_trajectories=709,
            number_of_samples=160359,
            features=[
                "unit_number", "time_cycles", "setting_1", "setting_2", "setting_3", "T24", "T30", "T50",
                "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32"
            ],
            limitations=[
                "Brayton-cycle gas turbine dynamics do not transfer directly to 4-stroke reciprocating piston engines.",
                "Utilized strictly as an algorithmic proxy for temporal ML model and Weibull verification."
            ],
            decision="SELECTIVE_PROXY",
        )
        self.register(cmapss_meta)
        self.register_alias("RUL_PROXY_CMAPSS", "NASA_CMAPSS")

        # 4. Case Western Reserve University (CWRU) Bearing Dataset
        cwru_meta = DatasetMetadata(
            dataset_id="CWRU_BEARING",
            name="CWRU Bearing Data Center Vibration Benchmark",
            purpose="Rotating-machinery/bearing vibration proxy.",
            source="Case Western Reserve University Bearing Data Center",
            domain="ROTATING_MACHINERY_VIBRATION",
            real_or_synthetic="CROSS_DOMAIN_PROXY",
            ground_truth_available=True,
            ground_truth_description="Electro-discharge machining seeded fault diameters (0.007 to 0.028 in) on drive-end and fan-end bearings.",
            target_engine_relevance="CROSS_DOMAIN_BEARING",
            license_status="Academic Open Access",
            generation_method="Dynamometer Test Stand with 12 kHz and 48 kHz High-Rate Accelerometers",
            version="1.0.0",
            number_of_trajectories=120,
            number_of_samples=240000,
            features=["Drive_End_Accel", "Fan_End_Accel", "Base_Accel", "Motor_Load", "Motor_Speed"],
            limitations=[
                "Electric motor drive stand; does not model reciprocating internal combustion harmonics.",
                "Static seeded faults rather than natural continuous progressive spalling."
            ],
            decision="METHODOLOGY_ONLY",
        )
        self.register(cwru_meta)
        self.register_alias("VIBRATION_PROXY_CWRU", "CWRU_BEARING")

        # 5. ALFA UAV Flight Anomaly Dataset
        alfa_meta = DatasetMetadata(
            dataset_id="ALFA_UAV",
            name="AirLab Failure and Anomaly (ALFA) Dataset for Fixed-Wing Autonomous UAVs",
            purpose="UAV flight/failure/anomaly proxy.",
            source="Carnegie Mellon University AirLab",
            domain="UAV_AVIONICS_FLIGHT",
            real_or_synthetic="CROSS_DOMAIN_PROXY",
            ground_truth_available=True,
            ground_truth_description="In-flight injected actuator and propulsion failures with GPS/IMU flight recorder ground truth.",
            target_engine_relevance="CROSS_DOMAIN_UAV_AVIONICS",
            license_status="MIT License",
            generation_method="CarbonZ T-28 Fixed-Wing Aircraft Autonomous Flight Missions with Pixhawk Autopilot",
            version="1.0.0",
            number_of_trajectories=47,
            number_of_samples=95000,
            features=["GPS_Lat", "GPS_Lon", "GPS_Alt", "Roll", "Pitch", "Yaw", "Airspeed", "Battery_Voltage", "Throttle"],
            limitations=[
                "Electric propulsion aircraft; lacks internal combustion engine thermodynamic channels.",
                "Used for autopilot trajectory replay, wind estimation, and navigation waypoint testing only."
            ],
            decision="NAVIGATION_ONLY",
        )
        self.register(alfa_meta)
        self.register_alias("NAVIGATION_PROXY_ALFA", "ALFA_UAV")

        # 6. Marine Engine Fault Dataset (Aalto University - Rejected Reference)
        marine_meta = DatasetMetadata(
            dataset_id="REFERENCE_MARINE",
            name="Marine Engine Fault Dataset (Aalto University)",
            purpose="Multi-fault terminology and maritime diesel taxonomy reference.",
            source="Aalto University Maritime Testbed (Zenodo)",
            domain="HEAVY_MARINE_DIESEL",
            real_or_synthetic="RESEARCH_REFERENCE",
            ground_truth_available=True,
            ground_truth_description="5 maritime diesel fault classes (air-cooler fouling, injector clogging, cavitation).",
            target_engine_relevance="INCOMPATIBLE_MARINE",
            license_status="Open Access Research",
            generation_method="Controlled Maritime Diesel Test Stand",
            version="1.0.0",
            number_of_trajectories=1,
            number_of_samples=89468,
            features=["Engine Speed", "No.1 Exh.Gas Temp.", "Fuel Flow", "LO Temp. Engine Out", "Charge Air Press."],
            limitations=[
                "Heavy marine propulsion dynamics are physically incompatible with high-altitude aero-piston digital twin.",
                "Strictly barred from active machine learning training or inference pipelines."
            ],
            decision="REJECTED",
        )
        self._rejected_datasets[marine_meta.dataset_id] = marine_meta
        self.register_alias("MARINE_ENGINE", "REFERENCE_MARINE")

        # 7. UAV Propeller Blade Damage Dataset (Rejected Reference)
        prop_meta = DatasetMetadata(
            dataset_id="REFERENCE_PROPELLER",
            name="UAV Propeller Blade Damage and Unbalance Dataset",
            purpose="Propeller structural vibration and unbalance formula reference.",
            source="UAV Acoustic/Vibration Testbed",
            domain="PROPELLER_AEROACOUSTICS",
            real_or_synthetic="RESEARCH_REFERENCE",
            ground_truth_available=True,
            ground_truth_description="Seeded blade damages (chipped, unbalanced) on propeller test rig.",
            target_engine_relevance="INCOMPATIBLE_PROPELLER",
            license_status="Research Reference",
            generation_method="High-Rate Accelerometer and Acoustic Transducer Logging",
            version="1.0.0",
            number_of_trajectories=5,
            number_of_samples=50000,
            features=["Blade_Vibration_DE", "Blade_Vibration_FE", "Acoustic_Sound_Pressure"],
            limitations=[
                "Propeller blade acoustic vibration lacks reciprocating engine thermomechanical coupling.",
                "Strictly barred from engine-health diagnostic and prognostic pipelines."
            ],
            decision="REJECTED",
        )
        self._rejected_datasets[prop_meta.dataset_id] = prop_meta
        self.register_alias("UAV_PROPELLER", "REFERENCE_PROPELLER")

    def register(self, metadata: DatasetMetadata) -> None:
        self._datasets[metadata.dataset_id] = metadata

    def register_alias(self, alias: str, target_id: str) -> None:
        self._aliases[alias] = target_id

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        target = self._aliases.get(dataset_id, dataset_id)
        if target in self._datasets:
            return self._datasets[target]
        return self._rejected_datasets.get(target)

    def list_all(self, include_rejected: bool = False) -> List[Dict[str, Any]]:
        core = [ds.to_dict() for ds in self._datasets.values()]
        if include_rejected:
            core.extend([ds.to_dict() for ds in self._rejected_datasets.values()])
        return core

    def get_summary(self, include_rejected: bool = False) -> Dict[str, Any]:
        catalog = self.list_all(include_rejected=include_rejected)
        return {
            "total_datasets": len(self._datasets) if not include_rejected else len(self._datasets) + len(self._rejected_datasets),
            "primary_datasets": sum(1 for ds in self._datasets.values() if ds.target_engine_relevance == "PRIMARY_TARGET_ROTAX_914"),
            "operational_context_datasets": sum(1 for ds in self._datasets.values() if ds.target_engine_relevance == "OPERATIONAL_ENVELOPE_CONTEXT"),
            "cross_domain_proxies": sum(1 for ds in self._datasets.values() if ds.real_or_synthetic == "CROSS_DOMAIN_PROXY"),
            "rejected_references": len(self._rejected_datasets),
            "total_trajectories": sum(ds.number_of_trajectories for ds in self._datasets.values()),
            "total_samples": sum(ds.number_of_samples for ds in self._datasets.values()),
            "catalog": catalog,
        }
