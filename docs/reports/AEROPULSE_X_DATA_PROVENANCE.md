# AeroPulse-X Data & Parameter Provenance Document
## Scientific Pedigree, Dataset Boundaries & Verification Status

**Document ID:** `APX-DP-2026-V1`  
**Classification:** Scientific Data Provenance & Dataset Registry  
**Standard:** Strict Transparency / Traceable Aviation Engineering  

---

## 1. Provenance Classification Key

Every dataset, physical constant, and engine parameter in AeroPulse-X is assigned one of the following provenance categories:

1. **`MEASURED`**: Physical sensor measurements recorded during actual flight campaigns or laboratory test-rig experiments.
2. **`MANUFACTURER`**: Published OEM specifications, Type Certificate Data Sheets (TCDS), or official operator manuals.
3. **`LITERATURE`**: Empirical engineering relationships published in peer-reviewed aerospace textbooks or journal articles.
4. **`DERIVED`**: Values calculated directly from first-principles thermodynamic and physical laws.
5. **`SYNTHETIC`**: Deterministic physics simulation data synthesized under controlled mathematical ground truth.
6. **`ASSUMED`**: Engineering estimates adopted where measured or published data is currently unavailable.

---

## 2. Dataset Provenance Registry

| Dataset ID | Dataset Name | Domain / Origin | Sample Count | Provenance Classification | Physical Failure Truth | Operational Role & Boundary |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `DS-ACES-01` | **NASA ACES (Altus II UAV)** | Real Flight Telemetry (Dryden Flight Research Center) | 173,878 frames (12 flights) | `MEASURED` | **NONE (0 Failures)** | Validates healthy flight operating envelope, RPM dynamics, and altitude lapse bounds. No failure ground truth. |
| `DS-SYNTH-02`| **Continuous Physics Degradation Corpus** | AeroPulse-X Coupled Thermodynamic Wear Model | 4,800 frames (60 trajectories) | `SYNTHETIC` | **EXACT MATHEMATICAL** | Primary benchmark for RUL trajectory tracking, monotonic wear degradation, and prognostic horizon. |
| `DS-CMAPSS-03`| **NASA C-MAPSS FD001** | Turbofan Run-to-Failure Benchmark (NASA Ames) | 100 train / 100 test engines | `MEASURED` *(Cross-Domain)*| **YES (Turbofan Only)** | Used strictly as an algorithmic proxy to benchmark non-linear trend extrapolation algorithms on aviation data. |
| `DS-CWRU-04` | **Case Western Reserve Vibration** | Bearing Test-Rig Accelerometer Measurements | 120,000 vibration samples | `MEASURED` *(Industrial)* | **YES (Bearing Spalling)** | Supporting bearing wear classifier; isolated from MALE UAV aero-piston digital twin. |

---

## 3. Physical Engine Parameter Provenance Table

| Parameter Name | Value | Unit | Engine / Profile | Provenance Source | Category | Confidence Status | Operational Role |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| `displacement_l` | 1.211 | L | Rotax 914 Turbo | Rotax 914 F/UL Operator Manual / EASA TCDS E.121 | `MANUFACTURER` | `VALIDATED_SPEC` | Total swept volume |
| `displacement_l` | 1.352 | L | AeroPiston 4C 1.35L | Published Generic Aero-Piston Standard | `LITERATURE` | `VALIDATED_SPEC` | Total swept volume |
| `displacement_l` | 1.991 | L | Generic AeroDiesel | Literature Inline-4 Diesel Specification | `LITERATURE` | `LITERATURE_INFORMED` | Total swept volume |
| `bore_mm` | 79.5 | mm | Rotax 914 Turbo | Rotax 914 F/UL Operator Manual | `MANUFACTURER` | `VALIDATED_SPEC` | Cylinder bore |
| `stroke_mm` | 61.0 | mm | Rotax 914 Turbo | Rotax 914 F/UL Operator Manual | `MANUFACTURER` | `VALIDATED_SPEC` | Piston stroke |
| `compression_ratio`| 9.0:1 | ratio | Rotax 914 Turbo | Rotax 914 F/UL Operator Manual | `MANUFACTURER` | `VALIDATED_SPEC` | Geometric compression |
| `compression_ratio`| 18.0:1| ratio | Generic AeroDiesel | Heavy Fuel Aero-Diesel Standards (Austin Ch 27) | `LITERATURE` | `LITERATURE_INFORMED` | Diesel compression |
| `base_power_kw` | 84.5 | kW | Rotax 914 Turbo | Rotax 914 Takeoff Power (115 HP @ 5800 RPM) | `MANUFACTURER` | `VALIDATED_SPEC` | Rated maximum power |
| `nominal_rpm` | 5500.0 | RPM | Rotax 914 Turbo | Continuous Maximum Operating Speed | `MANUFACTURER` | `VALIDATED_SPEC` | Continuous rated RPM |
| `max_rpm` | 5800.0 | RPM | Rotax 914 Turbo | 5-Minute Takeoff Maximum Speed | `MANUFACTURER` | `VALIDATED_SPEC` | Redline limit |
| `tbo_hours` | 1200.0 | h | Rotax 914 Turbo | Rotax Maintenance Manual (SB-914-001) | `MANUFACTURER` | `VALIDATED_SPEC` | Certified TBO limit |
| `tbo_hours` | 2000.0 | h | AeroPiston 1.35L | Standard FAA Part 33 Certified Piston TBO | `LITERATURE` | `VALIDATED_SPEC` | General aviation TBO |
| `fuel_lhv_mj_kg` | 43.5 | MJ/kg | Avgas 100LL | ASTM D910 Aviation Gasoline Specification | `LITERATURE` | `VALIDATED_SPEC` | Lower heating value |
| `afr_stoich` | 14.7 | ratio | Gasoline / Avgas | Chemical Stoichiometry ($C_8H_{18}$) | `DERIVED` | `DERIVED_THEORETICAL`| Air-fuel ratio |
| `gamma` | 1.33 | ratio | Exhaust Gas | High-Temperature Combustion Gas Specific Heat Ratio | `LITERATURE` | `LITERATURE_INFORMED` | Isentropic expansion |
| `mass_power_ratio` | 0.88 | kg/kW | Rotax 914 Turbo | Austin (2010) Ch 6 Fig 6.6 | `LITERATURE` | `LITERATURE_INFORMED` | Dry mass / power |
| `bsfc_nominal` | 0.33 | kg/kWh| Rotax 914 Turbo | Austin (2010) Ch 6.5.1 (4-stroke range: 0.3 - 0.4) | `LITERATURE` | `LITERATURE_INFORMED` | Fuel burn efficiency |
| `cooling_area_m2` | 0.85 | m² | AeroPiston 1.35L | Lumped Nacelle Cowl Geometry | `ASSUMED` | `LITERATURE_INFORMED` | Heat rejection surface |
| `oil_volume_l` | 3.5 | L | Rotax 914 Turbo | Dry Sump Tank Capacity | `MANUFACTURER` | `VALIDATED_SPEC` | Lubrication reservoir |
| `turbo_critical_alt`| 16,000 | ft | Rotax 914 Turbo | Critical Altitude for Maximum Boost (1.25 bar MAP) | `MANUFACTURER` | `VALIDATED_SPEC` | Turbocharger envelope |

---

## 4. Integrity Declarations

1. **No Assumed Value Disguised as Measured**: All assumed parameters (e.g. nacelle heat transfer coefficients) are explicitly tagged `ASSUMED` or `LITERATURE_INFORMED`.
2. **No Claim of Aircraft Physical Certification**: All synthetic degradation algorithms are transparently identified as simulation benchmarks pending access to physical engine test dynamometers.
