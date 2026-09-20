# Data Resource Specification (DATA_RESOURCE_SPEC)
## AeroPulse-X Authoritative Dataset Catalog, Domain Boundaries & Utility Specification

**Document Version:** 2.0.0-SPEC  
**Status:** AUTHORITATIVE & BINDING  
**Target Platform:** AeroPulse-X MALE Aero-Piston UAV Digital Twin & Diagnostic Framework  
**Primary Engine Target:** Rotax 914 F Turbocharged 4-Stroke Piston Engine / Continental TSIO-360 Operational Baseline  

---

## 1. Master Dataset Catalog & Integration Matrix

| Resource ID | Dataset Name | Physical Domain | Engine Architecture | Source Type | Availability in FS | Sampling | Ground Truth Status | Intended Use | Compatibility | Leakage Risk | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`REAL_ACES`** | NASA ACES Telemetry | Real Airborne Operational Telemetry | Continental TSIO-360-MB (Twin-Turbo 6-Cyl Piston, 5.9L) | Real Operational Flight Logs | `FINAL_DATASET/ACES/aces_health.csv` (116.4 MB) | 1 Hz continuous | 4-Class `Health_State` (Normal, Watch, Warning, Critical); No run-to-failure RUL | Primary operational envelope baseline & real flight health classifier training | HIGH (Reciprocating Piston) | RESOLVED (GroupKFold by Flight, no ground-truth penalty) | **`ACTIVE`** |
| **`AEROPULSE_SYNTHETIC`** | AeroPulse-X Synthetic Master Engine Corpus | Physics-Informed Reciprocating IC Simulation | Rotax 914 F (Turbocharged 4-Stroke Boxer-4, 1.2L, 115 HP) | Physics-Grounded Simulation (ODE Wear + Otto) | `app/data_engine.py` (Dynamic Generator) | 1 Hz kinematic trajectories | Exact mathematical RUL ($y_{\text{true}} = \max(0, t_{\text{fail}} - t)$ at $H=35.0$); 7 discrete fault modes | Primary physics-informed aero-piston digital twin, SIL degradation, and RUL demonstrator | DIRECT / EXACT (Target Rotax 914) | ZERO (Mathematical ground truth strictly isolated from features) | **`ACTIVE`** |
| **`RUL_PROXY_CMAPSS`** | NASA C-MAPSS v1 / C-MAPSS-2 (N-CMAPSS) | Commercial Aviation Gas Turbine | High-Bypass Commercial Turbofan (Brayton Cycle, 90,000 lbf) | Simulated Turbofan Degradation | `AeroPulse-Datasets/C-MAPSS/` (31.6 GB) | 1 sample/cycle (v1) / 1 Hz (v2) | Cycle-based run-to-failure ground truth ($Y$) | Algorithmic RUL methodology benchmark & Weibull parameter verification only | ZERO for piston diagnostics; HIGH for RUL algorithms | HIGH if merged (Strictly isolated by script `train_rul_cmapss.py`) | **`SELECTIVE_PROXY`** |
| **`VIBRATION_PROXY_CWRU`** | CWRU Bearing Data Center | Rotating Machinery Test Stand | 2 HP Reliance Electric Induction Motor | Laboratory Seeded Fault Test Rig | `data_sample/cwru_demo.csv` (22 KB), `AeroPulse-Datasets/CWRU/` (74.9 MB) | 12 kHz / 48 kHz Accelerometer | Seeded EDM fault location (Ball, Inner Race, Outer Race) & diameter (0.007"-0.028") | Rotordynamic vibration DSP feature extraction pipeline validation (RMS, Kurtosis, Crest Factor) | ZERO for engine health; HIGH for vibration DSP algorithms | ZERO (Isolated in `app/vibration.py`) | **`METHODOLOGY_ONLY`** |
| **`NAVIGATION_PROXY_ALFA`** | CMU AirLab Failure & Anomaly (ALFA) | Fixed-Wing Autonomous UAV Avionics | CarbonZ T-28 Trojan (Brushless Electric Motor + Pixhawk) | Real Autonomous Flight Telemetry | `AeroPulse-Datasets/ALFA/` (2.58 GB) | 10 Hz – 50 Hz (ROS Topics) | Timestamped actuator failures, engine-out emergencies, GPS/IMU tracks | Flight path cross-track error modeling, wind vector estimation ($V_w, \theta_w$), and RTL divert planning | ZERO for IC engine thermodynamics; HIGH for flight risk/GCS | ZERO (Isolated in mission risk & flight planner layers) | **`NAVIGATION_ONLY`** |
| **`REFERENCE_MARINE`** | Marine Engine Fault Dataset (Aalto University) | Maritime Heavy Diesel Testbed | Multi-Megawatt Low/Medium-Speed Heavy Marine Diesel | Controlled Marine Test Stand | `data_sample/dataset_summary.csv` (Reference entry only) | Variable | 5 maritime fault classes (air-cooler fouling, injector clogging, cavitation) | Research reference only for multi-fault terminology taxonomy | INCOMPATIBLE (Marine diesel vs high-altitude aero-piston) | HIGH if merged (Prohibited from model training) | **`REJECTED`** |
| **`REFERENCE_PROPELLER`** | UAV Propeller Blade Damage Dataset | UAV Multi-Rotor / Propeller Testbed | Brushless Electric Motor Propeller Stand | Acoustic / Accelerometer Test Stand | `AeroPulse-Datasets/UAV-Vibration/` (66.7 MB) | High-rate acceleration | Blade damage classification (Healthy, Damaged Blade, Unbalanced) | Reference only for propeller harmonic order unbalance equations | INCOMPATIBLE (Propeller structural acoustic vs 4-stroke engine digital twin) | HIGH if merged (Prohibited from engine health models) | **`REJECTED`** |

---

## 2. Strict Anti-Concatenation & Architectural Boundary Rules

### 2.1 Rule 1: Prohibition of Blind Multi-Source Merging
```python
# FORBIDDEN OPERATION:
# NEVER concatenate cross-domain dataframes into engine-health training vectors:
pd.concat([ACES, CMAPSS, CWRU, ALFA, Marine, Propeller])  # STRICTLY PROHIBITED
```
*Physical Rationale*: Turbofan bypass ratios ($P_{15}/P_2$), electric motor bearing spalling frequencies, and marine engine cooling water cavitation have **zero physical or thermodynamic meaning** in an air-cooled/water-cooled turbocharged 4-stroke spark-ignited aero-piston engine.

### 2.2 Rule 2: Explicit Namespaces & Model Isolation
Each dataset is restricted to its dedicated architectural namespace:
- `REAL_ACES/` $\rightarrow$ Core engine health classifiers (`aces_health.joblib`, `aces_tcn_residual.pt`, `aces_tcn_autoencoder.pt`, `aces_anomaly.joblib`).
- `AEROPULSE_SYNTHETIC/` $\rightarrow$ Virtual Data Lab, SIL simulation, physics-informed kinematics, and RUL extrapolation demonstrator.
- `RUL_PROXY_CMAPSS/` $\rightarrow$ Isolated prognostic benchmarking via `scripts/train_rul_cmapss.py` and `app/rul_validation.py`.
- `VIBRATION_PROXY_CWRU/` $\rightarrow$ Isolated vibration DSP validation via `app/vibration.py`.
- `NAVIGATION_PROXY_ALFA/` $\rightarrow$ Flight path risk, wind vector integration, and tactical waypoint contingency logic.
- `REFERENCE_MARINE/` & `REFERENCE_PROPELLER/` $\rightarrow$ Retained for literature reference only; strictly barred from all ML pipelines.

### 2.3 Rule 3: Anti-Leakage Ground Truth Isolation
Under no circumstances may the following ground-truth target variables be utilized as predictor inputs:
- `Health_State` (target label)
- `Degradation_Severity` (simulator ground-truth state)
- `fault_severity` / `target_severity` (injected fault ground truth)
- `true_RUL` / `true_failure_time` (analytical RUL targets)
- Post-fault future sequence observations or timestamps.

---

## 3. Physical Domain Specifications

### 3.1 `REAL_ACES` (NASA ACES Flight Telemetry)
- **Engine**: Continental TSIO-360-MB Twin-Turbocharged 6-Cylinder Horizontally Opposed Reciprocating Engine.
- **Displacement / Power**: 5.9 L (360 cu in), 210 HP @ 2700 RPM.
- **Operating Cycle**: 4-stroke spark ignition, turbocharged, fuel-injected.
- **Aircraft**: Altus II Civilian UAV (NASA Dryden Flight Research Center).
- **Sensors (60 Channels)**: `Engine_RPM`, `EGT1..4`, `CHT`, `Oil_Temp`, `Oil_Pressure`, `MAP_Injector`, `Fuel_Flow`, `Alternator_Temp`, `Battery_Voltage`, `Battery_Current`, `Operating_State`.
- **Physical Role**: Validates operational envelope boundaries (RPM, MAP, CHT, EGT, Oil Pressure) under real atmospheric lapse rates ($R^2 > 0.93$).

### 3.2 `AEROPULSE_SYNTHETIC` (Rotax 914 F ODE Simulation)
- **Engine**: Rotax 914 F Turbocharged 4-Cylinder Boxer Engine with automatic turbo wastegate controller.
- **Displacement / Power**: 1211 cc, 115 HP @ 5800 RPM (Takeoff) / 100 HP @ 5500 RPM (Max Continuous).
- **Operating Cycle**: 4-stroke spark ignition, liquid-cooled cylinder heads, air-cooled cylinders.
- **Governing Physics**:
  - Speed-Density air mass flow: $\dot{m}_{\text{air}} = \frac{V_d \cdot N \cdot \eta_v \cdot P_{\text{man}}}{2 R T_{\text{man}}}$
  - Arrhenius thermomechanical wear kinetics: $\frac{dD}{dt} = A \cdot \exp\left(-\frac{E_a}{R T}\right) \cdot \sigma_{\text{mech}}^n$
  - Kinematic mission phase schedule: `CLIMB` $\rightarrow$ `CRUISE` $\rightarrow$ `DESCENT` $\rightarrow$ `LANDING`
- **Physical Role**: High-fidelity Software-in-the-Loop (SIL) test corpus with deterministic ground-truth RUL ($H_{\text{crit}} = 35.0$).

### 3.3 `RUL_PROXY_CMAPSS` (NASA Turbofan Degradation)
- **Engine**: 90,000 lbf High-Bypass Commercial Turbofan.
- **Operating Cycle**: Continuous Brayton Cycle (Fan, Low-Pressure Compressor, High-Pressure Compressor, Combustor, Turbines, Nozzle).
- **Physical Boundary**: Gas turbine thermodynamic degradation (blade clearance growth, compressor fouling) does not model reciprocating piston ring/cylinder friction. Retained exclusively to benchmark Weibull hazard modeling and multi-step prognostic algorithms.

### 3.4 `VIBRATION_PROXY_CWRU` (Bearing Test Rig)
- **Apparatus**: 2 HP Reliance Electric Induction Motor with torque transducer and dynamometer.
- **Physical Boundary**: Stationary electric motor with seeded EDM pits; lacks 4-cylinder reciprocating inertial torque pulses ($f_{\text{fire}} = \frac{N}{60} \frac{N_{\text{cyl}}}{2}$). Retained exclusively for DSP feature extraction verification (RMS, kurtosis, crest factor, envelope spectra).

### 3.5 `NAVIGATION_PROXY_ALFA` (Fixed-Wing UAV)
- **Vehicle**: CarbonZ T-28 Trojan fixed-wing autonomous UAV with Pixhawk PX4 autopilot.
- **Physical Boundary**: Electric brushless motor; lacks piston thermodynamic channels (no CHT, EGT, oil pressure, manifold pressure). Retained exclusively for tactical waypoint cross-track deviation and wind triangle decomposition ($V_w, \theta_w$).

---

## 4. Integration Verification Criteria & Quality Gates

| Quality Gate | Requirement | Verification Method |
| :--- | :--- | :--- |
| **QG-1: Non-Concatenation** | No cross-dataset row merging in any pipeline | Automated test `test_dataset_boundaries_and_provenance.py` |
| **QG-2: Zero Leakage** | No ground-truth targets (`Health_State`, `Degradation_Severity`, `true_RUL`) as features | Data validator audit + automated leakage tests |
| **QG-3: Group Isolation** | Train/test splits partitioned strictly by flight/trajectory ID | `GroupKFold` / `GroupShuffleSplit` verification |
| **QG-4: Regression Baseline** | Full test suite passes completely | `pytest -q` $\rightarrow$ 100% passing tests |
| **QG-5: Disclaimers** | Every output report explicitly states data provenance and proxy limitations | Contract assertions on API and validator outputs |
