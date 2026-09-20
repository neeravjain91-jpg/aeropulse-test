# Dataset Registry & Scientific Provenance Catalog
## Master Dataset Registry Synchronized with `docs/DATA_RESOURCE_SPEC.md`

### 1. Master Dataset Inventory & Provenance Catalog

| Resource ID | Dataset Name | Physical Domain | Engine Architecture | Source Type | Ground Truth Status | AeroPulse Subsystem Role | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`REAL_ACES`** | NASA ACES Telemetry | Real Operational Flight Logs | Continental TSIO-360-MB (Twin-Turbo 6-Cyl Piston, 5.9L) | Real Flight Data | 4-Class `Health_State` (No run-to-failure RUL) | Primary operational envelope baseline & real flight health classification | **`ACTIVE`** |
| **`AEROPULSE_SYNTHETIC`** | AeroPulse-X Synthetic Master Corpus | Reciprocating IC Simulation (ODE Wear + Otto) | Rotax 914 F (Turbocharged 4-Stroke Boxer-4, 1.2L, 115 HP) | Physics-Grounded Simulation | Exact Mathematical RUL ($H=35.0$) & 7 Discrete Fault Modes | Primary physics-informed digital twin, SIL degradation, and RUL demonstrator | **`ACTIVE`** |
| **`RUL_PROXY_CMAPSS`** | NASA C-MAPSS v1 / N-CMAPSS | Commercial Gas Turbine | High-Bypass Commercial Turbofan (90,000 lbf, Brayton Cycle) | Cross-Domain Proxy | Run-to-Failure Remaining Cycles ($Y$) | Algorithmic RUL methodology benchmark & Weibull parameter verification only | **`SELECTIVE_PROXY`** |
| **`VIBRATION_PROXY_CWRU`** | CWRU Bearing Data Center | Rotating Machinery Test Rig | 2 HP Reliance Electric Induction Motor | Cross-Domain Proxy | Seeded EDM Fault Location & Diameter (0.007"-0.028") | Rotordynamic vibration DSP feature extraction pipeline validation | **`METHODOLOGY_ONLY`** |
| **`NAVIGATION_PROXY_ALFA`** | CMU AirLab Failure & Anomaly (ALFA) | Fixed-Wing Autonomous Avionics | CarbonZ T-28 Trojan (Brushless Electric Motor + Pixhawk) | Cross-Domain Proxy | Timestamped Actuator Failures & In-Flight Emergency Glide | Flight path cross-track error modeling, wind vector estimation ($V_w, \theta_w$), and RTL divert planning | **`NAVIGATION_ONLY`** |
| **`REFERENCE_MARINE`** | Marine Engine Fault Dataset | Maritime Heavy Diesel | Multi-Megawatt Heavy Marine Diesel Engine | Research Reference | 5 Maritime Diesel Fault Classes | Research reference for multi-fault terminology taxonomy only (Excluded from ML) | **`REJECTED`** |
| **`REFERENCE_PROPELLER`** | UAV Propeller Blade Dataset | Propeller Structural / Acoustic | Brushless Motor Propeller Test Rig | Research Reference | Damaged / Unbalanced Propeller Blade States | Reference for propeller harmonic order equations only (Excluded from ML) | **`REJECTED`** |

---

### 2. Scientific Boundaries & Strict Anti-Concatenation Policy

1. **Target Engine Ground Truth**: Real run-to-failure telemetry for the Rotax 914 F is unavailable in open literature. Therefore, `AEROPULSE_SYNTHETIC` serves as the physics-informed benchmark.
2. **NASA ACES Operational Disclosure**: NASA ACES provides real Altus II operational flight telemetry used strictly for operational-envelope and contextual real-flight validation; it contains **NO run-to-failure RUL ground truth** and is **not** target-engine Rotax 914 data.
3. **Cross-Domain Proxy Separation**: NASA C-MAPSS (turbofan gas turbine), CWRU (electric motor bearing), and ALFA (electric fixed-wing UAV) are utilized strictly as cross-domain algorithmic proxies within their isolated namespaces.
4. **Strict Anti-Concatenation Mandate**: Prohibits merging multi-source datasets into a single engine-health feature dataframe. All dataset pipelines maintain strict physical and architectural boundaries.
5. **Anti-Leakage Enforcement**: Ground-truth target fields (`Health_State`, `Degradation_Severity`, `true_RUL`, `true_failure_time`) are strictly excluded from predictive model inputs.
