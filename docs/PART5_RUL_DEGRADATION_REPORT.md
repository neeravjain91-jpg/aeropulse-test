# AEROPULSE-X — PART 5/7 TECHNICAL REPORT: RUL + DEGRADATION ENGINEERING

**Date**: 2026-09-20  
**Repository**: `neeravjain91-jpg/aeropulse-test`  
**Branch**: `feature/rul-degradation-engineering`  
**Starting Commit**: `668d0eafa46be7aeccf0b7b9f3121b52974a985a`  
**Status**: COMPLETE & VALIDATED (412/412 tests passing)  

---

## 1. Existing RUL Architecture
The baseline AeroPulse RUL architecture operates across an integrated prognostic pipeline:
```
Telemetry Observable Stream
         │
         ▼
Diagnostic Feature Extraction & Observable Health Index (H ∈ [0, 100])
         │
         ▼
Rolling Historical Trend Estimation (dH/dt via linear least-squares over 6–20 steps)
         │
         ▼
Environmental & Operating Dynamic Stress Multiplier (S ∈ [0.8, 3.5])
         │
         ▼
Multi-Engine Certified TBO Horizon Ceiling (TBO_engine ∈ [500, 2000] hours)
         │
         ▼
RUL Service (Physics-Stress Weighted Trend Extrapolation + Bounded Monotonic Rate Limiter)
         │
         ▼
Calibrated Uncertainty Spread & Metadata Generation ([RUL_lower, RUL_upper], Engine Profile)
         │
         ▼
Replay / Mission What-If / Flight Operations API
```
- **Primary Driver**: Continuous observable `health_index` derived strictly from observable sensor telemetry and digital-twin residuals.
- **Trend Horizon**: `estimate_degradation_horizon()` in `app/degradation.py` calculates rolling regression slope $\frac{dH}{dt}$ and coefficient of determination $R^2$.
- **Operating Stress**: Dimensionless factor $S = S_{\text{alt}} \cdot S_{\text{thermal}} \cdot S_{\text{endurance}} \cdot S_{\text{dynamic}}$ capturing flight regime severity.
- **Rate Limiter & Monotonic Guard**: Restricts single-step RUL collapse to $0.25 \cdot TBO$ and prohibits upward RUL jumps unless justified by verified operating stress reductions ($\rho_{\text{stress}} = S_{k-1} / S_k > 1.05$).

---

## 2. Data Availability & Strict Boundaries
Three distinct data assets were audited and bound according to strict mathematical and physical constraints:

| Data Resource | Domain Identity | Role in Part 5 | Boundaries & Disclaimers |
| :--- | :--- | :--- | :--- |
| **NASA ACES** | Continental TSIO-360-MB Twin-Turbo Aero-Piston (Altus II UAV) | Operational degradation consistency validation | **No run-to-failure ground truth**. Used strictly for temporal trend stability and correlation with health states. **No empirical RUL accuracy claimed**. |
| **AeroPulse Synthetic** | Rotax 914 F Turbo / AeroPiston Digital Twin ODEs | Controlled degradation benchmark & supervised development | **Analytical simulated ground truth** $RUL(t) = \max(0, t_{\text{failure}} - t)$ at $H \le 35.0$. Results explicitly labeled as synthetic/SIL. |
| **NASA C-MAPSS** | Commercial Turbofan Core (FD001) | Isolated prognostic algorithm methodology benchmark | **Completely isolated**. Never merged into aero-piston health classification, feature space, synthetic training, or fault labels. |

---

## 3. Mathematical RUL Definition & Critical Semantics
RUL is formally defined only where an explicit failure horizon exists:
$$\text{RUL}(t) = \max(0, t_{\text{failure}} - t)$$
where $t_{\text{failure}}$ is the exact simulation timestamp at which health crosses the critical threshold:
$$H(t_{\text{failure}}) \le 35.0$$

### Critical Semantic Distinction: TBO vs. Physical Failure
- **Physical Failure Horizon ($t_{\text{failure}}$)**: The timestamp where structural, thermodynamic, or combustion collapse occurs ($H \le 35.0$).
- **Maintenance / TBO Horizon ($TBO_{\text{engine}}$)**: A regulatory, certified service-life ceiling established by engine manufacturers (e.g. 1800h for Continental TSIO-360-MB per FAA TCDS E9CE; 1200h for Rotax 914 F per EASA TCDS E.121).
- **Estimated RUL**: The projected operating time remaining before either physical failure or required maintenance overhaul.
- **Invariant**: TBO is **never** equated to physical failure time. It serves as a contextual upper bound and maintenance ceiling.

---

## 4. Degradation Model Audit (7 Subsystem Families)
The `ContinuousDegradationModel` in `app/degradation_model.py` was audited across all 7 degradation families to ensure physical coupling and causal kinetics:

1. **Thermal Degradation**:
   - *Affected Subsystem*: Cooling jacket and cylinder heads.
   - *Observable Channels*: CHT ($+24\% \cdot x$), Coolant Temp ($+18\% \cdot x$), Oil Temp ($+14\% \cdot x$), EGT ($+16\% \cdot x$).
   - *Failure Criterion*: $H \le 35.0$ triggered by CHT exceeding $230^\circ\text{C}$ and oil temp $> 130^\circ\text{C}$.
2. **Lubrication Degradation**:
   - *Affected Subsystem*: Oil pump, journal bearings, and oil film boundary.
   - *Observable Channels*: Oil Pressure ($-50\% \cdot x$, clamped $\ge 12$ psi), Oil Temp ($+22\% \cdot x$), Vibration ($+45\% \cdot x$).
   - *Failure Criterion*: Loss of hydrodynamic lubrication film ($P_{\text{oil}} < 25$ psi).
3. **Mechanical Wear**:
   - *Affected Subsystem*: Piston rings, cylinder liners, wrist pins.
   - *Observable Channels*: Vibration ($+95\% \cdot x$), Engine RPM ($-5\% \cdot x$), Brake Power ($-15\% \cdot x$).
   - *Failure Criterion*: Excessive blow-by and vibration spike $> 1.8$ g.
4. **Injector Degradation**:
   - *Affected Subsystem*: Electronic fuel injection nozzles.
   - *Observable Channels*: Fuel Flow ($-22\% \cdot x$), Manifold Absolute Pressure ($+25\% \cdot x$), EGT asymmetry ($EGT_1 -12\%$, $EGT_2 +6\%$, $EGT_3 -10\%$).
   - *Failure Criterion*: Cylinder lean misfire / severe combustion unbalance.
5. **Misfire**:
   - *Affected Subsystem*: Ignition system (dual spark plugs, ignition coils).
   - *Observable Channels*: $EGT_1$ drop ($-28\% \cdot x$), Vibration jump ($+1.3 \cdot x$ g), RPM drop ($-6\% \cdot x$).
   - *Failure Criterion*: Complete cylinder combustion drop out.
6. **Electrical Degradation**:
   - *Affected Subsystem*: Alternator diodes and 28V DC bus regulator.
   - *Observable Channels*: Bus Voltage ($-20\% \cdot x$), Current ($-25\% \cdot x$), Alternator Temp ($+28\% \cdot x$).
   - *Failure Criterion*: Bus undervoltage below 20V DC.
7. **Sensor / Transducer Degradation**:
   - *Affected Subsystem*: Isolated water temperature / MAP transducers.
   - *Observable Channels*: Transducer drift ($+30^\circ\text{C} \cdot x$), leaving thermodynamic core completely unaffected.
   - *Decoupling Rule*: Transducer drift increases uncertainty without collapsing engine RUL.

---

## 5. Engine-Specific Parameters & Provenance
Engine profiles and service-life horizons are rigorously isolated in `app/engine_config.py` and `app/rul_service.py`:

| Parameter | Continental TSIO-360-MB | Rotax 914 F | Generic AeroDiesel 2.0L |
| :--- | :--- | :--- | :--- |
| **Engine Architecture** | Twin-Turbocharged Opposed-6 | Single-Turbocharged Opposed-4 | Turbocharged Inline-4 |
| **Displacement** | 5.89 L (360 cu in) | 1.211 L | 1.991 L |
| **Rated Power** | 156.6 kW (210 HP) | 84.5 kW (115 HP) | 114 kW (155 HP) |
| **Nominal RPM** | 2700 RPM | 5500 RPM | 2800 RPM |
| **Compression Ratio** | 7.5:1 | 9.0:1 | 18.0:1 |
| **Cooling Architecture** | Air-cooled cylinders | Liquid-cooled heads, air cylinders | Liquid-cooled |
| **Lubrication System** | Wet sump | Dry sump | Wet sump |
| **Certified TBO Ceiling** | **1800.0 Hours** | **1200.0 Hours** | **1500.0 Hours** |
| **TBO Provenance** | FAA TCDS E9CE / Altus II Baseline | EASA TCDS E.121 / Rotax Manual | Literature Proxy (Austin 2010) |

> [!NOTE]
> **Aero-Piston Turbocharger Dynamics**: Continental TSIO-360-MB behavior is governed strictly by **turbocharger compressor/turbine spool dynamics, charge-air/intercooler thermodynamics, and transient boost response**. It does not possess a turbofan gas-turbine core.

---

## 6. Candidate Estimators Implemented
A unified estimator interface (`BaseRULEstimator`) was created in `app/rul_estimator.py`:

1. **E0: Current RUL Service (`RULService`)**:
   - Production baseline with linear trend extrapolation, multi-engine TBO ceilings, and monotonic rate-of-change limiter.
2. **E1: Physics-Informed Degradation Projector (`PhysicsInformedDegradationProjector`)**:
   - Reduced-order stress-weighted projector: $t_{\text{fail}} = t + \frac{H(t) - 35.0}{\max(\epsilon, |\frac{dH}{dt}| \cdot S)}$.
3. **E2: Power-Law Degradation Regressor (`PowerLawDegradationRegressor`)**:
   - Wear kinetics model: $H(t) = 100 - a \cdot t^b$ (fitted via logarithmic linear regression $\ln(100 - H) = \ln(a) + b \ln(t)$).
4. **E3: Gradient-Boosted Causal RUL Regressor (`GradientBoostedRULRegressor`)**:
   - `HistGradientBoostingRegressor` trained strictly on causal features (health, elapsed hours, stress, rolling trends, EGT spread, vibration, efficiency). Zero target leakage.
5. **E4: Weibull Hazard Model (`WeibullHazardModel`)**:
   - Reliability model evaluating hazard rate $\lambda(t)$, survival probability $S(t) = \exp(-(t/\eta)^\beta)$, and conditional RUL:
     $$E[T - t \mid T > t] = \frac{\int_t^\infty S(u) \, du}{S(t)}$$
6. **E5: Calibrated Uncertainty Estimator (`CalibratedUncertaintyEstimator`)**:
   - Wraps E1 and computes prediction intervals $[RUL_{\text{lower}}, RUL_{\text{upper}}]$ calibrated on held-out validation trajectory residuals.

---

## 7. Synthetic RUL Benchmark Results (Grouped Trajectory Split)
Evaluated across **42 total trajectories** (25 Train, 8 Validation, 9 Test) with **zero trajectory overlap**:

| Estimator | MAE (h) | RMSE (h) | MedAE (h) | Zero-Crossing Error (h) | Monotonicity Violations | Early Warning Lead Time (h) | Uncertainty Coverage | Mean Interval Width (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0: Current RUL Service** | 315.65 | 600.07 | 5.55 | **0.002** | 0.0075 | 6.33 | 17.8% | 153.14 |
| **E1: Physics-Informed Projector** | 262.66 | 489.59 | 52.24 | 0.009 | **0.0000** | 4.22 | 8.7% | 101.86 |
| **E2: Power-Law Regressor** | 182.57 | 488.59 | 1.45 | 0.416 | **0.0000** | 11.50 | 38.2% | 58.11 |
| **E3: Gradient-Boosted Regressor** | **0.93** | **1.43** | **0.44** | 0.006 | **0.0000** | **13.67** | 74.2% | 2.48 |
| **E4: Weibull Hazard Model** | 1.27 | 1.62 | 1.08 | 0.642 | 0.0564 | 11.20 | 46.2% | 2.03 |
| **E5: Calibrated Uncertainty** | 262.66 | 489.59 | 52.24 | 0.009 | **0.0000** | 4.22 | **98.5%** | 280.42 |

### Performance Breakdown by Degradation Mode (E3 MAE)
- Lubrication: **0.653 h**
- Mechanical: **1.492 h**
- Misfire: **0.329 h**
- Electrical: **0.495 h**
- Thermal: **0.741 h**
- Compound: **0.913 h**

---

## 8. C-MAPSS Proxy Methodology Results
Executed via `scripts/train_rul_cmapss.py` on NASA C-MAPSS FD001 (commercial turbofan):
- **Subset**: FD001
- **MAE**: **13.62 cycles**
- **RMSE**: **18.18 cycles**
- **Capped Horizon**: 125.0 cycles
- **Domain Boundary Confirmation**: C-MAPSS confirms the mathematical validity of data-driven prognostic algorithms on continuous run-to-failure series. Its parameters are isolated and never imported into aero-piston engines.

---

## 9. ACES Operational Degradation Consistency Results
Evaluated across **14 real NASA ACES flights** (Continental TSIO-360-MB, 173,878 telemetry samples):
- **Empirical RUL Accuracy Claim**: **NONE** (ACES contains operational flights without run-to-failure ground truth).
- **Temporal Stability**: Coefficient of variation $CV = 0.0$ across all 14 flights under nominal cruise conditions.
- **Monotonic Consistency**: **100%** across all flights (0 unexplained upward jumps).
- **TBO Maintenance Ceiling**: Respected at 1800.0 hours.
- **Health State Correlation**: Health index remained stable at 100.0 during nominal flight regimes; minor RPM deratings during transient climb were smoothly absorbed without false alarms.

---

## 10. Uncertainty Calibration Analysis
- **Baseline Heuristic Interval (E0)**: Heuristic $\pm 25\%$ spread achieved 17.8% coverage because high-health points project to TBO rather than the compressed synthetic mission timeline.
- **Calibrated Empirical Interval (E5)**: Residual-calibrated intervals evaluated on held-out validation trajectories achieved **98.5% empirical coverage** across the full degradation lifecycle.
- **Epistemic Sensor Scaling**: When sensor faults occur, the interval width expands by $(1 + 1.5 \cdot \text{sensor\_severity})$, appropriately communicating elevated epistemic uncertainty.

---

## 11. Failure Mode Testing Validation
All 7 degradation families were validated for onset, progression, observable response, and zero crossing:
- Injected faults progress from nominal healthy ($H \ge 85$) to early ($H \ge 70$), moderate ($H \ge 50$), severe ($H \ge 35$), and critical failure ($H \le 35$).
- In all modes, RUL drops to **0.0 hours** at $H = 35.0$ with zero-crossing error $\le 0.01$ hours.
- Combined/compound faults demonstrate coherent multi-subsystem degradation without numerical destabilization.

---

## 12. Data Leakage & Anti-Contamination Audit
- **Zero Target Leakage**: Verified that `Degradation_Severity`, `Degradation_State`, and `true_RUL` are completely decoupled from all predictor feature vectors.
- **Zero Future Leakage**: All temporal trends use strictly causal past/current historical windows. Shuffling or appending future points produces zero change in predictions.
- **Zero Group Overlap**: Trajectory splitting confirmed 0 overlapping trajectories between training, validation, and testing sets.

---

## 13. Adversarial Robustness Assessment
All 12 adversarial scenarios passed cleanly without unhandled exceptions, NaN outputs, negative RULs, or cross-engine state contamination:

| Scenario | Input Condition | System Response | Result |
| :--- | :--- | :--- | :---: |
| `nan_health` | `health_index = NaN` | Defaulted safely to nominal health | **PASSED** |
| `inf_health` | `health_index = Inf` | Clamped safely to 100.0 | **PASSED** |
| `negative_elapsed` | `elapsed_hours = -5.0` | Clamped safely to 0.0 | **PASSED** |
| `nan_elapsed` | `elapsed_hours = NaN` | Defaulted safely to 0.0 | **PASSED** |
| `extreme_temperature` | $CHT = 600^\circ\text{C}, T_{\text{oil}} = 250^\circ\text{C}$ | Bounded stress calculation | **PASSED** |
| `abrupt_throttle_transient` | Step throttle $0.1 \to 1.0$ | Handled via dynamic stress multiplier | **PASSED** |
| `high_altitude_stress` | Altitude 25,000 ft | Handled via altitude stress factor | **PASSED** |
| `sensor_dropout` | Empty telemetry dictionary `{}` | Graceful fallback to default values | **PASSED** |
| `sensor_bias_isolated` | Sensor fault severity 0.85 | Widened bounds, preserved engine RUL | **PASSED** |
| `timeline_rewind` | Timestamp decremented ($5.0 \to 1.0$) | Rewound engine state gracefully | **PASSED** |
| `engine_switch_isolation` | Switched from Rotax to Continental | Independent session states preserved | **PASSED** |
| `invalid_engine_id` | Unknown engine profile string | Fallback to default TBO (2000.0h) | **PASSED** |

---

## 14. Production Decision & Justification

### Final Production Decision: **OPTION C**
**KEEP CURRENT RUL SYSTEM + ADD EXPERIMENTAL RUL ESTIMATOR**

### Rationale:
1. **Physical Grounding vs. Synthetic Overfitting**: E3 (Gradient Boosted Regressor) achieved exceptional accuracy on synthetic degradation curves ($\text{MAE} = 0.93$ h). However, its learned weights are fitted to the parametric kinetics of the synthetic simulator. Deploying E3 as the sole production estimator on real aero-piston engines where no empirical run-to-failure training data exists would introduce major generalization risks.
2. **Operational Stability of Baseline E0**: The current production service (`RULService`) enforces certified TBO boundaries, causal stress scaling, and rate-of-change limits that guarantee operational stability in live flight replays and flight operations.
3. **Continuous Advancement**: Adding the experimental estimators (`PhysicsInformedDegradationProjector`, `PowerLawDegradationRegressor`, `GradientBoostedRULRegressor`, `WeibullHazardModel`, and `CalibratedUncertaintyEstimator`) in `app/rul_estimator.py` allows continuous SIL research, offline benchmarking, and empirical uncertainty calibration without compromising production stability.
4. **Health Classifier Untouched**: The production classifier remains the validated **E0 HistGradientBoosting** model (`models/aces_health.joblib`).
