# AeroPulse-X — Part 3/7: Physics + Temporal Feature Engineering Report

**Document Version:** 1.0.0-PART3  
**Status:** COMPLETED & BINDING  
**Target Repository:** `neeravjain91-jpg/aeropulse-test`  
**Working Branch:** `feature/physics-temporal-features`  
**Base Commit:** `65b54bf`  
**Authoritative Reference:** `docs/DATA_RESOURCE_SPEC.md`

---

## 1. Executive Summary & Production Decision

Part 3/7 evaluated whether causal temporal behavior ($dX/dt$, $d^2X/dt^2$, window trends), cross-sensor physical relationships, and physics-informed residuals can improve AeroPulse-X health/fault diagnosis over the Part 2 baseline.

### Production Decision: **KEEP THE CURRENT PRODUCTION MODEL (E0 Baseline HistGradientBoosting)**

| Evaluation Dimension | Production Baseline (E0 HGB) | Best Alternative (E3 HGB) | Delta | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8919 | **0.8991** | +0.0072 (+0.72%) | Marginal gain |
| **Balanced Accuracy** | 0.8767 | **0.8802** | +0.0035 (+0.35%) | Marginal gain |
| **Macro-F1** | **0.8518** | 0.8479 | -0.0039 (-0.46%) | **Baseline Superior** |
| **Critical Recall** | **0.9131** | 0.9087 | -0.0044 (-0.48%) | **Baseline Superior** |
| **Critical Precision** | **0.7078** | 0.6599 | -0.0479 (-6.77%) | **Baseline Superior** |
| **Critical F1** | **0.7974** | 0.7646 | -0.0328 (-4.11%) | **Baseline Superior** |
| **Inference Latency** | **10.79 $\mu$s/sample** | 11.45 $\mu$s/sample | +0.66 $\mu$s (+6.1%) | **Baseline Faster** |
| **Model Size** | **939.7 KB** | 1027.9 KB | +88.2 KB (+9.4%) | **Baseline Leaner** |
| **Feature Dimensionality** | **15 features** | 56 features | +41 features (+273%) | **Baseline Leaner** |
| **Anti-Leakage Risk** | Zero | Low (Train-only ref) | Moderate complexity | **Baseline Simpler** |

### Acceptance Rule Assessment:
Per the governing Acceptance Rule:
> *"Do not replace the current production model merely because raw accuracy increases. A candidate must be evaluated for: real-flight generalization, critical recall, balanced accuracy, Macro-F1, latency, model size, leakage, physical defensibility. If no candidate clearly improves the real-flight operating profile: KEEP THE CURRENT PRODUCTION MODEL."*

While Experiment E3 (Baseline + Physics Residuals) achieved a marginal +0.72% raw accuracy increase, it caused **degradation in Critical Recall (91.31% $\rightarrow$ 90.87%)**, **degradation in Critical Precision (70.78% $\rightarrow$ 65.99%)**, and a **drop in overall Macro-F1 (0.8518 $\rightarrow$ 0.8479)**, while requiring 3.7$\times$ more features. Therefore, the baseline production model (`models/aces_health.joblib`) is maintained as the production standard.

---

## 2. Immutable Baseline Benchmark (E0)

### Real ACES Telemetry Baseline (Held-Out Test Flights 191, 225, 235; 30,061 samples):
- **Model Architecture:** `HistGradientBoostingClassifier` (scikit-learn 1.8.0)
- **Features (15):** `Engine_RPM`, `EGT1`, `EGT2`, `EGT3`, `CHT`, `Fuel_Flow`, `Oil_Temp`, `Oil_Pressure`, `Battery_Voltage`, `Battery_Current`, `Alternator_Temp`, `EFI_Fuel_Temp`, `EFI_Water_Temp`, `MAP_Injector`, `Operating_State`.
- **Accuracy:** `0.8919`
- **Balanced Accuracy:** `0.8767`
- **Macro-F1:** `0.8518`
- **Weighted-F1:** `0.8927`
- **Critical Precision:** `0.7078`
- **Critical Recall:** `0.9131`
- **Critical F1:** `0.7974`
- **Critical False Negative Rate:** `0.0869`
- **Per-Class F1:**
  - `Critical`: 0.7974
  - `Normal`: 0.9358
  - `Warning`: 0.8765
  - `Watch`: 0.7976
- **Confusion Matrix (Labels: Critical, Normal, Warning, Watch):**
  ```
  [[  620,     0,    59,     0],
   [    0, 17308,     0,  1300],
   [  256,     2,  3108,   274],
   [    0,  1073,   285,  5776]]
  ```
- **Inference Latency:** `10.79 μs/sample` (tested over 5,000 samples)
- **Serialized Model Size:** `939.7 KB`

---

## 3. Experiment 1: Causal Temporal Features

### Physical Justification & Time Constants (1 Hz Telemetry):
1. **Engine_RPM:**
   - $dX/dt$ (`RPM_dt`): Crankshaft rotational acceleration ($\text{RPM}/\text{s}$). Fast indicator of sudden combustion power loss, governor hunting, or pilot throttle excursion.
   - $d^2X/dt^2$ (`RPM_d2t`): Angular jerk ($\text{RPM}/\text{s}^2$). Differentiates smooth pilot commands from violent rotational deceleration during cylinder misfire.
   - 5-sample trend (`RPM_trend_5s`): $X_t - X_{t-5}$. Distinguishes continuous climb acceleration from high-frequency shaft jitter.
2. **MAP_Injector (Manifold Absolute Pressure):**
   - $dX/dt$ (`MAP_dt`): Intake manifold pressure rate ($\text{inHg}/\text{s}$). Fast response to throttle plate position and turbocharger wastegate spooling.
   - $d^2X/dt^2$ (`MAP_d2t`): Manifold pressure shock. Detects turbo compressor surge or sudden induction restriction.
   - 5-sample trend (`MAP_trend_5s`): Short-term manifold charging rate.
3. **Fuel_Flow & Oil_Pressure:**
   - Fast hydraulic response ($dX/dt$ and 5s trend). Detects fuel pump cavitation, filter clogging, or pressure relief valve bypass.
4. **Battery_Voltage & Battery_Current:**
   - Electrical bus dynamics ($dX/dt$ and 5s trend). Fast transients capture load switching and alternator regulator dropouts.
5. **EGT1, EGT2, EGT3 (Combustion Exhaust Gas Temperature):**
   - Thermocouple lag time constant $\tau \approx 3\text{--}5\text{ s}$.
   - Evaluated: $dX/dt$, 5-sample trend, and 10-sample trend. Captures rapid thermal exhaust spikes preceding cylinder head conduction.
6. **CHT, Oil_Temp, Alternator_Temp (Thermal Inertia Masses):**
   - Cylinder head and oil reservoir thermal time constants $\tau \approx 15\text{--}60\text{ s}$.
   - Evaluated: $dX/dt$, 10-sample trend, and 20-sample trend ($X_t - X_{t-20}$). Filters out 1-Hz sensor jitter to observe genuine heat accumulation.

### Causality & Flight Boundary Rules:
- **Strictly Causal:** Features at step $t$ depend exclusively on observations $k \le t$. Zero future observations ($k > t$) are accessible.
- **Flight Boundary Isolation:** At $t=0$ of each flight trajectory, $dX/dt = 0$, $d^2X/dt^2 = 0$, and trends $= 0$. Lag operations never span across flight IDs.

---

## 4. Experiment 2: Cross-Sensor Interaction Features

All evaluated cross-sensor features have documented physical rationales:

| Feature Name | Mathematical Formula | Physical Rationale |
| :--- | :--- | :--- |
| `EGT_spread` | $\max(EGT_1..3) - \min(EGT_1..3)$ | Measures inter-cylinder thermal balance. Spreads $>100^\circ\text{F}$ pinpoint localized fuel injector clogging, induction air leaks, or single-cylinder valve seating degradation. |
| `EGT_mean` | $\frac{1}{3}\sum_{i=1}^3 EGT_i$ | Global exhaust thermal energy. Reflects overall air-fuel mixture stoichiometry (equivalence ratio $\Phi$). |
| `EGT_max_abs_deviation` | $\max_i |EGT_i - \text{EGT Mean}|$ | Quantifies peak individual cylinder imbalance (identifies isolated lean or misfiring cylinders). |
| `RPM_MAP_ratio` | $\text{RPM} / (\text{MAP} + \epsilon)$ | Engine induction efficiency. At cruise, propeller pitch and manifold pressure maintain a bounded ratio; excursions indicate governor failure or induction leak. |
| `Fuel_Flow_per_RPM` | $\text{Fuel\_Flow} / (\text{RPM} + \epsilon)$ | Fuel delivery per crankshaft revolution ($mg/\text{rev}$). Directly reflects cycle fueling density; detects injector over-fueling or fuel delivery starvation. |
| `Oil_Pressure_per_RPM` | $\text{Oil\_Pressure} / (\text{RPM} + \epsilon)$ | Positive displacement pump delivery. A declining ratio under operating temperature is the primary indicator of hydrodynamic bearing clearance wear. |
| `CHT_Oil_Temp_delta` | $\text{CHT} - \text{Oil\_Temp}$ | Head fin convective cooling vs crankcase oil cooler heat rejection. Isolates cowl baffle leaks from oil cooler blockage. |
| `EGT_to_CHT_ratio` | $\text{EGT\_mean} / (\text{CHT} + \epsilon)$ | Combustion flame heat vs cylinder metal conduction. Late ignition elevates EGT while cooling CHT; severe detonation dramatically elevates CHT while dropping EGT. |
| `Electrical_Power_est` | $\text{Battery\_Voltage} \times \text{Battery\_Current}$ | Net battery power. Negative power indicates battery discharging into bus (alternator failure); positive indicates battery charging. |
| `Alternator_Thermal_Stress` | $\text{Alternator\_Temp} - \text{Ambient\_Temp}$ | Stator winding temperature rise over ambient air temperature. Separates electrical ohmic heating from atmospheric altitude lapse. |

---

## 5. Experiment 3: Physics-Informed Residuals

Three distinct residual categories were implemented with strict unit alignment:

1. **Healthy-Reference Residuals:**
   $$R_{\text{healthy}}(C) = X(C) - \text{Median}_{\text{train}}(C \mid \text{Operating\_State})$$
   $$Z_{\text{healthy}}(C) = \frac{X(C) - \text{Median}_{\text{train}}(C \mid \text{Operating\_State})}{\text{Std}_{\text{train}}(C \mid \text{Operating\_State})}$$
   - Computed for all 14 core channels.
   - **Anti-Leakage Rule:** Medians and standard deviations were fitted **strictly on training flights** (`Health_State == 'Normal'`). Zero test data influenced the reference parameters.

2. **Engine-Model Residuals (Thermodynamic Reduced-Order Otto Cycle):**
   $$R_{\text{physics}}(C) = X(C) - \hat{X}_{\text{engine\_model}}(C)$$
   - Evaluated on variables predicted by `ReducedOrderPistonEngine`:
     - $\text{MAP}$ (inHg): Model predicted intake manifold pressure. Units match ($101.325\text{ kPa} \rightarrow 29.92\text{ inHg}$).
     - $\text{CHT}$ ($^\circ\text{F}$): Model predicted cylinder head temperature. Units match ($^\circ\text{F}$).
     - $\text{EGT}$ ($^\circ\text{F}$): Model predicted exhaust gas temperature. Units match ($^\circ\text{F}$).
     - $\text{Oil\_Pressure}$ ($\text{psi}$): Model predicted hydrodynamic oil pressure. Units match ($\text{psi}$).
     - $\text{Oil\_Temp}$ ($^\circ\text{F}$): Model predicted sump oil temperature. Units match ($^\circ\text{F}$).
     - $\text{Fuel\_Flow}$ ($\text{L/h}$): Model predicted fuel consumption rate. Units match ($\text{L/h}$).
   - **Vibration Exclusion:** ACES flight telemetry lacks an accelerometer channel. In accordance with instructions, vibration residuals were excluded from ACES evaluation to avoid synthetic hallucinations.

3. **Paired-Digital-Twin Residuals:**
   $$R_{\text{twin}}(C) = 0.50 \cdot R_{\text{healthy}}(C) + 0.50 \cdot R_{\text{physics}}(C)$$
   $$\text{Twin\_Residual\_RMS} = \sqrt{\frac{1}{K}\sum_{k=1}^K \left(\frac{R_{\text{twin}}(k)}{\sigma_k}\right)^2}$$

---

## 6. Experiment 4: Operating-Phase Conditioning Analysis

Performance across verified ground-truth flight phases was analyzed to examine whether phase conditioning reduces false alarms during flight transitions:

### Real ACES Test Telemetry Distribution & Phase Performance (Baseline E0):
| Operating State | Sample Count | Class Distribution | Accuracy | Balanced Accuracy | False Alarm Rate (Normal $\rightarrow$ Non-Normal) | Critical Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`CRUISE_LOW`** | 8,654 | Normal: 5175, Watch: 1943, Warning: 1536, Critical: 0 | 0.9301 | 0.9176 | 5.31% (275 / 5175) | N/A (0 samples) |
| **`CRUISE`** | 11,217 | Normal: 7315, Warning: 1951, Watch: 1272, Critical: 679 | 0.9197 | 0.8573 | **1.98%** (145 / 7315) | **91.31%** (620 / 679) |
| **`HIGH`** | 10,190 | Normal: 6118, Watch: 3919, Warning: 153, Critical: 0 | 0.8289 | 0.7782 | 14.38% (880 / 6118) | N/A (0 samples) |

### Transition Dynamics & False Alarm Findings:
1. **Steady Cruise Regimes:**
   During stable `CRUISE`, the baseline model achieves an exceptionally low false alarm rate of **1.98%**, with **91.31% Critical Recall**.
2. **Transition into `HIGH` Power Regime:**
   When transitioning from `CRUISE_LOW` to `HIGH`, the engine experiences large transient increases in RPM (up to 5800 RPM) and MAP ($>35\text{ inHg}$). Without operating phase conditioning, these elevated readings mimic "Watch" state deviations compared to global cruise medians.
3. **Benefit of Conditioning:**
   Including `Operating_State` as a conditioning variable allows decision trees to isolate high-power regimes, preserving an 85.6% true-negative rate in `HIGH` despite severe operating parameter shifts. Causal derivatives ($dX/dt$) verify that the transition rate matches commanded autopilot pitch/throttle rates rather than uncommanded mechanical failure.

---

## 7. Master Experiment Matrix (E0 to E6 Across 6 Model Families)

All 42 model-experiment combinations were evaluated against the identical held-out ACES test set (30,061 samples across flights 191, 225, 235):

| Exp | Features | Model | ACES Acc | Balanced Acc | Macro-F1 | Crit Recall | Crit Precision | Crit F1 | Latency | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E0** | 15 | **HistGradientBoosting** | 0.8919 | **0.8767** | **0.8518** | **0.9131** | **0.7078** | **0.7974** | 10.8 $\mu$s | 939.7 KB |
| E0 | 15 | ExtraTrees | 0.8836 | 0.8237 | 0.8092 | 0.7555 | 0.6975 | 0.7251 | 5.1 $\mu$s | 10.9 MB |
| E0 | 15 | RandomForest | 0.8751 | 0.8375 | 0.8059 | 0.8940 | 0.5899 | 0.7108 | 5.2 $\mu$s | 9.9 MB |
| E0 | 15 | XGBoost | 0.8740 | 0.8054 | 0.8259 | 0.7585 | 0.8175 | 0.7869 | 1.7 $\mu$s | 743.4 KB |
| E0 | 15 | LightGBM | 0.8804 | 0.8660 | 0.8288 | 0.9381 | 0.6402 | 0.7611 | 3.8 $\mu$s | 854.9 KB |
| E0 | 15 | CatBoost | 0.8680 | 0.8582 | 0.7895 | 0.9912 | 0.5060 | 0.6700 | 0.5 $\mu$s | 205.3 KB |
| **E1** | 46 | HistGradientBoosting | 0.8900 | 0.8583 | 0.8434 | 0.8424 | 0.7287 | 0.7814 | 12.4 $\mu$s | 967.4 KB |
| E1 | 46 | ExtraTrees | 0.8713 | 0.8138 | 0.7810 | 0.7747 | 0.6123 | 0.6840 | 5.3 $\mu$s | 9.1 MB |
| E1 | 46 | RandomForest | 0.8851 | 0.8252 | 0.8175 | 0.7599 | 0.7207 | 0.7398 | 5.2 $\mu$s | 10.7 MB |
| E1 | 46 | XGBoost | 0.8767 | 0.8125 | 0.8317 | 0.7599 | 0.8309 | 0.7938 | 1.3 $\mu$s | 765.0 KB |
| E1 | 46 | LightGBM | 0.8792 | 0.8553 | 0.8219 | 0.9131 | 0.6282 | 0.7443 | 5.0 $\mu$s | 864.9 KB |
| E1 | 46 | CatBoost | 0.8658 | 0.8484 | 0.7848 | 0.9617 | 0.5094 | 0.6660 | 0.6 $\mu$s | 220.4 KB |
| **E2** | 28 | HistGradientBoosting | 0.8665 | 0.8239 | 0.7781 | 0.8807 | 0.5283 | 0.6604 | 10.8 $\mu$s | 992.4 KB |
| E2 | 28 | ExtraTrees | 0.8770 | 0.8172 | 0.7844 | 0.7938 | 0.6557 | 0.7182 | 5.9 $\mu$s | 12.2 MB |
| E2 | 28 | RandomForest | 0.8644 | 0.8025 | 0.7790 | 0.7923 | 0.5699 | 0.6630 | 5.1 $\mu$s | 9.8 MB |
| E2 | 28 | XGBoost | 0.8675 | 0.8069 | 0.8100 | 0.7968 | 0.7535 | 0.7745 | 1.2 $\mu$s | 761.3 KB |
| E2 | 28 | LightGBM | 0.8705 | 0.8378 | 0.7903 | 0.9161 | 0.5179 | 0.6617 | 4.1 $\mu$s | 879.3 KB |
| E2 | 28 | CatBoost | 0.8627 | 0.8368 | 0.7742 | 0.9543 | 0.4825 | 0.6409 | 0.6 $\mu$s | 212.1 KB |
| **E3** | 56 | **HistGradientBoosting** | **0.8991** | **0.8802** | 0.8479 | 0.9087 | 0.6599 | 0.7646 | 11.4 $\mu$s | 1027.9 KB |
| E3 | 56 | ExtraTrees | 0.8900 | 0.8426 | 0.8221 | 0.8277 | 0.6401 | 0.7219 | 5.2 $\mu$s | 11.8 MB |
| E3 | 56 | RandomForest | 0.8709 | 0.8460 | 0.8010 | 0.9381 | 0.5563 | 0.6985 | 5.3 $\mu$s | 8.2 MB |
| E3 | 56 | XGBoost | 0.8908 | 0.8417 | **0.8513** | 0.8071 | **0.8216** | **0.8143** | 1.2 $\mu$s | 794.4 KB |
| E3 | 56 | LightGBM | 0.8963 | 0.8741 | 0.8363 | 0.9190 | 0.6124 | 0.7350 | 4.4 $\mu$s | 895.4 KB |
| E3 | 56 | CatBoost | 0.8670 | 0.8501 | 0.7889 | 0.9735 | 0.4918 | 0.6535 | 0.6 $\mu$s | 207.8 KB |
| **E4** | 59 | HistGradientBoosting | 0.8750 | 0.8329 | 0.7973 | 0.8513 | 0.5758 | 0.6873 | 18.3 $\mu$s | 1003.6 KB |
| E4 | 59 | ExtraTrees | 0.8699 | 0.8069 | 0.7752 | 0.7776 | 0.6027 | 0.6791 | 5.3 $\mu$s | 11.2 MB |
| E4 | 59 | RandomForest | 0.8674 | 0.8018 | 0.7889 | 0.7644 | 0.6384 | 0.6958 | 5.5 $\mu$s | 10.4 MB |
| E4 | 59 | XGBoost | 0.8603 | 0.7992 | 0.7865 | 0.8189 | 0.6691 | 0.7364 | 1.2 $\mu$s | 778.6 KB |
| E4 | 59 | LightGBM | 0.8625 | 0.8259 | 0.7794 | 0.8954 | 0.4903 | 0.6337 | 4.9 $\mu$s | 884.2 KB |
| E4 | 59 | CatBoost | 0.8646 | 0.8347 | 0.7796 | 0.9308 | 0.5040 | 0.6539 | 0.6 $\mu$s | 215.8 KB |
| **E5** | 87 | HistGradientBoosting | 0.8970 | 0.8673 | 0.8499 | 0.8586 | 0.7084 | 0.7763 | 12.9 $\mu$s | 1039.6 KB |
| E5 | 87 | ExtraTrees | 0.8852 | 0.8299 | 0.8073 | 0.7997 | 0.6515 | 0.7179 | 5.4 $\mu$s | 11.7 MB |
| E5 | 87 | RandomForest | 0.8716 | 0.8453 | 0.7978 | 0.9499 | 0.5434 | 0.6913 | 5.4 $\mu$s | 8.2 MB |
| E5 | 87 | XGBoost | 0.8861 | 0.8499 | 0.8287 | 0.8778 | 0.6464 | 0.7445 | 2.3 $\mu$s | 801.9 KB |
| E5 | 87 | LightGBM | 0.8961 | 0.8721 | 0.8371 | 0.9102 | 0.6242 | 0.7406 | 6.5 $\mu$s | 897.9 KB |
| E5 | 87 | CatBoost | 0.8776 | 0.8628 | 0.7973 | 0.9750 | 0.4951 | 0.6567 | 0.6 $\mu$s | 209.4 KB |
| **E6** | 100 | HistGradientBoosting | 0.8907 | 0.8518 | 0.8199 | 0.8527 | 0.5981 | 0.7031 | 11.7 $\mu$s | 1060.4 KB |
| E6 | 100 | ExtraTrees | 0.8852 | 0.8418 | 0.8069 | 0.8630 | 0.6094 | 0.7145 | 7.8 $\mu$s | 12.1 MB |
| E6 | 100 | RandomForest | 0.8649 | 0.8399 | 0.7871 | 0.9529 | 0.5168 | 0.6701 | 5.5 $\mu$s | 8.1 MB |
| E6 | 100 | XGBoost | 0.8837 | 0.8342 | 0.8351 | 0.8100 | 0.7422 | 0.7746 | 1.4 $\mu$s | 800.3 KB |
| E6 | 100 | LightGBM | 0.8823 | 0.8531 | 0.7950 | 0.9337 | 0.5000 | 0.6513 | 5.7 $\mu$s | 902.5 KB |
| E6 | 100 | CatBoost | 0.8752 | 0.8572 | 0.7928 | 0.9779 | 0.4840 | 0.6475 | 0.8 $\mu$s | 215.9 KB |

---

## 8. Real ACES Telemetry vs Synthetic Results

In strict adherence to `DATA_RESOURCE_SPEC.md` and Rule 1 (anti-concatenation), real-flight telemetry (`REAL_ACES`) and physics simulation (`AEROPULSE_SYNTHETIC`) results are maintained strictly separated:

| Model Architecture | REAL_ACES Accuracy (E0) | REAL_ACES Macro-F1 (E0) | REAL_ACES Crit-Recall (E0) | SYNTHETIC Accuracy (E0) | SYNTHETIC Macro-F1 (E0) | SYNTHETIC Crit-Recall (E0) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | 0.8919 | 0.8518 | 0.9131 | **0.9959** | **0.9838** | **0.9667** |
| **LightGBM** | 0.8804 | 0.8288 | 0.9381 | **0.9938** | **0.9741** | **0.9333** |
| **CatBoost** | 0.8680 | 0.7895 | 0.9912 | **0.9959** | **0.9838** | **0.9667** |
| **XGBoost** | 0.8740 | 0.8259 | 0.7585 | **0.9959** | **0.9838** | **0.9667** |
| **RandomForest** | 0.8751 | 0.8059 | 0.8940 | **0.9938** | **0.9754** | **0.9667** |
| **ExtraTrees** | 0.8836 | 0.8092 | 0.7555 | **0.9918** | **0.9649** | **0.9000** |

### Domain Divergence Rationale:
1. **Mathematical Determinism of Synthetic Trajectories:**
   In `AEROPULSE_SYNTHETIC`, degradation paths follow deterministic Arrhenius and Bishop-Heywood differential equations with controlled Gaussian measurement noise. Consequently, gradient-boosted decision trees easily identify degradation stage boundaries, achieving $>99.3\%$ accuracy across all models.
2. **Real-Flight Atmospheric and Operational Variance:**
   In `REAL_ACES`, true operational flights experience unmeasured ambient humidity, turbulent gust loads, pilot trim adjustments, and high-altitude lapse rate variability. This introduces real-world aleatoric uncertainty, bounding real flight test accuracy to $\approx 89\text{--}90\%$.
3. **Validation Truth:**
   Reporting synthetic accuracy as proof of real-world flight performance is scientifically invalid. The true benchmark is real-flight generalization on held-out ACES flights 191, 225, and 235.

---

## 9. Adversarial Leakage & Causality Audit

An exhaustive forensic audit was conducted on all feature extraction routines:
1. **Target Leakage:**
   `Health_State`, `Degradation_Severity`, `fault_severity`, `true_RUL`, `true_failure_time`, and `degradation_stage` were strictly barred from feature vectors. The automated validator `audit_feature_names()` asserted zero prohibited fields across all 42 experiment configurations.
2. **Temporal Causality Audit:**
   Mutating future observations $t > t_0$ produced **identically zero change** in feature values at $t \le t_0$ ($\Delta = 0.000000$).
3. **Flight Boundary Isolation Audit:**
   Temporal diffs and rolling windows were strictly grouped by `Flight`. Between the final sample of Flight A and the initial sample of Flight B, $dX/dt = 0.000$ and $\text{trend}_{5s} = 0.000$, confirming zero cross-flight data bleeding.
4. **Reference Statistics Isolation:**
   Healthy reference medians and standard deviations were fitted strictly on the 11 training flights. Modifying test set values produced zero change in reference parameters.

---

## 10. Summary of Files Changed

| File | Change Summary |
| :--- | :--- |
| `app/feature_engineering.py` | [NEW] Clean, modular, tested pipeline for causal temporal derivatives, cross-sensor physical features, and physics-informed residual extraction with automated anti-leakage audits. |
| `scripts/benchmark_physics_temporal.py` | [NEW] High-performance benchmarking harness executing all 7 experiment configurations across 6 model families on real ACES and synthetic corpora with machine-readable report generation. |
| `tests/test_feature_engineering_physics_temporal.py` | [NEW] 6 automated regression tests covering temporal causality, flight boundary isolation, train-only residual references, unit consistency, forbidden feature exclusions, and feature reproducibility. |
| `reports/part3_benchmark_results.json` | [NEW] Machine-readable forensic results payload covering all 42 configurations and 12 required metrics. |
| `reports/part3_experiment_matrix.json` | [NEW] Machine-readable summary matrix comparing E0–E6 models. |
| `reports/part3_operating_phase_analysis.json` | [NEW] Machine-readable operating phase distribution and false alarm analysis. |
| `docs/PART3_PHYSICS_TEMPORAL_FEATURE_REPORT.md` | [NEW] Authoritative Part 3 engineering report. |
| `ACCURACY_PROGRESS.md` | Updated with Part 3 findings, benchmark metrics, and next phase definition. |
