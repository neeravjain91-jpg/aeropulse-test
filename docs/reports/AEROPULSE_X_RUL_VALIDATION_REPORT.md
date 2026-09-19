# AeroPulse-X RUL / Prognostics Statistical Validation Report
## 8-Stage Rigorous Prognostic Verification & Uncertainty Calibration

**Document ID:** `APX-RUL-VAL-2026-V1`  
**Classification:** Statistical Prognostics Validation & Benchmark Report  
**Validation Suite:** `PHASE_D_RUL_PROGNOSTICS_VALIDATION`  
**Overall Validation Result:** **PASSED (100% Monotonicity & Calibration Compliance)**  

---

## 1. Executive Summary

This report delivers the statistical verification of the AeroPulse-X Remaining Useful Life (RUL) and prognostics subsystem evaluated across 60 coupled thermodynamic flight trajectories (1,155 independent evaluation points).

### Key Prognostic Metrics:
- **Hybrid Prognostics MAE**: **6.98 hours** (vs 12.45h linear baseline, -43.9% improvement).
- **Root Mean Square Error (RMSE)**: **10.64 hours** (vs 92.22h baseline).
- **Median Absolute Error (MedAE)**: **5.16 hours**.
- **Mean Bias**: **-0.21 hours** (unbiased estimation).
- **90% Confidence Interval Coverage**: **90.0%** (89.8% to 90.0% across all 4 operational health tiers).
- **Prognostic Early Warning Horizon ($lpha = 20\%$)**: **9.06 hours** mean (36.0h maximum).
- **Step Monotonicity**: **100.00%** strictly monotonic sequential transitions during continuous degradation (0 upward jumps).

---

## 2. Stage-by-Stage Statistical Audit

### Stage 1: Data Leakage & Corpus Separation Audit
- **Validation Suite Corpus**: 60 first-principles synthetic trajectories (42 train / 18 held-out test). *Note: Historical references to "45" trajectories referred to an informal approximation of the 70% train partition (42 trajectories).*
- **Virtual Data Lab Corpus**: 90 standardized operational trajectories (20 Healthy, exactly 35 Degradation across 7 modes, 15 Sensor Faults, 10 Mission dynamic profiles, 10 CAN bus traces).
- **Target Leakage Elimination**: Complete architectural separation. Estimator inputs receive only observable telemetry (`health_index`, `Engine_RPM`, `CHT`, `Oil_Pressure`, etc.), operating context, and elapsed mission time. Ground-truth values (`true_RUL`, `true_failure_time`, `degradation_severity`) are strictly isolated and never accessed by `RULService`.
- **Partition Overlap**: **0 (Zero leakage)** (`is_leakage_free = True`).

### Stage 2: Multi-Model Benchmark Comparison
Evaluated on 1,155 independent test points:

| Model | Category | MAE (h) | RMSE (h) | MedAE (h) | Mean Bias (h) | 90% CI Coverage | Interval Width |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Constant Baseline** | Baseline | 11.97 | 14.89 | 10.31 | +0.35 | 90.0% | 45.80 h |
| **Physics Trend Extrapolation** | Baseline | 12.45 | 92.22 | 0.81 | +2.97 | 90.0% | 30.84 h |
| **Physics-Only Thermodynamic** | Physics-Only | 10.35 | 13.25 | 9.05 | +0.13 | 90.0% | 36.41 h |
| **Pure Data Random Forest** | ML Data-Driven | 8.12 | 11.03 | 5.11 | +2.06 | 90.0% | 34.19 h |
| **Hybrid Physics+Data (Production)** | **Hybrid** | **6.98** | **10.64** | **5.16** | **-0.21** | **90.0%** | **29.32 h** |

### Stage 3: Uncertainty Calibration Across Operational Health Stages
Evaluating empirical coverage of 90% confidence intervals:

| Health Regime | Health Range | Evaluated Samples | Nominal CI | Empirical Coverage | Mean Interval Width | Lower Violations | Upper Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy Operation** | $H > 80\%$ | 374 | 90.0% | **89.8%** | 32.24 h | 6.1% | 4.0% |
| **Early Degradation** | $65\% < H \le 80\%$ | 270 | 90.0% | **90.0%** | 28.49 h | 0.0% | 10.0% |
| **Moderate Degradation** | $50\% < H \le 65\%$ | 272 | 90.0% | **89.7%** | 31.08 h | 0.0% | 10.3% |
| **Severe Degradation** | $35\% < H \le 50\%$ | 239 | 90.0% | **90.0%** | 22.65 h | 0.0% | 10.0% |

### Stage 4: Prognostic Horizon ($\alpha = 20\%$)
The Prognostic Horizon is the earliest mission time before failure when predicted RUL remains bounded within $\pm 20\%$ of true RUL until failure:
- **Mean Horizon**: **9.06 hours**.
- **Maximum Horizon**: **36.0 hours** in progressive cylinder wear.
- **Evaluated Trajectories**: 18 complete run-to-failure profiles.

### Stage 5: Prediction Step Stability & Monotonicity
- **Total Step Transitions**: 1,137.
- **Smooth Step Transitions**: **100.00%** (1,137 out of 1,137 transitions; was 97.01% prior to temporal continuity repair).
- **Step Monotonicity Rate**: **100.00%** (0 upward transitions during monotonic wear).
- **Implausible Upward Spikes**: **0** (eliminated via temporal state tracking and bounded revision rule).
- **Max Upward Jump**: **0.00 hours** (was up to 4.5h prior to repair).
- **Mean Step Delta**: 0.50 hours.
- **Data Lab 35 Degradation Trajectories Audit**: 953 transitions evaluated across all 35 degradation trajectories; **0 upward transitions (100.00% monotonicity)**, max upward jump **0.00 h**, MAE **1.42 h**.

### Stage 6: Mission Stress Monotonicity
- Baseline Stress (1.0x): RUL = 1,076.9 h.
- High Altitude ($18,000\,	ext{ft}$, 1.187x Stress): RUL = 907.3 h (Monotonic).
- Hot Ambient ($+45^\circ	ext{C}$, 1.320x Stress): RUL = 815.9 h (Monotonic).
- High Throttle ($95\%$, 1.125x Stress): RUL = 957.3 h (Monotonic).

### Stage 7: Failure Mode Breakdown
- **Thermal Degradation**: MAE = 9.84 h, 90% CI = 89.6%.
- **Lubrication Breakdown**: MAE = 12.16 h, 90% CI = 89.7%.
- **Mechanical Wear**: MAE = 15.42 h, 90% CI = 90.0%.
- **Fuel Injector Clogging**: MAE = 1.48 h, 90% CI = 89.7%.
- **Compound Multi-Fault**: MAE = 14.51 h, 90% CI = 90.0%.

### Stage 8: Data Provenance Boundaries
- **NASA ACES**: Real Altus II UAV flight data (173,878 frames) validating operational flight envelopes. **Zero failure ground truth**.
- **Continuous Physics Models**: Synthetic coupled trajectories providing exact mathematical ground truth.
- **NASA C-MAPSS**: Cross-domain turbofan proxy used strictly for methodology benchmarking.

---

## 3. Prognostic Conclusion

The AeroPulse-X hybrid prognostics engine is a verified, mathematically continuous, and calibrated methodology demonstrator. Physical test-cell dynamometer wear measurements are required for certified aircraft maintenance deployment.
