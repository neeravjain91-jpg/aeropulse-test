# AeroPulse-X — Accuracy Improvement Progress

## CURRENT_PHASE
**Part 1/7 — System Understanding + Baseline** ✅ COMPLETE

## COMPLETED_PHASES
- [x] Part 1/7 — System Understanding + Baseline

---

## ARCHITECTURE PIPELINE MAP

```
UAV Platform (app/uav_platform_config.py)
  ↓ engine_id
Engine Profile (app/engine_config.py)
  ↓ EngineConfig dataclass
Engine Physics Model (app/engine_model.py → ReducedOrderPistonEngine)
  ↓ predict() → 14-channel telemetry dict
Data Engine / Trajectory Generator (app/data_engine.py → VirtualDataLabEngine)
  ↓ generate_*_trajectory() → List[CanonicalTelemetryPoint]
Degradation Model (app/degradation_model.py → ContinuousDegradationModel)
  ↓ apply(telemetry, state) → degraded telemetry
Simulator / Fault Injection (app/simulator.py → inject_fault, mission_adjust)
  ↓ modified telemetry dict
Digital Twin (app/digital_twin.py → ReferenceTwin)
  ↓ compare() → residuals, z-scores, expected values
Inference Pipeline (app/inference.py → AeroTwinAI.analyze)
  ├─ Model A: HGB Health Classifier (aces_health.joblib)
  ├─ Model B: TCN Residual Classifier (aces_tcn_residual.pt)
  ├─ Model D: Isolation Forest Anomaly (aces_anomaly.joblib)
  ├─ Model E: TCN Autoencoder Anomaly (aces_tcn_autoencoder.pt)
  ├─ Sensor Health Assessment (app/sensor_health.py)
  ├─ Hybrid Fusion Engine (app/fusion.py → 0.70 HGB + 0.30 TCN)
  ├─ Fault Advisory (app/advisory.py)
  └─ Health Index Calculation (heuristic formula)
      ↓
RUL Service (app/rul_service.py → RULService.estimate_rul)
  ↓ rul_hours, confidence, status
Risk Analytics (app/risk.py → mission_risk)
  ↓ score, level, MTBF, availability
Mission Intelligence (app/mission_intelligence.py → EventExtractor, TrajectorySummarizer)
  ↓ events, summary
LLM Report Service (app/llm_report_service.py → generate_mission_report)
  ↓ markdown/html report
Report Export (static/index.html → client-side download)
```

---

## PER-MODEL DETAIL

### Model A: HGB Health Classifier
| Item | Detail |
|------|--------|
| File | `models/aces_health.joblib` (964 KB) |
| Architecture | `HistGradientBoostingClassifier` (scikit-learn 1.8.0) |
| Features | 14 sensor channels + Operating_State (15 total) |
| Target | `Health_State` (4 classes: Normal, Watch, Warning, Critical) |
| Training Data | NASA ACES telemetry, 14 flights |
| Split | `GroupShuffleSplit` by flight ID, 20% test (3 held-out flights) |
| Train Flights | 11 flights (192, 193, 214, 216, 218, 220, 222, 224, 227, 237, 242) |
| Test Flights | 3 flights (191, 225, 235) |
| Leakage Exclusions | Robust_Anomaly_Score, Robust_Max_Deviation, Sensors_Above_2/3Sigma, *_rz features |
| Inference | `app/inference.py` L110-137 |
| Post-processing | Probability threshold overrides (Critical≥0.25 forced, Warning≥0.35 + Watch→Warning) |

### Model B: TCN Residual Classifier
| Item | Detail |
|------|--------|
| File | `models/aces_tcn_residual.pt` (89 KB) |
| Architecture | `PhysicsResidualTCN` (13 channels, 4 classes, [32,32,32] TCN) |
| Input | 30-step rolling window of physics-normalized residual vectors |
| Training | Same ACES flight split as Model A |
| Inference | `app/inference.py` L175-181, via `TemporalSequenceBuffer` |

### Model D: Isolation Forest Anomaly
| Item | Detail |
|------|--------|
| File | `models/aces_anomaly.joblib` (358 KB) |
| Architecture | `IsolationForest` (100 trees, 5% contamination) |
| Input | 14 sensor channels (point-level) |
| Training | Normal-only samples from ACES training flights |
| Inference | `app/inference.py` L142-146 |

### Model E: TCN Autoencoder
| Item | Detail |
|------|--------|
| File | `models/aces_tcn_autoencoder.pt` (40 KB) |
| Architecture | `TemporalTCNAutoencoder` (13 in, 8 latent, 32 hidden) |
| Input | Same 30-step window as Model B |
| Threshold | τ = 0.667 reconstruction error |
| Inference | `app/inference.py` L189-197 |

### RUL Service (No pre-trained model)
| Item | Detail |
|------|--------|
| File | `app/rul_service.py` (422 lines) |
| Method | Physics-stress weighted trend extrapolation |
| Inputs | health_index, elapsed_hours, altitude, ambient_c, throttle, engine_id |
| Trend | `app/degradation.py` → linear polyfit on last 20 health-index values |
| Validation Gap | **No target-domain run-to-failure data** (manifest explicitly states this) |
| Monotonic Guard | MAX_RUL_DROP_FRACTION = 0.25 per step; upward revision only on stress reduction |

### Supporting Models (Research Only)
- `cwru_vibration.joblib`: CWRU bearing vibration RF (not used in production pipeline)
- `marine_fault_research.joblib`: Marine engine RF (not used in production pipeline)

---

## BASELINE_METRICS

### Test Suite
- **373 passed, 0 failures** (pytest -q, 64.44s)

### Model A (HGB) — Held-Out Flight Test Set (30,061 samples)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8919 |
| Balanced Accuracy | 0.8767 |
| Macro F1 | 0.8518 |
| Weighted F1 | 0.8927 |
| 5-Fold Group CV Mean | 0.8926 ± 0.0405 |
| ECE (calibration) | 0.0230 |
| Brier Score | 0.1562 |

Per-class F1: Critical=0.797, Normal=0.936, Warning=0.876, Watch=0.798

### Model B (TCN) — Same Test Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.8817 |
| Balanced Accuracy | 0.8354 |
| Macro F1 | 0.8384 |

Per-flight weakness: Flight 235 → TCN balanced_accuracy=0.760, macro_f1=0.583

### Anomaly Detection — Real ACES Benchmark
| Model | AUROC | AUPRC | F1 | False Alarm Rate |
|-------|-------|-------|-----|-----------------|
| Model D (IsoForest) | 0.8725 | 0.7102 | 0.6272 | 6.73% |
| Model E (TCN AE) | 0.9683 | 0.8677 | 0.5776 | 23.33% |
| Hybrid (D+E) | 0.9598 | 0.8193 | 0.6904 | 5.79% |

### RUL
| Metric | Value |
|--------|-------|
| Method | Trend extrapolation (no ML model) |
| Validation | Synthetic-only (self-fulfilling — see ROOT_CAUSES) |
| Domain Data | **None** (manifest: "target-domain run-to-failure trajectories are not available") |

### Digital Twin
| Metric | Value |
|--------|-------|
| Reference | `healthy_reference.json` (per-state medians + stds from ACES training flights) |
| Expected Calc | 50% contextual reference + 50% physics model (non-simulation mode) |
| Simulation Mode | 100% physics model with 3% relative std |

---

## ROOT_CAUSES_FOUND

### RC-1: Ground-Truth Degradation_Severity Leaks Into Health Index (CRITICAL)
- **Location**: `app/inference.py` L234-236
- **Mechanism**: `degradation_severity = telemetry.get("Degradation_Severity", 0.0)` → `degradation_penalty = severity * 45.0` → directly subtracted from base_health_index
- **Impact**: Simulator-generated ground truth label directly controls the final health state. The health_index is NOT purely inferred from observables.
- **Downstream**: Health index drives RUL estimation, risk score, and report generation. All inherit this leakage.

### RC-2: Double Degradation Penalty in Replay Pipeline (HIGH)
- **Location**: `app/replay.py` `_trajectory_health()` and `run_replay()`
- **Mechanism**: `point["Degradation_Severity"] = fault_severity` is set BEFORE `ai.analyze()`, which subtracts `45.0 * severity`. Then `_trajectory_health()` subtracts ANOTHER `55.0 * severity`. Total: up to `100.0 * severity` deducted from ground truth.
- **Impact**: Artificially forces health to zero at severity=1.0 regardless of actual sensor readings.

### RC-3: Self-Fulfilling RUL Validation (HIGH)
- **Location**: `app/rul_validation.py`
- **Mechanism**: Synthetic test trajectories are generated by the same `ContinuousDegradationModel` and `ReducedOrderPistonEngine` that the RUL model uses. The RF regressor is trained and tested on data from identical physics rules.
- **Impact**: Reported RUL metrics (MAE, RMSE, coverage) are tautological — they measure agreement with the simulator, not real-world accuracy.
- **Also**: 90% prediction interval coverage is computed from test-set error percentiles, guaranteeing ~90% by construction.

### RC-4: Vibration Model Inconsistency (MEDIUM)
- **Location**: `app/engine_model.py` L209-210 vs `app/data_engine.py` L101-102
- **Mechanism**: `engine_model.predict()` returns vibration via `0.85 + 0.75*(rpm/NOMINAL_RPM)^2 + 0.45*(load-0.5) + misfire_vib` ≈ 2.7g at cruise. `data_engine._calc_physics_point()` overrides with calibrated formula `1.05 + 0.25*(rpm/5800)^2 + 0.12*throttle` ≈ 1.15g.
- **Impact**: Digital twin uses `engine_model.predict()` for expected values (high vibration baseline), while telemetry from data_engine uses calibrated formula (low vibration). This creates systematic bias in vibration residuals.

### RC-5: ISA Atmosphere Formula Error in simulator.py (MEDIUM)
- **Location**: `app/simulator.py` L133
- **Mechanism**: `p_amb_ratio = math.pow(max(0.1, t_amb_k / (273.15 + amb_c)), 5.255)`. Standard ISA uses sea-level standard temperature (288.15 K) in denominator, not ambient ground temperature.
- **Impact**: At non-standard ambient temperatures, the pressure/density calculations deviate from ISA standard, affecting MAP, fuel flow, and efficiency calculations.

### RC-6: Unit Conversion Inconsistencies in data_engine (LOW-MEDIUM)
- **Location**: `app/data_engine.py` L~101-110
- **Mechanism**: EGT conversion from °F to °C only if value > 1000°F; CHT conversion only if > 150°F. This means some values may remain in Fahrenheit while the digital twin/reference expects Celsius.
- **Impact**: Potential systematic errors in EGT/CHT residual calculations when values are near the conversion thresholds.

### RC-7: Hardcoded C-MAPSS Proxy Metrics (LOW)
- **Location**: `app/rul_validation.py` `_evaluate_cmapss_proxy()`
- **Mechanism**: Values like `reported_mae_cycles: 14.85` and `reported_rmse_cycles: 18.42` are static constants, not live evaluations.
- **Impact**: Misleading if referenced as "validated" metrics.

### RC-8: causal_coupling_passed Hardcoded to True (LOW)
- **Location**: `app/data_validator.py` L191
- **Mechanism**: `causal_coupling_passed = True` without any dynamic causal test.
- **Impact**: Data quality reports falsely claim causal coupling validation has passed.

---

## LEAKAGE AUDIT

### Critical Leakage Paths Found

| # | Type | Source | Sink | Severity |
|---|------|--------|------|----------|
| 1 | Ground truth → prediction | `Degradation_Severity` in telemetry | `inference.py` health_index formula | **CRITICAL** |
| 2 | Ground truth → prediction | `Degradation_Severity` in telemetry + `_trajectory_health()` | Double penalty in replay | **HIGH** |
| 3 | Circular evaluation | Synthetic generator = evaluation target | RUL validation metrics | **HIGH** |

### Leakage NOT Found (Verified Clean)
- Train/test flight split: Disjoint by flight ID (GroupShuffleSplit) ✅
- No random point-level split (GroupKFold for CV) ✅
- Leakage exclusions documented: Robust_Anomaly_Score, Robust_Max_Deviation, *_rz features excluded from HGB features ✅
- TCN autoencoder trained on normal-only windows from training flights ✅
- RUL service uses per-engine state isolation (no cross-engine leakage) ✅
- RUL monotonic guard prevents unexplained upward jumps ✅
- `true_RUL` and `true_failure_time` are NOT used as features in any production ML model ✅

---

## FILES_CHANGED
None yet (Part 1 is observation only)

## TESTS_RUN
- `pytest -q` → 373 passed, 0 failures, 64.44s

## RESULTS
Part 1/7 deliverables complete:
1. ✅ Git status checked: clean working tree on `main` at `a9fb7a4`
2. ✅ Full architecture pipeline mapped (17 modules, 5 ML models, 1 physics engine, 1 fusion engine)
3. ✅ Per-model detail: inputs, outputs, ground truth, training, inference, post-processing
4. ✅ Baseline metrics established for all models
5. ✅ Leakage audit complete: 3 critical/high issues found, 7 clean paths verified
6. ✅ Root causes documented: 8 accuracy issues identified and ranked

## NEXT_PHASE
**Part 2/7 — Fix Ground-Truth Leakage (RC-1, RC-2)**

Priority order for fixes:
1. RC-1: Remove `Degradation_Severity` from health index calculation in `inference.py`
2. RC-2: Fix double degradation penalty in `replay.py`
3. RC-4: Harmonize vibration model between engine_model and data_engine
4. RC-5: Fix ISA atmosphere formula in simulator.py
5. RC-6: Standardize EGT/CHT unit conversions in data_engine
6. RC-3: Disclose synthetic RUL validation limitations clearly
7. RC-7/RC-8: Fix hardcoded proxy metrics and causal coupling

## IMPORTANT_DECISIONS
- Do NOT modify the health index formula in a way that removes all degradation sensitivity — the formula should respond to degradation, but via observable sensor deviations, not via ground-truth labels.
- RUL validation gap (no target-domain run-to-failure data) is an inherent data limitation, not a code bug. It should be disclosed, not "fixed."

## KNOWN_LIMITATIONS
1. All ML models are trained on NASA ACES telemetry from a single aircraft (Ikhana/Altus II). Generalization to other MALE UAV platforms is unvalidated.
2. RUL is a method demonstrator only — no target-domain run-to-failure trajectories exist.
3. TCN performance degrades on flight 235 (macro F1 drops to 0.583, balanced accuracy 0.760).
4. Anomaly detection hybrid has 5.79% false alarm rate on ACES benchmark.
5. Physics engine model is reduced-order (not validated against test-cell data).
6. Vibration model is heuristic calibration, not based on measured accelerometer data.
