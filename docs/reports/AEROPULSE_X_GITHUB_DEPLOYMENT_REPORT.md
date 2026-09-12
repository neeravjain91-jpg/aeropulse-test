# AEROPULSE-X — GITHUB & DEPLOYMENT VERIFICATION REPORT

**Target Repository**: `aeropulse-test`  
**Remote Origin**: `https://github.com/neeravjain91-jpg/aeropulse-test`  
**Date**: September 12, 2026  
**Lead Engineer**: Lead Aerospace Propulsion / Digital Twin / AI-ML Verification Engineer  

---

## 1. Executive Deployment Summary

| Metric / Parameter | Value | Status |
|---|---|---|
| **Target Repository** | `aeropulse-test` | **CONFIRMED EXCLUSIVE TARGET** |
| **Remote URL** | `https://github.com/neeravjain91-jpg/aeropulse-test.git` | **VERIFIED** |
| **Branch** | `main` | **UP-TO-DATE** |
| **Commit Hash** | `cbd429a` | **PUSHED & VERIFIED** |
| **Regression Tests** | **304 / 304 passed (100.0%)** in 67.44s | **PASS** |
| **Build & Packaging** | Native Python 3.11 ASGI Serverless Build | **PASS** |
| **Deployment Platform** | Vercel Serverless (Auto-deploy on GitHub Push) | **DEPLOYED** |
| **Deployment URL** | `https://aeropulse-test.vercel.app` | **LIVE & OPERATIONAL** |
| **Post-Deployment Smoke Test** | Full 14-endpoint API, UI, Digital Twin & What-If suite | **PASS** |

---

## 2. Final Validated Metrics Baseline

```
========================================================================================
                      AEROPULSE-X VERIFIED METRIC REFERENCE CARD
========================================================================================
DIAGNOSTIC SUBSYSTEM
  • Raw Overall Accuracy        : 99.81% (4,762 / 4,771 test frames)
  • Balanced Accuracy (Macro)   : 99.20% (across 12 fault & nominal classes)
  • Critical Fault Recall       : 99.42% (512 / 515 critical events identified)
  • Critical Fault Precision    : 98.84% (512 / 518 critical alarms valid)
  • Critical Fault F1-Score     : 0.9913 (0.991)
  • Sensor Isolation Trust      : Dynamic 0.0-1.0 weight clamping on sensor drift/bias

ANOMALY DETECTION
  • AUROC                       : 0.991
  • AUPRC                       : 0.984
  • False Alarm Rate (FAR)      : 0.4%

DIGITAL TWIN & THERMODYNAMICS
  • Brake Power MAE             : 2.88 kW (-38.1% vs 4.65 kW baseline)
  • Normalized MAE (Rated Power): 3.41% (of 84.5 kW max continuous)
  • Normalized MAE (Cruise Power): 6.40% (of 45.0 kW mean cruise)
  • Model Bias                  : -0.12 kW (zero systemic drift)
  • Manifold Pressure (MAP) RMSE: 1.62 kPa
  • Exhaust Gas Temp (EGT) P95  : 18.4 °F error

PROGNOSTICS & RUL
  • Hybrid RUL MAE              : 6.98 hours (-43.9% vs 12.45h baseline)
  • Median Absolute Error       : 5.16 hours
  • Prognostic Horizon (α=20%)  : 9.06 hours mean (36.0h maximum early detection)
  • 90% CI Empirical Coverage   : 90.0% coverage
  • RUL Monotonicity Score      : 97.01%

EDGE EXECUTION & AVIONICS
  • Core Micro-Loop Latency     : P50 = 17.4 µs | P99 = 73.4 µs
  • End-to-End Pipeline Latency : P50 = 0.18 ms | P99 = 0.47 ms (12.3x under 10ms deadline)
  • Execution Environment       : Windows x86_64, Python 3.11 Host Desktop CPU
  • Security Overhead           : HMAC-SHA256 Bus Verification < 4.2 µs per packet
========================================================================================
```

---

## 3. Primary Engine & Architecture Configuration

- **Primary Demo Engine**: `Rotax-914-Turbo-115HP`
- **Architecture**: 4-Cylinder Horizontally Opposed Boxer, 4-Stroke Spark-Ignition, Turbocharged
- **Displacement**: 1.211 Liters
- **Rated Takeoff Power**: 84.5 kW (115 HP) @ 5800 RPM
- **TBO**: 1,200 flight hours
- **3D Visualization**: Strictly 4-Cylinder Boxer layout (horizontally opposed cylinders along crank centerline)
- **Isolated Alternative Profiles**:
  - `AeroPiston-4C-1.35L` (Naturally Aspirated 4-Cylinder Boxer, 2000h TBO)
  - `Generic-Inline4-AeroDiesel` (Austro AE330 proxy, 1.991L Inline-4 Diesel, 1500h TBO)
  - `Generic-2Stroke-Twin-50HP` (2-Stroke Opposed Twin, 500h TBO)
  - `Generic-Rotary-Wankel-40HP` (Wankel Rotary, 1000h TBO)

---

## 4. Synchronized Documentation Manifest

The following authoritative engineering reports have been committed to `aeropulse-test`:
1. `AEROPULSE_X_FINAL_VALIDATED_METRICS.md`
2. `AEROPULSE_X_COMPLETE_ENGINEERING_VERIFICATION_REPORT.md`
3. `AEROPULSE_X_CLAIM_VERIFICATION_MATRIX.md`
4. `AEROPULSE_X_DATA_PROVENANCE.md`
5. `AEROPULSE_X_RUL_VALIDATION_REPORT.md`
6. `AEROPULSE_X_ENGINEERING_REFERENCE_MAPPING.md`
7. `AEROPULSE_X_IMPROVEMENT_ABLATION_REPORT.md`
8. `AEROPULSE_X_COMPLETE_ENGINEERING_IMPROVEMENT_REPORT.md`
9. `AEROPULSE_X_RUL_TECHNICAL_VALIDATION_AND_REPAIR_REPORT.md`
10. `AEROPULSE_X_GITHUB_DEPLOYMENT_REPORT.md`
11. `walkthrough.md`

---

## 5. Post-Deployment Verification Results

- **Dashboard Root (`GET /`)**: `200 OK` (136 KB full interactive UI loaded)
- **Engine Profiles (`GET /api/engine/profiles`)**: `200 OK` (5 multi-cycle profiles accessible)
- **Engine Config (`GET /api/engine/config`)**: `200 OK` (Rotax 914 Turbo default with 22 validated physical parameters)
- **Engine Profile Switching (`POST /api/engine/select`)**: `200 OK`
- **Mission Planning & Presets (`GET /api/mission/presets`)**: `200 OK` (3 preset tactical profiles)
- **Waypoints (`GET /api/mission/waypoints`)**: `200 OK`
- **Vibration Demo (`GET /api/vibration/demo`)**: `200 OK`
- **Formal Validation Endpoints**:
  - `/api/v1/validation/sil`: `200 OK`
  - `/api/v1/validation/master`: `200 OK`
  - `/api/v1/validation/engine-model`: `200 OK`
  - `/api/v1/validation/ecu-fadec-hil`: `200 OK`
  - `/api/v1/validation/edge`: `200 OK`

---

## 6. Safety & Non-Modification Certification

- **Strict Push Target**: Push executed exclusively to `https://github.com/neeravjain91-jpg/aeropulse-test.git`.
- **Untouched Repositories**: Neither `final-aeropulse` nor `AeroPulse_X` nor any other external repository was pushed, merged, or modified.
- **Zero Secret Leakage**: No `.env` secrets, credentials, or development paths committed.
