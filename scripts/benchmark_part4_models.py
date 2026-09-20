"""Part 4 Model Benchmark Suite: E0 vs E3 vs Corrected Engine-Specific Residuals.

Evaluates 3 feature configurations:
1. E0: Baseline (15 features)
2. E3_Legacy: Generic 1.35L Reduced-Order Residuals (56 features)
3. E3_Corrected: Engine-Specific Model Residuals (TSIO-360 on ACES, Rotax-914 on Synthetic) (56 features)

Across 6 model architectures:
- HistGradientBoosting
- ExtraTrees
- RandomForest
- XGBoost
- LightGBM
- CatBoost

Against immutable held-out ACES flights: 191, 225, 235.
And separately against held-out AEROPULSE_SYNTHETIC trajectories.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_engine import VirtualDataLabEngine
from app.engine_config import EngineConfig
from app.feature_engineering import (
    PhysicsResidualExtractor,
    assemble_experiment_dataset,
    audit_feature_names,
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "HistGradientBoosting",
    "ExtraTrees",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def load_aces(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    aces = pd.read_csv(data_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(aces, groups=aces["Flight"]))

    train_df = aces.iloc[train_idx].copy()
    test_df = aces.iloc[test_idx].copy()

    train_flights = sorted(train_df["Flight"].unique().tolist())
    test_flights = sorted(test_df["Flight"].unique().tolist())
    return train_df, test_df, train_flights, test_flights


def load_synthetic(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine = VirtualDataLabEngine(master_seed=seed)
    corpus = engine.generate_master_corpus(
        num_healthy=8,
        num_degradation=14,
        num_sensor_faults=6,
        num_missions=4,
        num_can_faults=0,
        master_seed=seed,
    )
    train_d, test_d = engine.split_corpus_trajectories(corpus, train_ratio=0.70, seed=seed)

    def to_df(traj_dict: dict) -> pd.DataFrame:
        rows = []
        for tid, pts in traj_dict.items():
            for p in pts:
                d = p.to_dict()
                row = {
                    "Flight": tid,
                    "GPS_Time": float(d["timestamp"]),
                    "Engine_RPM": float(d["RPM"]),
                    "MAP_Injector": float(d["MAP"]),
                    "CHT": float(d["CHT"]) * 1.8 + 32.0,
                    "EGT1": float(d["EGT"]) * 1.8 + 32.0,
                    "EGT2": float(d["EGT"]) * 1.8 + 32.0,
                    "EGT3": float(d["EGT"]) * 1.8 + 32.0,
                    "Oil_Temp": float(d["oil_temperature"]) * 1.8 + 32.0,
                    "Oil_Pressure": float(d["oil_pressure"]),
                    "Fuel_Flow": float(d["fuel_flow"]),
                    "Battery_Voltage": float(d["bus_voltage"]),
                    "Battery_Current": float(d["current"]),
                    "Alternator_Temp": float(d["alternator_temperature"]) * 1.8 + 32.0,
                    "EFI_Fuel_Temp": (float(d["ambient_temperature"]) + 10.0) * 1.8 + 32.0,
                    "EFI_Water_Temp": float(d["coolant_temperature"]) * 1.8 + 32.0,
                    "Ambient_Temp": float(d["ambient_temperature"]),
                    "Operating_State": (
                        "CRUISE" if d["mission_phase"] in ["CRUISE", "ENDURANCE"]
                        else ("HIGH" if d["mission_phase"] in ["TAKEOFF", "CLIMB"] else "CRUISE_LOW")
                    ),
                    "Mission_Phase": d["mission_phase"],
                }
                h = float(d.get("health_index", 100.0))
                stage = str(d.get("degradation_stage", "HEALTHY"))
                if stage == "HEALTHY" or h >= 90.0:
                    row["Health_State"] = "Normal"
                elif stage == "EARLY" or h >= 70.0:
                    row["Health_State"] = "Watch"
                elif stage == "MODERATE" or h >= 50.0:
                    row["Health_State"] = "Warning"
                else:
                    row["Health_State"] = "Critical"
                rows.append(row)
        return pd.DataFrame(rows)

    return to_df(train_d), to_df(test_d)


def instantiate_model(model_name: str, random_state: int = 42) -> Any:
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.10, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, class_weight="balanced", random_state=random_state,
        )
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=16, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=random_state,
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=random_state, eval_metric="mlogloss",
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.10, class_weight="balanced",
            n_jobs=-1, random_state=random_state, verbose=-1,
        )
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=150, depth=6, learning_rate=0.10, auto_class_weights="Balanced",
            thread_count=-1, verbose=0, random_seed=random_state,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], latency_us: float, model_size_kb: float) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    crit_prec = precision_score(y_true, y_pred, labels=["Critical"], average="macro", zero_division=0)
    crit_rec = recall_score(y_true, y_pred, labels=["Critical"], average="macro", zero_division=0)
    crit_f1 = f1_score(y_true, y_pred, labels=["Critical"], average="macro", zero_division=0)
    crit_fnr = 1.0 - crit_rec

    per_class_f1 = {}
    for cls in classes:
        per_class_f1[cls] = float(f1_score(y_true == cls, np.array(y_pred) == cls, average="binary", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "Accuracy": round(float(acc), 4),
        "Balanced_Accuracy": round(float(bal_acc), 4),
        "Macro_F1": round(float(macro_f1), 4),
        "Weighted_F1": round(float(weighted_f1), 4),
        "Critical_Precision": round(float(crit_prec), 4),
        "Critical_Recall": round(float(crit_rec), 4),
        "Critical_F1": round(float(crit_f1), 4),
        "Critical_FNR": round(float(crit_fnr), 4),
        "Per_Class_F1": {k: round(float(v), 4) for k, v in per_class_f1.items()},
        "Confusion_Matrix": cm.tolist(),
        "Classes": classes,
        "Inference_Latency_us": round(float(latency_us), 2),
        "Model_Size_KB": round(float(model_size_kb), 2),
    }


def measure_latency_and_size(model: Any, X: np.ndarray) -> Tuple[float, float]:
    sample_size = min(len(X), 5000)
    X_sub = X[:sample_size]
    _ = model.predict(X_sub[:100])  # Warmup
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = model.predict(X_sub)
        times.append(time.perf_counter() - t0)
    us_per_sample = (np.mean(times) / sample_size) * 1e6

    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    size_kb = len(buf.getvalue()) / 1024.0
    return float(us_per_sample), float(size_kb)


def run_benchmark_part4() -> Dict[str, Any]:
    aces_path = ROOT / "FINAL_DATASET" / "ACES" / "aces_health.csv"
    aces_train, aces_test, trn_flights, tst_flights = load_aces(aces_path)
    syn_train, syn_test = load_synthetic(seed=42)

    classes = ["Critical", "Normal", "Warning", "Watch"]
    label_to_int = {c: i for i, c in enumerate(classes)}

    y_aces_train = aces_train["Health_State"]
    y_aces_test = aces_test["Health_State"]
    y_aces_train_int = y_aces_train.map(label_to_int).values
    y_aces_test_int = y_aces_test.map(label_to_int).values

    y_syn_train = syn_train["Health_State"]
    y_syn_test = syn_test["Health_State"]
    y_syn_train_int = y_syn_train.map(label_to_int).values
    y_syn_test_int = y_syn_test.map(label_to_int).values

    # Fit extractors
    # 1. Legacy Generic 1.35L
    ext_gen_aces = PhysicsResidualExtractor(EngineConfig.default_135l()).fit_healthy_reference(aces_train)
    ext_gen_syn = PhysicsResidualExtractor(EngineConfig.default_135l()).fit_healthy_reference(syn_train)

    # 2. Corrected Engine-Specific (TSIO-360 for ACES, Rotax-914 for Synthetic)
    ext_corr_aces = PhysicsResidualExtractor(EngineConfig.continental_tsio_360()).fit_healthy_reference(aces_train)
    ext_corr_syn = PhysicsResidualExtractor(EngineConfig.rotax_914()).fit_healthy_reference(syn_train)

    configs = [
        ("E0_Baseline", "E0", None, None),
        ("E3_Generic_135L", "E3", ext_gen_aces, ext_gen_syn),
        ("E3_Engine_Matched_TSIO360", "E3", ext_corr_aces, ext_corr_syn),
    ]

    benchmark_results: Dict[str, Any] = {}
    summary_matrix: List[Dict[str, Any]] = []

    for cfg_label, exp_id, ext_aces, ext_syn in configs:
        print(f"\n=======================================================")
        print(f"   EVALUATING CONFIGURATION: {cfg_label}")
        print(f"=======================================================")

        X_aces_tr_raw = assemble_experiment_dataset(aces_train, exp_id, residual_extractor=ext_aces, flight_col="Flight")
        X_aces_ts_raw = assemble_experiment_dataset(aces_test, exp_id, residual_extractor=ext_aces, flight_col="Flight")

        X_syn_tr_raw = assemble_experiment_dataset(syn_train, exp_id, residual_extractor=ext_syn, flight_col="Flight")
        X_syn_ts_raw = assemble_experiment_dataset(syn_test, exp_id, residual_extractor=ext_syn, flight_col="Flight")

        num_features = X_aces_tr_raw.shape[1]
        feature_names = X_aces_tr_raw.columns.tolist()

        num_cols = [c for c in feature_names if c != "Operating_State"]
        cat_cols = ["Operating_State"] if "Operating_State" in feature_names else []

        pre_aces = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        X_aces_tr = pre_aces.fit_transform(X_aces_tr_raw)
        X_aces_ts = pre_aces.transform(X_aces_ts_raw)

        pre_syn = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        X_syn_tr = pre_syn.fit_transform(X_syn_tr_raw)
        X_syn_ts = pre_syn.transform(X_syn_ts_raw)

        benchmark_results[cfg_label] = {
            "features_count": num_features,
            "models": {},
        }

        for model_name in MODEL_NAMES:
            print(f"  Training {model_name} [{cfg_label}]...")
            model = instantiate_model(model_name)
            is_int = model_name in ["XGBoost", "LightGBM", "CatBoost"]

            t0 = time.perf_counter()
            if is_int:
                model.fit(X_aces_tr, y_aces_train_int)
                pred_raw = model.predict(X_aces_ts)
                preds = [classes[int(p)] for p in np.array(pred_raw).flatten()]
            else:
                model.fit(X_aces_tr, y_aces_train)
                preds = model.predict(X_aces_ts).tolist()
            fit_time = time.perf_counter() - t0

            lat_us, size_kb = measure_latency_and_size(model, X_aces_ts)
            aces_metrics = evaluate_predictions(y_aces_test, preds, classes, lat_us, size_kb)
            aces_metrics["fit_time_seconds"] = round(fit_time, 2)

            # Synthetic evaluation
            syn_model = instantiate_model(model_name)
            if is_int:
                syn_model.fit(X_syn_tr, y_syn_train_int)
                syn_pred_raw = syn_model.predict(X_syn_ts)
                syn_preds = [classes[int(p)] for p in np.array(syn_pred_raw).flatten()]
            else:
                syn_model.fit(X_syn_tr, y_syn_train)
                syn_preds = syn_model.predict(X_syn_ts).tolist()

            syn_lat_us, syn_size_kb = measure_latency_and_size(syn_model, X_syn_ts)
            syn_metrics = evaluate_predictions(y_syn_test, syn_preds, classes, syn_lat_us, syn_size_kb)

            benchmark_results[cfg_label]["models"][model_name] = {
                "real_aces": aces_metrics,
                "synthetic": syn_metrics,
            }

            summary_matrix.append({
                "config": cfg_label,
                "model": model_name,
                "features": num_features,
                "aces_acc": aces_metrics["Accuracy"],
                "aces_bal_acc": aces_metrics["Balanced_Accuracy"],
                "aces_macro_f1": aces_metrics["Macro_F1"],
                "aces_crit_rec": aces_metrics["Critical_Recall"],
                "aces_crit_prec": aces_metrics["Critical_Precision"],
                "aces_crit_f1": aces_metrics["Critical_F1"],
                "aces_latency_us": aces_metrics["Inference_Latency_us"],
                "model_size_kb": aces_metrics["Model_Size_KB"],
                "syn_acc": syn_metrics["Accuracy"],
                "syn_macro_f1": syn_metrics["Macro_F1"],
                "syn_crit_rec": syn_metrics["Critical_Recall"],
            })

            print(
                f"    ACES: Acc={aces_metrics['Accuracy']:.4f} | BalAcc={aces_metrics['Balanced_Accuracy']:.4f} | "
                f"MacroF1={aces_metrics['Macro_F1']:.4f} | CritRec={aces_metrics['Critical_Recall']:.4f} | "
                f"CritF1={aces_metrics['Critical_F1']:.4f} | Latency={aces_metrics['Inference_Latency_us']:.1f}us"
            )

    out_file = REPORTS_DIR / "part4_model_benchmark.json"
    payload = {
        "configurations": benchmark_results,
        "summary_matrix": summary_matrix,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved Part 4 model benchmark results to {out_file}")
    return payload


if __name__ == "__main__":
    run_benchmark_part4()
