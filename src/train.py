"""
Model Training Script:
Trains three distinct machine learning classifiers on the PE malware dataset:
1. Baseline: Random Forest Classifier
2. Standard XGBoost Classifier (Unconstrained interaction)
3. Constrained XGBoost Classifier (with domain-grounded Feature Interaction Constraints)

Saves all trained models and training metadata to the `models/` directory.
"""
import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    TRAIN_DATA_PATH, MODELS_DIR, TARGET_COL,
    RF_PARAMS, XGB_STANDARD_PARAMS, XGB_CONSTRAINED_PARAMS,
    INTERACTION_CONSTRAINTS, FEATURE_GROUPS
)
from src.preprocess import load_and_preprocess_data


def train_all_models():
    """Train RF, Standard XGBoost, and Constrained XGBoost."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DATA_PATH.exists():
        print("[*] Processed training data not found. Running preprocessor...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
    else:
        print(f"[*] Loading training data from: {TRAIN_DATA_PATH}")
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]

    print(f"[+] Loaded training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    training_logs = {}

    # -------------------------------------------------------------
    # 1. Baseline Model: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[1/3] Training Baseline Model: Random Forest...")
    print("=" * 60)
    rf_start = time.time()
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - rf_start
    print(f"[+] Random Forest training completed in {rf_time:.2f} seconds.")

    rf_path = MODELS_DIR / "random_forest_baseline.joblib"
    joblib.dump(rf_model, rf_path)
    print(f"[+] Saved Random Forest model to: {rf_path}")
    training_logs["Random_Forest"] = {
        "train_time_sec": round(rf_time, 3),
        "n_estimators": RF_PARAMS["n_estimators"],
        "max_depth": RF_PARAMS["max_depth"]
    }

    # -------------------------------------------------------------
    # 2. Standard XGBoost (Unconstrained)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2/3] Training Standard XGBoost (Unconstrained)...")
    print("=" * 60)
    xgb_std_start = time.time()
    xgb_standard = XGBClassifier(**XGB_STANDARD_PARAMS)
    xgb_standard.fit(X_train, y_train)
    xgb_std_time = time.time() - xgb_std_start
    print(f"[+] Standard XGBoost training completed in {xgb_std_time:.2f} seconds.")

    xgb_std_joblib_path = MODELS_DIR / "xgboost_standard.joblib"
    xgb_std_json_path = MODELS_DIR / "xgboost_standard.json"
    joblib.dump(xgb_standard, xgb_std_joblib_path)
    xgb_standard.save_model(str(xgb_std_json_path))
    print(f"[+] Saved Standard XGBoost model to: {xgb_std_joblib_path}")
    training_logs["XGBoost_Standard"] = {
        "train_time_sec": round(xgb_std_time, 3),
        "n_estimators": XGB_STANDARD_PARAMS["n_estimators"],
        "max_depth": XGB_STANDARD_PARAMS["max_depth"],
        "learning_rate": XGB_STANDARD_PARAMS["learning_rate"],
        "interaction_constraints": "None (Fully Unconstrained)"
    }

    # -------------------------------------------------------------
    # 3. Constrained XGBoost (Feature Interaction Constraints)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3/3] Training Constrained XGBoost (Feature Interaction Constraints)...")
    print("=" * 60)
    print("[*] Applying 4 domain-isolated feature interaction groups:")
    for group_name, feats in FEATURE_GROUPS.items():
        print(f"    - {group_name}: {len(feats)} features")

    xgb_const_start = time.time()
    xgb_constrained = XGBClassifier(**XGB_CONSTRAINED_PARAMS)
    xgb_constrained.fit(X_train, y_train)
    xgb_const_time = time.time() - xgb_const_start
    print(f"[+] Constrained XGBoost training completed in {xgb_const_time:.2f} seconds.")

    xgb_const_joblib_path = MODELS_DIR / "xgboost_constrained.joblib"
    xgb_const_json_path = MODELS_DIR / "xgboost_constrained.json"
    joblib.dump(xgb_constrained, xgb_const_joblib_path)
    xgb_constrained.save_model(str(xgb_const_json_path))
    print(f"[+] Saved Constrained XGBoost model to: {xgb_const_joblib_path}")
    training_logs["XGBoost_Constrained"] = {
        "train_time_sec": round(xgb_const_time, 3),
        "n_estimators": XGB_CONSTRAINED_PARAMS["n_estimators"],
        "max_depth": XGB_CONSTRAINED_PARAMS["max_depth"],
        "learning_rate": XGB_CONSTRAINED_PARAMS["learning_rate"],
        "interaction_constraints_groups": len(INTERACTION_CONSTRAINTS)
    }

    # Save summary
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_logs, f, indent=2)
    print(f"\n[+] Saved training summary to: {summary_path}")

    print("\n[+] All 3 models trained and serialized successfully!")
    return {
        "rf": rf_model,
        "xgb_standard": xgb_standard,
        "xgb_constrained": xgb_constrained
    }


if __name__ == "__main__":
    train_all_models()
