"""
Data Preprocessing Pipeline:
- Loads the raw ClaMP malware dataset
- Handles missing values and data integrity checks
- Encodes categorical attributes (e.g. packer_type)
- Performs stratified train/test split
- Persists processed datasets and encoder artifacts
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path so modules resolve cleanly when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    MODELS_DIR, RANDOM_SEED, TEST_SIZE, TARGET_COL,
    GROUP_DOS_COFF, GROUP_OPTIONAL_HEADER, GROUP_ENTROPY_FILE, GROUP_PACKER_SECTIONS
)
from data.download_data import download_dataset


class SafeLabelEncoder(LabelEncoder):
    """Robust LabelEncoder that maps unseen labels during inference to an unknown token."""
    def transform(self, y):
        classes_dict = {cls: idx for idx, cls in enumerate(self.classes_)}
        return np.array([classes_dict.get(item, len(self.classes_)) for item in y], dtype=int)


def load_and_preprocess_data():
    """Load raw dataset, clean, encode, and split into train/test sets."""
    if not RAW_DATA_PATH.exists():
        download_dataset()

    print(f"[*] Loading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    initial_shape = df.shape
    print(f"[+] Loaded raw dataset: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"[!] Warning: Found {total_nulls} missing values. Imputing with median/mode...")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
    else:
        print("[+] Data integrity check passed: 0 missing values.")

    # Drop potential duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"[+] Removed {initial_len - len(df)} duplicate records. Current rows: {len(df)}")

    # Encode categorical features
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if 'packer_type' in df.columns and df['packer_type'].dtype == 'object':
        print("[*] Encoding 'packer_type' categorical feature...")
        encoder = SafeLabelEncoder()
        df['packer_type'] = encoder.fit_transform(df['packer_type'].astype(str))
        joblib.dump(encoder, MODELS_DIR / "packer_encoder.joblib")
        print(f"[+] Fitted packer encoder with {len(encoder.classes_)} unique packer classes.")

    # Validate target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns!")

    # Ensure consistent column ordering matching our domain groups
    all_features = GROUP_DOS_COFF + GROUP_OPTIONAL_HEADER + GROUP_ENTROPY_FILE + GROUP_PACKER_SECTIONS
    missing_features = [f for f in all_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Expected features missing in dataset: {missing_features}")

    # Reorder columns: features first, target last
    ordered_df = df[all_features + [TARGET_COL]].copy()

    X = ordered_df[all_features]
    y = ordered_df[TARGET_COL].astype(int)

    # Stratified Train/Test split
    print(f"[*] Splitting dataset (Test size = {TEST_SIZE * 100:.0f}%, Stratified, Seed = {RANDOM_SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"[+] Train set: {X_train.shape[0]} samples (Malware: {(y_train == 1).sum()}, Benign: {(y_train == 0).sum()})")
    print(f"[+] Test set:  {X_test.shape[0]} samples (Malware: {(y_test == 1).sum()}, Benign: {(y_test == 0).sum()})")

    # Save processed CSVs
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"[+] Saved train dataset to: {TRAIN_DATA_PATH}")
    print(f"[+] Saved test dataset to:  {TEST_DATA_PATH}")

    # Save feature metadata for inference and UI presets
    metadata = {
        "feature_names": all_features,
        "n_features": len(all_features),
        "target_col": TARGET_COL,
        "classes": [0, 1],
        "class_labels": {0: "Benign", 1: "Malware"},
        "benign_sample_preset": X_test[y_test == 0].iloc[0].to_dict(),
        "malware_sample_preset": X_test[y_test == 1].iloc[0].to_dict(),
        "feature_medians": X_train.median().to_dict(),
        "feature_mins": X_train.min().to_dict(),
        "feature_maxs": X_train.max().to_dict()
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved feature metadata to: {MODELS_DIR / 'feature_metadata.json'}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()
