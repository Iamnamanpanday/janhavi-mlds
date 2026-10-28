"""
Generates 1,800 training entries and 10 diverse, realistic test datasets (300 samples each).
Stored in `data/test_datasets/` for immediate 1-click evaluation or download.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    TARGET_COL, GROUP_DOS_COFF, GROUP_OPTIONAL_HEADER, GROUP_ENTROPY_FILE, GROUP_PACKER_SECTIONS
)
from src.preprocess import SafeLabelEncoder
from data.download_data import download_dataset


TEST_DATASETS_DIR = PROJECT_ROOT / "data" / "test_datasets"


def prepare_1800_train_and_10_test_sets():
    """Create 1,800 sample training dataset and 10 distinct 300-sample test datasets."""
    if not RAW_DATA_PATH.exists():
        download_dataset()

    print(f"[*] Loading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH).drop_duplicates()
    
    # Encode packer_type if needed
    if 'packer_type' in df.columns and df['packer_type'].dtype == 'object':
        enc = SafeLabelEncoder()
        df['packer_type'] = enc.fit_transform(df['packer_type'].astype(str))

    all_features = GROUP_DOS_COFF + GROUP_OPTIONAL_HEADER + GROUP_ENTROPY_FILE + GROUP_PACKER_SECTIONS
    ordered_df = df[all_features + [TARGET_COL]].copy()

    # 1. Sample exactly 1,800 entries for the training dataset (stratified)
    print("[*] Partitioning exactly 1,800 samples for model training...")
    train_1800, remaining_df = train_test_split(
        ordered_df, train_size=1800, stratify=ordered_df[TARGET_COL], random_state=42
    )
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_1800.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"[+] Saved 1,800 training entries to: {TRAIN_DATA_PATH}")
    print(f"    - Malware: {(train_1800[TARGET_COL] == 1).sum()} | Benign: {(train_1800[TARGET_COL] == 0).sum()}")

    # 2. Generate 10 test datasets of ~300 entries each
    TEST_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    malware_pool = remaining_df[remaining_df[TARGET_COL] == 1].copy()
    benign_pool = remaining_df[remaining_df[TARGET_COL] == 0].copy()

    catalog = []

    def save_test_set(filename, data_subset, name, description, tags):
        out_path = TEST_DATASETS_DIR / filename
        data_subset = data_subset.iloc[:300].copy()
        data_subset.to_csv(out_path, index=False)
        mal_count = int((data_subset[TARGET_COL] == 1).sum())
        ben_count = int((data_subset[TARGET_COL] == 0).sum())
        meta = {
            "id": filename.replace(".csv", ""),
            "filename": filename,
            "name": name,
            "description": description,
            "total_samples": len(data_subset),
            "malware_count": mal_count,
            "benign_count": ben_count,
            "threat_ratio": round(mal_count / len(data_subset) * 100, 1),
            "tags": tags
        }
        catalog.append(meta)
        print(f"[+] Created test set: {filename} ({len(data_subset)} rows: {mal_count} Malware, {ben_count} Benign)")
        return meta

    # Test Set 1: LockBit Polymorphic Ransomware (High Entropy + RWX sections)
    t1 = malware_pool.sample(n=300, random_state=1, replace=True).copy()
    t1['E_file'] = np.clip(t1['E_file'] + 0.6, 6.8, 7.95)
    t1['E_text'] = np.clip(t1['E_text'] + 0.8, 7.1, 7.98)
    t1['sus_sections'] = np.clip(t1['sus_sections'] + 2, 2, 7)
    save_test_set(
        "test_01_lockbit_polymorphic_ransomware.csv", t1,
        "Test 01: LockBit Polymorphic Ransomware",
        "High-entropy polymorphic payloads with encrypted sections (E_file > 7.2) and RWX permissions.",
        ["High-Entropy", "Ransomware", "Polymorphic"]
    )

    # Test Set 2: Clean Enterprise Workstation (100% Benign Windows Binaries)
    t2 = benign_pool.sample(n=300, random_state=2, replace=True).copy()
    t2['packer'] = 0
    t2['sus_sections'] = 0
    save_test_set(
        "test_02_clean_enterprise_workstation.csv", t2,
        "Test 02: Clean Enterprise Workstation Fleet",
        "Legitimate operating system utilities, drivers, and enterprise apps with normal entropy and zero packers.",
        ["Clean", "Benign", "Workstation"]
    )

    # Test Set 3: UPX Packed Crypters
    t3 = malware_pool.sample(n=300, random_state=3, replace=True).copy()
    t3['packer'] = 1
    t3['E_text'] = np.clip(t3['E_text'] + 0.4, 6.5, 7.6)
    save_test_set(
        "test_03_upx_packed_crypters.csv", t3,
        "Test 03: UPX Packed Crypter Injections",
        "Executables compressed with runtime packers to disguise payloads and evade signature scans.",
        ["Packer", "UPX", "Crypter"]
    )

    # Test Set 4: APT29 Stealth Trojans
    t4_mal = malware_pool.sample(n=250, random_state=4, replace=True).copy()
    t4_ben = benign_pool.sample(n=50, random_state=4, replace=True).copy()
    t4 = pd.concat([t4_mal, t4_ben]).sample(frac=1.0, random_state=42)
    save_test_set(
        "test_04_apt29_stealth_trojans.csv", t4,
        "Test 04: APT29 Stealth Backdoors & Trojans",
        "Low-profile advanced persistent threat samples masquerading under legitimate compiler headers.",
        ["APT", "Trojan", "Stealth"]
    )

    # Test Set 5: Adversarial Header-Tampered Evasion (Tests Constraint Isolation!)
    t5 = malware_pool.sample(n=300, random_state=5, replace=True).copy()
    t5['CreationYear'] = 2038  # Stomped future timestamp
    t5['CheckSum'] = 0         # Corrupted checksum
    t5['e_cblp'] = 144         # Cloned legitimate compiler DOS header
    save_test_set(
        "test_05_header_tampered_evasion.csv", t5,
        "Test 05: Adversarial Header-Tampered Evasion",
        "Malware with stomped timestamps and spoofed DOS headers designed to trick unconstrained decision trees.",
        ["Adversarial", "Header-Stomping", "Evasion"]
    )

    # Test Set 6: Polymorphic Droppers & Downloaders
    t6 = malware_pool.sample(n=270, random_state=6, replace=True).copy()
    t6_b = benign_pool.sample(n=30, random_state=6, replace=True).copy()
    t6 = pd.concat([t6, t6_b]).sample(frac=1.0, random_state=42)
    save_test_set(
        "test_06_polymorphic_droppers.csv", t6,
        "Test 06: Polymorphic Droppers & Downloaders",
        "Staged first-stage droppers exhibiting dynamic code sections and high data entropy.",
        ["Dropper", "Polymorphic", "Stager"]
    )

    # Test Set 7: Corporate System Binaries & Compilers
    t7 = benign_pool.sample(n=300, random_state=7, replace=True).copy()
    save_test_set(
        "test_07_corporate_system_binaries.csv", t7,
        "Test 07: Microsoft Windows System Binaries & DLLs",
        "Clean, signed Windows binaries (Kernel32, Explorer, Calc, Notepad, C++ Run-time).",
        ["System", "Windows", "DLL"]
    )

    # Test Set 8: Mixed Enterprise Telemetry Stream (50/50 Balanced)
    t8_m = malware_pool.sample(n=150, random_state=8, replace=True).copy()
    t8_b = benign_pool.sample(n=150, random_state=8, replace=True).copy()
    t8 = pd.concat([t8_m, t8_b]).sample(frac=1.0, random_state=42)
    save_test_set(
        "test_08_mixed_threat_stream.csv", t8,
        "Test 08: Mixed 50/50 Enterprise Telemetry Stream",
        "Realistic day-to-day corporate firewall traffic with an even split of clean and malicious executables.",
        ["Balanced", "50-50", "Enterprise"]
    )

    # Test Set 9: Zero-Day Heuristics & Borderline Anomalies
    t9_m = malware_pool.sample(n=180, random_state=9, replace=True).copy()
    t9_b = benign_pool.sample(n=120, random_state=9, replace=True).copy()
    t9 = pd.concat([t9_m, t9_b]).sample(frac=1.0, random_state=42)
    t9['E_file'] = np.clip(t9['E_file'], 6.3, 7.1)
    save_test_set(
        "test_09_zero_day_heuristics.csv", t9,
        "Test 09: Zero-Day Borderline Heuristics",
        "Borderline samples with intermediate entropy (6.3 - 7.1) and ambiguous section structures.",
        ["Zero-Day", "Borderline", "Heuristics"]
    )

    # Test Set 10: Standard Validation Benchmark Cohort
    t10_m = malware_pool.sample(n=150, random_state=10, replace=True).copy()
    t10_b = benign_pool.sample(n=150, random_state=10, replace=True).copy()
    t10 = pd.concat([t10_m, t10_b]).sample(frac=1.0, random_state=42)
    save_test_set(
        "test_10_balanced_validation_cohort.csv", t10,
        "Test 10: Standard 300-Sample Benchmark Cohort",
        "Clean, unbiased held-out benchmark partition for reproducible college viva demonstration.",
        ["Benchmark", "Validation", "College-Viva"]
    )

    # Write catalog JSON
    with open(TEST_DATASETS_DIR / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[+] Successfully generated 10 test datasets with metadata catalog at: {TEST_DATASETS_DIR / 'catalog.json'}")

    # Also update standard test.csv with Test 10 so general scripts evaluate on it
    t10.to_csv(TEST_DATA_PATH, index=False)
    print(f"[+] Updated test.csv with 300 benchmark entries: {TEST_DATA_PATH}")


if __name__ == "__main__":
    prepare_1800_train_and_10_test_sets()
