"""
Download script for authentic PE malware benchmark dataset (ClaMP Integrated Dataset).
Reference: Ajit Kumar, K.S. Kuppusamy, G. Aghila - "A learning approach to detect malware using PE header"
"""
import os
import sys
import urllib.request
import pandas as pd
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/urwithajit9/ClaMP/master/dataset/ClaMP_Integrated-5184.csv"
RAW_DATA_PATH = Path(__file__).resolve().parent / "raw" / "ClaMP_Integrated-5184.csv"


def download_dataset(force_download: bool = False) -> Path:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_PATH.exists() and not force_download:
        print(f"[+] Dataset already present at: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"[+] Verified local dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return RAW_DATA_PATH

    print(f"[*] Downloading authentic PE malware dataset from:\n    {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"[+] Download complete! Saved to: {RAW_DATA_PATH}")
        
        # Verify integrity
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"[+] Successfully loaded: {df.shape[0]} samples with {df.shape[1]} features.")
        if "class" not in df.columns:
            raise ValueError("Target column 'class' missing from downloaded dataset.")
        print(f"[+] Class breakdown: Benign (0) = {(df['class'] == 0).sum()}, Malware (1) = {(df['class'] == 1).sum()}")
        return RAW_DATA_PATH
    except Exception as e:
        print(f"[-] Error downloading dataset: {e}", file=sys.stderr)
        print("[!] Instructions for manual download:", file=sys.stderr)
        print(f"    1. Download CSV directly from: {DATA_URL}", file=sys.stderr)
        print(f"    2. Place it in: {RAW_DATA_PATH}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    download_dataset()
