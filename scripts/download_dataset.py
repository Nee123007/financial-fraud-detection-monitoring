"""
Automated KaggleHub Dataset Downloader for Financial Fraud Monitoring Dataset
Dataset: neel1237/indian-fraud-transactions-for-monitoring
"""

import os
import sys
from pathlib import Path

def download_data():
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub is not installed. Please install it using: pip install kagglehub")
        sys.exit(1)

    dataset_handle = 'neel1237/indian-fraud-transactions-for-monitoring'
    print(f"[*] Downloading dataset '{dataset_handle}' via kagglehub...")
    path = kagglehub.dataset_download(dataset_handle)
    print(f"[✓] Dataset successfully downloaded to: {path}")

    csv_files = list(Path(path).rglob("*.csv"))
    if csv_files:
        for f in csv_files:
            print(f"    - Found data file: {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    return path

if __name__ == "__main__":
    download_data()
