#!/usr/bin/env python3
"""Generate synthetic workforce dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generator import generate_all


def main():
    print("Generating workforce data (500 employees, 50 projects)...")
    datasets = generate_all()
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows")
    print("Done. Data saved to data/")


if __name__ == "__main__":
    main()
