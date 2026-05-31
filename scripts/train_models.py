#!/usr/bin/env python3
"""Train XGBoost risk models and demand forecasts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import WorkforceIntelligencePipeline


def main():
    pipeline = WorkforceIntelligencePipeline()
    if not (pipeline.data_dir / "employees.csv").exists():
        print("No data found. Generating...")
        pipeline.generate_data()
    print("Training models...")
    metrics = pipeline.train()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Models saved to models/")


if __name__ == "__main__":
    main()
