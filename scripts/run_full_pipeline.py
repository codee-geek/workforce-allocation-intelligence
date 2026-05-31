#!/usr/bin/env python3
"""Generate data, train, predict, and optimize in one run."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import WorkforceIntelligencePipeline


def main():
    pipeline = WorkforceIntelligencePipeline()
    print("Running full Workforce Allocation Intelligence pipeline...")
    output = pipeline.run_full()
    print("\n=== Training Metrics ===")
    for k, v in output["train_metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    opt = output["results"]["optimization"]
    print("\n=== Optimization ===")
    print(f"  Status: {opt['status']}")
    for k, v in opt["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nComplete. Launch dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
