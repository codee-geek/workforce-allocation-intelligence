#!/usr/bin/env python3
"""Run risk prediction, forecasting, and OR-Tools optimization."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import WorkforceIntelligencePipeline


def main():
    pipeline = WorkforceIntelligencePipeline()
    if not (pipeline.data_dir / "employees.csv").exists():
        pipeline.generate_data()
    if not (pipeline.models_dir / "risk_predictor.joblib").exists():
        print("Training models first...")
        pipeline.train()

    print("Running inference + optimization...")
    results = pipeline.run_inference()
    opt = results["optimization"]
    print(f"Solver status: {opt['status']}")
    print(f"Metrics: {opt['metrics']}")
    emp = results["employee_risks"]
    print(f"\nHigh bench risk: {emp['bench_alert'].sum()} employees")
    print(f"High burnout risk: {emp['burnout_alert'].sum()} employees")
    proj = results["project_risks"]
    print(f"High completion risk: {proj['completion_alert'].sum()} projects")
    print(f"Recommended assignments: {len(results['recommended_allocations'])}")
    print("\nOutputs written to data/")


if __name__ == "__main__":
    main()
