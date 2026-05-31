"""REST API for workforce allocation intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline.orchestrator import WorkforceIntelligencePipeline

app = FastAPI(
    title="Workforce Allocation Intelligence API",
    description="Bench, burnout, and completion risk prediction with OR-Tools allocation",
    version="1.0.0",
)

pipeline: WorkforceIntelligencePipeline | None = None
_cache: dict | None = None


def get_pipeline() -> WorkforceIntelligencePipeline:
    global pipeline
    if pipeline is None:
        pipeline = WorkforceIntelligencePipeline()
    return pipeline


class HealthResponse(BaseModel):
    status: str
    employees: int
    projects: int


class RiskSummary(BaseModel):
    bench_alerts: int
    burnout_alerts: int
    completion_alerts: int
    optimization_status: str


@app.get("/health", response_model=HealthResponse)
def health():
    p = get_pipeline()
    try:
        data = p.load_data()
        n_emp = len(data["employees"])
        n_proj = len(data["projects"])
    except FileNotFoundError:
        n_emp, n_proj = 0, 0
    return HealthResponse(status="ok", employees=n_emp, projects=n_proj)


@app.post("/pipeline/generate")
def generate_data():
    p = get_pipeline()
    datasets = p.generate_data()
    return {"message": "Data generated", "datasets": {k: len(v) for k, v in datasets.items()}}


@app.post("/pipeline/train")
def train():
    p = get_pipeline()
    try:
        metrics = p.train()
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    return {"metrics": metrics}


@app.post("/pipeline/run")
def run_pipeline():
    global _cache
    p = get_pipeline()
    try:
        if not (p.data_dir / "employees.csv").exists():
            p.generate_data()
        if not (p.models_dir / "risk_predictor.joblib").exists():
            p.train()
        _cache = p.run_inference()
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    emp = _cache["employee_risks"]
    proj = _cache["project_risks"]
    return RiskSummary(
        bench_alerts=int(emp["bench_alert"].sum()),
        burnout_alerts=int(emp["burnout_alert"].sum()),
        completion_alerts=int(proj["completion_alert"].sum()),
        optimization_status=_cache["optimization"]["status"],
    )


@app.get("/risks/employees")
def employee_risks(limit: int = 50, alert_only: bool = False):
    if _cache is None:
        raise HTTPException(400, "Run POST /pipeline/run first")
    df = _cache["employee_risks"]
    if alert_only:
        df = df[df["bench_alert"] | df["burnout_alert"]]
    return df.head(limit).to_dict(orient="records")


@app.get("/risks/projects")
def project_risks(limit: int = 50, alert_only: bool = False):
    if _cache is None:
        raise HTTPException(400, "Run POST /pipeline/run first")
    df = _cache["project_risks"]
    if alert_only:
        df = df[df["completion_alert"]]
    return df.head(limit).to_dict(orient="records")


@app.get("/forecast/demand")
def demand_forecast(skill: str | None = None):
    p = get_pipeline()
    path = p.data_dir / "demand_forecasts.csv"
    if not path.exists():
        raise HTTPException(400, "Forecasts not available. Train models first.")
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["week_start"])
    if skill:
        df = df[df["skill"] == skill]
    return df.to_dict(orient="records")


@app.get("/allocations/recommended")
def recommended_allocations(limit: int = 100):
    p = get_pipeline()
    path = p.data_dir / "recommended_allocations.csv"
    if not path.exists():
        raise HTTPException(400, "No allocations. Run pipeline first.")
    import pandas as pd
    df = pd.read_csv(path)
    return df.head(limit).to_dict(orient="records")


@app.get("/optimization/metrics")
def optimization_metrics():
    p = get_pipeline()
    path = p.data_dir / "optimization_metrics.json"
    if not path.exists():
        raise HTTPException(400, "No optimization results.")
    import json
    return json.loads(path.read_text())
