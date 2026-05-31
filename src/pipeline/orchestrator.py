"""End-to-end pipeline: data → train → predict → forecast → optimize."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.data.generator import generate_all
from src.models.demand_forecast import DemandForecaster
from src.models.risk_predictor import RiskPredictor
from src.optimization.allocator import WorkforceAllocator


class WorkforceIntelligencePipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.data_dir = ROOT / self.config["paths"]["data_dir"]
        self.models_dir = ROOT / self.config["paths"]["models_dir"]
        self.risk_predictor = RiskPredictor(self.config)
        self.forecaster = DemandForecaster(self.config)
        self.allocator = WorkforceAllocator(self.config)

    def load_data(self) -> dict[str, pd.DataFrame]:
        names = [
            "employees", "projects", "utilization_history",
            "current_allocations", "skill_demand",
            "employee_risk_labels", "project_risk_labels",
        ]
        date_cols = {"utilization_history": ["week_start"], "skill_demand": ["week_start"]}
        data = {}
        for n in names:
            path = self.data_dir / f"{n}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run generate_data first.")
            df = pd.read_csv(path)
            for col in date_cols.get(n, []):
                df[col] = pd.to_datetime(df[col])
            data[n] = df
        return data

    def generate_data(self) -> dict[str, pd.DataFrame]:
        return generate_all(self.data_dir)

    def train(self, data: dict[str, pd.DataFrame] | None = None) -> dict:
        data = data or self.load_data()
        metrics = self.risk_predictor.train(
            data["employees"],
            data["projects"],
            data["utilization_history"],
            data["current_allocations"],
            data["employee_risk_labels"],
            data["project_risk_labels"],
        )
        self.risk_predictor.save(self.models_dir)
        forecasts = self.forecaster.fit_predict(data["skill_demand"])
        forecasts.to_csv(self.data_dir / "demand_forecasts.csv", index=False)
        self.forecaster.save(self.models_dir)
        summary = self.forecaster.get_skill_summary()
        summary.to_csv(self.data_dir / "skill_demand_summary.csv", index=False)
        metrics["forecast_skills"] = len(self.forecaster.forecasts)
        (self.models_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def run_inference(self, data: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
        data = data or self.load_data()
        if not (self.models_dir / "risk_predictor.joblib").exists():
            self.train(data)
        else:
            self.risk_predictor.load(self.models_dir)
            if not (self.data_dir / "demand_forecasts.csv").exists():
                self.forecaster.fit_predict(data["skill_demand"])
                self.forecaster.save(self.models_dir)
            else:
                self.forecaster.load(self.models_dir)

        emp_risks = self.risk_predictor.predict_employee_risks(
            data["employees"],
            data["utilization_history"],
            data["current_allocations"],
            data["projects"],
        )
        proj_risks = self.risk_predictor.predict_project_risks(
            data["projects"],
            data["current_allocations"],
            data["employees"],
        )

        demand_fc = pd.read_csv(self.data_dir / "demand_forecasts.csv", parse_dates=["week_start"]) \
            if (self.data_dir / "demand_forecasts.csv").exists() \
            else self.forecaster.fit_predict(data["skill_demand"])

        allocation = self.allocator.optimize(
            data["employees"],
            data["projects"],
            emp_risks,
            proj_risks,
            data["current_allocations"],
            demand_fc,
        )

        emp_risks.to_csv(self.data_dir / "employee_risk_predictions.csv", index=False)
        proj_risks.to_csv(self.data_dir / "project_risk_predictions.csv", index=False)
        if len(allocation["assignments"]):
            allocation["assignments"].to_csv(
                self.data_dir / "recommended_allocations.csv", index=False
            )
        (self.data_dir / "optimization_metrics.json").write_text(
            json.dumps(allocation["metrics"], indent=2)
        )

        return {
            "employee_risks": emp_risks,
            "project_risks": proj_risks,
            "demand_forecasts": demand_fc,
            "recommended_allocations": allocation["assignments"],
            "optimization": allocation,
        }

    def run_full(self) -> dict:
        data = self.generate_data()
        train_metrics = self.train(data)
        results = self.run_inference(data)
        return {"train_metrics": train_metrics, "results": results}
