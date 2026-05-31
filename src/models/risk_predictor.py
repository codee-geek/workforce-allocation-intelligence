"""XGBoost models for bench, burnout, and project completion risk."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import ROOT, load_config
from src.features.engineering import (
    build_employee_features,
    build_project_features,
    employee_model_matrix,
    project_model_matrix,
)


class RiskPredictor:
    TARGETS_EMP = ("bench_risk", "burnout_risk")
    TARGET_PROJ = "completion_risk"

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        xgb_cfg = self.config["models"]["xgboost"]
        self.params = {
            "n_estimators": xgb_cfg["n_estimators"],
            "max_depth": xgb_cfg["max_depth"],
            "learning_rate": xgb_cfg["learning_rate"],
            "objective": "reg:squarederror",
            "random_state": self.config["simulation"]["seed"],
            "n_jobs": -1,
        }
        self.employee_models: dict[str, XGBRegressor] = {}
        self.project_model: XGBRegressor | None = None
        self.employee_features: list[str] = []
        self.project_features: list[str] = []

    def train(
        self,
        employees: pd.DataFrame,
        projects: pd.DataFrame,
        util_history: pd.DataFrame,
        allocations: pd.DataFrame,
        emp_labels: pd.DataFrame,
        proj_labels: pd.DataFrame,
    ) -> dict[str, float]:
        emp_feat = build_employee_features(employees, util_history, allocations, projects)
        emp_feat = emp_feat.merge(emp_labels, on="employee_id")
        X_emp, self.employee_features = employee_model_matrix(emp_feat)

        proj_feat = build_project_features(projects, allocations, employees, util_history)
        proj_feat = proj_feat.merge(proj_labels, on="project_id")
        X_proj, self.project_features = project_model_matrix(proj_feat)

        metrics: dict[str, float] = {}

        for target in self.TARGETS_EMP:
            y = emp_feat[target].values
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_emp, y, test_size=0.2, random_state=42
            )
            model = XGBRegressor(**self.params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            pred = np.clip(model.predict(X_te), 0, 1)
            self.employee_models[target] = model
            metrics[f"{target}_mae"] = mean_absolute_error(y_te, pred)
            try:
                metrics[f"{target}_auc"] = roc_auc_score(
                    (y_te > 0.5).astype(int), pred
                )
            except ValueError:
                metrics[f"{target}_auc"] = 0.5

        y_p = proj_feat[self.TARGET_PROJ].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_proj, y_p, test_size=0.2, random_state=42
        )
        self.project_model = XGBRegressor(**self.params)
        self.project_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred_p = np.clip(self.project_model.predict(X_te), 0, 1)
        metrics[f"{self.TARGET_PROJ}_mae"] = mean_absolute_error(y_te, pred_p)
        try:
            metrics[f"{self.TARGET_PROJ}_auc"] = roc_auc_score(
                (y_te > 0.5).astype(int), pred_p
            )
        except ValueError:
            metrics[f"{self.TARGET_PROJ}_auc"] = 0.5

        return metrics

    def predict_employee_risks(
        self,
        employees: pd.DataFrame,
        util_history: pd.DataFrame,
        allocations: pd.DataFrame,
        projects: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.employee_models:
            raise RuntimeError("Employee models not trained. Call train() or load() first.")

        feat = build_employee_features(employees, util_history, allocations, projects)
        X, _ = employee_model_matrix(feat)
        X = X.reindex(columns=self.employee_features, fill_value=0)

        out = feat[["employee_id", "name", "primary_skill", "seniority"]].copy()
        out["bench_risk_score"] = np.clip(
            self.employee_models["bench_risk"].predict(X), 0, 1
        )
        out["burnout_risk_score"] = np.clip(
            self.employee_models["burnout_risk"].predict(X), 0, 1
        )

        th = self.config["risk_thresholds"]
        out["bench_alert"] = out["bench_risk_score"] >= th["bench_high"]
        out["burnout_alert"] = out["burnout_risk_score"] >= th["burnout_high"]
        return out

    def predict_project_risks(
        self,
        projects: pd.DataFrame,
        allocations: pd.DataFrame,
        employees: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.project_model is None:
            raise RuntimeError("Project model not trained. Call train() or load() first.")

        feat = build_project_features(projects, allocations, employees)
        X, _ = project_model_matrix(feat)
        X = X.reindex(columns=self.project_features, fill_value=0)

        out = feat[
            ["project_id", "name", "status", "required_fte", "assigned_fte", "fte_gap"]
        ].copy()
        out["completion_risk_score"] = np.clip(self.project_model.predict(X), 0, 1)
        out["completion_alert"] = (
            out["completion_risk_score"] >= self.config["risk_thresholds"]["completion_risk_high"]
        )
        return out

    def save(self, directory: Path | None = None) -> None:
        d = directory or ROOT / self.config["paths"]["models_dir"]
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "employee_models": self.employee_models,
                "project_model": self.project_model,
                "employee_features": self.employee_features,
                "project_features": self.project_features,
                "params": self.params,
            },
            d / "risk_predictor.joblib",
        )

    def load(self, directory: Path | None = None) -> None:
        d = directory or ROOT / self.config["paths"]["models_dir"]
        payload = joblib.load(d / "risk_predictor.joblib")
        self.employee_models = payload["employee_models"]
        self.project_model = payload["project_model"]
        self.employee_features = payload["employee_features"]
        self.project_features = payload["project_features"]
