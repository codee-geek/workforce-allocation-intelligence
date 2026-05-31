"""Synthetic workforce data for 500 employees × 50 projects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_config


class WorkforceDataGenerator:
    def __init__(self, config: dict | None = None, seed: int | None = None):
        self.config = config or load_config()
        self.rng = np.random.default_rng(seed or self.config["simulation"]["seed"])
        self.skills = self.config["skills"]
        self.n_employees = self.config["company"]["num_employees"]
        self.n_projects = self.config["company"]["num_projects"]
        self.history_weeks = self.config["simulation"]["history_weeks"]

    def generate_employees(self) -> pd.DataFrame:
        seniority = self.rng.choice(
            ["Junior", "Mid", "Senior", "Lead"],
            self.n_employees,
            p=[0.35, 0.35, 0.22, 0.08],
        )
        primary_idx = self.rng.integers(0, len(self.skills), self.n_employees)
        skill_matrix = self.rng.random((self.n_employees, len(self.skills))) < 0.35
        for i, p in enumerate(primary_idx):
            skill_matrix[i, p] = True
            for j in range(len(self.skills)):
                if j != p and self.rng.random() < 0.25:
                    skill_matrix[i, j] = True

        rows = []
        for i in range(self.n_employees):
            row = {
                "employee_id": f"E{i+1:04d}",
                "name": f"Employee {i+1}",
                "seniority": seniority[i],
                "primary_skill": self.skills[primary_idx[i]],
                "location": self.rng.choice(["US", "UK", "India", "Poland", "Canada"]),
                "hourly_cost": float(
                    {"Junior": 45, "Mid": 65, "Senior": 95, "Lead": 120}[seniority[i]]
                    * self.rng.uniform(0.9, 1.1)
                ),
                "max_weekly_hours": 40.0,
            }
            for j, skill in enumerate(self.skills):
                row[f"skill_{skill.replace(' ', '_').replace('/', '_')}"] = bool(skill_matrix[i, j])
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_projects(self) -> pd.DataFrame:
        statuses = self.rng.choice(
            ["Active", "Active", "Active", "Planning", "At Risk"],
            self.n_projects,
        )
        rows = []
        for i in range(self.n_projects):
            n_req = int(self.rng.integers(2, 5))
            req_skills = self.rng.choice(self.skills, n_req, replace=False)
            priority = float(self.rng.choice([1.0, 1.5, 2.0, 3.0], p=[0.2, 0.35, 0.3, 0.15]))
            rows.append(
                {
                    "project_id": f"P{i+1:03d}",
                    "name": f"Project {i+1}",
                    "status": statuses[i],
                    "priority": priority,
                    "deadline_weeks": int(self.rng.integers(4, 20)),
                    "required_fte": float(self.rng.uniform(2, 12)),
                    "completion_pct": float(self.rng.uniform(0.15, 0.85)),
                    "required_skills": ",".join(req_skills),
                }
            )
        return pd.DataFrame(rows)

    def generate_utilization_history(
        self, employees: pd.DataFrame, projects: pd.DataFrame
    ) -> pd.DataFrame:
        project_ids = projects["project_id"].tolist()
        rows = []
        base_date = pd.Timestamp("2025-01-06")

        for week in range(self.history_weeks):
            week_start = base_date + pd.Timedelta(weeks=week)
            for _, emp in employees.iterrows():
                util = float(self.rng.beta(2, 2) * 1.1)
                on_bench = util < 0.25
                overloaded = util > 0.95
                assigned = (
                    None
                    if on_bench
                    else self.rng.choice(project_ids)
                )
                rows.append(
                    {
                        "employee_id": emp["employee_id"],
                        "week_start": week_start,
                        "utilization": min(util, 1.15),
                        "billable_hours": util * emp["max_weekly_hours"],
                        "project_id": assigned,
                        "on_bench": on_bench,
                        "overloaded": overloaded,
                    }
                )
        return pd.DataFrame(rows)

    def generate_allocations(
        self, employees: pd.DataFrame, projects: pd.DataFrame
    ) -> pd.DataFrame:
        """Current-week allocation snapshot."""
        rows = []
        for _, emp in employees.iterrows():
            util = float(self.rng.beta(2.5, 2))
            if util < 0.2:
                rows.append(
                    {
                        "employee_id": emp["employee_id"],
                        "project_id": None,
                        "allocation_pct": 0.0,
                        "role_on_project": None,
                    }
                )
                continue
            proj = projects.sample(1, random_state=int(emp["employee_id"][1:])).iloc[0]
            rows.append(
                {
                    "employee_id": emp["employee_id"],
                    "project_id": proj["project_id"],
                    "allocation_pct": min(util, 1.0),
                    "role_on_project": emp["primary_skill"],
                }
            )
        return pd.DataFrame(rows)

    def generate_skill_demand_weekly(self, projects: pd.DataFrame) -> pd.DataFrame:
        """Weekly FTE demand per skill (for forecasting)."""
        base_date = pd.Timestamp("2025-01-06")
        rows = []
        for week in range(self.history_weeks + 8):
            week_start = base_date + pd.Timedelta(weeks=week)
            for skill in self.skills:
                active = projects[projects["status"].isin(["Active", "At Risk"])]
                base_demand = 0.0
                for _, p in active.iterrows():
                    if skill in p["required_skills"].split(","):
                        base_demand += p["required_fte"] / max(
                            len(p["required_skills"].split(",")), 1
                        )
                noise = self.rng.normal(0, 0.5)
                trend = 0.02 * week
                seasonal = 2 * np.sin(2 * np.pi * week / 13)
                demand_fte = max(0.5, base_demand + noise + trend + seasonal)
                rows.append(
                    {
                        "week_start": week_start,
                        "skill": skill,
                        "demand_fte": demand_fte,
                    }
                )
        return pd.DataFrame(rows)

    def generate_risk_labels(
        self, employees: pd.DataFrame, util_history: pd.DataFrame, projects: pd.DataFrame
    ) -> pd.DataFrame:
        """Ground-truth risk labels derived from recent utilization patterns."""
        recent = util_history.sort_values("week_start").groupby("employee_id").tail(4)
        agg = recent.groupby("employee_id").agg(
            avg_util=("utilization", "mean"),
            util_std=("utilization", "std"),
            bench_weeks=("on_bench", "sum"),
            overload_weeks=("overloaded", "sum"),
        ).reset_index()
        agg["util_std"] = agg["util_std"].fillna(0)

        labels = []
        for _, emp in employees.iterrows():
            row = agg[agg["employee_id"] == emp["employee_id"]]
            if row.empty:
                avg_util, util_std, bench_w, overload_w = 0.5, 0.1, 0, 0
            else:
                r = row.iloc[0]
                avg_util, util_std = r["avg_util"], r["util_std"]
                bench_w, overload_w = r["bench_weeks"], r["overload_weeks"]

            bench_risk = min(1.0, 0.4 * (1 - avg_util) + 0.15 * bench_w + 0.1 * util_std)
            burnout_risk = min(1.0, 0.5 * max(0, avg_util - 0.85) + 0.2 * overload_w + 0.1 * util_std)

            labels.append(
                {
                    "employee_id": emp["employee_id"],
                    "bench_risk": float(np.clip(bench_risk + self.rng.normal(0, 0.05), 0, 1)),
                    "burnout_risk": float(np.clip(burnout_risk + self.rng.normal(0, 0.05), 0, 1)),
                }
            )
        emp_labels = pd.DataFrame(labels)

        proj_labels = []
        for _, p in projects.iterrows():
            gap = max(0, p["required_fte"] - p["required_fte"] * p["completion_pct"] * 0.3)
            completion_risk = min(
                1.0,
                0.3 * (1 - p["completion_pct"])
                + 0.2 * (p["status"] == "At Risk")
                + 0.15 * (p["deadline_weeks"] < 6)
                + 0.1 * gap / max(p["required_fte"], 1),
            )
            proj_labels.append(
                {
                    "project_id": p["project_id"],
                    "completion_risk": float(
                        np.clip(completion_risk + self.rng.normal(0, 0.05), 0, 1)
                    ),
                }
            )
        return emp_labels, pd.DataFrame(proj_labels)


def generate_all(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    config = load_config()
    out = output_dir or ROOT / config["paths"]["data_dir"]
    out.mkdir(parents=True, exist_ok=True)

    gen = WorkforceDataGenerator(config)
    employees = gen.generate_employees()
    projects = gen.generate_projects()
    util_history = gen.generate_utilization_history(employees, projects)
    allocations = gen.generate_allocations(employees, projects)
    skill_demand = gen.generate_skill_demand_weekly(projects)
    emp_risk, proj_risk = gen.generate_risk_labels(employees, util_history, projects)

    datasets = {
        "employees": employees,
        "projects": projects,
        "utilization_history": util_history,
        "current_allocations": allocations,
        "skill_demand": skill_demand,
        "employee_risk_labels": emp_risk,
        "project_risk_labels": proj_risk,
    }
    for name, df in datasets.items():
        df.to_csv(out / f"{name}.csv", index=False)

    meta = {
        "n_employees": len(employees),
        "n_projects": len(projects),
        "history_weeks": config["simulation"]["history_weeks"],
        "skills": config["skills"],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return datasets
