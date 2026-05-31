"""Feature engineering for risk prediction models."""

from __future__ import annotations

import pandas as pd

from src.config import load_config


def _skill_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("skill_")]


def build_employee_features(
    employees: pd.DataFrame,
    util_history: pd.DataFrame,
    allocations: pd.DataFrame,
    projects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    config = load_config()
    skills = config["skills"]

    recent = (
        util_history.sort_values("week_start")
        .groupby("employee_id")
        .tail(8)
    )
    util_agg = recent.groupby("employee_id").agg(
        avg_util_8w=("utilization", "mean"),
        std_util_8w=("utilization", "std"),
        min_util_8w=("utilization", "min"),
        max_util_8w=("utilization", "max"),
        bench_weeks_8w=("on_bench", "sum"),
        overload_weeks_8w=("overloaded", "sum"),
        weeks_tracked=("utilization", "count"),
    ).reset_index()
    util_agg["std_util_8w"] = util_agg["std_util_8w"].fillna(0)

    trend_rows = []
    for eid, grp in recent.groupby("employee_id"):
        if len(grp) < 2:
            trend_rows.append({"employee_id": eid, "util_trend": 0.0})
            continue
        y = grp.sort_values("week_start")["utilization"].values
        x = range(len(y))
        slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
        trend_rows.append({"employee_id": eid, "util_trend": slope})
    util_trend = pd.DataFrame(trend_rows)

    alloc = allocations.copy()
    alloc["is_benched"] = alloc["project_id"].isna() | (alloc["allocation_pct"] < 0.1)
    alloc_feat = alloc.groupby("employee_id").agg(
        current_allocation=("allocation_pct", "max"),
        is_benched_now=("is_benched", "max"),
    ).reset_index()

    feat = employees.merge(util_agg, on="employee_id", how="left")
    feat = feat.merge(util_trend, on="employee_id", how="left")
    feat = feat.merge(alloc_feat, on="employee_id", how="left")

    feat["seniority_ord"] = feat["seniority"].map(
        {"Junior": 0, "Mid": 1, "Senior": 2, "Lead": 3}
    ).fillna(1)
    feat["skill_count"] = feat[_skill_columns(feat)].sum(axis=1)
    feat["cost_per_util"] = feat["hourly_cost"] / feat["avg_util_8w"].clip(lower=0.1)

    if projects is not None:
        skill_demand_map = {}
        for skill in skills:
            count = sum(1 for rs in projects["required_skills"] if skill in rs.split(","))
            skill_demand_map[skill] = count
        feat["primary_skill_demand"] = feat["primary_skill"].map(skill_demand_map).fillna(0)

    fill_cols = [
        "avg_util_8w", "std_util_8w", "min_util_8w", "max_util_8w",
        "bench_weeks_8w", "overload_weeks_8w", "util_trend",
        "current_allocation", "is_benched_now",
    ]
    for c in fill_cols:
        if c in feat.columns:
            feat[c] = feat[c].fillna(0 if c != "is_benched_now" else False)

    return feat


def build_project_features(
    projects: pd.DataFrame,
    allocations: pd.DataFrame,
    employees: pd.DataFrame,
    util_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    alloc = allocations.dropna(subset=["project_id"])
    staffed = alloc.groupby("project_id").agg(
        assigned_fte=("allocation_pct", "sum"),
        headcount=("employee_id", "nunique"),
    ).reset_index()

    feat = projects.merge(staffed, on="project_id", how="left")
    feat["assigned_fte"] = feat["assigned_fte"].fillna(0)
    feat["headcount"] = feat["headcount"].fillna(0)
    feat["fte_gap"] = feat["required_fte"] - feat["assigned_fte"]
    feat["staffing_ratio"] = feat["assigned_fte"] / feat["required_fte"].clip(lower=0.5)
    feat["status_at_risk"] = (feat["status"] == "At Risk").astype(int)
    feat["status_active"] = (feat["status"] == "Active").astype(int)
    feat["urgency"] = feat["priority"] / feat["deadline_weeks"].clip(lower=1)
    feat["n_required_skills"] = feat["required_skills"].str.count(",") + 1

    return feat


def employee_model_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = {
        "employee_id", "name", "seniority", "primary_skill", "location",
        "bench_risk", "burnout_risk", "completion_risk",
    }
    skill_cols = _skill_columns(features)
    cat_cols = ["seniority_ord", "skill_count", "is_benched_now"] + skill_cols
    num_cols = [
        c for c in features.columns
        if c not in exclude
        and c not in skill_cols
        and features[c].dtype in ["float64", "float32", "int64", "int32", "bool"]
    ]
    use_cols = list(dict.fromkeys(
        [c for c in num_cols if c in features.columns]
        + [c for c in skill_cols + ["seniority_ord", "skill_count", "is_benched_now"] if c in features.columns]
    ))
    X = features[use_cols].copy()
    for c in X.columns:
        if getattr(X[c], "dtype", None) == bool or str(X[c].dtype) == "bool":
            X[c] = X[c].astype(int)
    X = X.fillna(0)
    return X, list(X.columns)


def project_model_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = {
        "project_id", "name", "status", "required_skills", "completion_risk",
    }
    use_cols = [
        c for c in features.columns
        if c not in exclude and features[c].dtype in ["float64", "float32", "int64", "int32"]
    ]
    X = features[use_cols].fillna(0)
    return X, list(X.columns)
