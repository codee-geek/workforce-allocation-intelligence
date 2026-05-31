"""
Workforce Allocation Intelligence Dashboard
Run: streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Streamlit Cloud secrets / deploy env
if "WORKFORCE_CONFIG" not in os.environ:
    try:
        if "WORKFORCE_CONFIG" in st.secrets:
            os.environ["WORKFORCE_CONFIG"] = st.secrets["WORKFORCE_CONFIG"]
    except Exception:
        pass
os.environ.setdefault("WORKFORCE_CONFIG", "config.cloud.yaml")

from src.config import load_config
from src.pipeline.orchestrator import WorkforceIntelligencePipeline

st.set_page_config(
    page_title="Workforce Allocation Intelligence",
    page_icon="📊",
    layout="wide",
)

config = load_config()
DATA_DIR = ROOT / config["paths"]["data_dir"]
MODELS_DIR = ROOT / config["paths"]["models_dir"]


@st.cache_resource
def get_pipeline():
    return WorkforceIntelligencePipeline(config)


def ensure_results():
    pipeline = get_pipeline()
    if not (DATA_DIR / "employee_risk_predictions.csv").exists():
        with st.spinner("Running pipeline (generate → train → optimize)..."):
            if not (DATA_DIR / "employees.csv").exists():
                pipeline.generate_data()
            if not (MODELS_DIR / "risk_predictor.joblib").exists():
                pipeline.train()
            pipeline.run_inference()
    return pipeline


def main():
    st.title("Workforce Allocation Intelligence System")
    st.caption(
        f"IT Services workforce optimizer — {config['company']['num_employees']} employees, "
        f"{config['company']['num_projects']} projects | "
        "XGBoost · Prophet · OR-Tools"
    )

    pipeline = get_pipeline()

    col_run, col_refresh = st.columns([3, 1])
    with col_run:
        if st.button("🔄 Run Full Pipeline", type="primary"):
            with st.spinner("Generating data, training models, optimizing..."):
                pipeline.run_full()
            st.success("Pipeline complete!")
            st.rerun()
    with col_refresh:
        if st.button("Refresh View"):
            st.rerun()

    ensure_results()

    employees = pd.read_csv(DATA_DIR / "employees.csv")
    projects = pd.read_csv(DATA_DIR / "projects.csv")
    emp_risks = pd.read_csv(DATA_DIR / "employee_risk_predictions.csv")
    proj_risks = pd.read_csv(DATA_DIR / "project_risk_predictions.csv")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees", len(employees))
    c2.metric("Bench Alerts", int(emp_risks["bench_alert"].sum()))
    c3.metric("Burnout Alerts", int(emp_risks["burnout_alert"].sum()))
    c4.metric("Project Risk Alerts", int(proj_risks["completion_alert"].sum()))
    opt_path = DATA_DIR / "optimization_metrics.json"
    if opt_path.exists():
        opt = json.loads(opt_path.read_text())
        c5.metric("Solver", opt.get("solver_status", "—").upper())
    else:
        c5.metric("Solver", "—")

    tab_overview, tab_risks, tab_forecast, tab_alloc, tab_data = st.tabs(
        ["Overview", "Risk Analysis", "Demand Forecast", "Allocation", "Raw Data"]
    )

    with tab_overview:
        st.subheader("Risk Distribution")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_bench = px.histogram(
                emp_risks, x="bench_risk_score", nbins=30,
                title="Bench Risk Scores", color_discrete_sequence=["#e74c3c"],
            )
            fig_bench.add_vline(
                x=config["risk_thresholds"]["bench_high"],
                line_dash="dash", annotation_text="Alert threshold",
            )
            st.plotly_chart(fig_bench, use_container_width=True)
        with col_b:
            fig_burn = px.histogram(
                emp_risks, x="burnout_risk_score", nbins=30,
                title="Burnout Risk Scores", color_discrete_sequence=["#f39c12"],
            )
            fig_burn.add_vline(
                x=config["risk_thresholds"]["burnout_high"],
                line_dash="dash", annotation_text="Alert threshold",
            )
            st.plotly_chart(fig_burn, use_container_width=True)

        fig_scatter = px.scatter(
            emp_risks,
            x="bench_risk_score",
            y="burnout_risk_score",
            color="seniority",
            hover_data=["name", "primary_skill"],
            title="Bench vs Burnout Risk (employees)",
            opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Project Completion Risk")
        fig_proj = px.bar(
            proj_risks.sort_values("completion_risk_score", ascending=False).head(15),
            x="name", y="completion_risk_score", color="status",
            title="Top 15 Projects by Completion Risk",
        )
        fig_proj.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_proj, use_container_width=True)

    with tab_risks:
        risk_type = st.selectbox("View", ["Bench alerts", "Burnout alerts", "All high risk"])
        if risk_type == "Bench alerts":
            show = emp_risks[emp_risks["bench_alert"]].sort_values("bench_risk_score", ascending=False)
        elif risk_type == "Burnout alerts":
            show = emp_risks[emp_risks["burnout_alert"]].sort_values("burnout_risk_score", ascending=False)
        else:
            show = emp_risks[
                emp_risks["bench_alert"] | emp_risks["burnout_alert"]
            ].sort_values("burnout_risk_score", ascending=False)
        st.dataframe(show, use_container_width=True, height=400)

        st.subheader("At-Risk Projects")
        st.dataframe(
            proj_risks[proj_risks["completion_alert"]].sort_values(
                "completion_risk_score", ascending=False
            ),
            use_container_width=True,
        )

    with tab_forecast:
        fc_path = DATA_DIR / "demand_forecasts.csv"
        if fc_path.exists():
            fc = pd.read_csv(fc_path, parse_dates=["week_start"])
            skill = st.selectbox("Skill", sorted(fc["skill"].unique()))
            skill_fc = fc[fc["skill"] == skill]
            hist = pd.read_csv(DATA_DIR / "skill_demand.csv", parse_dates=["week_start"])
            hist_skill = hist[hist["skill"] == skill]

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_skill["week_start"], y=hist_skill["demand_fte"],
                name="Historical", mode="lines", line=dict(color="#3498db"),
            ))
            fig_fc.add_trace(go.Scatter(
                x=skill_fc["week_start"], y=skill_fc["forecast_fte"],
                name="Forecast", mode="lines", line=dict(color="#2ecc71", dash="dash"),
            ))
            fig_fc.add_trace(go.Scatter(
                x=skill_fc["week_start"], y=skill_fc["forecast_upper"],
                fill=None, mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_fc.add_trace(go.Scatter(
                x=skill_fc["week_start"], y=skill_fc["forecast_lower"],
                fill="tonexty", mode="lines", line=dict(width=0),
                name="Confidence band", fillcolor="rgba(46,204,113,0.2)",
            ))
            fig_fc.update_layout(
                title=f"Skill Demand Forecast: {skill}",
                xaxis_title="Week", yaxis_title="FTE Demand",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            summary_path = DATA_DIR / "skill_demand_summary.csv"
            if summary_path.exists():
                st.subheader("Next-Week Demand by Skill")
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        else:
            st.warning("Run pipeline to generate forecasts.")

    with tab_alloc:
        alloc_path = DATA_DIR / "recommended_allocations.csv"
        if alloc_path.exists():
            alloc = pd.read_csv(alloc_path)
            st.subheader("OR-Tools Recommended Assignments")
            if opt_path.exists():
                st.json(json.loads(opt_path.read_text()))
            filter_skill = st.text_input("Filter by employee skill (optional)")
            display = alloc
            if filter_skill:
                emp_skills = employees[
                    employees["primary_skill"].str.contains(filter_skill, case=False, na=False)
                ]["employee_id"]
                display = alloc[alloc["employee_id"].isin(emp_skills)]
            st.dataframe(display, use_container_width=True, height=450)

            proj_counts = alloc.groupby("project_id").size().reset_index(name="assignments")
            fig_alloc = px.bar(proj_counts, x="project_id", y="assignments", title="Assignments per Project")
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.warning("No allocation recommendations yet.")

    with tab_data:
        st.subheader("Employees (sample)")
        st.dataframe(employees.head(20), use_container_width=True)
        st.subheader("Projects")
        st.dataframe(projects, use_container_width=True)
        metrics_path = MODELS_DIR / "training_metrics.json"
        if metrics_path.exists():
            st.subheader("Model Training Metrics")
            st.json(json.loads(metrics_path.read_text()))


if __name__ == "__main__":
    main()
