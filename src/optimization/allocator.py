"""OR-Tools based workforce allocation optimizer."""

from __future__ import annotations

import pandas as pd
from ortools.linear_solver import pywraplp

from src.config import load_config
from src.features.engineering import build_project_features


class WorkforceAllocator:
    """
    Assigns employees to projects to:
    - Fill project FTE gaps (weighted by completion risk & priority)
    - Minimize bench (under-utilization)
    - Minimize overload (burnout)
  - Respect skill match constraints
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.opt_cfg = self.config["optimization"]
        self.skills = self.config["skills"]

    def _employee_has_skill(self, emp_row: pd.Series, skill: str) -> bool:
        col = f"skill_{skill.replace(' ', '_').replace('/', '_')}"
        return bool(emp_row.get(col, False)) or emp_row.get("primary_skill") == skill

    def optimize(
        self,
        employees: pd.DataFrame,
        projects: pd.DataFrame,
        employee_risks: pd.DataFrame,
        project_risks: pd.DataFrame,
        allocations: pd.DataFrame,
        demand_forecast: pd.DataFrame | None = None,
        max_assignments_per_employee: int = 1,
    ) -> dict:
        proj_feat = build_project_features(projects, allocations, employees)
        active_projects = proj_feat[
            proj_feat["status"].isin(["Active", "At Risk", "Planning"])
        ].copy()
        proj_risk_map = project_risks.set_index("project_id")["completion_risk_score"].to_dict()
        bench_map = employee_risks.set_index("employee_id")["bench_risk_score"].to_dict()
        burnout_map = employee_risks.set_index("employee_id")["burnout_risk_score"].to_dict()

        current_alloc = allocations.set_index("employee_id")["allocation_pct"].to_dict()

        emp_ids = employees["employee_id"].tolist()
        proj_ids = active_projects["project_id"].tolist()

        if not proj_ids:
            return {
                "status": "no_active_projects",
                "assignments": pd.DataFrame(),
                "metrics": {},
            }

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            raise RuntimeError("No OR-Tools solver available")

        # Decision: fraction of employee e assigned to project p (0-1)
        x = {}
        for e in emp_ids:
            for p in proj_ids:
                x[e, p] = solver.NumVar(0, 1, f"x_{e}_{p}")

        # Slack: bench and overload per employee
        bench_slack = {e: solver.NumVar(0, 1, f"bench_{e}") for e in emp_ids}
        overload_slack = {e: solver.NumVar(0, 1, f"overload_{e}") for e in emp_ids}

        # Each employee total allocation <= max_utilization
        max_util = self.opt_cfg["max_utilization"]
        target = self.opt_cfg["target_utilization"]

        for e in emp_ids:
            solver.Add(sum(x[e, p] for p in proj_ids) <= max_util)

        if max_assignments_per_employee == 1:
            for e in emp_ids:
                # At most one "primary" project: sum of binary indicators approximated
                # via sum of fractions <= 1 (fractional allowed for partial FTE)
                pass  # fractional model already caps at max_util

        # Project staffing: assigned FTE should meet gap (soft)
        proj_slack = {}
        for _, proj in active_projects.iterrows():
            p = proj["project_id"]
            gap = max(0, float(proj["fte_gap"]))
            proj_slack[p] = solver.NumVar(0, gap + 5, f"gap_{p}")
            assigned = sum(x[e, p] for e in emp_ids)
            solver.Add(assigned + proj_slack[p] >= gap * 0.7)

        # Bench / overload linking
        for e in emp_ids:
            total = sum(x[e, p] for p in proj_ids)
            cur = current_alloc.get(e, 0)
            # bench when total < target
            solver.Add(bench_slack[e] >= target - total)
            solver.Add(overload_slack[e] >= total - target)

        # Skill mismatch penalty via constraints: discourage assignment if no skill match
        mismatch = {}
        proj_skill_req = {}
        for _, proj in active_projects.iterrows():
            p = proj["project_id"]
            req = proj["required_skills"].split(",")
            proj_skill_req[p] = req
            for e in emp_ids:
                emp_row = employees[employees["employee_id"] == e].iloc[0]
                match = any(self._employee_has_skill(emp_row, s.strip()) for s in req)
                mismatch[e, p] = 0 if match else 1

        objective = solver.Objective()
        bench_pen = self.opt_cfg["bench_penalty"]
        overload_pen = self.opt_cfg["overload_penalty"]
        mismatch_pen = self.opt_cfg["skill_mismatch_penalty"]
        completion_w = self.opt_cfg["completion_weight"]

        for e in emp_ids:
            br = bench_map.get(e, 0.3)
            bo = burnout_map.get(e, 0.3)
            objective.SetCoefficient(bench_slack[e], bench_pen * (1 + br))
            objective.SetCoefficient(overload_slack[e], overload_pen * (1 + bo))

        for p in proj_ids:
            risk = proj_risk_map.get(p, 0.5)
            priority = float(
                active_projects[active_projects["project_id"] == p]["priority"].iloc[0]
            )
            objective.SetCoefficient(proj_slack[p], completion_w * risk * priority)

        for e in emp_ids:
            for p in proj_ids:
                if mismatch.get((e, p), 0):
                    # Small assignment allowed but penalized
                    objective.SetCoefficient(x[e, p], mismatch_pen)
                else:
                    objective.SetCoefficient(x[e, p], -5 * proj_risk_map.get(p, 0.5))

        objective.SetMinimization()
        status = solver.Solve()

        status_name = {
            pywraplp.Solver.OPTIMAL: "optimal",
            pywraplp.Solver.FEASIBLE: "feasible",
        }.get(status, "infeasible")

        assignments = []
        for e in emp_ids:
            for p in proj_ids:
                val = x[e, p].solution_value()
                if val > 0.05:
                    emp_row = employees[employees["employee_id"] == e].iloc[0]
                    assignments.append(
                        {
                            "employee_id": e,
                            "employee_name": emp_row["name"],
                            "project_id": p,
                            "recommended_allocation": round(val, 3),
                            "bench_risk": round(bench_map.get(e, 0), 3),
                            "burnout_risk": round(burnout_map.get(e, 0), 3),
                            "skill_match": mismatch.get((e, p), 0) == 0,
                        }
                    )

        assign_df = pd.DataFrame(assignments)

        metrics = {
            "solver_status": status_name,
            "total_assignments": len(assign_df),
            "employees_placed": assign_df["employee_id"].nunique() if len(assign_df) else 0,
            "avg_bench_slack": sum(bench_slack[e].solution_value() for e in emp_ids) / len(emp_ids),
            "avg_overload_slack": sum(overload_slack[e].solution_value() for e in emp_ids) / len(emp_ids),
            "total_project_gap_slack": sum(proj_slack[p].solution_value() for p in proj_ids),
        }

        return {
            "status": status_name,
            "assignments": assign_df,
            "metrics": metrics,
        }
