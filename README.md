# Workforce Allocation Intelligence System

AI-powered workforce planning for IT services companies. Predicts **bench risk**, **burnout risk**, and **project completion risk**, forecasts skill demand, and recommends optimal employee–project allocations.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Synthetic Data  │────▶│ Feature Engine   │────▶│ XGBoost Models  │
│ 500 emp × 50 pr │     │ Utilization, FTE │     │ Bench/Burnout/  │
└─────────────────┘     └──────────────────┘     │ Completion Risk │
                              │                    └────────┬────────┘
                              ▼                             │
                       ┌──────────────────┐                 │
                       │ Prophet / ARIMA  │                 ▼
                       │ Skill Demand Fcst│          ┌──────────────┐
                       └────────┬─────────┘          │  OR-Tools    │
                                │                    │  Optimizer   │
                                └───────────────────▶│  Allocation  │
                                                     └──────────────┘
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| Risk prediction | **XGBoost** | Bench, burnout, completion risk scores (0–1) |
| Demand forecasting | **Prophet** (or ARIMA) | Weekly FTE demand per skill, 8-week horizon |
| Allocation | **OR-Tools** (SCIP/GLOP) | Minimize bench/overload while filling project gaps |

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run everything (data → train → predict → optimize)
python scripts/run_full_pipeline.py

# Launch interactive dashboard
streamlit run app/dashboard.py
```

## Project Structure

```
├── config.yaml              # Company size, skills, model params
├── data/                    # Generated CSVs & predictions
├── models/                  # Trained XGBoost & forecast artifacts
├── src/
│   ├── data/generator.py    # Synthetic 500×50 dataset
│   ├── features/            # Feature engineering
│   ├── models/              # XGBoost + Prophet/ARIMA
│   ├── optimization/        # OR-Tools allocator
│   ├── pipeline/            # End-to-end orchestration
│   └── api/main.py          # FastAPI REST API
├── app/dashboard.py         # Streamlit UI
└── scripts/                 # CLI entry points
```

## Scripts

| Command | Description |
|---------|-------------|
| `python scripts/generate_data.py` | Generate synthetic workforce data |
| `python scripts/train_models.py` | Train XGBoost + Prophet models |
| `python scripts/run_allocation.py` | Predict risks & run optimizer |
| `python scripts/run_full_pipeline.py` | All steps in sequence |

## API

```bash
uvicorn src.api.main:app --reload --app-dir .
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status |
| `/pipeline/generate` | POST | Generate data |
| `/pipeline/train` | POST | Train models |
| `/pipeline/run` | POST | Full inference + optimization |
| `/risks/employees` | GET | Employee risk scores |
| `/risks/projects` | GET | Project completion risks |
| `/forecast/demand` | GET | Skill demand forecasts |
| `/allocations/recommended` | GET | OR-Tools assignments |

## Configuration

Edit `config.yaml` to adjust:

- `company.num_employees` / `num_projects`
- `models.forecast.method`: `prophet` or `arima`
- `risk_thresholds`: alert cutoffs
- `optimization.*`: penalty weights for bench, overload, skill mismatch

## Models

### XGBoost (Risk Prediction)

**Employee targets:** `bench_risk`, `burnout_risk`  
Features: 8-week utilization stats, allocation %, seniority, skill breadth, trends.

**Project target:** `completion_risk`  
Features: FTE gap, staffing ratio, deadline urgency, status flags.

### Prophet (Demand Forecasting)

Per-skill weekly FTE demand with 8-week forward forecast and confidence bands.

### OR-Tools (Optimization)

Linear program minimizing:
- Bench slack (under-target utilization)
- Overload slack (burnout)
- Unfilled project FTE gaps (weighted by completion risk × priority)
- Skill mismatch penalties

## License

MIT
