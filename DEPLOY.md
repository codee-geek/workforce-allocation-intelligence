# Deployment Guide

Deploy the **Streamlit dashboard** (recommended) or the **FastAPI** service.

| Platform | Best for | Cost |
|----------|----------|------|
| [Streamlit Community Cloud](#1-streamlit-community-cloud-recommended) | Demo / portfolio | Free |
| [Render](#2-render-docker) | Public URL + Docker | Free tier |
| [Docker local](#3-docker-local) | Testing before cloud | Free |

Cloud deploys use `config.cloud.yaml` (100 employees, 15 projects) for faster builds and cold starts.

---

## 1. Streamlit Community Cloud (recommended)

1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select repo `workforce-allocation-intelligence`.
4. Settings:
   - **Main file path:** `streamlit_app.py`
   - **Branch:** `main`
5. Click **Advanced settings** → **Secrets** and paste:

```toml
WORKFORCE_CONFIG = "config.cloud.yaml"
```

6. Click **Deploy**.

**First load:** The app runs the ML pipeline once if outputs are missing (~2–5 min). Later loads are fast if the instance stays warm.

**Live URL:** `https://<your-app>.streamlit.app`

---

## 2. Render (Docker)

1. Fork or use your GitHub repo.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect the repo; Render reads `render.yaml` and creates:
   - `workforce-dashboard` (Streamlit on port 8501)
   - `workforce-api` (FastAPI on port 8000)
4. Wait for the Docker build (pre-trains models in the image).

Or deploy only the dashboard:

- **New Web Service** → Docker → Dockerfile path: `Dockerfile`

---

## 3. Docker (local)

```bash
# Dashboard only
docker compose up dashboard

# Dashboard + API
docker compose up

# Dashboard: http://localhost:8501
# API:       http://localhost:8000/docs
```

Build manually:

```bash
docker build -t workforce-dashboard .
docker run -p 8501:8501 -e WORKFORCE_CONFIG=config.cloud.yaml workforce-dashboard
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `WORKFORCE_CONFIG` | Path to YAML config (`config.yaml` or `config.cloud.yaml`) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App times out on first open | Use `config.cloud.yaml`; click **Run Full Pipeline** once, or redeploy with Docker (models baked in at build) |
| Prophet / cmdstan errors | Use Docker or Render (includes `build-essential`); Streamlit Cloud usually works but first run is slow |
| Out of memory on free tier | Use `config.cloud.yaml`; reduce `num_employees` further in that file |

---

## Resume / portfolio

After deploy, add to your resume:

> Live demo: `https://<your-app>.streamlit.app` | GitHub: `https://github.com/codee-geek/workforce-allocation-intelligence`
