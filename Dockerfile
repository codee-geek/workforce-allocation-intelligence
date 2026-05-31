FROM python:3.11-slim-bookworm

WORKDIR /app

# Prophet / cmdstan build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train models at build time so the container starts quickly
ENV WORKFORCE_CONFIG=config.cloud.yaml
RUN python scripts/run_full_pipeline.py

ENV WORKFORCE_CONFIG=config.cloud.yaml
EXPOSE 8501

HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
