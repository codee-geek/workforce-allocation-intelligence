"""
Streamlit Cloud entry point.
Set main file to streamlit_app.py in Streamlit Cloud settings, or use app/dashboard.py directly.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Default to cloud-sized config on hosted platforms
os.environ.setdefault("WORKFORCE_CONFIG", "config.cloud.yaml")

runpy.run_path(str(ROOT / "app" / "dashboard.py"), run_name="__main__")
