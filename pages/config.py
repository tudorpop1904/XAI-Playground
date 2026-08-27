"""
pages/config.py
===============
Centralized configuration for the Streamlit frontend.
Reads the API base URL from the environment so it works
both locally (localhost) and inside Docker (service name).
"""

import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
