"""Central configuration for the AGCT backend."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
BACKEND_DATA_DIR = BACKEND_DIR / "data"
BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE_DIR = BACKEND_DATA_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

DATA_PATH = str(DATA_DIR)
DB_PATH = str(BACKEND_DATA_DIR / "session.db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "240"))

CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
FAISS_K = 3
DEFAULT_DIFFICULTY = "moderate"
DEFAULT_VERIFICATION_THRESHOLD = 0.72

DIFFICULTY_THRESHOLDS = {
    "basic": 0.6,
    "moderate": 0.8,
    "advanced": 1.0,
}
