"""
FIN.DR — App-wide configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Score thresholds ──────────────────────────────────────────
APPROVE_THRESHOLD   = 0.30
BLOCK_THRESHOLD     = 0.70

# ── Model paths ───────────────────────────────────────────────
TF_MODEL_PATH          = os.path.join(BASE_DIR, "models", "tensorflow_model.keras")
TRANSFORMER_MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_model.pt")
ISOLATION_MODEL_PATH   = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
SCALER_PATH            = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_COLS_PATH      = os.path.join(BASE_DIR, "models", "feature_cols.json")

# ── Database ──────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "database", "findr.db")

# ── Fusion weights ────────────────────────────────────────────
TF_WEIGHT          = 0.45
TRANSFORMER_WEIGHT = 0.45
ANOMALY_WEIGHT     = 0.10

# ── Voice settings ────────────────────────────────────────────
VOICE_LISTEN_TIMEOUT   = 15   # seconds
VOICE_MAX_RETRIES      = 3

# ── Ollama (local LLaMA3) ─────────────────────────────────────
'''OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
USE_OLLAMA   = True   # set False to skip LLM explanation
'''

OLLAMA_EXE = os.path.join(
    BASE_DIR,
    "ollama",
    "ollama.exe"
)

OLLAMA_MODEL = "llama3.2:3b"
# Host for Ollama HTTP API (local Ollama server). Keep as localhost when
# running Ollama alongside the server. If you use a remote Ollama, change this.
OLLAMA_HOST = "http://localhost:11434"

# Toggle whether to call Ollama for explanations. Set False to skip LLM calls.
USE_OLLAMA = True

USE_ONLY_LOCAL_OLLAMA = True