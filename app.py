"""
FIN.DR — Flask application entry point
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS
from database.db import init_db
from routes.predict   import predict_bp
from routes.voice     import voice_bp
from routes.dashboard import dashboard_bp
import subprocess
import time

from config import OLLAMA_EXE


def start_local_ollama():

    subprocess.Popen(
        [OLLAMA_EXE, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = "findr-secret-2026"
    app.config["JSON_SORT_KEYS"] = False

    CORS(app)

    # Blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(voice_bp)
    app.register_blueprint(dashboard_bp)

    # DB
    init_db()

    # Warm up models (non-blocking — loads in background)
    import threading
    def _warm():
        from utils.model_loader import get_loader
        get_loader()
    threading.Thread(target=_warm, daemon=True).start()

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 55)
    print("  FIN.DR - AI Fraud Detection System")
    print("  Dashboard -> http://localhost:5000")
    print("=" * 55 + "\n")
    start_local_ollama()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
