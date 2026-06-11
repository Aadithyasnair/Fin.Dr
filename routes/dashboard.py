"""
FIN.DR — Dashboard + Analytics routes
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
from flask import Blueprint, request, jsonify, render_template
from database.db import get_recent_transactions, get_analytics_data

dashboard_bp = Blueprint("dashboard", __name__)

# Retrain job tracker
_retrain_jobs: dict = {}
_retrain_lock = threading.Lock()


@dashboard_bp.route("/")
def index():
    return render_template("dashboard.html")


@dashboard_bp.route("/api/transactions", methods=["GET"])
def transactions():
    limit        = int(request.args.get("limit", 50))
    status_filter = request.args.get("status", "ALL")
    rows = get_recent_transactions(limit=limit, status_filter=status_filter)
    return jsonify({"success": True, "transactions": rows, "count": len(rows)})


@dashboard_bp.route("/api/analytics", methods=["GET"])
def analytics():
    data = get_analytics_data()
    return jsonify({"success": True, **data})


@dashboard_bp.route("/api/retrain/<model_name>", methods=["POST"])
def retrain(model_name: str):
    valid = {"tensorflow", "transformer", "isolation"}
    if model_name not in valid:
        return jsonify({"success": False, "error": f"Unknown model: {model_name}"}), 400

    with _retrain_lock:
        if _retrain_jobs.get(model_name) == "running":
            return jsonify({"success": False, "error": "Already retraining"}), 409
        _retrain_jobs[model_name] = "running"

    def _do_retrain():
        try:
            if model_name == "tensorflow":
                from training.train_tensorflow import train
            elif model_name == "transformer":
                from training.train_transformer import train
            else:
                from training.train_isolation import train
            train()
            # Reload
            from utils.model_loader import get_loader
            get_loader().reload()
            with _retrain_lock:
                _retrain_jobs[model_name] = "done"
        except Exception as e:
            print(f"[Retrain] Error: {e}")
            with _retrain_lock:
                _retrain_jobs[model_name] = f"error: {e}"

    t = threading.Thread(target=_do_retrain, daemon=True)
    t.start()
    return jsonify({"success": True, "model": model_name, "status": "started"})


@dashboard_bp.route("/api/retrain/status", methods=["GET"])
def retrain_status():
    with _retrain_lock:
        return jsonify({"jobs": dict(_retrain_jobs)})


@dashboard_bp.route("/api/reload_models", methods=["POST"])
def reload_models():
    try:
        from utils.model_loader import get_loader
        get_loader().reload()
        return jsonify({"success": True, "message": "Models reloaded"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
