"""
FIN.DR — /api/predict route
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from flask import Blueprint, request, jsonify
from database.db import insert_transaction, insert_prediction
from utils.model_loader import get_loader
from utils.fusion import fuse_scores
from config import OLLAMA_HOST, OLLAMA_MODEL, USE_OLLAMA

predict_bp = Blueprint("predict", __name__)


def _get_llm_explanation(txn: dict, scores: dict, decision: str, reason: str) -> str:
    """Call local Ollama LLaMA3 for a concise plain-English explanation."""
    if not USE_OLLAMA:
        return reason
    try:
        import requests as req
        prompt = f"""You are a senior fraud analyst at a bank.
Transaction: ₹{txn.get('amount')} at {txn.get('merchant')} on {txn.get('timestamp')}.
AI scores: TensorFlow={scores['tf_score']:.2%}, Transformer={scores['transformer_score']:.2%}, Anomaly={scores['anomaly_score']:.2%}.
Final risk score: {scores['final_score']:.2%}. Decision: {decision}.
Provide a concise (2-3 sentence) plain-English explanation for this decision that would be shown to a bank analyst.
Focus on WHY this transaction was flagged. Be specific and professional."""

        resp = req.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("response", reason).strip()
    except Exception as e:
        print(f"[LLM] Ollama unavailable: {e}")
    return reason


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    # Build transaction record
    txn_id = str(uuid.uuid4())[:8].upper()
    txn = {
        "id":        txn_id,
        "user_id":   data.get("user_id", "USR001"),
        "amount":    data.get("amount", 100.0),
        "merchant":  data.get("merchant", "Unknown Merchant"),
        "timestamp": data.get("timestamp", "2026-06-10 10:00"),
        "card_type": data.get("card_type", "VISA"),
        "location":  data.get("location", "Unknown"),
        "txn_type":  data.get("txn_type", "purchase"),
    }

    # ML prediction
    loader = get_loader()
    raw_scores = loader.predict({**txn, **data})

    # Fusion
    fusion = fuse_scores(
        tf_score=raw_scores["tf_score"],
        transformer_score=raw_scores["transformer_score"],
        anomaly_score=raw_scores["anomaly_score"],
        txn_type=txn.get("txn_type", "purchase"),
        merchant=txn.get("merchant", ""),
    )

    all_scores = {**raw_scores, "final_score": fusion["final_score"]}
    decision = fusion["decision"]
    reason   = fusion["reason"]

    # LLM explanation (non-blocking, fast timeout)
    llm_exp = _get_llm_explanation(txn, all_scores, decision, reason)

    # Persist
    insert_transaction(txn)
    insert_prediction(txn_id, all_scores, decision, reason, llm_exp)

    return jsonify({
        "success":        True,
        "transaction_id": txn_id,
        "scores":         all_scores,
        "fusion":         fusion,
        "llm_explanation": llm_exp,
        "transaction":    txn,
    })
