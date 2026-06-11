"""
FIN.DR — /api/voice/* routes (browser-driven voice flow)

New flow:
  - POST /api/voice/start    -> returns first question (phase=QUESTION)
  - POST /api/voice/respond  -> accepts recognized text and returns next question or final result
  - GET  /api/voice/status/<txn_id> -> returns final DB result or not-started
  - POST /api/voice/input/<txn_id>  -> keyboard fallback that forwards to respond
  - POST /api/voice/verify/<txn_id>  -> initiate voice verification for any existing transaction
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from voice import verifier
from database.db import get_voice_status, get_transaction_by_id

voice_bp = Blueprint("voice", __name__)


@voice_bp.route("/api/voice/start", methods=["POST"])
def voice_start():
    data = request.get_json(force=True) or {}
    txn_id = data.get("transaction_id") or data.get("txn_id") or "UNKNOWN"
    amount = float(data.get("amount", 0))
    merchant = data.get("merchant", "Unknown")
    timestamp = data.get("timestamp", "Unknown time")

    res = verifier.start_session(txn_id, amount, merchant, timestamp)
    return jsonify({"success": True, "transaction_id": txn_id, **res})


@voice_bp.route("/api/voice/respond", methods=["POST"])
def voice_respond():
    data = request.get_json(force=True) or {}
    txn_id = data.get("transaction_id") or data.get("txn_id") or "UNKNOWN"
    text = data.get("response", "").strip()

    res = verifier.respond_to_session(txn_id, text)
    # normalize response shape
    return jsonify({"success": True, "transaction_id": txn_id, **res})


@voice_bp.route("/api/voice/status/<txn_id>", methods=["GET"])
def voice_status(txn_id: str):
    # Check DB for finalised result
    db_result = get_voice_status(txn_id)
    if db_result:
        return jsonify({"phase": "COMPLETE", **db_result})
    return jsonify({"phase": "NOT_STARTED"})


@voice_bp.route("/api/voice/input/<txn_id>", methods=["POST"])
def voice_keyboard_input(txn_id: str):
    """Keyboard fallback — forwards the text as a response to the session."""
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    res = verifier.respond_to_session(txn_id, text)
    return jsonify({"success": True, "transaction_id": txn_id, **res})


@voice_bp.route("/api/voice/verify/<txn_id>", methods=["POST"])
def voice_verify_transaction(txn_id: str):
    """
    Initiate voice verification for an existing transaction (from transactions list).
    Looks up transaction details and starts a new verification session.
    """
    try:
        txn = get_transaction_by_id(txn_id)
        if not txn:
            return jsonify({
                "success": False, 
                "error": f"Transaction {txn_id} not found"
            }), 404
        
        # Start a new voice verification session for this transaction
        res = verifier.start_session(
            txn_id=txn_id,
            amount=txn['amount'],
            merchant=txn['merchant'],
            timestamp=txn['timestamp']
        )
        return jsonify({"success": True, "transaction_id": txn_id, **res, "transaction": dict(txn)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

