"""
FIN.DR — Fusion Engine
Combines TF + Transformer + Anomaly scores into a single risk decision.
Supports context-aware score adjustment based on transaction type and merchant.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (APPROVE_THRESHOLD, BLOCK_THRESHOLD,
                    TF_WEIGHT, TRANSFORMER_WEIGHT, ANOMALY_WEIGHT)


DECISION_LABELS = {
    "APPROVE":           {"label": "APPROVE",            "color": "#00ff88", "icon": "\u2705"},
    "CALL_USER":         {"label": "CALL USER",          "color": "#ffaa00", "icon": "\U0001f4de"},
    "BLOCK_TRANSACTION": {"label": "BLOCK TRANSACTION",  "color": "#ff4444", "icon": "\U0001f6ab"},
}

# ── Context tables ─────────────────────────────────────────────

# Multipliers applied to the raw fused score BEFORE threshold comparison.
# < 1.0 = lower risk for this transaction type; 1.0 = no change.
TXN_TYPE_MULTIPLIERS = {
    "self_transfer":   0.30,   # Sending to own account — almost always legit
    "salary_credit":  0.20,   # Salary / income credit
    "bill_payment":   0.50,   # Utility / insurance / subscription bills
    "emi":            0.55,   # Scheduled EMI payment
    "upi_transfer":   0.75,   # UPI to known contact
    "purchase":       1.00,   # Standard retail purchase
    "cash_withdrawal":1.00,   # ATM withdrawal — no reduction
}

# Known-trusted merchant keywords (case-insensitive substring match)
TRUSTED_MERCHANTS = {
    "amazon", "flipkart", "myntra", "meesho", "nykaa", "swiggy", "zomato",
    "apple", "samsung", "sony", "lg", "croma", "vijay sales", "reliance digital",
    "hdfc", "icici", "sbi", "axis", "kotak", "yes bank",
    "phonepe", "paytm", "gpay", "google pay", "bhim",
    "neft", "imps", "rtgs", "irctc", "makemytrip", "bookmyshow",
    "lic", "nps", "mutual fund", "zerodha", "groww",
}


def fuse_scores(
    tf_score: float,
    transformer_score: float,
    anomaly_score: float,
    txn_type: str = "purchase",
    merchant: str = "",
) -> dict:
    """
    Weighted fusion:
        raw = 0.45*tf + 0.45*transformer + 0.10*anomaly
    Context adjustment:
        final = raw * txn_type_multiplier * merchant_multiplier
    Decision thresholds:
        < 0.30  -> APPROVE
        0.30-0.70 -> CALL_USER
        > 0.70  -> BLOCK_TRANSACTION
    """
    raw_score = (
        TF_WEIGHT          * tf_score +
        TRANSFORMER_WEIGHT * transformer_score +
        ANOMALY_WEIGHT     * anomaly_score
    )
    raw_score = max(0.0, min(1.0, raw_score))

    # ── Context adjustment ─────────────────────────────────────
    txn_mult = TXN_TYPE_MULTIPLIERS.get(txn_type, 1.0)

    # Trusted merchant keyword check
    merchant_lower = merchant.lower()
    merchant_mult = 1.0
    for kw in TRUSTED_MERCHANTS:
        if kw in merchant_lower:
            merchant_mult = 0.70   # known merchant -> 30% score reduction
            break

    final_score = round(max(0.0, min(1.0, raw_score * txn_mult * merchant_mult)), 4)

    context_note = ""
    if txn_mult < 1.0:
        label_map = {
            "self_transfer":  "Self-transfer to own account",
            "salary_credit":  "Salary / income credit",
            "bill_payment":   "Scheduled bill payment",
            "emi":            "Scheduled EMI payment",
            "upi_transfer":   "UPI transfer to contact",
        }
        context_note = label_map.get(txn_type, "")
    if merchant_mult < 1.0:
        context_note = (context_note + "; " if context_note else "") + f"Recognised trusted merchant ({merchant})"

    if final_score < APPROVE_THRESHOLD:
        decision = "APPROVE"
    elif final_score <= BLOCK_THRESHOLD:
        decision = "CALL_USER"
    else:
        decision = "BLOCK_TRANSACTION"

    reason = _build_reason(
        tf_score, transformer_score, anomaly_score,
        raw_score, final_score, decision, context_note
    )

    return {
        "final_score":        final_score,
        "raw_score":          round(raw_score, 4),
        "context_note":       context_note,
        "decision":           decision,
        "decision_meta":      DECISION_LABELS[decision],
        "reason":             reason,
        "weights": {
            "tf":          TF_WEIGHT,
            "transformer": TRANSFORMER_WEIGHT,
            "anomaly":     ANOMALY_WEIGHT,
        },
    }


def _build_reason(
    tf: float, trans: float, ano: float,
    raw: float, final: float, decision: str,
    context_note: str = ""
) -> str:
    flags = []

    if tf > 0.6:
        flags.append(f"Deep learning model flagged high fraud probability ({tf:.0%})")
    if trans > 0.6:
        flags.append(f"Transformer model detected suspicious pattern ({trans:.0%})")
    if ano > 0.5:
        flags.append(f"Anomaly detector found unusual behaviour ({ano:.0%})")
    if tf > 0.4 and trans > 0.4:
        flags.append("Both neural networks agree on elevated risk")

    if not flags:
        if raw < 0.15:
            flags.append("All models report low-risk transaction profile")
        else:
            flags.append(f"Combined model score ({raw:.0%}) within normal range")

    if context_note:
        flags.append(f"Context reduced risk: {context_note}")

    prefix = {
        "APPROVE":           "Transaction approved",
        "CALL_USER":         "Verification required",
        "BLOCK_TRANSACTION": "Transaction blocked",
    }[decision]

    return prefix + " — " + "; ".join(flags) + "."
