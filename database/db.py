"""
FIN.DR — SQLite database setup and CRUD helpers
"""
import sqlite3
import os
from datetime import datetime
from config import DB_PATH


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id           TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL,
            amount       REAL NOT NULL,
            merchant     TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            card_type    TEXT DEFAULT 'VISA',
            location     TEXT DEFAULT 'Unknown',
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id     TEXT NOT NULL,
            tf_score           REAL,
            transformer_score  REAL,
            anomaly_score      REAL,
            final_score        REAL,
            decision           TEXT,
            reason             TEXT,
            llm_explanation    TEXT,
            created_at         TEXT NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions(id)
        );

        CREATE TABLE IF NOT EXISTS voice_verifications (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id  TEXT NOT NULL,
            status          TEXT,
            reason          TEXT,
            followup        TEXT,
            raw_response    TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions(id)
        );
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialised at", DB_PATH)


# ── Transactions ──────────────────────────────────────────────

def insert_transaction(txn: dict) -> str:
    import uuid
    txn_id = txn.get("id") or str(uuid.uuid4())[:8].upper()
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO transactions
            (id, user_id, amount, merchant, timestamp, card_type, location, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        txn_id,
        txn.get("user_id", "USR001"),
        float(txn.get("amount", 0)),
        txn.get("merchant", "Unknown"),
        txn.get("timestamp", datetime.now().isoformat()),
        txn.get("card_type", "VISA"),
        txn.get("location", "Unknown"),
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()
    return txn_id


def insert_prediction(txn_id: str, scores: dict, decision: str, reason: str, llm_exp: str = ""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO predictions
            (transaction_id, tf_score, transformer_score, anomaly_score,
             final_score, decision, reason, llm_explanation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        txn_id,
        round(float(scores.get("tf_score", 0)), 4),
        round(float(scores.get("transformer_score", 0)), 4),
        round(float(scores.get("anomaly_score", 0)), 4),
        round(float(scores.get("final_score", 0)), 4),
        decision,
        reason,
        llm_exp,
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


def insert_voice_result(txn_id: str, result: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO voice_verifications
            (transaction_id, status, reason, followup, raw_response, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        txn_id,
        result.get("status", "UNKNOWN"),
        result.get("reason", ""),
        result.get("followup", ""),
        result.get("raw_response", ""),
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


# ── Query helpers ─────────────────────────────────────────────

def get_transaction_by_id(txn_id: str) -> dict:
    """Fetch a single transaction by ID."""
    conn = get_conn()
    row = conn.execute("""
        SELECT t.id, t.user_id, t.amount, t.merchant, t.timestamp,
               t.card_type, t.location, t.created_at,
               p.tf_score, p.transformer_score, p.anomaly_score,
               p.final_score, p.decision, p.reason, p.llm_explanation,
               v.status AS voice_status
        FROM transactions t
        LEFT JOIN predictions p ON p.transaction_id = t.id
        LEFT JOIN voice_verifications v ON v.transaction_id = t.id
        WHERE t.id = ?
    """, (txn_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_transactions(limit: int = 50, status_filter: str = None) -> list:
    conn = get_conn()
    query = """
        SELECT t.id, t.user_id, t.amount, t.merchant, t.timestamp,
               t.card_type, t.location, t.created_at,
               p.tf_score, p.transformer_score, p.anomaly_score,
               p.final_score, p.decision, p.reason, p.llm_explanation,
               v.status AS voice_status
        FROM transactions t
        LEFT JOIN predictions p ON p.transaction_id = t.id
        LEFT JOIN voice_verifications v ON v.transaction_id = t.id
    """
    params = []
    if status_filter and status_filter != "ALL":
        query += " WHERE p.decision = ?"
        params.append(status_filter)
    query += " ORDER BY t.created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analytics_data() -> dict:
    from datetime import timedelta
    conn = get_conn()

    # Counts by decision
    counts = {}
    for row in conn.execute("""
        SELECT decision, COUNT(*) as cnt FROM predictions GROUP BY decision
    """).fetchall():
        counts[row["decision"]] = row["cnt"]

    # 7-day daily fraud trend
    trend = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN decision='BLOCK_TRANSACTION' THEN 1 ELSE 0 END) as fraud
            FROM predictions
            WHERE DATE(created_at) = ?
        """, (day,)).fetchone()
        trend.append({
            "date": day,
            "total": row["total"] or 0,
            "fraud": row["fraud"] or 0,
        })

    # Avg risk scores last 20
    timeline = conn.execute("""
        SELECT created_at, final_score, decision
        FROM predictions
        ORDER BY created_at DESC LIMIT 20
    """).fetchall()

    conn.close()
    return {
        "counts": counts,
        "trend": trend,
        "timeline": [dict(r) for r in timeline],
        "total": sum(counts.values()),
    }


def get_voice_status(txn_id: str) -> dict:
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM voice_verifications
        WHERE transaction_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (txn_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}
