"""
FIN.DR — Model Loader (singleton, thread-safe)
Loads TF, Transformer, and IsolationForest once at startup.
Falls back to calibrated random scores if models are not yet trained.
"""
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from config import (TF_MODEL_PATH, TRANSFORMER_MODEL_PATH,
                    ISOLATION_MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH)

_lock = threading.Lock()
_instance = None


class ModelLoader:
    def __init__(self):
        self.tf_model          = None
        self.transformer_model = None
        self.iso_data          = None
        self.scaler            = None
        self.feature_cols      = []
        self._load_all()

    # ── Loaders ────────────────────────────────────────────────

    def _load_tf(self):
        try:
            import tensorflow as tf
            self.tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
            print("[ModelLoader] ✅ TensorFlow model loaded")
        except Exception as e:
            print(f"[ModelLoader] ⚠️  TF model not found: {e}")

    def _load_transformer(self):
        try:
            import torch
            from training.train_transformer import TabularTransformer
            checkpoint = torch.load(TRANSFORMER_MODEL_PATH, map_location="cpu", weights_only=False)
            model = TabularTransformer(
                n_features=checkpoint["n_features"],
                d_model=checkpoint.get("d_model", 64),
                nhead=checkpoint.get("nhead", 4),
                num_layers=checkpoint.get("num_layers", 2),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.transformer_model = model
            print("[ModelLoader] ✅ Transformer model loaded")
        except Exception as e:
            print(f"[ModelLoader] ⚠️  Transformer model not found: {e}")

    def _load_iso(self):
        try:
            self.iso_data = joblib.load(ISOLATION_MODEL_PATH)
            print("[ModelLoader] ✅ Isolation Forest loaded")
        except Exception as e:
            print(f"[ModelLoader] ⚠️  IsolationForest not found: {e}")

    def _load_scaler(self):
        try:
            self.scaler = joblib.load(SCALER_PATH)
            with open(FEATURE_COLS_PATH) as f:
                self.feature_cols = json.load(f)
            print(f"[ModelLoader] ✅ Scaler loaded ({len(self.feature_cols)} features)")
        except Exception as e:
            print(f"[ModelLoader] ⚠️  Scaler not found: {e}")

    def _load_all(self):
        self._load_tf()
        self._load_transformer()
        self._load_iso()
        self._load_scaler()

    # ── Feature builder ────────────────────────────────────────

    def _build_feature_vector(self, txn: dict) -> np.ndarray:
        """Map incoming API dict → numpy array aligned to training features."""
        if not self.feature_cols:
            return np.zeros((1, 26), dtype=np.float32)

        row = {}
        for col in self.feature_cols:
            # Try exact match first
            if col in txn:
                row[col] = float(txn[col])
            elif col == "TransactionAmt":
                row[col] = float(txn.get("amount", 100.0))
            elif col == "TransactionDT":
                row[col] = float(txn.get("timestamp_dt", 86400.0))
            elif col == "DeviceType":
                row[col] = 1.0 if str(txn.get("device", "desktop")).lower() == "mobile" else 0.0
            else:
                row[col] = 0.0

        vec = np.array([[row[c] for c in self.feature_cols]], dtype=np.float32)

        if self.scaler:
            try:
                vec = self.scaler.transform(vec)
            except Exception:
                pass
        return vec

    # ── Prediction ─────────────────────────────────────────────

    def predict(self, txn: dict) -> dict:
        """
        Returns dict with tf_score, transformer_score, anomaly_score (all 0-1).
        Falls back to calibrated random if a model isn't loaded.
        """
        X = self._build_feature_vector(txn)
        amount = float(txn.get("amount", 100))

        # ── TensorFlow ──
        if self.tf_model is not None:
            try:
                tf_score = float(self.tf_model.predict(X, verbose=0)[0][0])
            except Exception as e:
                print(f"[ModelLoader] TF predict error: {e}")
                tf_score = self._synthetic_score(amount)
        else:
            tf_score = self._synthetic_score(amount)

        # ── Transformer ──
        if self.transformer_model is not None:
            try:
                import torch
                with torch.no_grad():
                    t = torch.tensor(X, dtype=torch.float32)
                    transformer_score = float(self.transformer_model(t).item())
            except Exception as e:
                print(f"[ModelLoader] Transformer predict error: {e}")
                transformer_score = self._synthetic_score(amount)
        else:
            transformer_score = self._synthetic_score(amount)

        # ── Isolation Forest ──
        if self.iso_data is not None:
            try:
                iso_model = self.iso_data["model"]
                min_s     = self.iso_data["min_score"]
                max_s     = self.iso_data["max_score"]
                raw = iso_model.decision_function(X)[0]
                anomaly_score = float(1 - (raw - min_s) / (max_s - min_s + 1e-8))
                anomaly_score = max(0.0, min(1.0, anomaly_score))
            except Exception as e:
                print(f"[ModelLoader] IsoForest predict error: {e}")
                anomaly_score = self._synthetic_score(amount, noise=0.3)
        else:
            anomaly_score = self._synthetic_score(amount, noise=0.3)

        return {
            "tf_score":          round(tf_score, 4),
            "transformer_score": round(transformer_score, 4),
            "anomaly_score":     round(anomaly_score, 4),
        }

    @staticmethod
    def _synthetic_score(amount: float, noise: float = 0.15) -> float:
        """Calibrated fallback score — high amounts = higher risk."""
        base = min(amount / 5000.0, 0.9)
        score = base + np.random.normal(0, noise)
        return float(max(0.01, min(0.99, score)))

    def reload(self):
        self._load_all()


def get_loader() -> ModelLoader:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ModelLoader()
    return _instance
