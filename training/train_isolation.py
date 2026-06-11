"""
FIN.DR — Isolation Forest Training Script (anomaly detection)
Enhanced version using advanced preprocessing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from training.train_advanced import preprocess_advanced

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")


def train(data_dir: str = None):
    print("[IsoForest-Enhanced] Starting Isolation Forest training...")
    X_train, X_test, y_train, y_test, feature_cols, _ = preprocess_advanced(data_dir=data_dir)

    fraud_rate = y_train.mean()
    print(f"[IsoForest-Enhanced] Training with fraud rate: {fraud_rate:.2%}")
    
    model = IsolationForest(
        n_estimators=300,  # Increased from 200
        contamination=max(0.01, fraud_rate),  # At least 1% contamination
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train)

    # Evaluate: convert scores to fraud probability
    raw_scores_test = model.decision_function(X_test)
    # Map: lower score = more anomalous → higher fraud prob
    min_s, max_s = raw_scores_test.min(), raw_scores_test.max()
    probs_test = 1 - (raw_scores_test - min_s) / (max_s - min_s + 1e-8)
    probs_test = np.clip(probs_test, 0, 1)

    # Also evaluate on training set for comparison
    raw_scores_train = model.decision_function(X_train)
    probs_train = 1 - (raw_scores_train - min_s) / (max_s - min_s + 1e-8)
    probs_train = np.clip(probs_train, 0, 1)

    auc_train = roc_auc_score(y_train, probs_train)
    auc_test = roc_auc_score(y_test, probs_test)
    
    y_pred = (probs_test > 0.5).astype(int)
    
    print(f"\n[IsoForest-Enhanced] Results:")
    print(f"  Train AUC: {auc_train:.4f}")
    print(f"  Test AUC:  {auc_test:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model,
        "min_score": min_s,
        "max_score": max_s,
        "train_auc": auc_train,
        "test_auc": auc_test,
    }, MODEL_PATH)
    print(f"\n[IsoForest-Enhanced] Saved → {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()

