"""
FIN.DR — Preprocessing pipeline
Generates synthetic IEEE-CIS-style data if no CSV is found.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_data(n_samples: int = 50000) -> pd.DataFrame:
    """Synthetic dataset mimicking IEEE-CIS schema (fraud ~3.5%)."""
    np.random.seed(42)
    n_fraud = int(n_samples * 0.035)
    n_legit = n_samples - n_fraud

    def make_block(n, fraud=False):
        base_amount = np.random.lognormal(4.5, 1.2, n) if not fraud else np.random.lognormal(6.0, 1.5, n)
        return {
            "TransactionAmt":     base_amount,
            "card1":              np.random.randint(1000, 9999, n),
            "card2":              np.random.choice([111, 222, 333, 555], n),
            "card3":              np.random.choice([150, 185, 224], n),
            "card5":              np.random.choice([100, 102, 117, 226], n),
            "addr1":              np.random.randint(100, 500, n),
            "addr2":              np.random.choice([87, 96, 65], n),
            "dist1":              np.random.exponential(50, n) if not fraud else np.random.exponential(200, n),
            "P_emaildomain":      np.random.choice([0, 1, 2, 3], n),   # label-encoded
            "R_emaildomain":      np.random.choice([0, 1, 2, 3, 4], n),
            "DeviceType":         np.random.choice([0, 1], n),
            "DeviceInfo":         np.random.choice(range(20), n),
            "TransactionDT":      np.random.randint(86400, 86400 * 180, n),
            "C1":                 np.random.randint(0, 10, n),
            "C2":                 np.random.randint(0, 12, n),
            "C6":                 np.random.randint(0, 5, n),
            "C11":                np.random.randint(0, 8, n),
            "D1":                 np.random.randint(0, 200, n),
            "D3":                 np.random.randint(0, 100, n),
            "M1":                 np.random.choice([0, 1], n),
            "M2":                 np.random.choice([0, 1], n),
            "M3":                 np.random.choice([0, 1], n),
            "V1":                 np.random.randn(n),
            "V2":                 np.random.randn(n),
            "V3":                 np.random.randn(n),
            "V4":                 np.random.randn(n),
            "isFraud":            [1 if fraud else 0] * n,
        }

    legit = pd.DataFrame(make_block(n_legit, fraud=False))
    fraud = pd.DataFrame(make_block(n_fraud, fraud=True))
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    print(f"[Preprocess] Synthetic data: {len(df)} rows, fraud rate {df['isFraud'].mean():.2%}")
    return df


def load_raw_data(data_dir: str = None) -> pd.DataFrame:
    """Try to load real CSV; fall back to synthetic."""
    if data_dir:
        candidates = [
            os.path.join(data_dir, "train_transaction.csv"),
            os.path.join(data_dir, "creditcard.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"[Preprocess] Loading dataset: {path}")
                df = pd.read_csv(path, nrows=200000)
                # normalise column name for credit-card dataset
                if "Class" in df.columns:
                    df = df.rename(columns={"Class": "isFraud", "Amount": "TransactionAmt"})
                return df
    print("[Preprocess] No dataset found — generating synthetic data...")
    return generate_synthetic_data()


def preprocess(data_dir: str = None):
    """Full preprocessing pipeline. Returns (X_train, X_test, y_train, y_test, feature_cols)."""
    df = load_raw_data(data_dir)

    target = "isFraud"
    drop_cols = [target, "TransactionID"] if "TransactionID" in df.columns else [target]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    # Drop columns with >50% nulls
    null_pct = feature_df.isnull().mean()
    feature_df = feature_df.drop(columns=null_pct[null_pct > 0.5].index)

    # Fill numerics with median, categoricals with mode
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = feature_df[col].fillna(feature_df[col].mode()[0] if not feature_df[col].mode().empty else "Unknown")
            feature_df[col] = feature_df[col].astype("category").cat.codes
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    feature_cols = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save scaler + feature cols
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    print(f"[Preprocess] Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols, scaler


if __name__ == "__main__":
    preprocess()
