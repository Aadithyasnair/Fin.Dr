"""
FIN.DR — Advanced Model Training with Enhanced Fraud Detection
Combines improved data generation, better architectures, and ensemble methods.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EnhancedDataGenerator:
    """Generate realistic fraud patterns based on known fraud characteristics."""
    
    @staticmethod
    def generate_realistic_data(n_samples: int = 100000, fraud_rate: float = 0.035) -> pd.DataFrame:
        """
        Generate synthetic data with more realistic fraud patterns:
        - Unusual transaction amounts
        - Unusual merchant categories
        - Unusual locations
        - Multiple rapid transactions
        - Round amounts (often fraud)
        """
        np.random.seed(42)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        def make_legitimate(n):
            """Legitimate transaction patterns."""
            return {
                "TransactionAmt": np.random.lognormal(4.5, 1.2, n),  # Right-skewed like real data
                "card1": np.random.randint(1000, 9999, n),
                "card2": np.random.choice([111, 222, 333, 555, 666], n),
                "card3": np.random.choice([150, 185, 224, 300, 400], n),
                "card5": np.random.choice([100, 102, 117, 226, 250], n),
                "addr1": np.random.randint(100, 500, n),
                "addr2": np.random.choice([87, 96, 65, 120, 150], n),
                "dist1": np.random.exponential(50, n),  # Distance from home
                "P_emaildomain": np.random.choice([0, 1, 2, 3, 4, 5], n),
                "R_emaildomain": np.random.choice([0, 1, 2, 3, 4, 5, 6], n),
                "DeviceType": np.random.choice([0, 1], n, p=[0.4, 0.6]),  # Mobile more common
                "DeviceInfo": np.random.choice(range(30), n),
                "TransactionDT": np.random.randint(86400, 86400 * 180, n),
                "hour": np.random.choice(range(24), n, p=np.array([1/30]*6 + [1/20]*12 + [1/30]*6)),
                "day_of_week": np.random.choice(range(7), n),
                "C1": np.random.randint(0, 10, n),
                "C2": np.random.randint(0, 12, n),
                "C6": np.random.randint(0, 5, n),
                "C11": np.random.randint(0, 8, n),
                "C14": np.random.exponential(1, n),
                "D1": np.random.randint(0, 200, n),
                "D3": np.random.randint(0, 100, n),
                "D4": np.random.randint(0, 50, n),
                "D9": np.random.exponential(20, n),
                "M1": np.random.choice([0, 1], n),
                "M2": np.random.choice([0, 1], n),
                "M3": np.random.choice([0, 1], n),
                "M4": np.random.choice([0, 1], n),
                "M5": np.random.choice([0, 1], n),
                "V1": np.random.randn(n) * 0.5,
                "V2": np.random.randn(n) * 0.5,
                "V3": np.random.randn(n) * 0.5,
                "V4": np.random.randn(n) * 0.5,
                "V5": np.random.randn(n) * 0.5,
                "isFraud": np.zeros(n, dtype=int),
            }

        def make_fraudulent(n):
            """Fraudulent transaction patterns - distinctive features."""
            return {
                "TransactionAmt": np.where(
                    np.random.rand(n) > 0.7,
                    np.random.uniform(10000, 500000, n),  # Very high amounts (30%)
                    np.random.choice([999, 9999, 49999, 99999], n)  # Round amounts (70%)
                ),
                "card1": np.random.randint(1000, 9999, n),
                "card2": np.random.choice([111, 222, 333], n),  # Limited cards used
                "card3": np.random.choice([150, 224], n),
                "card5": np.random.choice([100, 102], n),
                "addr1": np.random.randint(1, 100, n),  # Different addresses
                "addr2": np.random.choice([87, 96], n),
                "dist1": np.random.exponential(150, n),  # Very far from home
                "P_emaildomain": np.random.choice([0, 1, 7, 8], n),  # Specific domains
                "R_emaildomain": np.random.choice([0, 1, 2], n),
                "DeviceType": np.ones(n, dtype=int),  # Mostly mobile fraud
                "DeviceInfo": np.random.choice(range(20), n),
                "TransactionDT": np.random.randint(86400, 86400 * 180, n),
                "hour": np.random.choice([2, 3, 4, 5], n),  # Off-hours fraud
                "day_of_week": np.random.choice([0, 6], n),  # Weekends
                "C1": np.random.randint(0, 5, n),
                "C2": np.random.randint(0, 8, n),
                "C6": np.random.randint(2, 5, n),
                "C11": np.random.randint(4, 8, n),
                "C14": np.random.exponential(3, n),  # Different dist
                "D1": np.random.randint(150, 200, n),
                "D3": np.random.randint(70, 100, n),
                "D4": np.random.randint(30, 50, n),
                "D9": np.random.exponential(60, n),
                "M1": np.random.choice([0, 1], n, p=[0.2, 0.8]),
                "M2": np.random.choice([0, 1], n, p=[0.8, 0.2]),
                "M3": np.random.choice([0, 1], n, p=[0.3, 0.7]),
                "M4": np.random.choice([0, 1], n, p=[0.6, 0.4]),
                "M5": np.ones(n, dtype=int),  # Always 1 for fraud
                "V1": np.random.randn(n) * 1.5,  # Higher variance
                "V2": np.random.randn(n) * 1.5,
                "V3": np.random.randn(n) * 1.5,
                "V4": np.random.randn(n) * 1.5,
                "V5": np.random.randn(n) * 1.5,
                "isFraud": np.ones(n, dtype=int),
            }

        legit_df = pd.DataFrame(make_legitimate(n_legit))
        fraud_df = pd.DataFrame(make_fraudulent(n_fraud))
        
        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"[DataGen] Generated {len(df)} transactions")
        print(f"[DataGen] Fraud rate: {df['isFraud'].mean():.2%}")
        print(f"[DataGen] Features: {len(df.columns) - 1}")
        return df


def preprocess_advanced(data_dir: str = None):
    """Advanced preprocessing with feature engineering."""
    # Try to load real data first
    if data_dir:
        candidates = [
            os.path.join(data_dir, "train_transaction.csv"),
            os.path.join(data_dir, "creditcard.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"[Preprocess] Loading real dataset: {path}")
                df = pd.read_csv(path, nrows=200000)
                if "Class" in df.columns:
                    df = df.rename(columns={"Class": "isFraud", "Amount": "TransactionAmt"})
                break
        else:
            df = EnhancedDataGenerator.generate_realistic_data(n_samples=100000)
    else:
        df = EnhancedDataGenerator.generate_realistic_data(n_samples=100000)

    target = "isFraud"
    drop_cols = [target, "TransactionID"] if "TransactionID" in df.columns else [target]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    # Remove high-null columns
    null_pct = feature_df.isnull().mean()
    feature_df = feature_df.drop(columns=null_pct[null_pct > 0.5].index)

    # Handle missing values
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = feature_df[col].fillna(feature_df[col].mode()[0] if not feature_df[col].mode().empty else "Unknown")
            feature_df[col] = feature_df[col].astype("category").cat.codes
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    feature_cols = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    print(f"[Preprocess] Features: {len(feature_cols)}")
    print(f"[Preprocess] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[Preprocess] Train fraud rate: {y_train.mean():.2%} | Test fraud rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_tensorflow_advanced():
    """Train improved TensorFlow model with better architecture."""
    import tensorflow as tf
    from tensorflow import keras
    
    print("[TF-Advanced] Starting enhanced TensorFlow training...")
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess_advanced()
    
    n_features = X_train.shape[1]
    neg, pos = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: (neg / pos) * 2.0}  # More weight on fraud
    print(f"[TF-Advanced] Class weights: {class_weight}")

    # Improved architecture
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_auc", mode="max", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, monitor="val_auc", mode="max", verbose=1, min_lr=1e-6
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    loss, acc, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[TF-Advanced] Test Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")
    
    # Save
    model_path = os.path.join(BASE_DIR, "models", "tensorflow_model.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[TF-Advanced] Model saved: {model_path}")
    return model


if __name__ == "__main__":
    train_tensorflow_advanced()
