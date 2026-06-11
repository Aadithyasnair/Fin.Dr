"""
FIN.DR — TensorFlow DNN Training Script (Enhanced)
Uses advanced preprocessing and improved architecture.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from training.train_advanced import preprocess_advanced

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tensorflow_model.keras")


def train(data_dir: str = None):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report

    print("[TF-Enhanced] Starting TensorFlow DNN training...")
    X_train, X_test, y_train, y_test, feature_cols, _ = preprocess_advanced(data_dir=data_dir)

    n_features = X_train.shape[1]
    neg, pos = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: (neg / pos) * 2.0}  # Increased fraud weight
    print(f"[TF-Enhanced] Class weights: {class_weight}")

    # ── Improved model architecture ──────────────────────────────────────
    inputs = keras.Input(shape=(n_features,))
    
    # Deep architecture with regularization
    x = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001),
                           kernel_initializer="he_normal")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001),
                           kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001),
                           kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True, monitor="val_auc", mode="max", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=7, factor=0.5, monitor="val_auc", mode="max", verbose=1, min_lr=1e-6
        ),
    ]

    print("[TF-Enhanced] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=512,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n[TF-Enhanced] Evaluating on test set...")
    loss, acc, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"\n[TF-Enhanced] Test Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[TF-Enhanced] Model saved → {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
