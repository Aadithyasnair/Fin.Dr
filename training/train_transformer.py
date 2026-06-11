"""
FIN.DR — PyTorch Tabular Transformer Training Script (Enhanced)
Uses advanced preprocessing and improved training.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report
from training.train_advanced import preprocess_advanced

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Architecture ─────────────────────────────────────────

class TabularTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_embed = nn.Linear(n_features, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 1.5),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_features)
        x = self.input_embed(x).unsqueeze(1)      # → (batch, 1, d_model)
        x = self.feature_norm(x)
        x = x + self.pos_embed
        x = self.transformer(x)                    # → (batch, 1, d_model)
        x = x.squeeze(1)                           # → (batch, d_model)
        return self.classifier(x).squeeze(-1)      # → (batch,)


# ── Training ───────────────────────────────────────────────────

def train(data_dir: str = None):
    print(f"[Transformer-Enhanced] Device: {DEVICE}")
    X_train, X_test, y_train, y_test, feature_cols, _ = preprocess_advanced(data_dir=data_dir)

    neg, pos = np.bincount(y_train.astype(int))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
    print(f"[Transformer-Enhanced] Positive weight: {neg/pos:.2f}")

    def to_loader(X, y, batch=256, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, batch=256, shuffle=True)
    test_loader  = to_loader(X_test, y_test, batch=256, shuffle=False)

    n_features = X_train.shape[1]
    model = TabularTransformer(n_features=n_features, d_model=64, nhead=4, num_layers=2, dropout=0.15).to(DEVICE)
    
    # Use BCE with logits for better numerical stability
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Modify classifier to output logits instead of probabilities
    model.classifier[-1] = nn.Identity()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    patience = 15

    print(f"[Transformer-Enhanced] Starting training...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Eval
        model.eval()
        all_logits, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_logits.extend(logits.cpu().numpy())
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1:02d} | loss={avg_loss:.4f} | val_AUC={auc:.4f} | patience={patience_counter}")

        if patience_counter >= patience:
            print(f"[Transformer-Enhanced] Early stopping at epoch {epoch+1}")
            break

    # Restore best weights and replace Identity with Sigmoid for inference
    if best_state:
        model.load_state_dict(best_state)
    model.classifier[-1] = nn.Sigmoid()
    model.eval()

    # Evaluate on test set
    with torch.no_grad():
        all_probs = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            probs = model(xb).cpu().numpy()
            all_probs.extend(probs)
    
    y_pred = (np.array(all_probs) > 0.5).astype(int)
    print(f"\n[Transformer-Enhanced] Results:")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"\n{classification_report(all_labels, y_pred, target_names=['Legitimate', 'Fraud'])}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_features":  n_features,
        "d_model":     64,
        "nhead":       4,
        "num_layers":  2,
        "best_auc":    best_auc,
    }, MODEL_PATH)
    print(f"\n[Transformer-Enhanced] Saved → {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()

