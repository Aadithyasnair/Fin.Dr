# FIN.DR — AI Fraud Detection System

A production-level banking fraud detection engine simulating fintech security systems. It features a multi-model AI ensemble, weighted decision fusion, a voice verification agent, and a high-fidelity glassmorphic dashboard.

---

## 🚀 Key Features

1. **AI Ensemble Models:**
   - **TensorFlow DNN:** Deep classification network (Dense + BatchNorm + Dropout).
   - **PyTorch Tabular Transformer:** Feature embeddings + Multi-head attention.
   - **Isolation Forest (scikit-learn):** Unsupervised anomaly detection.
2. **Fusion Risk Scoring:**
   - Combines scores using weighted fusion (`0.45 * TF + 0.45 * Transformer + 0.10 * Anomaly`).
   - Dynamic threshold decision routing: `APPROVE` (<0.30), `CALL USER` (0.30 - 0.70), `BLOCK` (>0.70).
3. **Voice Verification Agent:**
   - Offline Speech synthesis (`pyttsx3`) + Speech recognition (`speech_recognition`).
   - Parallel microphone & keyboard listening fallback.
   - Intent classifier logic & escalation flows.
4. **Interactive Dashboard:**
   - Premium glassmorphic neon-dark UI.
   - Live analytics (Volume, fraud rates, decision distributions via Chart.js).
   - Searchable, sortable, and filterable log feed.
   - In-dashboard model control panel (retraining/reloading).

---

## 📂 Project Structure

```
Fin.Dr/
├── app.py                     # Flask backend entry point
├── config.py                  # Thresholds, paths & parameters
├── train_all.py               # Master training script
├── requirements.txt           # Virtual environment requirements
│
├── training/
│   ├── preprocess.py          # Data cleaning, scaling, synthetic data generator
│   ├── train_tensorflow.py    # TensorFlow DNN trainer
│   ├── train_transformer.py   # PyTorch Transformer trainer
│   └── train_isolation.py     # Isolation Forest trainer
│
├── voice/
│   └── verifier.py            # FraudVerificationAgent logic
│
├── routes/
│   ├── predict.py             # Transaction evaluation API
│   ├── voice.py               # Voice trigger & polling APIs
│   └── dashboard.py           # Dashboard stats & table logs
│
├── database/
│   └── db.py                  # SQLite management layer
│
├── utils/
│   ├── fusion.py              # Fusion scoring engine
│   └── model_loader.py        # Model loading singleton
│
├── templates/
│   └── dashboard.html         # Jinja dashboard UI
└── static/
    ├── css/style.css          # Custom styling theme
    └── js/dashboard.js        # Dashboard state & charts controller
```

---

## 🛠️ Setup Instructions

### 1. Initialise the Environment
Ensure you run within the local folder's virtual environment:
```powershell
# Create venv (already done)
py -3.12 -m venv .venv --prompt "findr"

# Activate
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
Dependencies are listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 3. Run Pre-training (Optional)
If you want to train the models on synthetic data first:
```powershell
python train_all.py
```
*(Note: If weights are missing at startup, the system will fall back to calibrated random scores, so the app remains fully interactive).*

### 4. Launch the Server
```powershell
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.
