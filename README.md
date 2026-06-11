# FIN.DR — AI-Powered Fraud Detection & Risk Analysis Platform

## Overview

**FIN.DR** is a production-inspired fraud detection platform that combines machine learning, anomaly detection, and intelligent verification workflows to identify suspicious banking transactions in real time.

The system simulates modern fintech fraud-prevention architectures by integrating deep learning models, transformer-based classification, anomaly detection, risk fusion, voice verification, and an interactive analytics dashboard.

---

# Key Features

## Multi-Model Fraud Detection

FIN.DR utilizes an ensemble architecture consisting of:

* TensorFlow Deep Neural Network (DNN)
* PyTorch Tabular Transformer
* Isolation Forest Anomaly Detector

Each model contributes to a unified fraud risk score through a weighted fusion engine.

---

## Intelligent Risk Scoring

```text
Risk Score =
0.45 × TensorFlow Score +
0.45 × Transformer Score +
0.10 × Anomaly Score
```

### Decision Routing

| Risk Score  | Action            |
| ----------- | ----------------- |
| < 0.30      | APPROVE           |
| 0.30 – 0.70 | CALL USER         |
| > 0.70      | BLOCK TRANSACTION |

---

## Voice Verification Workflow

For medium-risk transactions, FIN.DR initiates an intelligent verification process.

### Capabilities

* Voice-based user verification
* Speech recognition
* Text-to-speech responses
* Multi-step confirmation workflow
* Keyboard fallback support
* Fraud escalation handling

---

## Analytics Dashboard

A modern glassmorphic dashboard provides:

* Real-time transaction monitoring
* Fraud analytics
* Decision distribution charts
* Searchable transaction logs
* Risk score visualization
* Model management controls

---

# System Architecture

```text
Transaction
      │
      ▼
Preprocessing
      │
      ▼
 ┌───────────────┐
 │ TensorFlow DNN│
 └───────────────┘
      │
      ▼
 ┌───────────────┐
 │ Transformer   │
 └───────────────┘
      │
      ▼
 ┌───────────────┐
 │IsolationForest│
 └───────────────┘
      │
      ▼
  Fusion Engine
      │
      ▼
   Risk Score
      │
 ┌────┼────┐
 ▼    ▼    ▼
Approve Call Block
```

---

# Project Structure

```text
FIN.DR/
│
├── app.py
├── config.py
├── train_all.py
├── requirements.txt
│
├── training/
│   ├── preprocess.py
│   ├── train_tensorflow.py
│   ├── train_transformer.py
│   ├── train_isolation.py
│   └── train_advanced.py
│
├── routes/
│   ├── predict.py
│   ├── voice.py
│   └── dashboard.py
│
├── voice/
│   └── verifier.py
│
├── database/
│   └── db.py
│
├── utils/
│   ├── fusion.py
│   └── model_loader.py
│
├── templates/
│   └── dashboard.html
│
└── static/
    ├── css/
    └── js/
```

---

# Machine Learning Pipeline

## TensorFlow Deep Neural Network

### Architecture

```text
Input
 ↓
Dense(512)
 ↓
Dense(256)
 ↓
Dense(128)
 ↓
Dense(64)
 ↓
Sigmoid Output
```

### Features

* Batch Normalization
* Dropout Regularization
* L2 Regularization
* Early Stopping
* Class Weighting

---

## Transformer Model

### Features

* Feature Embeddings
* Multi-Head Attention
* Layer Normalization
* GELU Activation
* Attention-Based Learning

---

## Isolation Forest

Unsupervised anomaly detection model used to identify previously unseen fraud patterns.

### Capabilities

* Unknown fraud detection
* Outlier identification
* Anomaly risk scoring

---

# Dataset Support

## Synthetic Dataset

Automatically generates:

* 100,000+ transactions
* Realistic fraud behaviors
* Multiple attack patterns

## Supported Public Datasets

### IEEE-CIS Fraud Detection

```text
train_transaction.csv
```

### Credit Card Fraud Dataset

```text
creditcard.csv
```

---

## Custom Dataset Requirements

### Required Columns

```text
Amount
isFraud
```

### Optional Columns

```text
TransactionDT
DeviceType
card1
card2
addr1
addr2
```

---

# Fraud Detection Capabilities

FIN.DR can detect:

* High-value transactions
* Unusual spending behavior
* Off-hour activity
* Weekend fraud patterns
* Card testing attacks
* Geographic anomalies
* Account takeover attempts
* Rapid transaction bursts
* Merchant risk anomalies

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/FIN.DR.git

cd FIN.DR
```

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Model Training

## Train Complete Pipeline

```bash
python training/train_advanced.py
```

### Generated Artifacts

```text
models/
├── tensorflow_model.keras
├── scaler.pkl
└── feature_cols.json
```

---

# Running the Application

```bash
python app.py
```

Application URL:

```text
http://localhost:5000
```

---

# API Endpoints

## Predict Transaction

```http
POST /api/predict
```

### Example Request

```json
{
  "amount": 5000,
  "merchant": "Amazon India",
  "user_id": "USR001"
}
```

---

## Voice Verification

```http
POST /api/voice/verify/<transaction_id>
```

---

## Reload Models

```http
POST /api/reload_models
```

---

## Retrain Models

```http
POST /api/retrain/tensorflow

POST /api/retrain/transformer

POST /api/retrain/isolation
```

---

# Performance

| Metric             | Value  |
| ------------------ | ------ |
| Accuracy           | 98%+   |
| AUC-ROC            | 0.97+  |
| Precision          | 0.82+  |
| Recall             | 0.79+  |
| Prediction Latency | <100ms |

---

# Future Enhancements

* Browser-based voice verification
* SHAP explainability
* Real-time streaming analytics
* Docker deployment
* PostgreSQL integration
* Redis caching
* JWT authentication
* Role-based access control (RBAC)
* Kubernetes deployment

---

# Technology Stack

### Backend

* Flask
* Python

### Machine Learning

* TensorFlow
* PyTorch
* Scikit-Learn

### Frontend

* Bootstrap
* Chart.js
* HTML/CSS/JavaScript

### Database

* SQLite (Current)
* PostgreSQL (Planned)

---

# License

Released under the **MIT License**.

---

# Acknowledgements

* TensorFlow
* PyTorch
* Scikit-Learn
* Flask
* Bootstrap
* Chart.js
