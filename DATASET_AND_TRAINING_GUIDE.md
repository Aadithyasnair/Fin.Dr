# FIN.DR Model Training & Dataset Guide

## Quick Start

### Using Synthetic Data (Default)
The system generates realistic synthetic data automatically:
```bash
python training/train_advanced.py
```

This creates 100,000 transactions with:
- **Fraud rate**: 3.5% (realistic)
- **Fraud patterns**: Unusual amounts, round numbers, off-hour transactions, unusual locations
- **Features**: 35+ engineered features mimicking real fraud indicators

## Using Real Datasets

### Supported Datasets

#### 1. IEEE-CIS Fraud Detection Dataset
- **Source**: Kaggle (requires download)
- **Format**: `train_transaction.csv` with 400+ features
- **Fraud Rate**: ~3.5%
- **Usage**:
```bash
# Place file in: datasets/ieee_cis/train_transaction.csv
python -c "
from training.train_advanced import preprocess_advanced
X_train, X_test, y_train, y_test, _, _ = preprocess_advanced(data_dir='datasets/ieee_cis')
"
```

#### 2. Credit Card Fraud (UCI Dataset)
- **Source**: UCI Machine Learning Repository
- **Format**: `creditcard.csv` with columns: Time, Amount, V1-V28, Class
- **Fraud Rate**: ~0.17%
- **Usage**:
```bash
# Place file in: datasets/credit_card/creditcard.csv
python -c "
from training.train_advanced import preprocess_advanced
X_train, X_test, y_train, y_test, _, _ = preprocess_advanced(data_dir='datasets/credit_card')
"
```

## Providing Your Own Dataset

### Required Format
Create a CSV file with:
```
TransactionAmt,card1,card2,...,isFraud
100.50,1234,111,...,0
5000.00,5678,222,...,1
...
```

### Minimum Required Columns
- `TransactionAmt` or `Amount`: Transaction amount (float)
- `isFraud` or `Class`: Target label (0=legitimate, 1=fraud)

### Optional but Recommended
- `DeviceType`: 0=desktop, 1=mobile
- `card1`, `card2`, ...: Card identifiers
- `addr1`, `addr2`: Address identifiers
- `dist1`: Distance from home
- `P_emaildomain`, `R_emaildomain`: Email domain
- `TransactionDT`: Time of transaction
- `V1-V5`: Feature vectors from fraud detection models

### Step-by-Step: Add Your Dataset

1. **Create directory**:
```bash
mkdir -p datasets/my_data
```

2. **Place CSV file**:
```bash
cp your_fraud_data.csv datasets/my_data/data.csv
```

3. **Update `train_advanced.py`** (optional - add CSV detection):
```python
def preprocess_advanced(data_dir: str = None):
    if data_dir:
        candidates = [
            os.path.join(data_dir, "train_transaction.csv"),
            os.path.join(data_dir, "creditcard.csv"),
            os.path.join(data_dir, "data.csv"),  # ADD THIS
        ]
```

4. **Train models**:
```bash
python -c "
from training.train_advanced import preprocess_advanced
X_train, X_test, y_train, y_test, _, _ = preprocess_advanced(data_dir='datasets/my_data')
"
```

## Training Models

### 1. Train Enhanced Models
```bash
python training/train_advanced.py
```

Trains:
- ✅ TensorFlow DNN (improved architecture)
- ✅ Isolation Forest (unsupervised anomaly detection)
- ✅ Transformer (attention-based)

**Output**:
- `models/tensorflow_model.keras` - Main deep learning model
- `models/scaler.pkl` - Feature scaler
- `models/feature_cols.json` - Feature names

### 2. Train Transformer Model
```bash
python training/train_transformer.py
```

**Architecture**:
- Multi-head attention (4 heads, 2 layers)
- Feature embeddings
- Tabular data optimized

### 3. Train Isolation Forest
```bash
python training/train_isolation.py
```

**Method**:
- Unsupervised anomaly detection
- 200 estimators
- Contamination ratio matches fraud rate

## Model Architecture Details

### TensorFlow DNN (45% weight)
```
Input (features) 
  ↓
Dense(512, ReLU) → BatchNorm → Dropout(0.4)
  ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
  ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
  ↓
Dense(64, ReLU) → BatchNorm → Dropout(0.2)
  ↓
Dense(1, Sigmoid) → Output [0-1]
```

**Features**:
- L2 regularization on large layers
- Early stopping on validation AUC
- Class weighting for imbalanced data
- Learning rate reduction on plateau

### Transformer (45% weight)
```
Input → Embedding Layer
  ↓
Multi-Head Attention (4 heads, 2 layers)
  ↓
Feed-Forward Network
  ↓
Classification Head → Output [0-1]
```

### Isolation Forest (10% weight)
```
Unsupervised anomaly detection
- 200 decision trees
- Isolation by random feature selection
- Normalized anomaly scores [0-1]
```

## Score Fusion (How Decisions Are Made)

```
Raw Score = 0.45 × TF_score + 0.45 × Transformer_score + 0.10 × Anomaly_score

Context Adjustment:
- Transaction type multiplier (0.20-1.00)
  - Self-transfer: 0.30 (very safe)
  - Salary credit: 0.20 (very safe)
  - Bill payment: 0.50
  - UPI transfer: 0.75
  - Purchase: 1.00 (standard)
  
- Merchant multiplier (0.70-1.00)
  - Trusted merchants (Amazon, HDFC, etc.): 0.70
  - Unknown merchant: 1.00

Final Score = Raw Score × Transaction_Type_Multiplier × Merchant_Multiplier

Decision Thresholds:
- < 0.30  → APPROVE (green)
- 0.30-0.70 → CALL_USER (yellow, voice verification)
- > 0.70  → BLOCK_TRANSACTION (red)
```

## Improving Model Performance

### Data Augmentation
The system includes synthetic fraud patterns:
- Round amounts (9999, 49999) → Common in fraud
- High amounts (100K+) → Unusual
- Off-hour transactions → 2-5 AM
- Weekend transactions → More fraud
- Unusual locations → Far from home
- Limited card reuse → Stolen cards

### Feature Engineering
Add custom features to your data:
```python
# Time-based features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

# Aggregate features
df['transactions_per_hour'] = df.groupby(['user_id', 'hour']).size()
df['avg_amount_card'] = df.groupby('card_id')['amount'].transform('mean')

# Behavioral features
df['amount_deviation'] = (df['amount'] - df['avg_amount_card']) / df['std_amount_card']
```

### Retraining with Your Data

1. **Place dataset in** `datasets/my_dataset/`
2. **Update path in training script**
3. **Run**:
```bash
python training/train_advanced.py
```

4. **Reload models**:
   - Dashboard → Models panel → "Reload All Models"
   - Or API: `POST /api/reload_models`

## Model Evaluation Metrics

### Key Metrics Tracked
- **Accuracy**: Overall correctness
- **AUC-ROC**: Sensitivity/Specificity trade-off
- **Precision**: False positive rate (↑ = fewer false alarms)
- **Recall**: Fraud detection rate (↑ = catch more fraud)

### Checking Model Performance
After training, the system prints:
```
Test Results:
  Accuracy:  0.9856
  AUC-ROC:   0.9741
  Precision: 0.8234
  Recall:    0.7892
```

## Troubleshooting

### Models Not Improving?
1. **Check data quality**: Is fraud labeled correctly?
2. **Check fraud rate**: Should be 1-5%
3. **Add features**: More diverse features → better models
4. **Increase training data**: 100K+ samples recommended
5. **Adjust thresholds**: Edit `config.py` APPROVE_THRESHOLD, BLOCK_THRESHOLD

### Voice Verification Not Working?
- Ensure browser allows microphone access
- Check that transaction is flagged as CALL_USER
- Verify user says "yes" or "no" clearly (see voice guide)

### Slow Predictions?
- Models load once at startup (check logs)
- First prediction might be slow (TensorFlow initialization)
- Subsequent predictions: <100ms

## API Integration

### Make Predictions
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000,
    "merchant": "Amazon India",
    "timestamp": "2026-06-11 10:30:00",
    "card_type": "VISA",
    "user_id": "USR001"
  }'
```

### Retrain Models
```bash
curl -X POST http://localhost:5000/api/retrain/tensorflow
curl -X POST http://localhost:5000/api/retrain/transformer
curl -X POST http://localhost:5000/api/retrain/isolation
```

### Reload Models
```bash
curl -X POST http://localhost:5000/api/reload_models
```

## Next Steps

1. ✅ **Generate synthetic data**: `python training/train_advanced.py`
2. 🔄 **Collect real transaction data** from your institution
3. 📊 **Add your dataset**: `datasets/my_data/transactions.csv`
4. 🚀 **Retrain models** with real data
5. 🔧 **Fine-tune thresholds** based on business requirements
6. 📈 **Monitor performance** via analytics dashboard

