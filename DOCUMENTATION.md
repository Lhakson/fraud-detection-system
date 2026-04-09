# Fraud Detection System — Build Journal

**Built in 48 hours | IEEE-CIS Dataset | XGBoost + FastAPI + Docker**

---

## The Honest Story

This is not a clean tutorial. This is what actually happened when building a real fraud detection system from scratch — the failures, the fixes, the decisions, and the lessons. Every mistake is documented because that's how real data science works.

---

## What We Built

A real-time fraud detection API that:
- Takes a payment transaction as input
- Returns a fraud score, risk tier, decision, and human-readable reasons
- Runs in under 20ms
- Deployed in a Docker container

```
Input:  transaction JSON
Output: { fraud_score: 0.31, risk_tier: "HIGH", decision: "BLOCK", reasons: [...] }
```

---

## The Dataset

**IEEE-CIS Fraud Detection** — Kaggle 2019 competition dataset from Vesta Corporation (a real payment processor).

| Fact | Value |
|------|-------|
| Total transactions | 590,540 |
| Fraud transactions | 20,663 (3.5%) |
| Legit transactions | 569,877 (96.5%) |
| Total features | 434 |
| Date range | 2017–2018 |

### What the columns actually mean

- `TransactionAmt` — how much money changed hands
- `ProductCD` — type of product: W=physical, C=digital, H=hotel, R=travel, S=service
- `card1-card6` — card account number, type, brand, country (all anonymized)
- `P_emaildomain` — purchaser email provider (gmail.com, protonmail.com etc)
- `R_emaildomain` — recipient email provider
- `C1-C14` — count features (how many times something happened — anonymized)
- `D1-D15` — time delta features (days between events — anonymized)
- `M1-M9` — match features (does name match billing? T/F)
- `V1-V339` — Vesta's proprietary engineered features (most powerful, fully anonymized)
- `DeviceType` — desktop or mobile
- `id_01-id_38` — device, browser, network signals

**The hard truth about V-features:** Nobody outside Vesta knows exactly what V317 or V258 mean. They're black boxes. But SHAP told us they matter — so we used them.

---

## Day 1 — Data, Features, Models

### Step 1: Environment Setup

**What worked:** Python 3.13 + virtual environment on Windows.

**What failed:** Running bash commands in PowerShell.

```powershell
# FAILED — && doesn't work in PowerShell
git init fraud-detection && cd fraud-detection

# FIXED — run separately
git init fraud-detection
cd fraud-detection
```

**Lesson:** Always check your shell. PowerShell ≠ bash. Most tutorials assume bash.

---

### Step 2: Getting the Data

**What worked:** Kaggle CLI with `kaggle.json` credentials.

**What failed:** `.env` file not being read by Python.

```python
# FAILED — env vars were None because .env wasn't loading
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')  # None!

# FIXED — bypass .env entirely, use Kaggle CLI directly
kaggle competitions download -c ieee-fraud-detection -p data\raw
```

**Lesson:** Don't over-engineer config files for a 2-day build. Use the simplest path that works.

**Result:**
```
Transaction shape: (590540, 394)
Identity shape:    (144233, 41)
Merged shape:      (590540, 434)
Fraud rate:        3.50%
```

---

### Step 3: Exploratory Data Analysis

**Key findings from EDA:**

**Finding 1 — The class imbalance problem**
```
Legit: 569,877  (96.5%)
Fraud:  20,663  ( 3.5%)
```
A model that predicts "legit" for everything gets 96.5% accuracy. Accuracy is useless here. We need Precision, Recall, and AUC-ROC.

**Finding 2 — Fraud amounts are sneaky**
```
Legit median:  $68.50
Fraud median:  $75.00
```
Fraudsters don't always make small test charges. The $37.098 repeated 3 times in fraud samples is a card-testing pattern — same card, same amount, rapid succession.

**Finding 3 — Product C is a massive red flag**
```
ProductCD = C (digital goods): 11.7% fraud rate
Overall fraud rate:              3.5%
```
Digital goods = instant delivery, no shipping address, easy to resell. Perfect for fraudsters.

**Finding 4 — The V-features are powerful but mysterious**
SHAP analysis later revealed V317, V258, V306 as top predictors. We still don't know exactly what they measure. We used them anyway because the data said so.

---

### Step 4: Feature Engineering

**What we built and why:**

| Feature | Why it matters |
|---------|---------------|
| `hour`, `is_night` | Fraud spikes at 2-4am when victims sleep |
| `amt_decimal` | $37.098 repeated = card testing script |
| `amt_zscore_card` | $4,000 charge on a $50/month card = suspicious |
| `card1_count` | New card used once = higher risk |
| `high_risk_combo` | Digital + credit + mobile = highest fraud rate |
| `email_match` | Buyer and recipient on different domains = suspicious |
| `p_email_high_risk` | protonmail.com = anonymous = red flag |

**What failed — SMOTE errors**

```
# FAILED — string values breaking SMOTE
ValueError: could not convert string to float: 'NotFound'

# FAILED — infinity values breaking SMOTE  
ValueError: Input X contains infinity or a value too large for dtype('float64')

# Root cause: amt_zscore_card divided by zero std for single-transaction cards
# Fix:
X = X.replace([np.inf, -np.inf], -999)
```

**Lesson:** Always check for infinity values after any division operation. `std=0` is common with rare categories.

**SMOTE result:**
```
Before: Fraud 16,530  (3.50%)   ← imbalanced
After:  Fraud 455,902 (50.00%)  ← balanced
Training size: 911,804 rows
```

---

### Step 5: Model Training — The Full Story

#### Attempt 1 — 3-model ensemble (Isolation Forest + XGBoost + LightGBM)

```
Isolation Forest AUC:  0.709
XGBoost AUC:           0.902
LightGBM AUC:          0.897
Ensemble AUC:          0.879  ← WORSE than XGBoost alone
Precision:             12%    ← terrible
False Positive Rate:   20%    ← blocking 23,422 innocent customers
```

**What went wrong:** Isolation Forest (AUC 0.709) dragged down the ensemble. Including a weak model in a weighted average hurts the strong models.

**Fix:** Drop Isolation Forest from ensemble. Keep XGBoost + LightGBM only.

#### Attempt 2 — XGBoost + LightGBM ensemble

```
Ensemble AUC:  0.900
Precision:     69%
Recall:        47%
FPR:           0.75%   ← massively better
Legit blocked: 857     ← down from 23,422
```

**Lesson:** More models ≠ better ensemble. Only include models that add signal, not noise.

#### Attempt 3 — Finetuned hyperparameters

Changed:
- `n_estimators: 300 → 500` (more trees)
- `max_depth: 6 → 8` (deeper patterns)
- `learning_rate: 0.05 → 0.03` (slower, more careful)
- `min_child_weight: 5 → 3` (catch rarer fraud patterns)

```
XGBoost AUC:   0.902 → 0.925  (+0.023)
Ensemble AUC:  0.900 → 0.918  (+0.018)
Precision:     69%   → 73%
Legit blocked: 857   → 768
```

**Lesson:** XGBoost was still learning at tree 300. Always check if the model is still improving before stopping.

---

## Day 2 — API, Docker, Deployment

### Step 6: FastAPI Service — The Hard Part

**What failed — module import error**

```python
# FAILED
uvicorn.run('src.api.main:app', reload=True)
# ModuleNotFoundError: No module named 'src'

# FIXED — pass the app object directly
uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
```

**What failed — port already in use**

```
[WinError 10048] Only one usage of each socket address is permitted
```

Fix: `taskkill /PID <process_id> /F` or change port to 8001.

**What failed — wrong config being loaded**

The API kept loading the OLD ensemble config (threshold 0.1027, 3 models) instead of the new one (threshold 0.3691, 2 models). Even after rewriting the code.

Root cause: We ran training twice. Second run saved a NEW `ensemble_config.pkl` but the API was pointing at the old one from a cached import.

Fix: Created `fix_config.py` to force-overwrite with correct values.

---

### Step 7: The Feature Mismatch Problem

**The biggest technical challenge of the build.**

Our full model used 450 features. The API could only send 15-20. We filled the rest with `-999`.

Result: The model saw `-999` everywhere and got confused — it had never seen that pattern in training.

```
# Real test results with 450-feature model + -999 padding:
Legit $50 mastercard:    fraud_score = 0.82  → BLOCK  ❌ (wrong)
Fraud protonmail mobile: fraud_score = 0.35  → APPROVE ❌ (wrong)
```

**Three attempts to fix this:**

**Attempt 1 — Use 20 hand-crafted features**
Trained a new XGBoost on 20 features we engineered ourselves.
Result: AUC 0.91 but scores were backwards — legit scored higher than fraud.
Root cause: Our hand-crafted features didn't have enough signal without the V-features.

**Attempt 2 — Use top 20 SHAP features**
Let SHAP tell us which 20 features the full model actually relied on.
```
Top features by SHAP importance:
  0.49  card6
  0.44  TransactionAmt
  0.43  is_night
  0.36  hour
  0.29  V317
  0.29  C14
  ...
```
Retrained on these 20 features using raw data + `scale_pos_weight=27`.
Result: AUC 0.903, scores behaved correctly.

**Attempt 3 — Fix default values**
Was sending `-999` for V317 etc. Real median = 0.
```python
# WRONG — -999 is not a real value
'V317': t.V317 if t.V317 is not None else -999

# RIGHT — use real median from training data
'V317': t.V317 if t.V317 is not None else 0
```

**Final working results:**
```
Fraud transaction:  fraud_score = 0.31  → BLOCK   ✅
Legit transaction:  fraud_score = 0.04  → APPROVE ✅
Suspicious:         fraud_score = 0.21  → BLOCK   ✅
API latency:        ~15ms               ✅
```

**Lesson:** Feature distribution at inference must match training distribution. `-999` for missing values only works if the model was trained with `-999` for those same features.

---

### Step 8: Docker

**What failed — WSL 2 not installed**

```
Docker Desktop is unable to start
```

Fix: `wsl --install` → restart computer → Docker Desktop started.

**What worked perfectly:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi uvicorn xgboost pandas numpy scikit-learn
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["python", "src/api/main.py"]
```

```
docker build -t fraud-detection .   # 92 seconds
docker run -p 8000:8000 fraud-detection
# → running at http://localhost:8000
```

**Lesson:** Docker on Windows requires WSL 2. Always install WSL first before Docker.

---

## Final Metrics

| Metric | Full Model (450 features) | API Model (20 features) |
|--------|--------------------------|------------------------|
| AUC-ROC | 0.9176 | 0.9028 |
| Precision | 73% | 63% |
| Recall | 51% | 46% |
| FPR | 0.67% | 0.96% |
| API latency | N/A | ~15ms |
| Features | 450 | 20 |

**Why two models?**
The full model is the most accurate — used for batch scoring.
The API model uses only features available at request time — used for real-time decisions.
This is standard practice in production fraud systems.

---

## What I Would Do Differently

**1. Start with fewer features**
We spent hours debugging SMOTE errors caused by 450 features.
Starting with 20 SHAP features from the beginning would have saved 3 hours.

**2. Test the API contract first**
Before training any model, define exactly what fields the API will receive.
Then engineer only those features. Work backwards from deployment.

**3. Use a feature store**
The biggest limitation of this system — velocity features like
"how many times has this card been used today?" require a database
of recent transactions. We used a placeholder value of 1.
A Redis feature store would make these real.

**4. Add Kafka for streaming**
Current architecture: HTTP request → model → response.
Production architecture: Kafka topic → consumer → model → Kafka topic.
The code is ready for it — the predict function just needs to be wrapped
as a Kafka consumer.

---

## Architecture Decisions Explained

### Why XGBoost over neural networks?
Tabular data with mixed types (numbers + categories + missing values).
XGBoost handles all of this natively. Neural networks need extensive preprocessing.
For fraud detection on structured data, XGBoost consistently outperforms
deep learning unless you have 10M+ rows.

### Why SMOTE over class weights?
SMOTE creates synthetic fraud examples in feature space, giving the model
more diverse fraud patterns to learn from.
Class weights just tell the model "fraud matters more" but show it
the same 16,530 fraud examples repeatedly.
In practice: SMOTE gave +2% AUC on the full model.
For the API model we used scale_pos_weight because training on raw data
without SMOTE was faster and still worked well.

### Why threshold tuning matters more than model selection
Default threshold = 0.5 gave terrible results on every model.
The optimal threshold was found by scanning the Precision-Recall curve
and picking the point that maximized F1.
This single step improved practical performance more than switching
between XGBoost, LightGBM, or any ensemble.

### Why we dropped Isolation Forest
AUC 0.709 — only slightly better than random (0.5).
Including it in the ensemble dragged down precision from 69% to 12%.
The lesson: unsupervised anomaly detection works better as a
separate alert layer, not combined with supervised models.

---

## Business Impact

Assumptions: $25M annual GMV, 0.1% fraud rate = $25K/month fraud loss.

| Scenario | Fraud caught | Monthly saving |
|----------|-------------|----------------|
| No model | 0% | $0 |
| Our model (46% recall) | 46% | $11,500 |
| Perfect model (100% recall) | 100% | $25,000 |

API infrastructure cost on Render free tier: $0/month.
ROI: infinite (positive return on $0 cost).

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data | pandas, pyarrow | Fast parquet I/O |
| Imbalance | SMOTE, scale_pos_weight | Two strategies, different tradeoffs |
| Model | XGBoost, LightGBM | Best for tabular fraud data |
| Explainability | SHAP | Feature importance + API feature selection |
| Experiment tracking | MLflow | Log every run, compare metrics |
| API | FastAPI, Pydantic | Auto docs, type validation, fast |
| Container | Docker, WSL2 | Reproducible deployment |
| Language | Python 3.13 | Latest stable |

---

## Files Reference

```
fraud-detection/
├── data/
│   ├── raw/                     ← Kaggle CSV files (gitignored)
│   └── processed/
│       ├── fraud_merged.parquet ← merged transaction + identity
│       ├── train_features.parquet ← SMOTE-balanced training set
│       └── test_features.parquet  ← held-out test set
├── models/
│   ├── xgboost.pkl              ← full 450-feature model
│   ├── lightgbm.pkl             ← full 450-feature model
│   ├── xgboost_api.pkl          ← lightweight 20-feature API model
│   ├── ensemble_config.pkl      ← weights + threshold + metrics
│   └── api_config.pkl           ← API model config
├── notebooks/
│   ├── 01_eda.py                ← full EDA with 7 plots
│   └── explore.py               ← quick data exploration
├── reports/
│   ├── eda/                     ← EDA plots (4 charts)
│   └── models/                  ← SHAP plots
├── src/
│   ├── data_ingestion.py        ← download + merge + validate
│   ├── features/
│   │   └── engineering.py       ← feature engineering + SMOTE
│   ├── models/
│   │   ├── train.py             ← full model training
│   │   └── train_api_model.py   ← lightweight API model
│   └── api/
│       └── main.py              ← FastAPI service
├── Dockerfile                   ← container definition
├── docker-compose.yml           ← multi-service config
├── requirements.txt             ← all dependencies
└── .env                         ← credentials (gitignored)
```

---

## Running It Yourself

**Option 1 — Local Python:**
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection
cd fraud-detection
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
python src/api/main.py
# open http://localhost:8000/docs
```

**Option 2 — Docker:**
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
# open http://localhost:8000/docs
```

**Test it:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 37.098,
    "ProductCD": "C",
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "protonmail.com",
    "R_emaildomain": "gmail.com",
    "DeviceType": "mobile",
    "TransactionDT": 7200,
    "M4": "M2",
    "M6": "F"
  }'
```

Expected response:
```json
{
  "fraud_score": 0.3149,
  "risk_tier": "HIGH",
  "decision": "BLOCK",
  "reasons": [
    "Digital product — high fraud category",
    "Credit card — higher risk than debit",
    "Mobile device transaction",
    "Unusual decimal amount: $37.098"
  ],
  "latency_ms": 17.0
}
```