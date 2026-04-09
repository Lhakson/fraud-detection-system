# Real-Time Fraud Detection System

> End-to-end ML system: IEEE-CIS data → XGBoost ensemble → FastAPI → Docker  
> Built in 48 hours. AUC 0.90. Production-grade.

## Live API
https://fraud-detection-api-fjg8.onrender.com/docs

## Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.90 |
| Precision | 63% |
| Recall | 46% |
| False Positive Rate | 0.96% |
| API Latency | <15ms |

## Architecture

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection
cd fraud-detection
pip install -r requirements.txt
python src/api/main.py
# open http://localhost:8000/docs
```

## Sample API Call
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

## Response
```json
{
  "fraud_score": 0.3149,
  "risk_tier": "CRITICAL",
  "decision": "BLOCK",
  "reasons": [
    "Digital product — high fraud category",
    "Credit card — higher risk than debit",
    "Mobile device transaction",
    "Unusual decimal amount: $37.098"
  ],
  "latency_ms": 15.96
}
```

## Tech Stack
`Python 3.13` `XGBoost` `SHAP` `FastAPI` `Docker` `MLflow`

## Dataset
IEEE-CIS Fraud Detection — Kaggle (590K transactions, 3.5% fraud rate)

## Key Design Decisions
- **20 SHAP features** over 450 — top features identified by SHAP explainability
- **scale_pos_weight=27** — handles class imbalance without SMOTE for API model  
- **Threshold tuning** — optimized F1 score, not default 0.5
- **Risk tiers** — CRITICAL/HIGH/MEDIUM/LOW/VERY_LOW for business decisions

## Business Impact
Catching 46% of fraud on $25M GMV = ~$11.5K/month saved.  
API cost on Render free tier = $0/month.''

## Demo
[Watch 5-minute demo]https://www.loom.com/share/d161675b65794281bc64927560b00d54
