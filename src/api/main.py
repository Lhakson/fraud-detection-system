from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Load model ────────────────────────────────────────────────
MODELS_DIR = Path('models')
model      = pickle.load(open(MODELS_DIR / 'xgboost_api.pkl', 'rb'))
config     = pickle.load(open(MODELS_DIR / 'api_config.pkl', 'rb'))
THRESHOLD  = float(config['threshold'])
FEATURES   = config['feature_cols']

print(f'Model loaded.')
print(f'Threshold: {THRESHOLD}')
print(f'Features:  {FEATURES}')

app = FastAPI(title='Fraud Detection API', version='3.0.0')

class Transaction(BaseModel):
    TransactionAmt:  float
    ProductCD:       str
    card4:           str
    card6:           str
    P_emaildomain:   str
    R_emaildomain:   str
    DeviceType:      Optional[str]   = None
    TransactionDT:   Optional[int]   = 86400
    V317:            Optional[float] = None
    V258:            Optional[float] = None
    V306:            Optional[float] = None
    V312:            Optional[float] = None
    V128:            Optional[float] = None
    V127:            Optional[float] = None
    C14:             Optional[float] = None
    C11:             Optional[float] = None
    C8:              Optional[float] = None
    M4:              Optional[str]   = None  # M0, M1, M2
    M6:              Optional[str]   = None  # T or F

def featurize(t: Transaction) -> pd.DataFrame:
    amt  = t.TransactionAmt
    dt   = t.TransactionDT or 86400
    hour = (dt // 3600) % 24

    # M4 encoding — M0=0, M1=1, M2=2
    m4_map = {'M0': 0, 'M1': 1, 'M2': 2}
    m4_val = m4_map.get(t.M4, 0) if t.M4 else 0

    # M6 encoding — T=1, F=0
    m6_val = 1 if t.M6 == 'T' else 0

    row = {
        'TransactionAmt': amt,
        'amt_log':        np.log1p(amt),
        'hour':           hour,
        'day_of_week':    (dt // 86400) % 7,
        'is_night':       1 if hour < 6 else 0,
        'is_weekend':     1 if (dt // 86400) % 7 >= 5 else 0,
        'is_credit':      1 if t.card6 == 'credit' else 0,
        'card6':          1 if t.card6 == 'credit' else 0,
        'card1_count':    1,
        # Use 0 as default — that's the real median value
        'V317':           t.V317 if t.V317 is not None else 0,
        'V258':           t.V258 if t.V258 is not None else 0,
        'V306':           t.V306 if t.V306 is not None else 0,
        'V312':           t.V312 if t.V312 is not None else 0,
        'V128':           t.V128 if t.V128 is not None else 0,
        'V127':           t.V127 if t.V127 is not None else 0,
        'C14':            t.C14 if t.C14 is not None else 1,
        'C11':            t.C11 if t.C11 is not None else 1,
        'C8':             t.C8  if t.C8  is not None else 1,
        'M4':             m4_val,
        'M6':             m6_val,
    }

    return pd.DataFrame([row])[FEATURES]

def get_reasons(t: Transaction, score: float) -> list:
    out = []
    if t.ProductCD == 'C':
        out.append('Digital product — high fraud category')
    if t.card6 == 'credit':
        out.append('Credit card — higher risk than debit')
    if t.DeviceType == 'mobile':
        out.append('Mobile device transaction')
    if t.TransactionAmt % 1 != 0:
        out.append(f'Unusual decimal amount: ${t.TransactionAmt}')
    hour = ((t.TransactionDT or 86400) // 3600) % 24
    if hour < 6:
        out.append(f'Night-time transaction: {hour}:00am')
    if t.P_emaildomain in ['protonmail.com', 'anonymous.com']:
        out.append(f'High risk email: {t.P_emaildomain}')
    if t.P_emaildomain != t.R_emaildomain:
        out.append('Purchaser and recipient emails do not match')
    if t.TransactionAmt > 500:
        out.append(f'High amount: ${t.TransactionAmt}')
    if not out:
        out.append('Anomalous pattern detected by model')
    return out[:4]

@app.get('/health')
def health():
    return {
        'status':    'healthy',
        'threshold': THRESHOLD,
        'features':  len(FEATURES),
        'model':     'XGBoost (top 20 SHAP features)',
        'metrics':   config['metrics']
    }

@app.post('/predict')
def predict(txn: Transaction):
    start       = time.time()
    X           = featurize(txn)
    score       = float(model.predict_proba(X)[0][1])
    decision    = 'BLOCK' if score >= THRESHOLD else 'APPROVE'

    if score >= 0.40:   tier = 'CRITICAL'
    elif score >= 0.25: tier = 'HIGH'
    elif score >= 0.15: tier = 'MEDIUM'
    elif score >= 0.08: tier = 'LOW'
    else:               tier = 'VERY_LOW'

    return {
        'transaction_id': f'txn_{int(time.time()*1000)}',
        'fraud_score':    round(score, 4),
        'risk_tier':      tier,
        'decision':       decision,
        'reasons':        get_reasons(txn, score) if decision == 'BLOCK' else [],
        'latency_ms':     round((time.time() - start) * 1000, 2),
        'model_version':  '3.0.0'
    }

@app.get('/metrics')
def get_metrics():
    return {
        'model':     'XGBoost top 20 SHAP features',
        'auc':       round(config['metrics']['auc'], 4),
        'precision': round(config['metrics']['precision'], 4),
        'recall':    round(config['metrics']['recall'], 4),
        'f1':        round(config['metrics']['f1'], 4),
        'threshold': THRESHOLD,
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)