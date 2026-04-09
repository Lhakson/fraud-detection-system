"""
src/models/train_api_model.py
──────────────────────────────
Trains a lightweight XGBoost on only the 20 features
we can realistically send through the API.
Lower AUC than full model but honest real-time behavior.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             precision_recall_curve, confusion_matrix)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
MODELS_DIR    = Path('models')

# ── These are the ONLY features we send via API ───────────────
API_FEATURES = [
    # Amount signals
    'TransactionAmt',
    'amt_log',
    'amt_decimal',
    'amt_is_round',

    # Time signals
    'hour',
    'day_of_week',
    'is_night',
    'is_weekend',

    # Card signals
    'is_credit',
    'is_visa',
    'is_mobile',
    'high_risk_combo',

    # Email signals
    'p_email_high_risk',
    'p_email_low_risk',
    'email_match',

    # Velocity signals (we send defaults but still useful)
    'card1_count',
    'card_amt_count',

    # Optional numerics
    'C1', 'C2', 'D1',
]

print('='*55)
print('LOADING DATA')
print('='*55)
df = pd.read_parquet(PROCESSED_DIR / 'fraud_merged.parquet')
print(f'Loaded: {df.shape}')

# ── Rebuild only the features we need ────────────────────────
print('\nBuilding API features...')

# Time
df['hour']         = (df['TransactionDT'] // 3600) % 24
df['day_of_week']  = (df['TransactionDT'] // 86400) % 7
df['is_night']     = df['hour'].between(0, 6).astype(int)
df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)

# Amount
df['amt_log']      = np.log1p(df['TransactionAmt'])
df['amt_decimal']  = df['TransactionAmt'] % 1
df['amt_is_round'] = (df['TransactionAmt'] % 1 == 0).astype(int)

# Card
df['is_credit']    = (df['card6'] == 'credit').astype(int)
df['is_visa']      = (df['card4'] == 'visa').astype(int)
df['is_mobile']    = (df['DeviceType'] == 'mobile').astype(int)
df['high_risk_combo'] = (
    (df['ProductCD'] == 'C') &
    (df['card6'] == 'credit') &
    (df['DeviceType'] == 'mobile')
).astype(int)

# Email
high_risk = ['protonmail.com', 'anonymous.com', 'guerrillamail.com']
low_risk  = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
df['p_email_high_risk'] = df['P_emaildomain'].isin(high_risk).astype(int)
df['p_email_low_risk']  = df['P_emaildomain'].isin(low_risk).astype(int)
df['email_match']       = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)

# Velocity
df['card1_count']    = df.groupby('card1')['card1'].transform('count')
df['card_amt_count'] = df.groupby(['card1', 'TransactionAmt'])['TransactionAmt'].transform('count')

# Keep only API features + target
available = [f for f in API_FEATURES if f in df.columns]
print(f'Features available: {len(available)}')

X = df[available].copy()
y = df['isFraud'].copy()

# Fill missing
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

print(f'Fraud rate: {y.mean():.2%}')

# ── Train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost with scale_pos_weight instead of SMOTE ──────
# scale_pos_weight = legit count / fraud count
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f'\nscale_pos_weight: {scale:.1f}')

print('\n' + '='*55)
print('TRAINING LIGHTWEIGHT XGBOOST')
print('='*55)

model = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    scale_pos_weight  = scale,
    tree_method       = 'hist',
    random_state      = 42,
    n_jobs            = -1,
    eval_metric       = 'auc',
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── Evaluate ──────────────────────────────────────────────────
prob  = model.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, prob)
print(f'\nAUC: {auc:.4f}')

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, prob)
best_threshold = 0.5
best_f1        = 0
best_precision = 0
best_recall    = 0

for p, r, t in zip(precisions, recalls, thresholds):
    f1 = 2 * p * r / (p + r + 1e-9)
    if f1 > best_f1:
        best_f1        = f1
        best_threshold = t
        best_precision = p
        best_recall    = r

y_pred = (prob >= best_threshold).astype(int)
cm     = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr    = fp / (fp + tn)

print(f'Threshold: {best_threshold:.4f}')
print(f'Precision: {best_precision:.4f}')
print(f'Recall:    {best_recall:.4f}')
print(f'F1:        {best_f1:.4f}')
print(f'FPR:       {fpr:.4f}')
print(f'Fraud caught:  {tp:,}')
print(f'Legit blocked: {fp:,}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Legit','Fraud']))

# ── Save ──────────────────────────────────────────────────────
pickle.dump(model, open(MODELS_DIR / 'xgboost_api.pkl', 'wb'))

api_config = {
    'threshold':    best_threshold,
    'feature_cols': available,
    'weights':      {'xgb': 1.0},
    'metrics': {
        'auc':       float(auc),
        'precision': float(best_precision),
        'recall':    float(best_recall),
        'f1':        float(best_f1),
        'fpr':       float(fpr),
    }
}
pickle.dump(api_config, open(MODELS_DIR / 'api_config.pkl', 'wb'))

print('\nSaved: models/xgboost_api.pkl')
print('Saved: models/api_config.pkl')
print('\n' + '='*55)
print('DONE — update main.py to load xgboost_api.pkl')
print('='*55)