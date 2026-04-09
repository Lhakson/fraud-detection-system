"""
Feature Engineering for Fraud Detection
Builds velocity features, encodes categoricals,
handles missing values, applies SMOTE
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')

# ── Columns to use ────────────────────────────────────────────
CAT_COLS = ['ProductCD', 'card4', 'card6',
            'P_emaildomain', 'R_emaildomain',
            'M1','M2','M3','M4','M5','M6','M7','M8','M9',
            'DeviceType']

NUM_COLS = ['TransactionAmt', 'dist1', 'dist2',
            'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
            'D1','D2','D3','D4','D5','D10','D11','D15',
            'V1','V2','V3','V4','V5','V6','V12','V13','V14',
            'V17','V19','V20','V29','V30','V33','V34']


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print('Building features...')
    df = df.copy()

    # ── 1. Time features ──────────────────────────────────────
    # TransactionDT is seconds offset — extract hour and day
    df['hour']        = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
    df['is_night']    = df['hour'].between(0, 6).astype(int)
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    print('  ✓ Time features')

    # ── 2. Amount features ────────────────────────────────────
    df['amt_log']     = np.log1p(df['TransactionAmt'])
    df['amt_decimal'] = df['TransactionAmt'] % 1  # decimal part — .098 is suspicious
    df['amt_is_round']= (df['TransactionAmt'] % 1 == 0).astype(int)

    # Amount vs card average — is this transaction unusual for this card?
    card_amt_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
    card_amt_std  = df.groupby('card1')['TransactionAmt'].transform('std').fillna(1)
    df['amt_zscore_card'] = (df['TransactionAmt'] - card_amt_mean) / card_amt_std
    print('  ✓ Amount features')

    # ── 3. Velocity features ──────────────────────────────────
    # How many times has this card been used? (high velocity = fraud signal)
    df['card1_count']    = df.groupby('card1')['card1'].transform('count')
    df['card_addr_count']= df.groupby(['card1','addr1'])['TransactionAmt'].transform('count')

    # How many transactions from this email domain?
    df['p_email_count']  = df.groupby('P_emaildomain')['isFraud'].transform('count')

    # Has this exact amount been charged before on this card? (repeated = testing)
    df['card_amt_count'] = df.groupby(['card1','TransactionAmt'])['TransactionAmt'].transform('count')
    print('  ✓ Velocity features')

    # ── 4. Email features ─────────────────────────────────────
    # Free/anonymous email providers are higher risk
    high_risk_emails = ['protonmail.com', 'anonymous.com', 'guerrillamail.com']
    low_risk_emails  = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']

    df['p_email_high_risk'] = df['P_emaildomain'].isin(high_risk_emails).astype(int)
    df['p_email_low_risk']  = df['P_emaildomain'].isin(low_risk_emails).astype(int)

    # Do purchaser and recipient use same email domain?
    df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
    print('  ✓ Email features')

    # ── 5. Card features ──────────────────────────────────────
    df['is_credit']   = (df['card6'] == 'credit').astype(int)
    df['is_visa']     = (df['card4'] == 'visa').astype(int)
    df['is_mobile']   = (df['DeviceType'] == 'mobile').astype(int)

    # High risk combo: digital product + credit + mobile
    df['high_risk_combo'] = (
        (df['ProductCD'] == 'C') &
        (df['card6'] == 'credit') &
        (df['DeviceType'] == 'mobile')
    ).astype(int)
    print('  ✓ Card features')

    # ── 6. Encode categoricals ────────────────────────────────
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('unknown')
            df[col] = le.fit_transform(df[col])
    print('  ✓ Categorical encoding')

    # ── 7. Fill missing numerics ──────────────────────────────
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    print('  ✓ Missing values filled')

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (everything except target and ID)."""
    drop_cols = ['TransactionID', 'isFraud', 'TransactionDT',
                 'amt_bucket']
    return [c for c in df.columns if c not in drop_cols]


def prepare_train_test(df: pd.DataFrame):
    print('\n' + '='*50)
    print('PREPARING TRAIN/TEST SPLIT')
    print('='*50)

    # Build features
    df = build_features(df)

    # Get features and target
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df['isFraud'].copy()

    # Force ALL columns to numeric — no strings allowed
    print('Converting all columns to numeric...')
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Now fill NaNs safely
    # Now fill NaNs safely
    X = X.fillna(-999)

    # Replace infinity values
    X = X.replace([np.inf, -np.inf], -999)
    
    # Final check
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f'Dropping remaining non-numeric cols: {non_numeric}')
        X = X.drop(columns=non_numeric)
    
    feature_cols = X.columns.tolist()

    print(f'\nFeatures: {len(feature_cols)}')
    print(f'Samples:  {len(X):,}')
    print(f'Fraud:    {y.sum():,} ({y.mean():.2%})')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'\nTrain: {len(X_train):,} rows')
    print(f'Test:  {len(X_test):,} rows')

    # SMOTE on training set ONLY
    print('\nApplying SMOTE to training set...')
    print(f'Before SMOTE — Fraud: {y_train.sum():,} ({y_train.mean():.2%})')

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f'After SMOTE  — Fraud: {y_train_resampled.sum():,} ({y_train_resampled.mean():.2%})')
    print(f'Training size after SMOTE: {len(X_train_resampled):,}')

    # Save
    print('\nSaving processed data...')
    train_out = pd.DataFrame(X_train_resampled, columns=feature_cols)
    train_out['isFraud'] = y_train_resampled.values
    train_out.to_parquet(PROCESSED_DIR / 'train_features.parquet', index=False)

    test_out = pd.DataFrame(X_test, columns=feature_cols)
    test_out['isFraud'] = y_test.values
    test_out.to_parquet(PROCESSED_DIR / 'test_features.parquet', index=False)

    print('Saved train_features.parquet and test_features.parquet')

    return X_train_resampled, X_test.values, y_train_resampled, y_test, feature_cols


if __name__ == '__main__':
    print('Loading merged data...')
    df = pd.read_parquet(PROCESSED_DIR / 'fraud_merged.parquet')
    X_train, X_test, y_train, y_test, features = prepare_train_test(df)
    print('\n' + '='*50)
    print('FEATURE ENGINEERING COMPLETE')
    print('='*50)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape:  {X_test.shape}')
    print(f'Features built: {len(features)}')
    print('\nNew features created:')
    new_features = ['hour', 'is_night', 'amt_log', 'amt_zscore_card',
                    'card1_count', 'high_risk_combo', 'email_match']
    for f in new_features:
        if f in features:
            print(f'  ✓ {f}')
    print('\nNext step: python src/models/train.py')