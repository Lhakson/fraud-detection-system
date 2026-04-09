from pathlib import Path
import pandas as pd

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print('Loading transaction data...')
txn = pd.read_csv(RAW_DIR / 'train_transaction.csv')
print(f'Transaction shape: {txn.shape}')

print('Loading identity data...')
idn = pd.read_csv(RAW_DIR / 'train_identity.csv')
print(f'Identity shape: {idn.shape}')

print('Merging...')
df = txn.merge(idn, on='TransactionID', how='left')
print(f'Merged shape: {df.shape}')

fraud_rate = df['isFraud'].mean() * 100
print(f'Fraud rate: {fraud_rate:.2f}%')
print(f'Total rows: {len(df):,}')

out = PROCESSED_DIR / 'fraud_merged.parquet'
df.to_parquet(out, index=False)
print(f'Saved to {out}')
print('Done!')
