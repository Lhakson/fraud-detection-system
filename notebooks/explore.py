import pandas as pd
from pathlib import Path

df = pd.read_parquet('data/processed/fraud_merged.parquet')

# Basic shape
print('='*50)
print('DATASET OVERVIEW')
print('='*50)
print(f'Rows:     {df.shape[0]:,}')
print(f'Columns:  {df.shape[1]}')
print(f'Size:     {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')

# Fraud breakdown
print('\n' + '='*50)
print('FRAUD BREAKDOWN')
print('='*50)
print(f'Legit transactions:  {(df.isFraud==0).sum():,}  ({(df.isFraud==0).mean():.1%})')
print(f'Fraud transactions:  {(df.isFraud==1).sum():,}  ({(df.isFraud==1).mean():.1%})')

# What does a transaction look like
print('\n' + '='*50)
print('SAMPLE TRANSACTION (FRAUD)')
print('='*50)
print(df[df.isFraud==1][['TransactionAmt','ProductCD','card4','card6',
                          'P_emaildomain','DeviceType']].head(5).to_string())

print('\n' + '='*50)
print('SAMPLE TRANSACTION (LEGIT)')
print('='*50)
print(df[df.isFraud==0][['TransactionAmt','ProductCD','card4','card6',
                          'P_emaildomain','DeviceType']].head(5).to_string())

# Column groups explained
print('\n' + '='*50)
print('COLUMN GROUPS')
print('='*50)
print(f'Transaction info (TransactionDT, Amt, ProductCD): basic purchase info')
print(f'Card features    (card1-card6):  card type, bank, country')
print(f'Address          (addr1, addr2): billing/shipping address')
print(f'Email domains    (P_, R_):       purchaser vs recipient email')
print(f'C features       (C1-C14):       count-based features (anonymized)')
print(f'D features       (D1-D15):       timedelta features (days between events)')
print(f'M features       (M1-M9):        match features (name, address match?)')
print(f'V features       (V1-V339):      Vesta engineered features (anonymized)')
print(f'Identity         (id_01-id_38):  device, browser, network info')

# Amount stats
print('\n' + '='*50)
print('TRANSACTION AMOUNT STATS')
print('='*50)
print(df.groupby('isFraud')['TransactionAmt'].describe().round(2).to_string())

# Top fraud product categories
print('\n' + '='*50)
print('FRAUD RATE BY PRODUCT CATEGORY')
print('='*50)
print(df.groupby('ProductCD')['isFraud'].agg(['sum','mean','count'])
      .rename(columns={'sum':'fraud_count','mean':'fraud_rate','count':'total'})
      .sort_values('fraud_rate', ascending=False).to_string())