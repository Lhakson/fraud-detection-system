from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
REPORTS_DIR = Path('reports/eda')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FRAUD_COLOR = '#E24B4A'
LEGIT_COLOR = '#378ADD'

print('Loading data...')
df = pd.read_parquet(PROCESSED_DIR / 'fraud_merged.parquet')
print(f'Shape: {df.shape}')

# 1. Class imbalance
fraud_counts = df['isFraud'].value_counts()
fraud_rate = df['isFraud'].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Legit', 'Fraud'], fraud_counts.values, color=[LEGIT_COLOR, FRAUD_COLOR], width=0.5)
axes[0].set_title('Class distribution')
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center')
axes[1].pie(fraud_counts.values, labels=['Legit', 'Fraud'],
            colors=[LEGIT_COLOR, FRAUD_COLOR], autopct='%1.2f%%',
            startangle=90, wedgeprops={'edgecolor': 'white'})
axes[1].set_title(f'Fraud rate = {fraud_rate:.2%}')
plt.suptitle('Class Imbalance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_DIR / '01_class_imbalance.png', bbox_inches='tight')
print('Saved: 01_class_imbalance.png')
plt.show()

# 2. Transaction amount
fraud_amt = df[df['isFraud'] == 1]['TransactionAmt']
legit_amt = df[df['isFraud'] == 0]['TransactionAmt']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(np.log1p(legit_amt), bins=60, alpha=0.6, color=LEGIT_COLOR, label='Legit', density=True)
axes[0].hist(np.log1p(fraud_amt), bins=60, alpha=0.7, color=FRAUD_COLOR, label='Fraud', density=True)
axes[0].set_title('Amount distribution (log scale)')
axes[0].set_xlabel('log(1 + TransactionAmt)')
axes[0].legend()
bp = axes[1].boxplot([legit_amt.clip(upper=1000), fraud_amt.clip(upper=1000)],
                      labels=['Legit', 'Fraud'], patch_artist=True)
bp['boxes'][0].set_facecolor(LEGIT_COLOR + '88')
bp['boxes'][1].set_facecolor(FRAUD_COLOR + '88')
axes[1].set_title('Amount boxplot (capped $1K)')
bins = [0, 10, 50, 100, 250, 500, 1000, 5000, 100000]
labels = ['<$10','$10-50','$50-100','$100-250','$250-500','$500-1K','$1K-5K','>$5K']
df['amt_bucket'] = pd.cut(df['TransactionAmt'], bins=bins, labels=labels)
bucket_fraud = df.groupby('amt_bucket', observed=True)['isFraud'].mean() * 100
axes[2].bar(bucket_fraud.index, bucket_fraud.values, color=FRAUD_COLOR, alpha=0.8)
axes[2].set_title('Fraud rate by amount bucket')
axes[2].set_ylabel('Fraud rate (%)')
axes[2].tick_params(axis='x', rotation=40)
plt.suptitle('Transaction Amount Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_DIR / '02_transaction_amounts.png', bbox_inches='tight')
print('Saved: 02_transaction_amounts.png')
plt.show()

# 3. Time patterns
df['hour'] = (df['TransactionDT'] // 3600) % 24
df['day']  = (df['TransactionDT'] // 86400) % 7
hour_stats = df.groupby('hour')['isFraud'].agg(['mean','count']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax2 = axes[0].twinx()
axes[0].bar(hour_stats['hour'], hour_stats['count'], alpha=0.3, color=LEGIT_COLOR)
ax2.plot(hour_stats['hour'], hour_stats['mean'] * 100, color=FRAUD_COLOR, marker='o', ms=4)
axes[0].set_title('Volume vs fraud rate by hour')
axes[0].set_xlabel('Hour (0=midnight)')
axes[0].set_ylabel('Transaction count', color=LEGIT_COLOR)
ax2.set_ylabel('Fraud rate (%)', color=FRAUD_COLOR)
day_stats = df.groupby('day')['isFraud'].mean().reset_index()
day_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
axes[1].bar(day_labels, day_stats['isFraud'] * 100, color=FRAUD_COLOR, alpha=0.8)
axes[1].axhline(fraud_rate * 100, color='black', ls='--', lw=1, label='Overall avg')
axes[1].set_title('Fraud rate by day of week')
axes[1].set_ylabel('Fraud rate (%)')
axes[1].legend()
plt.suptitle('Time Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_DIR / '03_time_patterns.png', bbox_inches='tight')
print('Saved: 03_time_patterns.png')
plt.show()

# 4. Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = missing / len(df) * 100

fig, ax = plt.subplots(figsize=(10, 6))
colors = [FRAUD_COLOR if p > 50 else LEGIT_COLOR for p in missing_pct.values[:30]]
ax.barh(missing_pct.index[:30], missing_pct.values[:30], color=colors, alpha=0.85)
ax.axvline(50, color='black', ls='--', lw=1, label='>50% drop candidate')
ax.set_title('Missing value % (top 30 columns)')
ax.set_xlabel('Missing (%)')
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / '04_missing_values.png', bbox_inches='tight')
print('Saved: 04_missing_values.png')
plt.show()

# Summary
print()
print('='*50)
print('EDA KEY FINDINGS')
print('='*50)
print(f'Fraud rate:        {fraud_rate:.2%}  (use SMOTE)')
print(f'Fraud median amt:  ${fraud_amt.median():.2f}')
print(f'Legit median amt:  ${legit_amt.median():.2f}')
print(f'Cols >50% missing: {(missing_pct > 50).sum()}')
print(f'Total features:    {df.shape[1]}')
print(f'Reports saved to:  {REPORTS_DIR.resolve()}')
print('='*50)
print('Next: Feature engineering + SMOTE + XGBoost')