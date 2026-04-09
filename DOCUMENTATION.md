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

---

## Key Failures and Fixes

### 1. PowerShell vs bash
FAILED: git init fraud-detection and and cd fraud-detection
FIXED: Run commands separately — and and does not work in PowerShell

### 2. .env file not loading
FAILED: os.getenv returning None for Kaggle credentials
FIXED: Bypassed .env entirely, used Kaggle CLI directly

### 3. SMOTE string error
FAILED: ValueError could not convert string to float NotFound
FIXED: Force all columns to numeric before SMOTE, replace inf with -999

### 4. 3-model ensemble was worse
FAILED: Adding Isolation Forest dragged ensemble AUC from 0.90 to 0.879
FIXED: Dropped Isolation Forest, kept only XGBoost and LightGBM

### 5. Feature mismatch at API time
FAILED: 450-feature model receiving -999 for 430 unknown features
FIXED: Retrained on top 20 SHAP features using real median defaults

### 6. Docker WSL2 error
FAILED: Docker Desktop unable to start on Windows
FIXED: wsl install then restart fixed it

---

## What I Would Do Differently

1. Start with SHAP feature selection before training any model
2. Define the API contract before engineering features
3. Add Redis feature store for real velocity features
4. Wrap predict function as Kafka consumer for streaming

---

## Live API
https://fraud-detection-api-fjg8.onrender.com/docs

## GitHub
https://github.com/Lhakson/fraud-detection-system
