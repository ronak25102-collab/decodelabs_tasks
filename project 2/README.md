# Project 2: Fraud Detection Pipeline

**Author:** Ronak Kumar | DecodeLabs Batch 2026

## What this does
Builds a classification system to catch fraudulent transactions in a dataset where only 2% are actually fraud.

The main challenge is **class imbalance** - if a model just says "not fraud" every time, it gets 98% accuracy but catches zero fraud. So accuracy is useless here.

### Steps:
1. **Baseline models** (no balancing) - shows that accuracy looks great but recall is terrible
2. **SMOTE** - generates synthetic fraud samples to balance the training set (test set stays untouched)
3. **Hyperparameter tuning** - GridSearchCV on both Logistic Regression and Random Forest
4. **Evaluation** - Precision, Recall, F1-Score, ROC-AUC (not accuracy)

### Results:
- Logistic Regression: high recall (0.92) but lower precision
- Random Forest: high precision (0.94) but lower recall
- Both models got ROC-AUC above 0.98

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python project2_fraud_detection.py
```

Outputs go to `outputs/` folder (6 plots + comparison CSV).
