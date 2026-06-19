# DecodeLabs - Data Science Industrial Training (Batch 2026)
# Project 2: Supervised Learning - Fraud Detection Pipeline
#
# Author: Ronak Kumar
#
# Goal: Build a classification model that can catch fraudulent
#       transactions in a dataset where fraud is super rare (~2%).

import os, sys, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, RocCurveDisplay)
from imblearn.over_sampling import SMOTE

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Project 2 - Fraud Detection Pipeline")
print("=" * 60)


# ---------------------------------------------------------------
#  STEP 1: Generate synthetic fraud data
# ---------------------------------------------------------------
# Making a dataset where only 2% of transactions are fraud.
# This is realistic - in real credit card data fraud is usually <1%.

print("\n>> Step 1: Creating transaction dataset...")
np.random.seed(42)
N = 10000
fraud_ratio = 0.02

n_fraud = int(N * fraud_ratio)
n_legit = N - n_fraud

# normal transactions
legit = pd.DataFrame({
    "amount": np.random.lognormal(mean=4.5, sigma=1.0, size=n_legit),
    "hour_of_day": np.random.randint(0, 24, n_legit),
    "day_of_week": np.random.randint(0, 7, n_legit),
    "merchant_category": np.random.randint(0, 15, n_legit),
    "distance_from_home": np.abs(np.random.normal(20, 15, n_legit)),
    "distance_from_last_txn": np.abs(np.random.normal(10, 10, n_legit)),
    "ratio_to_median_amount": np.random.lognormal(0, 0.4, n_legit),
    "txn_velocity_1h": np.random.poisson(2, n_legit),
    "txn_velocity_24h": np.random.poisson(5, n_legit),
    "is_online": np.random.choice([0, 1], n_legit, p=[0.55, 0.45]),
    "is_international": np.random.choice([0, 1], n_legit, p=[0.88, 0.12]),
    "card_age_months": np.random.randint(1, 120, n_legit),
    "num_cards": np.random.randint(1, 5, n_legit),
    "credit_limit": np.random.lognormal(9, 0.5, n_legit),
    "is_fraud": 0
})

# fraudulent transactions - slightly different patterns but with overlap
# (overlap is important so it's not trivially separable)
fraud = pd.DataFrame({
    "amount": np.random.lognormal(mean=5.2, sigma=1.2, size=n_fraud),
    "hour_of_day": np.random.choice([0,1,2,3,4,5,6,20,21,22,23,8,10,12,14,16], n_fraud),
    "day_of_week": np.random.randint(0, 7, n_fraud),
    "merchant_category": np.random.choice([0,1,2,3,5,7,9,12,14], n_fraud),
    "distance_from_home": np.abs(np.random.normal(45, 30, n_fraud)),
    "distance_from_last_txn": np.abs(np.random.normal(35, 25, n_fraud)),
    "ratio_to_median_amount": np.random.lognormal(0.6, 0.6, n_fraud),
    "txn_velocity_1h": np.random.poisson(4, n_fraud),
    "txn_velocity_24h": np.random.poisson(8, n_fraud),
    "is_online": np.random.choice([0, 1], n_fraud, p=[0.35, 0.65]),
    "is_international": np.random.choice([0, 1], n_fraud, p=[0.60, 0.40]),
    "card_age_months": np.random.randint(1, 60, n_fraud),
    "num_cards": np.random.randint(1, 6, n_fraud),
    "credit_limit": np.random.lognormal(9, 0.5, n_fraud),
    "is_fraud": 1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   {df.shape[0]} transactions, {df.shape[1]} features")
print(f"   Fraud: {n_fraud} ({fraud_ratio*100:.1f}%) | Legit: {n_legit} ({(1-fraud_ratio)*100:.1f}%)")

# show the imbalance visually
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["is_fraud"].value_counts()
bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{val:,}", ha="center", fontsize=13, weight="bold")
ax.set_title("Class Distribution - Massive Imbalance", fontsize=14, weight="bold")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_class_imbalance.png"), dpi=150)
plt.close(fig)
print("   Saved: 01_class_imbalance.png")


# ---------------------------------------------------------------
#  STEP 2: Preprocessing
# ---------------------------------------------------------------
print("\n>> Step 2: Scaling and splitting data...")
features = [c for c in df.columns if c != "is_fraud"]
X = df[features].values
y = df["is_fraud"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print(f"   Train: {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]}")
print(f"   Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")


# ---------------------------------------------------------------
#  STEP 3: Baseline models WITHOUT SMOTE
# ---------------------------------------------------------------
# Training on raw imbalanced data first to show why accuracy
# is misleading. A model that predicts "not fraud" every time
# gets 98% accuracy but catches zero fraud.

print("\n>> Step 3: Baseline models (no SMOTE)...")
print("   (showing why accuracy is misleading on imbalanced data)\n")

lr_base = LogisticRegression(max_iter=1000, random_state=42)
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lr_base.fit(X_train_sc, y_train)
rf_base.fit(X_train_sc, y_train)

for name, model in [("Logistic Regression", lr_base), ("Random Forest", rf_base)]:
    preds = model.predict(X_test_sc)
    acc = (preds == y_test).mean()
    report = classification_report(y_test, preds, target_names=["Legitimate", "Fraud"], output_dict=True)
    print(f"   {name} (no SMOTE):")
    print(f"     Accuracy = {acc:.4f} <-- looks good but misleading!")
    print(f"     Fraud Recall = {report['Fraud']['recall']:.4f} <-- this matters more")
    print(f"     Fraud Precision = {report['Fraud']['precision']:.4f}\n")


# ---------------------------------------------------------------
#  STEP 4: Apply SMOTE
# ---------------------------------------------------------------
# SMOTE creates synthetic fraud samples by interpolating between
# existing ones. Only applied to training data - test stays real.

print(">> Step 4: Applying SMOTE to training data...")
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"   Before: {np.bincount(y_train)}")
print(f"   After:  {np.bincount(y_train_sm)}")
print("   (SMOTE only on training set, test set untouched)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (title, labels) in zip(axes, [("Before SMOTE", y_train), ("After SMOTE", y_train_sm)]):
    c = np.bincount(labels)
    bars = ax.bar(["Legitimate", "Fraud"], c, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    for bar, val in zip(bars, c):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f"{val:,}", ha="center", fontsize=12, weight="bold")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel("Count")
fig.suptitle("SMOTE Rebalancing Effect", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_smote_effect.png"), dpi=150)
plt.close(fig)
print("   Saved: 02_smote_effect.png")


# ---------------------------------------------------------------
#  STEP 5: Train tuned models on balanced data
# ---------------------------------------------------------------
print("\n>> Step 5: Training with hyperparameter tuning (GridSearchCV)...")

# logistic regression
lr_params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                       lr_params, scoring="roc_auc", cv=StratifiedKFold(5), n_jobs=-1)
lr_grid.fit(X_train_sm, y_train_sm)
lr_best = lr_grid.best_estimator_
print(f"   LR best: {lr_grid.best_params_} (CV AUC: {lr_grid.best_score_:.4f})")

# random forest
rf_params = {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                       rf_params, scoring="roc_auc", cv=StratifiedKFold(5), n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)
rf_best = rf_grid.best_estimator_
print(f"   RF best: {rf_grid.best_params_} (CV AUC: {rf_grid.best_score_:.4f})")


# ---------------------------------------------------------------
#  STEP 6: Evaluate with proper metrics
# ---------------------------------------------------------------
# NOT using accuracy. Using precision, recall, F1, and ROC-AUC
# because those actually tell you how well the model catches fraud.

print("\n>> Step 6: Evaluation (Precision / Recall / ROC-AUC)...")
print("   (accuracy intentionally not used - misleading for imbalanced data)\n")

results = {}
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 5))
fig_roc, ax_roc = plt.subplots(figsize=(9, 7))

for i, (name, model) in enumerate([("Logistic Regression", lr_best), ("Random Forest", rf_best)]):
    preds = model.predict(X_test_sc)
    probs = model.predict_proba(X_test_sc)[:, 1]
    report = classification_report(y_test, preds, target_names=["Legitimate", "Fraud"], output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc_val = auc(fpr, tpr)
    
    results[name] = {
        "Precision (Fraud)": report["Fraud"]["precision"],
        "Recall (Fraud)": report["Fraud"]["recall"],
        "F1 (Fraud)": report["Fraud"]["f1-score"],
        "ROC-AUC": roc_auc_val,
    }
    
    print(f"   {name} (SMOTE + tuned):")
    print(f"     Precision = {report['Fraud']['precision']:.4f}")
    print(f"     Recall    = {report['Fraud']['recall']:.4f}")
    print(f"     F1        = {report['Fraud']['f1-score']:.4f}")
    print(f"     ROC-AUC   = {roc_auc_val:.4f}\n")
    
    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[i],
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    axes_cm[i].set_title(f"{name}", fontsize=12, weight="bold")
    axes_cm[i].set_ylabel("Actual")
    axes_cm[i].set_xlabel("Predicted")
    
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc_val:.3f})")

fig_cm.suptitle("Confusion Matrices (with SMOTE)", fontsize=14, weight="bold")
fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUT_DIR, "03_confusion_matrices.png"), dpi=150)
plt.close(fig_cm)
print("   Saved: 03_confusion_matrices.png")

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
ax_roc.set_xlabel("False Positive Rate", fontsize=12)
ax_roc.set_ylabel("True Positive Rate", fontsize=12)
ax_roc.set_title("ROC Curves", fontsize=14, weight="bold")
ax_roc.legend(fontsize=11)
ax_roc.grid(True, alpha=0.3)
fig_roc.tight_layout()
fig_roc.savefig(os.path.join(OUTPUT_DIR, "04_roc_curves.png"), dpi=150)
plt.close(fig_roc)
print("   Saved: 04_roc_curves.png")

# precision-recall curve (more informative than ROC for imbalanced data)
fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
for name, model in [("Logistic Regression", lr_best), ("Random Forest", rf_best)]:
    probs = model.predict_proba(X_test_sc)[:, 1]
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax_pr.plot(recall_arr, precision_arr, linewidth=2, label=f"{name} (AP={ap:.3f})")
ax_pr.set_xlabel("Recall", fontsize=12)
ax_pr.set_ylabel("Precision", fontsize=12)
ax_pr.set_title("Precision-Recall Curves", fontsize=14, weight="bold")
ax_pr.legend(fontsize=11)
ax_pr.grid(True, alpha=0.3)
fig_pr.tight_layout()
fig_pr.savefig(os.path.join(OUTPUT_DIR, "05_precision_recall_curves.png"), dpi=150)
plt.close(fig_pr)
print("   Saved: 05_precision_recall_curves.png")

# which features matter most? (from random forest)
fig_fi, ax_fi = plt.subplots(figsize=(10, 7))
importances = rf_best.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
feat_imp.plot.barh(ax=ax_fi, color="#3498db", edgecolor="white")
ax_fi.set_title("Feature Importance (Random Forest)", fontsize=13, weight="bold")
ax_fi.set_xlabel("Importance")
fig_fi.tight_layout()
fig_fi.savefig(os.path.join(OUTPUT_DIR, "06_feature_importance.png"), dpi=150)
plt.close(fig_fi)
print("   Saved: 06_feature_importance.png")


# ---------------------------------------------------------------
#  STEP 7: Final comparison
# ---------------------------------------------------------------
print("\n>> Step 7: Model comparison\n")
comparison = pd.DataFrame(results).T.round(4)
print(comparison.to_string())

comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
print("\n   Saved: model_comparison.csv")

print("\n" + "=" * 60)
print("  Done! Project 2 complete.")
print("=" * 60)
print(f"""
  Key takeaway:
  Accuracy on this dataset would be ~{(y_test==0).mean()*100:.0f}% even if the model
  predicted "not fraud" for everything. That's why we used ROC-AUC
  and Recall instead - those actually measure fraud detection ability.
  
  SMOTE helped a lot by giving the model more fraud examples to learn from.
""")
