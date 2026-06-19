# DecodeLabs - Data Science Industrial Training (Batch 2026)
# Project 1: Advanced EDA & Feature Engineering
#
# Author: Ronak Kumar
# 
# Goal: Clean a raw dataset and make it ready for ML by handling
#       missing values, removing outliers, and creating new features.

import os, sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

# save plots and csv here
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Project 1 - Advanced EDA & Feature Engineering")
print("=" * 60)


# ---------------------------------------------------------------
#  STEP 1: Create a synthetic e-commerce dataset
# ---------------------------------------------------------------
# Using synthetic data so anyone can reproduce the results without
# needing an external CSV file. Added NaN values and outliers on
# purpose to practice cleaning techniques.

print("\n>> Step 1: Creating synthetic e-commerce dataset...")

np.random.seed(42)
N = 5000

data = pd.DataFrame({
    "customer_id"          : np.arange(1, N + 1),
    "age"                  : np.random.randint(18, 72, N).astype(float),
    "gender"               : np.random.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04]),
    "annual_income"        : np.round(np.random.lognormal(mean=10.8, sigma=0.5, size=N), 2),
    "total_spend"          : np.round(np.random.exponential(scale=500, size=N), 2),
    "num_orders"           : np.random.poisson(lam=8, size=N),
    "avg_order_value"      : np.round(np.random.gamma(shape=5, scale=20, size=N), 2),
    "days_since_last_order": np.random.randint(0, 365, N).astype(float),
    "website_visits"       : np.random.poisson(lam=25, size=N),
    "pages_per_visit"      : np.round(np.random.normal(loc=5.5, scale=1.8, size=N), 2),
    "session_duration_min" : np.round(np.random.gamma(shape=3, scale=4, size=N), 2),
    "discount_pct_used"    : np.round(np.random.beta(a=2, b=5, size=N) * 50, 2),
    "customer_satisfaction": np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.25, 0.35, 0.25]),
    "is_returning_customer": np.random.choice([0, 1], N, p=[0.35, 0.65]),
    "region"               : np.random.choice(["North", "South", "East", "West", "Central"], N),
})

# inject missing values (roughly 6-12% in selected columns)
missing_cols = ["age", "annual_income", "total_spend", "avg_order_value",
                "days_since_last_order", "pages_per_visit", "session_duration_min"]
for col in missing_cols:
    mask = np.random.random(N) < np.random.uniform(0.06, 0.12)
    data.loc[mask, col] = np.nan

# inject some outliers to practice removal
outlier_indices = np.random.choice(N, 80, replace=False)
data.loc[outlier_indices[:20], "annual_income"]   *= np.random.uniform(5, 10, 20)
data.loc[outlier_indices[20:40], "total_spend"]   *= np.random.uniform(8, 15, 20)
data.loc[outlier_indices[40:60], "avg_order_value"] *= np.random.uniform(6, 12, 20)
data.loc[outlier_indices[60:], "session_duration_min"] *= np.random.uniform(5, 10, 20)

print(f"   Shape: {data.shape}")
print(f"   Columns: {list(data.columns)}")


# ---------------------------------------------------------------
#  STEP 2: Initial EDA
# ---------------------------------------------------------------
print("\n>> Step 2: Exploratory Data Analysis...")

print("\nData types:")
print(data.dtypes.to_string())

print("\nBasic statistics:")
print(data.describe().round(2).to_string())

# how much data is missing?
missing = data.isnull().sum()
missing_pct = (missing / len(data) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("percent", ascending=False)
print("\nMissing values:")
print(missing_df.to_string())

# plot: missing value heatmap
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(data.isnull().T, cbar=True, cmap="YlOrRd", yticklabels=True, ax=ax)
ax.set_title("Missing Value Heatmap (yellow = NaN)", fontsize=13, weight="bold")
ax.set_xlabel("Record Index")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_missing_value_heatmap.png"), dpi=150)
plt.close(fig)
print("   Saved: 01_missing_value_heatmap.png")

# plot: distributions before cleaning
numeric_cols = data.select_dtypes(include=[np.number]).columns.drop("customer_id")
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):
        data[col].dropna().hist(bins=40, ax=axes[i], color="#3498db", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, weight="bold")
        axes[i].tick_params(labelsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Distributions Before Cleaning", fontsize=15, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_distributions_before.png"), dpi=150)
plt.close(fig)
print("   Saved: 02_distributions_before.png")


# ---------------------------------------------------------------
#  STEP 3: Missing Data Imputation (Mean vs Median vs KNN)
# ---------------------------------------------------------------
# Dropping rows with NaN would throw away ~30-40% of data, so
# imputation is the way to go. Comparing 3 approaches:
#   - Mean:   simple but gets pulled by outliers
#   - Median: more robust to skew
#   - KNN:    uses nearby rows to fill in, usually most realistic

print("\n>> Step 3: Comparing imputation methods (Mean / Median / KNN)...")

impute_cols = ["age", "annual_income", "total_spend", "avg_order_value",
               "days_since_last_order", "pages_per_visit", "session_duration_min"]

df_mean   = data.copy()
df_median = data.copy()
df_knn    = data.copy()

# mean imputation
for col in impute_cols:
    df_mean[col].fillna(df_mean[col].mean(), inplace=True)

# median imputation
for col in impute_cols:
    df_median[col].fillna(df_median[col].median(), inplace=True)

# KNN imputation (k=5, weighted by distance)
knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
df_knn[impute_cols] = knn_imputer.fit_transform(df_knn[impute_cols])

# plot: compare distributions after each method
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
compare_cols = ["annual_income", "total_spend", "pages_per_visit"]
for i, col in enumerate(compare_cols):
    ax = axes[0, i]
    data[col].dropna().plot.kde(ax=ax, label="Original", color="gray", linestyle="--", linewidth=2)
    df_mean[col].plot.kde(ax=ax, label="Mean", color="#e74c3c", alpha=0.7)
    df_median[col].plot.kde(ax=ax, label="Median", color="#2ecc71", alpha=0.7)
    df_knn[col].plot.kde(ax=ax, label="KNN", color="#3498db", alpha=0.7)
    ax.set_title(f"{col} - Density Comparison", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    
    ax2 = axes[1, i]
    box_data = pd.DataFrame({
        "Original": data[col], "Mean": df_mean[col],
        "Median": df_median[col], "KNN": df_knn[col],
    })
    box_data.boxplot(ax=ax2, patch_artist=True,
                     boxprops=dict(facecolor="#3498db", alpha=0.5),
                     medianprops=dict(color="red", linewidth=2))
    ax2.set_title(f"{col} - Box Plots", fontsize=11, weight="bold")
    ax2.tick_params(labelsize=9)

fig.suptitle("Imputation Comparison: Mean vs Median vs KNN", fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_imputation_comparison.png"), dpi=150)
plt.close(fig)
print("   Saved: 03_imputation_comparison.png")

# Going with KNN - it keeps the distribution shape the closest to original
print("   >> Chose KNN imputation (k=5) - best distribution match")
df_clean = df_knn.copy()

assert df_clean[impute_cols].isnull().sum().sum() == 0, "Still has NaNs!"
print(f"   Remaining NaN values: 0 (verified)")


# ---------------------------------------------------------------
#  STEP 4: Outlier Detection & Treatment (Z-Score + IQR)
# ---------------------------------------------------------------
# Using two methods:
# 1) Z-Score: flags points more than 3 std devs from mean (good for normal data)
# 2) IQR: Q1-1.5*IQR to Q3+1.5*IQR (works even on skewed distributions)
# I'll use IQR for the actual capping since most of our features are skewed.

print("\n>> Step 4: Outlier detection and capping...")

outlier_cols = ["annual_income", "total_spend", "avg_order_value", "session_duration_min"]

# Z-Score check first (just to show both methods)
print("\n   Z-Score results (|z| > 3):")
z_score_summary = {}
for col in outlier_cols:
    z = np.abs(stats.zscore(df_clean[col].dropna()))
    n_outliers = (z > 3).sum()
    z_score_summary[col] = n_outliers
    print(f"     {col}: {n_outliers} outliers")

# IQR-based capping
print("\n   IQR capping (Q1 - 1.5*IQR to Q3 + 1.5*IQR):")
iqr_summary = {}
for col in outlier_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_below = (df_clean[col] < lower).sum()
    n_above = (df_clean[col] > upper).sum()
    iqr_summary[col] = {"lower": lower, "upper": upper, "n_outliers": n_below + n_above}
    
    # winsorize - cap at bounds instead of removing
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    
    print(f"     {col}: {n_below + n_above} values capped [{lower:.2f}, {upper:.2f}]")

# plot: before vs after
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, col in enumerate(outlier_cols):
    df_knn[col].plot.box(ax=axes[0, i], patch_artist=True,
                         boxprops=dict(facecolor="#e74c3c", alpha=0.5),
                         medianprops=dict(color="black", linewidth=2))
    axes[0, i].set_title(f"BEFORE - {col}", fontsize=10, weight="bold")
    
    df_clean[col].plot.box(ax=axes[1, i], patch_artist=True,
                           boxprops=dict(facecolor="#2ecc71", alpha=0.5),
                           medianprops=dict(color="black", linewidth=2))
    axes[1, i].set_title(f"AFTER - {col}", fontsize=10, weight="bold")

fig.suptitle("Outlier Treatment: Before vs After IQR Capping", fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_outlier_treatment.png"), dpi=150)
plt.close(fig)
print("   Saved: 04_outlier_treatment.png")


# ---------------------------------------------------------------
#  STEP 5: Feature Engineering (5 new features)
# ---------------------------------------------------------------
# Creating new columns that might be more useful for ML models than
# the raw data. Each one has a business logic behind it.

print("\n>> Step 5: Creating 5 new features...")

# 1. how much revenue does each website visit generate?
df_clean["revenue_per_visit"] = (df_clean["total_spend"] / df_clean["website_visits"].replace(0, 1)).round(2)
print("   + revenue_per_visit = total_spend / website_visits")

# 2. average basket size per order
df_clean["spend_per_order"] = (df_clean["total_spend"] / df_clean["num_orders"].replace(0, 1)).round(2)
print("   + spend_per_order = total_spend / num_orders")

# 3. engagement score - weighted combo of visits, pages, and time spent
df_clean["engagement_score"] = (
    (df_clean["website_visits"] / df_clean["website_visits"].max()) * 0.4 +
    (df_clean["pages_per_visit"] / df_clean["pages_per_visit"].max()) * 0.3 +
    (df_clean["session_duration_min"] / df_clean["session_duration_min"].max()) * 0.3
).round(4)
print("   + engagement_score = weighted(visits 40%, pages 30%, duration 30%)")

# 4. spending tier - bin customers into quartiles
spend_labels = ["Low", "Medium", "High", "Premium"]
df_clean["spending_tier"] = pd.qcut(df_clean["total_spend"], q=4, labels=spend_labels)
print("   + spending_tier = quartile bins (Low/Medium/High/Premium)")

# 5. recency score - higher means they bought more recently
df_clean["recency_score"] = (1 / np.log1p(df_clean["days_since_last_order"] + 1)).round(4)
print("   + recency_score = 1 / log(1 + days_since_last_order)")

new_features = ["revenue_per_visit", "spend_per_order", "engagement_score",
                "spending_tier", "recency_score"]

print(f"\n   Created {len(new_features)} features total")
for feat in new_features:
    if df_clean[feat].dtype in [np.float64, np.int64, np.float32]:
        print(f"     {feat}: min={df_clean[feat].min():.4f}, max={df_clean[feat].max():.4f}")
    else:
        print(f"     {feat}: {list(df_clean[feat].unique())}")

# plot: distributions of new features
numeric_new = ["revenue_per_visit", "spend_per_order", "engagement_score", "recency_score"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
for i, col in enumerate(numeric_new):
    ax = axes[i // 2, i % 2]
    ax.hist(df_clean[col], bins=40, color=colors[i], edgecolor="white", alpha=0.85)
    ax.axvline(df_clean[col].mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {df_clean[col].mean():.2f}")
    ax.axvline(df_clean[col].median(), color="red", linestyle="-.", linewidth=1.5, label=f"Median: {df_clean[col].median():.2f}")
    ax.set_title(f"{col}", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
fig.suptitle("Engineered Feature Distributions", fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_engineered_features.png"), dpi=150)
plt.close(fig)
print("   Saved: 05_engineered_features.png")

# spending tier breakdown
fig, ax = plt.subplots(figsize=(8, 5))
tier_counts = df_clean["spending_tier"].value_counts().sort_index()
tier_counts.plot.bar(ax=ax, color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"],
                     edgecolor="white", alpha=0.9)
ax.set_title("Customers by Spending Tier", fontsize=13, weight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Tier")
for i, v in enumerate(tier_counts):
    ax.text(i, v + 20, str(v), ha="center", fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "06_spending_tiers.png"), dpi=150)
plt.close(fig)
print("   Saved: 06_spending_tiers.png")


# ---------------------------------------------------------------
#  STEP 6: Correlation Analysis
# ---------------------------------------------------------------
print("\n>> Step 6: Correlation matrix...")

corr_cols = ["age", "annual_income", "total_spend", "num_orders", "avg_order_value",
             "website_visits", "pages_per_visit", "session_duration_min",
             "discount_pct_used", "customer_satisfaction",
             "revenue_per_visit", "spend_per_order", "engagement_score", "recency_score"]

corr_matrix = df_clean[corr_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Correlation Heatmap (Original + Engineered Features)", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "07_correlation_heatmap.png"), dpi=150)
plt.close(fig)
print("   Saved: 07_correlation_heatmap.png")

# find strongest correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
corr_pairs = corr_pairs[corr_pairs.Feature_1 != corr_pairs.Feature_2]
corr_pairs["abs_corr"] = corr_pairs.Correlation.abs()
corr_pairs = corr_pairs.drop_duplicates(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(10)
print("\n   Strongest correlations:")
for _, row in corr_pairs.iterrows():
    print(f"     {row.Feature_1} <-> {row.Feature_2}: r = {row.Correlation:+.3f}")


# ---------------------------------------------------------------
#  STEP 7: Save the clean dataset
# ---------------------------------------------------------------
print("\n>> Step 7: Saving cleaned dataset...")

output_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
df_clean.to_csv(output_path, index=False)
print(f"   Saved: cleaned_dataset.csv ({len(df_clean)} rows x {len(df_clean.columns)} cols)")

print("\n" + "=" * 60)
print("  Done! Project 1 complete.")
print("=" * 60)
print(f"""
  What was done:
  - Started with {N} records, 15 columns
  - Imputed {data[impute_cols].isnull().sum().sum()} missing values using KNN (k=5)
  - Capped {sum(v['n_outliers'] for v in iqr_summary.values())} outliers using IQR method
  - Created {len(new_features)} new features
  - Final dataset: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns
  - All outputs in: {OUTPUT_DIR}
""")
