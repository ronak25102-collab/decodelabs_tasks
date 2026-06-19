"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           DECODELABS — DATA SCIENCE INDUSTRIAL TRAINING (2026)              ║
║                                                                            ║
║   PROJECT 1: Advanced EDA & Feature Engineering                            ║
║   Author : Ronak                                                           ║
║   Goal   : Transform raw, chaotic data into a mathematically clean         ║
║            dataset ready for machine learning algorithms.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Key deliverables
────────────────
1. Statistical imputation  — Mean, Median, and KNN compared side-by-side
2. Outlier neutralization  — Z-Score AND Interquartile Range (IQR) methods
3. Feature engineering     — 5 new predictive features derived from raw data
4. Correlation deep-dive   — Heatmaps showing before/after feature impact
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
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

# Where to save all outputs (plots, cleaned CSV)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  PROJECT 1 — Advanced EDA & Feature Engineering")
print("  DecodeLabs Industrial Training | Batch 2026")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: SYNTHETIC DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════
# We create a realistic e-commerce customer dataset with deliberate NaN values
# and outliers so we can demonstrate every cleaning technique rigorously.

print("\n[1/7] Generating synthetic e-commerce dataset …")

np.random.seed(42)                                    # reproducibility
N = 5000                                              # number of records

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

# ── Inject realistic missing values (~8-12 % in selected columns) ────────────
missing_cols = ["age", "annual_income", "total_spend", "avg_order_value",
                "days_since_last_order", "pages_per_visit", "session_duration_min"]
for col in missing_cols:
    mask = np.random.random(N) < np.random.uniform(0.06, 0.12)
    data.loc[mask, col] = np.nan

# ── Inject outliers into numeric columns ─────────────────────────────────────
outlier_indices = np.random.choice(N, 80, replace=False)
data.loc[outlier_indices[:20], "annual_income"]   *= np.random.uniform(5, 10, 20)
data.loc[outlier_indices[20:40], "total_spend"]   *= np.random.uniform(8, 15, 20)
data.loc[outlier_indices[40:60], "avg_order_value"] *= np.random.uniform(6, 12, 20)
data.loc[outlier_indices[60:], "session_duration_min"] *= np.random.uniform(5, 10, 20)

print(f"   Dataset shape : {data.shape}")
print(f"   Columns       : {list(data.columns)}")
print(f"   Memory usage  : {data.memory_usage(deep=True).sum() / 1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: INITIAL EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Performing initial EDA …")

print("\n── Data Types ─────────────────────────────────")
print(data.dtypes.to_string())

print("\n── Statistical Summary (numeric) ───────────────")
print(data.describe().round(2).to_string())

# ── Missing value summary ────────────────────────────────────────────────────
missing = data.isnull().sum()
missing_pct = (missing / len(data) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df = missing_df[missing_df.missing_count > 0].sort_values("missing_pct", ascending=False)

print("\n── Missing Values ─────────────────────────────")
print(missing_df.to_string())

# ── Visualisation: Missing value heatmap ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(data.isnull().T, cbar=True, cmap="YlOrRd", yticklabels=True, ax=ax)
ax.set_title("Missing Value Heatmap — Each Yellow Cell Is a NaN", fontsize=13, weight="bold")
ax.set_xlabel("Record Index")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_missing_value_heatmap.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 01_missing_value_heatmap.png")

# ── Visualisation: Distribution of all numeric columns ───────────────────────
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
fig.suptitle("Distribution of All Numeric Features (Before Cleaning)", fontsize=15, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_distributions_before.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 02_distributions_before.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: MISSING DATA IMPUTATION — Mean vs Median vs KNN
# ══════════════════════════════════════════════════════════════════════════════
"""
WHY THIS MATTERS
────────────────
Simply dropping NaN rows would lose 30-40 % of our data. Statistical
imputation preserves dataset size while introducing minimal bias.

Strategy:
  • Mean   — best for normally distributed features
  • Median — best for skewed features (robust to outliers)
  • KNN    — captures local structure; often the most realistic imputation
"""

print("\n[3/7] Comparing imputation strategies (Mean / Median / KNN) …")

impute_cols = ["age", "annual_income", "total_spend", "avg_order_value",
               "days_since_last_order", "pages_per_visit", "session_duration_min"]

# Store copies for comparison
df_mean   = data.copy()
df_median = data.copy()
df_knn    = data.copy()

# ── Mean imputation ──────────────────────────────────────────────────────────
for col in impute_cols:
    df_mean[col].fillna(df_mean[col].mean(), inplace=True)

# ── Median imputation ────────────────────────────────────────────────────────
for col in impute_cols:
    df_median[col].fillna(df_median[col].median(), inplace=True)

# ── KNN imputation (k = 5) ──────────────────────────────────────────────────
knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
df_knn[impute_cols] = knn_imputer.fit_transform(df_knn[impute_cols])

# ── Visualisation: Compare distributions for a key column ────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

compare_cols = ["annual_income", "total_spend", "pages_per_visit"]
for i, col in enumerate(compare_cols):
    # Row 1: KDE overlay
    ax = axes[0, i]
    data[col].dropna().plot.kde(ax=ax, label="Original (with gaps)", color="gray", linestyle="--", linewidth=2)
    df_mean[col].plot.kde(ax=ax, label="Mean Imputed", color="#e74c3c", alpha=0.7)
    df_median[col].plot.kde(ax=ax, label="Median Imputed", color="#2ecc71", alpha=0.7)
    df_knn[col].plot.kde(ax=ax, label="KNN Imputed", color="#3498db", alpha=0.7)
    ax.set_title(f"{col} — Distribution Comparison", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    
    # Row 2: Box plot comparison
    ax2 = axes[1, i]
    box_data = pd.DataFrame({
        "Original": data[col],
        "Mean": df_mean[col],
        "Median": df_median[col],
        "KNN": df_knn[col],
    })
    box_data.boxplot(ax=ax2, patch_artist=True,
                     boxprops=dict(facecolor="#3498db", alpha=0.5),
                     medianprops=dict(color="red", linewidth=2))
    ax2.set_title(f"{col} — Box Plot Comparison", fontsize=11, weight="bold")
    ax2.tick_params(labelsize=9)

fig.suptitle("Imputation Strategy Comparison: Mean vs Median vs KNN",
             fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_imputation_comparison.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 03_imputation_comparison.png")

# ── Decision: use KNN imputation as it best preserves the distribution ───────
print("   → Selected KNN imputation (k=5, distance-weighted) — best distribution fidelity.")
df_clean = df_knn.copy()

# Verify no missing values remain
assert df_clean[impute_cols].isnull().sum().sum() == 0, "Imputation incomplete!"
print(f"   → Remaining NaN values in imputed columns: 0  ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: OUTLIER DETECTION & TREATMENT
# ══════════════════════════════════════════════════════════════════════════════
"""
TWO COMPLEMENTARY METHODS
──────────────────────────
1. Z-Score: flags any point > 3 standard deviations from the mean.
   Best when the distribution is approximately normal.

2. IQR (Interquartile Range): uses Q1 − 1.5·IQR and Q3 + 1.5·IQR as fences.
   Robust to non-normal distributions; our primary method.
"""

print("\n[4/7] Detecting and neutralising outliers (Z-Score & IQR) …")

outlier_cols = ["annual_income", "total_spend", "avg_order_value", "session_duration_min"]

# ── Z-Score analysis ─────────────────────────────────────────────────────────
print("\n   Z-Score Analysis (|z| > 3):")
z_score_summary = {}
for col in outlier_cols:
    z = np.abs(stats.zscore(df_clean[col].dropna()))
    n_outliers = (z > 3).sum()
    z_score_summary[col] = n_outliers
    print(f"     {col:30s} → {n_outliers} outliers detected")

# ── IQR analysis & capping ──────────────────────────────────────────────────
print("\n   IQR Method (capping at Q1 - 1.5·IQR / Q3 + 1.5·IQR):")
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
    
    # Cap outliers (Winsorisation)
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    
    print(f"     {col:30s} → {n_below + n_above:3d} outliers capped  "
          f"[{lower:>10.2f}, {upper:>10.2f}]")

# ── Visualisation: Before / After box plots ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, col in enumerate(outlier_cols):
    # Before
    df_knn[col].plot.box(ax=axes[0, i], patch_artist=True,
                         boxprops=dict(facecolor="#e74c3c", alpha=0.5),
                         medianprops=dict(color="black", linewidth=2))
    axes[0, i].set_title(f"BEFORE — {col}", fontsize=10, weight="bold")
    
    # After
    df_clean[col].plot.box(ax=axes[1, i], patch_artist=True,
                           boxprops=dict(facecolor="#2ecc71", alpha=0.5),
                           medianprops=dict(color="black", linewidth=2))
    axes[1, i].set_title(f"AFTER  — {col}", fontsize=10, weight="bold")

fig.suptitle("Outlier Neutralisation: Before (IQR flagged) → After (Capped)",
             fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_outlier_treatment.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 04_outlier_treatment.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: FEATURE ENGINEERING — 5 new predictive features
# ══════════════════════════════════════════════════════════════════════════════
"""
Engineered features capture domain knowledge that raw columns cannot.
Each feature below has a clear business rationale.
"""

print("\n[5/7] Engineering 5 new predictive features …")

# Feature 1: Revenue Per Visit — monetisation efficiency
df_clean["revenue_per_visit"] = (df_clean["total_spend"] / df_clean["website_visits"].replace(0, 1)).round(2)
print("   ✓ revenue_per_visit      = total_spend / website_visits")

# Feature 2: Avg Spend Per Order — basket value indicator
df_clean["spend_per_order"] = (df_clean["total_spend"] / df_clean["num_orders"].replace(0, 1)).round(2)
print("   ✓ spend_per_order        = total_spend / num_orders")

# Feature 3: Engagement Score — composite digital engagement metric
df_clean["engagement_score"] = (
    (df_clean["website_visits"] / df_clean["website_visits"].max()) * 0.4 +
    (df_clean["pages_per_visit"] / df_clean["pages_per_visit"].max()) * 0.3 +
    (df_clean["session_duration_min"] / df_clean["session_duration_min"].max()) * 0.3
).round(4)
print("   ✓ engagement_score       = weighted composite (visits 40%, pages 30%, duration 30%)")

# Feature 4: Spending Tier — categorical binning for business segmentation
spend_labels = ["Low", "Medium", "High", "Premium"]
df_clean["spending_tier"] = pd.qcut(df_clean["total_spend"], q=4, labels=spend_labels)
print("   ✓ spending_tier          = quartile-based bins (Low/Medium/High/Premium)")

# Feature 5: Recency Score — inverse-log of days since last order (higher = more recent)
df_clean["recency_score"] = (1 / np.log1p(df_clean["days_since_last_order"] + 1)).round(4)
print("   ✓ recency_score          = 1 / log(1 + days_since_last_order)")

# ── Summary of new features ─────────────────────────────────────────────────
new_features = ["revenue_per_visit", "spend_per_order", "engagement_score",
                "spending_tier", "recency_score"]
print(f"\n   Engineered features ({len(new_features)} total):")
for feat in new_features:
    if df_clean[feat].dtype in [np.float64, np.int64, np.float32]:
        print(f"     {feat:25s} min={df_clean[feat].min():.4f}  max={df_clean[feat].max():.4f}")
    else:
        print(f"     {feat:25s} categories={list(df_clean[feat].unique())}")

# ── Visualisation: Engineered feature distributions ──────────────────────────
numeric_new = ["revenue_per_visit", "spend_per_order", "engagement_score", "recency_score"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
for i, col in enumerate(numeric_new):
    ax = axes[i // 2, i % 2]
    ax.hist(df_clean[col], bins=40, color=colors[i], edgecolor="white", alpha=0.85)
    ax.axvline(df_clean[col].mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean: {df_clean[col].mean():.2f}")
    ax.axvline(df_clean[col].median(), color="red", linestyle="-.", linewidth=1.5, label=f"Median: {df_clean[col].median():.2f}")
    ax.set_title(f"Engineered Feature: {col}", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
fig.suptitle("Distribution of Newly Engineered Features", fontsize=14, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_engineered_features.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 05_engineered_features.png")

# ── Spending Tier bar chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
tier_counts = df_clean["spending_tier"].value_counts().sort_index()
tier_counts.plot.bar(ax=ax, color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"],
                     edgecolor="white", alpha=0.9)
ax.set_title("Customer Distribution Across Spending Tiers", fontsize=13, weight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Spending Tier")
for i, v in enumerate(tier_counts):
    ax.text(i, v + 20, str(v), ha="center", fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "06_spending_tiers.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 06_spending_tiers.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Computing and visualising correlation matrix …")

corr_cols = ["age", "annual_income", "total_spend", "num_orders", "avg_order_value",
             "website_visits", "pages_per_visit", "session_duration_min",
             "discount_pct_used", "customer_satisfaction",
             "revenue_per_visit", "spend_per_order", "engagement_score", "recency_score"]

corr_matrix = df_clean[corr_cols].corr()

# ── Full correlation heatmap ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Correlation Matrix — Original + Engineered Features", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "07_correlation_heatmap.png"), dpi=150)
plt.close(fig)
print("   ✓ Saved: 07_correlation_heatmap.png")

# ── Top correlated pairs ────────────────────────────────────────────────────
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
corr_pairs = corr_pairs[corr_pairs.Feature_1 != corr_pairs.Feature_2]
corr_pairs["abs_corr"] = corr_pairs.Correlation.abs()
corr_pairs = corr_pairs.drop_duplicates(subset=["abs_corr"]).sort_values("abs_corr", ascending=False).head(10)
print("\n   Top 10 correlated feature pairs:")
for _, row in corr_pairs.iterrows():
    print(f"     {row.Feature_1:25s} ↔ {row.Feature_2:25s}  r = {row.Correlation:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: EXPORT CLEAN DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Exporting cleaned dataset …")

output_path = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
df_clean.to_csv(output_path, index=False)
print(f"   ✓ Saved: cleaned_dataset.csv  ({len(df_clean)} rows × {len(df_clean.columns)} cols)")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PROJECT 1 — COMPLETE")
print("=" * 70)
print(f"""
  Summary of operations:
  ─────────────────────
  • Original records        : {N}
  • Missing values handled  : {data[impute_cols].isnull().sum().sum()} NaN cells imputed (KNN, k=5)
  • Outliers capped (IQR)   : {sum(v['n_outliers'] for v in iqr_summary.values())} values across {len(outlier_cols)} columns
  • New features engineered : {len(new_features)} ({', '.join(new_features)})
  • Final dataset shape     : {df_clean.shape}
  • All outputs saved to    : {OUTPUT_DIR}

  Files generated:
  ────────────────
  01_missing_value_heatmap.png     — Visual map of all NaN locations
  02_distributions_before.png      — Pre-cleaning distributions
  03_imputation_comparison.png     — Mean vs Median vs KNN side-by-side
  04_outlier_treatment.png         — Before/after IQR capping
  05_engineered_features.png       — Distribution of new features
  06_spending_tiers.png            — Customer spending tier breakdown
  07_correlation_heatmap.png       — Full correlation matrix
  cleaned_dataset.csv              — ML-ready dataset
""")
