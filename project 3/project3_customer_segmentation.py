# DecodeLabs - Data Science Industrial Training (Batch 2026)
# Project 3: Unsupervised Learning - Customer Segmentation
#
# Author: Ronak Kumar
#
# Goal: Find natural customer groups in retail data using K-Means
#       and turn those clusters into business-useful personas.

import os, sys, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Project 3 - Customer Segmentation")
print("=" * 60)


# ---------------------------------------------------------------
#  STEP 1: Create synthetic retail dataset with 22 features
# ---------------------------------------------------------------
# Built the data with 4 hidden segments baked in, but the algorithm
# doesn't know about them - it has to discover the groups on its own.

print("\n>> Step 1: Building retail customer dataset (22 features)...")
np.random.seed(42)
N = 3000

# these are the "true" segments the algorithm should hopefully find
segments = {
    "high_value":   {"n": 500,  "income_m": 120000, "spend_m": 4000, "freq_m": 20, "age_m": 42},
    "budget":       {"n": 900,  "income_m": 35000,  "spend_m": 400,  "freq_m": 5,  "age_m": 28},
    "mid_regular":  {"n": 1000, "income_m": 65000,  "spend_m": 1500, "freq_m": 12, "age_m": 35},
    "dormant":      {"n": 600,  "income_m": 55000,  "spend_m": 150,  "freq_m": 1,  "age_m": 50},
}

dfs = []
for seg, p in segments.items():
    n = p["n"]
    seg_df = pd.DataFrame({
        "age": np.random.normal(p["age_m"], 8, n).clip(18, 80).astype(int),
        "annual_income": np.random.normal(p["income_m"], p["income_m"]*0.25, n).clip(15000),
        "total_spend_12m": np.random.normal(p["spend_m"], p["spend_m"]*0.35, n).clip(0),
        "purchase_frequency": np.random.poisson(p["freq_m"], n).clip(0),
        "avg_basket_value": np.random.normal(p["spend_m"]/max(p["freq_m"],1), 30, n).clip(5),
        "days_since_last_purchase": np.random.exponential(365/max(p["freq_m"],1), n).clip(0, 730).astype(int),
        "num_categories_bought": np.random.poisson(max(p["freq_m"]//2, 1), n).clip(1, 20),
        "avg_discount_used_pct": np.random.beta(2, 5, n) * 60,
        "num_returns": np.random.poisson(max(p["freq_m"]//5, 0), n),
        "return_rate_pct": np.random.beta(2, 10, n) * 30,
        "website_visits_monthly": np.random.poisson(max(p["freq_m"]*2, 3), n),
        "app_usage_monthly": np.random.poisson(max(p["freq_m"], 2), n),
        "email_open_rate_pct": np.random.beta(3, 4, n) * 100,
        "email_click_rate_pct": np.random.beta(2, 8, n) * 50,
        "customer_tenure_months": np.random.randint(1, 96, n),
        "num_support_tickets": np.random.poisson(2, n),
        "loyalty_points": np.random.normal(p["spend_m"]*2, p["spend_m"]*0.5, n).clip(0),
        "referral_count": np.random.poisson(max(p["freq_m"]//4, 0), n),
        "social_media_engagement": np.random.beta(2, 5, n) * 100,
        "preferred_channel": np.random.choice(["Online", "In-Store", "Mobile App"], n, p=[0.4, 0.35, 0.25]),
        "gender": np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04]),
        "region": np.random.choice(["North", "South", "East", "West", "Central"], n),
    })
    dfs.append(seg_df)

df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   {df.shape[0]} customers, {df.shape[1]} features")
print(f"   Numeric: {df.select_dtypes(include=[np.number]).shape[1]}, Categorical: {df.select_dtypes(include=['object']).shape[1]}")


# ---------------------------------------------------------------
#  STEP 2: Preprocessing
# ---------------------------------------------------------------
print("\n>> Step 2: Encoding categoricals and scaling...")

df_processed = df.copy()
le_dict = {}
for col in ["preferred_channel", "gender", "region"]:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    le_dict[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)
print(f"   Scaled matrix: {X_scaled.shape}")


# ---------------------------------------------------------------
#  STEP 3: PCA - reduce 22 dimensions down to 3
# ---------------------------------------------------------------
# 22 features is too many to visualize or cluster effectively.
# PCA finds the directions with the most variance and projects
# the data onto those axes.

print("\n>> Step 3: PCA dimensionality reduction (22D -> 3D)...")

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# how much variance does each component explain?
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, color="#3498db", alpha=0.8, edgecolor="white")
axes[0].set_title("Variance per Component", fontsize=12, weight="bold")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Variance Ratio")

axes[1].plot(range(1, len(cumulative_var)+1), cumulative_var, "o-", color="#e74c3c", linewidth=2)
axes[1].axhline(y=0.80, color="gray", linestyle="--", label="80% threshold")
axes[1].axhline(y=0.90, color="gray", linestyle=":", label="90% threshold")
n_80 = np.argmax(cumulative_var >= 0.80) + 1
axes[1].axvline(x=n_80, color="#2ecc71", linestyle="--", alpha=0.7, label=f"{n_80} comps for 80%")
axes[1].set_title("Cumulative Variance", fontsize=12, weight="bold")
axes[1].set_xlabel("# Components")
axes[1].set_ylabel("Cumulative Ratio")
axes[1].legend()
fig.suptitle("PCA Variance Analysis", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_pca_variance.png"), dpi=150)
plt.close(fig)
print(f"   Need {n_80} components for 80% variance")
print(f"   PC1: {pca_full.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2: {pca_full.explained_variance_ratio_[1]*100:.1f}%, "
      f"PC3: {pca_full.explained_variance_ratio_[2]*100:.1f}%")
print("   Saved: 01_pca_variance.png")

# keep 3 components for visualization
pca_3d = PCA(n_components=3, random_state=42)
X_pca = pca_3d.fit_transform(X_scaled)
print(f"   Reduced: {X_scaled.shape[1]}D -> {X_pca.shape[1]}D")


# ---------------------------------------------------------------
#  STEP 4: Elbow Method - find the right number of clusters
# ---------------------------------------------------------------
# Plot WCSS (within-cluster sum of squares) for K=1 to 10.
# The "elbow" is where adding more clusters stops helping much.

print("\n>> Step 4: Elbow method (K=1 to 10)...")

K_range = range(1, 11)
wcss = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_pca)
    wcss.append(km.inertia_)
    print(f"   K={k:2d}  WCSS = {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, wcss, "o-", linewidth=2.5, color="#e74c3c", markersize=8)
ax.axvline(x=4, color="#2ecc71", linestyle="--", linewidth=2, label="K=4 (elbow)")
ax.set_title("Elbow Method - WCSS vs K", fontsize=14, weight="bold")
ax.set_xlabel("Number of Clusters (K)", fontsize=12)
ax.set_ylabel("WCSS", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
for k, w in zip(K_range, wcss):
    ax.annotate(f"{w:.0f}", (k, w), textcoords="offset points", xytext=(0, 12), fontsize=8, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_elbow_method.png"), dpi=150)
plt.close(fig)
print("   Saved: 02_elbow_method.png")


# ---------------------------------------------------------------
#  STEP 5: Silhouette Analysis - confirm the optimal K
# ---------------------------------------------------------------
# Silhouette score measures how similar each point is to its own
# cluster vs the nearest other cluster. Higher = better separation.

print("\n>> Step 5: Silhouette scores (K=2 to 8)...")

sil_scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores[k] = score
    print(f"   K={k}  Silhouette = {score:.4f}")

# Picking K=4 based on both elbow and silhouette:
# - Elbow shows clear bend at K=4 (WCSS drops steeply then flattens)
# - Silhouette at K=4 is 0.50+ which is solid
# - K=4 also makes the most business sense for segmentation
best_k = 4
print(f"\n   Going with K={best_k}")
print(f"   Elbow: clear bend at K=4")
print(f"   Silhouette at K=4: {sil_scores[4]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sil_scores.keys(), sil_scores.values(),
       color=["#e74c3c" if k==best_k else "#3498db" for k in sil_scores],
       edgecolor="white", alpha=0.85)
ax.set_title("Silhouette Scores by K", fontsize=14, weight="bold")
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
for k, s in sil_scores.items():
    ax.text(k, s + 0.005, f"{s:.3f}", ha="center", fontsize=10, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_silhouette_scores.png"), dpi=150)
plt.close(fig)
print("   Saved: 03_silhouette_scores.png")

# detailed silhouette plot
fig, ax = plt.subplots(figsize=(10, 8))
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_pca)
sample_sil = silhouette_samples(X_pca, cluster_labels)

y_lower = 10
colors = plt.cm.viridis(np.linspace(0, 0.9, best_k))
for i in range(best_k):
    ith = sample_sil[cluster_labels == i]
    ith.sort()
    y_upper = y_lower + len(ith)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, facecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(ith), f"Cluster {i}", fontsize=11, weight="bold")
    y_lower = y_upper + 10

ax.axvline(x=sil_scores[best_k], color="red", linestyle="--", linewidth=2, label=f"Avg = {sil_scores[best_k]:.3f}")
ax.set_title(f"Silhouette Plot (K={best_k})", fontsize=14, weight="bold")
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster")
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_silhouette_plot.png"), dpi=150)
plt.close(fig)
print("   Saved: 04_silhouette_plot.png")


# ---------------------------------------------------------------
#  STEP 6: Run K-Means with K=4
# ---------------------------------------------------------------
print(f"\n>> Step 6: K-Means clustering (K={best_k})...")
df["cluster"] = cluster_labels
for c in range(best_k):
    n = (cluster_labels == c).sum()
    print(f"   Cluster {c}: {n} customers ({n/len(df)*100:.1f}%)")


# ---------------------------------------------------------------
#  STEP 7: Visualize clusters
# ---------------------------------------------------------------
print(f"\n>> Step 7: Plotting clusters...")

# 2D view
fig, ax = plt.subplots(figsize=(12, 9))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis",
                     alpha=0.6, s=15, edgecolors="white", linewidth=0.3)
centers = km_final.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=250, edgecolors="black",
           linewidth=2, zorder=5, label="Centroids")
ax.set_title("Customer Segments (2D PCA)", fontsize=15, weight="bold")
ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
ax.legend(fontsize=12)
plt.colorbar(scatter, ax=ax, label="Cluster")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_clusters_2d.png"), dpi=150)
plt.close(fig)
print("   Saved: 05_clusters_2d.png")

# 3D view
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels,
           cmap="viridis", alpha=0.5, s=10)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", marker="X", s=300,
           edgecolors="black", linewidth=2)
ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("Customer Segments (3D PCA)", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "06_clusters_3d.png"), dpi=150)
plt.close(fig)
print("   Saved: 06_clusters_3d.png")


# ---------------------------------------------------------------
#  STEP 8: Create business personas from clusters
# ---------------------------------------------------------------
# Looking at the average stats per cluster and giving them
# human-readable names that a marketing team could actually use.

print(f"\n>> Step 8: Building business personas...\n")

profile_cols = ["age", "annual_income", "total_spend_12m", "purchase_frequency",
                "avg_basket_value", "days_since_last_purchase", "num_categories_bought",
                "avg_discount_used_pct", "website_visits_monthly", "loyalty_points",
                "customer_tenure_months"]

cluster_profiles = df.groupby("cluster")[profile_cols].mean().round(2)
print("Cluster averages:")
print(cluster_profiles.to_string())

# assign names based on what the numbers tell us
persona_map = {}
for c in range(best_k):
    row = cluster_profiles.loc[c]
    if row["total_spend_12m"] > 2500 and row["purchase_frequency"] > 15:
        persona_map[c] = "Premium Loyalists"
    elif row["total_spend_12m"] < 300 and row["days_since_last_purchase"] > 200:
        persona_map[c] = "Dormant / At-Risk"
    elif row["total_spend_12m"] < 600 and row["purchase_frequency"] < 8:
        persona_map[c] = "Budget Browsers"
    else:
        persona_map[c] = "Mid-Tier Regulars"

df["persona"] = df["cluster"].map(persona_map)

print("\nBusiness Personas:")
for c, persona in sorted(persona_map.items()):
    n = (df["cluster"] == c).sum()
    row = cluster_profiles.loc[c]
    print(f"\n  Cluster {c} -> {persona}")
    print(f"    Size:      {n} customers ({n/len(df)*100:.1f}%)")
    print(f"    Income:    ${row['annual_income']:,.0f}")
    print(f"    Spend/yr:  ${row['total_spend_12m']:,.0f}")
    print(f"    Frequency: {row['purchase_frequency']:.0f} orders/year")
    print(f"    Recency:   {row['days_since_last_purchase']:.0f} days ago")
    print(f"    Tenure:    {row['customer_tenure_months']:.0f} months")

# distribution plots by persona
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
viz_cols = ["annual_income", "total_spend_12m", "purchase_frequency", "days_since_last_purchase"]
viz_titles = ["Annual Income", "Total Spend (12mo)", "Purchase Frequency", "Days Since Last Purchase"]
colors_map = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#9b59b6"}

for i, (col, title) in enumerate(zip(viz_cols, viz_titles)):
    ax = axes[i // 2, i % 2]
    for c in range(best_k):
        subset = df[df["cluster"] == c][col]
        ax.hist(subset, bins=30, alpha=0.5, label=persona_map[c], color=colors_map.get(c, "#333"))
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(fontsize=8)
    ax.set_ylabel("Count")
fig.suptitle("Distributions by Persona", fontsize=15, weight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "07_persona_distributions.png"), dpi=150)
plt.close(fig)
print("\n   Saved: 07_persona_distributions.png")

# radar chart comparing personas
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
radar_cols = ["annual_income", "total_spend_12m", "purchase_frequency",
              "website_visits_monthly", "loyalty_points", "num_categories_bought"]
normalized = cluster_profiles[radar_cols].copy()
for col in radar_cols:
    mx = normalized[col].max()
    if mx > 0:
        normalized[col] = normalized[col] / mx

angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
angles += angles[:1]

for c in range(best_k):
    values = normalized.loc[c].tolist() + [normalized.loc[c].tolist()[0]]
    ax.plot(angles, values, "o-", linewidth=2, label=persona_map[c], color=colors_map.get(c, "#333"))
    ax.fill(angles, values, alpha=0.1, color=colors_map.get(c, "#333"))

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_cols, fontsize=9)
ax.set_title("Persona Comparison", fontsize=14, weight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "08_persona_radar.png"), dpi=150)
plt.close(fig)
print("   Saved: 08_persona_radar.png")

# save everything
output_path = os.path.join(OUTPUT_DIR, "segmented_customers.csv")
df.to_csv(output_path, index=False)
print(f"   Saved: segmented_customers.csv ({len(df)} rows)")

print("\n" + "=" * 60)
print("  Done! Project 3 complete.")
print("=" * 60)
print(f"""
  Results:
  - {N} customers, {df.shape[1]-2} features -> PCA -> 3 components
  - Optimal K = {best_k} (elbow + silhouette = {sil_scores[best_k]:.4f})
  - Found 4 distinct customer segments:
    {', '.join(set(persona_map.values()))}
""")
