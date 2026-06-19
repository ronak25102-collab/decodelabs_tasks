# DecodeLabs — Data Science Industrial Training (Batch 2026)

**Author:** Ronak Kumar  
**Organization:** DecodeLabs  
**Track:** Data Science  

---

## 📁 Project Overview

This repository contains **3 complete Data Science projects** built as part of the DecodeLabs Industrial Training program. Each project demonstrates progressive mastery — from data wrangling to supervised and unsupervised machine learning.

| # | Project | Key Techniques | Status |
|---|---------|----------------|--------|
| 1 | [Advanced EDA & Feature Engineering](./project%201/) | KNN Imputation, Z-Score/IQR Outlier Treatment, Feature Engineering | ✅ Complete |
| 2 | [Supervised Learning — Fraud Detection](./project%202/) | SMOTE, Logistic Regression, Random Forest, ROC-AUC Evaluation | ✅ Complete |
| 3 | [Unsupervised Learning — Customer Segmentation](./project%203/) | PCA, K-Means, Elbow Method, Silhouette Analysis, Business Personas | ✅ Complete |

---

## 🚀 Project 1: Advanced EDA & Feature Engineering

**Goal:** Transform raw, chaotic data into a mathematically clean dataset ready for ML algorithms.

### What I Did:
- Generated a realistic **5,000-record e-commerce dataset** with injected NaN values and outliers
- Compared **3 imputation strategies** (Mean vs Median vs KNN) — selected KNN for best distribution fidelity
- Detected outliers using both **Z-Score** and **IQR methods** — capped 779 extreme values
- Engineered **5 new predictive features**: `revenue_per_visit`, `spend_per_order`, `engagement_score`, `spending_tier`, `recency_score`
- Produced **7 publication-quality visualizations** and a clean CSV output

### Sample Output:
| Imputation Method | Distribution Preservation |
|---|---|
| Mean | Shifts distribution center |
| Median | Robust but loses variance info |
| **KNN (k=5)** | **Best — preserves local structure** |

---

## 🔍 Project 2: Supervised Learning — Fraud Detection Pipeline

**Goal:** Build and tune a classification model to identify fraudulent transactions in a highly imbalanced dataset.

### What I Did:
- Created a **10,000-transaction dataset** with realistic **2% fraud rate** (extreme imbalance)
- Demonstrated why **Accuracy is DECEPTIVE** on imbalanced data (99.3% accuracy but only 66% Recall!)
- Applied **SMOTE** to the training set only (never the test set) to balance classes
- Trained and tuned **Logistic Regression** and **Random Forest** with GridSearchCV
- Evaluated using **Precision, Recall, F1-Score, and ROC-AUC** — deliberately discarded Accuracy

### Results:
| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.40 | **0.92** | 0.55 | **0.99** |
| Random Forest | **0.94** | 0.64 | **0.76** | 0.98 |

---

## 📊 Project 3: Unsupervised Learning — Customer Segmentation

**Goal:** Use distance-based algorithms to discover hidden mathematical groupings in unlabeled retail data.

### What I Did:
- Built a **3,000-customer dataset with 22 features** across spending, engagement, and demographics
- Applied **PCA** to reduce 22 dimensions → 3 principal components
- Used the **Elbow Method** (WCSS) and **Silhouette Score** to mathematically prove **K=4** as optimal
- Ran **K-Means clustering** and translated raw clusters into **4 actionable business personas**

### Discovered Personas:
| Persona | Size | Avg Income | Avg Spend/yr | Frequency |
|---------|------|------------|-------------|-----------|
| 💎 Premium Loyalists | 16.6% | $121K | $4,155 | 20/yr |
| 🛒 Mid-Tier Regulars | 33.3% | $63K | $1,504 | 12/yr |
| 💤 Dormant / At-Risk | 19.4% | $55K | $152 | 1/yr |
| 🏷️ Budget Browsers | 30.6% | $36K | $392 | 5/yr |

---

## 🛠️ Tech Stack

- **Language:** Python 3.14
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn
- **Techniques:** KNN Imputation, Z-Score, IQR, SMOTE, GridSearchCV, PCA, K-Means, Silhouette Analysis

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/ronak25102-collab/decode_labs_projects.git
cd decode_labs_projects

# Install dependencies
pip install -r requirements.txt

# Run any project
python "project 1/project1_eda_feature_engineering.py"
python "project 2/project2_fraud_detection.py"
python "project 3/project3_customer_segmentation.py"
```

Each script generates all outputs (plots + CSV) in an `outputs/` subfolder automatically.

---

*Built with dedication during the DecodeLabs Industrial Training Program, Batch 2026.*
