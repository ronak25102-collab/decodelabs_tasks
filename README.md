# DecodeLabs - Data Science Projects

**Ronak Kumar** | DecodeLabs Industrial Training, Batch 2026

---

## Projects

### Project 1: Advanced EDA & Feature Engineering
Cleaned a messy e-commerce dataset (5000 records) by:
- Comparing **Mean, Median, and KNN imputation** for missing values - went with KNN since it preserved distributions best
- Removing outliers using both **Z-Score** and **IQR** methods
- Created **5 new features** like revenue_per_visit, engagement_score, spending_tier etc.
- 7 visualizations + cleaned CSV output

### Project 2: Fraud Detection Pipeline
Built a fraud detection system on 10K transactions with only 2% fraud rate:
- Showed why **accuracy is misleading** on imbalanced data (98% accuracy means nothing if you miss all the fraud)
- Used **SMOTE** to balance the training data
- Trained **Logistic Regression** and **Random Forest** with GridSearchCV tuning
- Evaluated with **Precision, Recall, and ROC-AUC** instead of accuracy
- Random Forest got 0.94 precision, LR got 0.92 recall

### Project 3: Customer Segmentation
Segmented 3000 retail customers into groups using unsupervised learning:
- **PCA** to reduce 22 features down to 3 components
- Used **Elbow Method** and **Silhouette Scores** to find optimal K=4
- **K-Means** clustering produced 4 clear segments
- Translated clusters into business personas: Premium Loyalists, Mid-Tier Regulars, Budget Browsers, Dormant/At-Risk

---

## How to Run

```bash
git clone https://github.com/ronak25102-collab/decode_labs_projects.git
cd decode_labs_projects
pip install -r requirements.txt

python "project 1/project1_eda_feature_engineering.py"
python "project 2/project2_fraud_detection.py"
python "project 3/project3_customer_segmentation.py"
```

Each script saves its plots and CSV files in an `outputs/` folder.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn
