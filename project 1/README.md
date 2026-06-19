# Project 1: Advanced EDA & Feature Engineering

**Author:** Ronak Kumar | DecodeLabs Batch 2026

## What this does
Takes a raw e-commerce dataset (5000 customers) and cleans it up for machine learning:

1. **Missing value handling** - compared Mean, Median, and KNN imputation side by side. KNN worked best since it looks at similar rows to fill gaps instead of just using the average.

2. **Outlier removal** - used both Z-Score (flags anything 3+ std devs away) and IQR method (Q1-1.5*IQR to Q3+1.5*IQR). Went with IQR capping since most features are skewed.

3. **Feature engineering** - created 5 new columns:
   - `revenue_per_visit` - how much money each website visit generates
   - `spend_per_order` - average basket size
   - `engagement_score` - weighted combo of visits, pages viewed, and time spent
   - `spending_tier` - quartile-based categories (Low/Medium/High/Premium)
   - `recency_score` - higher = bought more recently

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python project1_eda_feature_engineering.py
```

Outputs go to `outputs/` folder (7 plots + cleaned CSV).
