# Project 3: Customer Segmentation

**Author:** Ronak Kumar | DecodeLabs Batch 2026

## What this does
Groups 3000 retail customers into segments using unsupervised learning, then turns those segments into business personas.

### Steps:
1. **PCA** - reduced 22 features down to 3 principal components for clustering and visualization
2. **Elbow Method** - plotted WCSS for K=1 to 10, clear bend at K=4
3. **Silhouette Analysis** - confirmed K=4 gives good cluster separation (score = 0.50)
4. **K-Means** - clustered with K=4
5. **Persona creation** - looked at each cluster's stats and gave them business-friendly names

### Personas found:
| Segment | % of customers | Avg spend/yr | How often they buy |
|---------|---------------|-------------|-------------------|
| Premium Loyalists | 16.6% | $4,155 | 20x/year |
| Mid-Tier Regulars | 33.3% | $1,504 | 12x/year |
| Budget Browsers | 30.6% | $392 | 5x/year |
| Dormant / At-Risk | 19.4% | $152 | 1x/year |

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python project3_customer_segmentation.py
```

Outputs go to `outputs/` folder (8 plots + segmented CSV).
