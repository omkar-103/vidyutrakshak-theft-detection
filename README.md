# VidyutRakshak — AI-Powered Electricity Theft Detection

**ML Bubble 2026 — Machine Learning Awareness & Skill Building Challenge**
Army Institute of Technology (AIT), Pune
Team: Kaivalya

---

## Problem Statement

Electricity theft and non-technical losses cost Indian power distribution companies (DISCOMs) an estimated ₹1.5 lakh crore annually. Current detection relies heavily on manual field inspections, which are slow, expensive, and easy to evade. VidyutRakshak uses machine learning on smart meter consumption data to flag likely theft cases automatically, with **explainable risk scores** that field inspectors can act on with confidence — not just a black-box "fraud" flag.

## Why This Domain

Most teams gravitate toward healthcare, fraud, or agriculture. Energy & power management is comparatively under-explored despite being one of India's largest and most concrete infrastructure problems, with a natural fit for real ML techniques: time-series anomaly detection, imbalanced classification, and explainability — all directly demanded by this challenge's evaluation criteria.

## Dataset

**SGCC (State Grid Corporation of China) Electricity Theft Detection Dataset** — a public, real-world benchmark dataset containing daily consumption records for 42,372 customers over 1,035 days (Jan 1, 2014 – Oct 31, 2016), with confirmed theft/non-theft labels (~8.5% theft, reflecting real-world class imbalance).

**Important note on dataset origin:** No public Indian DISCOM dataset with confirmed electricity-theft labels currently exists — this data is commercially sensitive and not released publicly by any state utility. SGCC is the standard, most widely cited public benchmark used in electricity-theft-detection research worldwide, and is used here as a training/validation proxy. The pipeline and features are designed to be format-agnostic, so the same approach is intended to be fine-tuned on real Indian DISCOM data (e.g. under India's Smart Metering National Programme, in partnership with utilities like MSEDCL, BSES, or Tata Power) once such labeled data becomes available.

**Citation:** Zheng, Z., Yang, Y., Niu, X., Dai, H.N., Zhou, Y. "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids." *IEEE Transactions on Industrial Informatics*, vol. 14, no. 4, pp. 1606–1615, April 2018.

## Methodology

1. **Data cleaning:** Missing values (~25% of the raw dataset) handled via linear interpolation per customer; outliers capped using the 3-sigma rule.
2. **Feature engineering:** 12 human-readable statistical features per customer (mean, std, min, max, median, skewness, kurtosis, zero-consumption-day %, first/second-half consumption drop %, weekly variance) — chosen specifically to remain interpretable for SHAP explainability.
3. **Modeling — two architectures compared:**
   - **XGBoost** on engineered features, with `scale_pos_weight` to handle class imbalance.
   - **LSTM** on raw daily consumption sequences, to compare feature-engineered vs. sequence-learning approaches.
4. **Explainability:** SHAP (TreeExplainer) applied to the XGBoost model for both global feature importance and individual-customer risk explanations.

## Results

| Model | AUC-ROC | F1 (theft class) |
|---|---|---|
| XGBoost | 0.749 | 0.299 |
| LSTM | 0.690 | — |

![XGBoost vs LSTM ROC Curve Comparison](model_comparison.png)

XGBoost on engineered statistical features outperformed the LSTM on raw sequences, suggesting that explicit statistical signals (consumption variance, drop percentage) capture theft-indicative patterns more effectively than sequence learning alone on this dataset. F1 for the theft class is evaluated separately from overall accuracy, since accuracy alone is misleading on an ~8.5%-imbalanced dataset — a model that always predicts "no theft" would score ~91% accuracy while being practically useless.

Full metrics are in `metrics.json`.

## Explainability Highlights

SHAP analysis identifies `weekly_variance`, `consumption_drop_pct`, and `max_consumption` as the strongest theft-indicative features — consistent with real-world theft patterns such as irregular tampering and sudden usage drops.

### Global Feature Importance
![Global SHAP Feature Importance](shap_summary.png)

### Single-Customer Risk Explanation
![Individual Customer SHAP Explanation](shap_individual_example.png)

The SHAP explanations show exactly which statistical factors drove an individual customer's risk score up or down, providing actionable transparency for field inspectors.

## Deployment Considerations

- **Data ingestion:** Designed to plug into smart meter data feeds from Indian DISCOMs under the Smart Metering National Programme.
- **Model refresh:** Feedback loop where field inspector outcomes (confirmed theft / false positive) are logged and used to periodically retrain the model, improving accuracy on real Indian consumption patterns over time.
- **Inspector-facing output:** Risk score + SHAP-based explanation per flagged customer, so inspections are prioritized and justified, not just a raw model score.
- **Scalability:** XGBoost inference is lightweight enough for near-real-time scoring across large customer bases; LSTM retained for periodic deeper sequence-level audits.

## Repository Structure

```
vidyutrakshak-theft-detection/
├── README.md
├── VidyutRakshak_Notebook.ipynb        # Full Colab notebook — data cleaning, feature engineering, training, SHAP
├── vidyutrakshak_xgb_model.json        # Trained XGBoost model
├── metrics.json                        # Performance metrics + dataset/deployment notes
├── model_comparison.png                # XGBoost vs LSTM ROC curve comparison
├── shap_summary.png                    # Global SHAP feature importance plot
├── shap_individual_example.png         # Individual customer SHAP explanation
└── docs/
    └── project_documentation.pdf       # Detailed project report
```

## Team

**Kaivalya**
Omkar Parelkar — [github.com/omkar-103](https://github.com/omkar-103) — omkarparelkar@gmail.com

## Submitted For

ML Bubble 2026 — Machine Learning Awareness & Skill Building Challenge, hosted by Army Institute of Technology (AIT), Pune.