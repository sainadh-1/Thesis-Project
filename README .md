# Digital Advertising Campaign Conversion Rate Prediction

Predicting the **conversion rate** of digital advertising campaigns using machine learning, with an explicit focus on detecting and controlling for **target leakage**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Overview

Digital ad platforms generate huge volumes of campaign data, but predicting **conversion rate** accurately is hard: it depends on a complex mix of engagement, targeting, and spend variables — several of which can leak information about the outcome itself.

This project builds and compares three regression models — **Linear Regression**, **Random Forest**, and **XGBoost** — to predict conversion rate, and runs a dedicated leakage analysis to separate genuine predictive signal from variables that are mathematically derived from the target.

## Dataset

| | |
|---|---|
| Source | [Kaggle — Digital Advertising Campaign Performance Dataset](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset) |
| Size | 10,000 rows × 41 columns |
| Unit of observation | One advertising campaign per row |
| Target | `conversion_rate` |

Features cover campaign setup (platform, objective, placement), device and audience characteristics, impressions/clicks, spend, engagement metrics, and performance measures.

## Target Leakage

Target leakage occurs when a predictor is directly or indirectly derived from the target itself, producing unrealistically strong performance that won't generalize.

**Excluded as direct leakage:** `conversions`, `revenue`, `profit`, `ROAS`, `CPA`

`CPA` (Cost Per Acquisition) gets special treatment: it isn't dropped by default, since it's a standard campaign metric, but it's mathematically derivable from `conversions` and `ad_spend`. Model performance is therefore reported **with and without CPA** so the effect of this leakage risk is transparent rather than hidden.

## Workflow

1. Load the dataset
2. Inspect structure, data types, missing values, and duplicates
3. Drop the campaign ID
4. Engineer date-based features from `start_date`
5. Identify and separate potential leakage variables
6. Split predictors (`X`) and target (`y`)
7. Train/test split (80/20, `random_state=42`)
8. Preprocess numerical and categorical features
9. Train Linear Regression, Random Forest, and XGBoost
10. Evaluate with R², RMSE, and MAE
11. Run a CPA sensitivity analysis

## Preprocessing

- **Numerical features:** median imputation → standard scaling
- **Categorical features:** most-frequent imputation → one-hot encoding
- Combined via a `ColumnTransformer`, chained into a `Pipeline` with each model

## Results

### CPA sensitivity analysis (R²)

| Model | With CPA | Without CPA |
|---|---:|---:|
| Random Forest | 0.9849 | 0.6863 |
| XGBoost | 0.9766 | 0.8013 |
| Linear Regression | 0.6218 | 0.6229 |

Removing CPA sharply reduces R² for both ensemble models, confirming it carries strong leakage-adjacent signal (it's derivable from `conversions` and `ad_spend`). **The without-CPA results are treated as the primary, leakage-controlled results** — the with-CPA numbers are reported only to demonstrate the leakage effect.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · KaggleHub

## Repository structure

```
.
├── notebooks/
│   └── conversion_rate_prediction.ipynb   # full analysis & modeling
├── data/                                  # dataset (downloaded via KaggleHub)
├── README.md
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/conversion_rate_prediction.ipynb
```

The dataset is pulled automatically via `kagglehub` when the notebook runs — no manual download needed (a free Kaggle account/API token is required).

## A note on interpretation

Results should be read in the context of this specific dataset. Kaggle provides limited documentation on how the data was collected, so high R² values — even after leakage control — shouldn't be treated as evidence of real-world advertising behavior without validation against industry data.

## License

MIT — see [LICENSE](LICENSE) for details.
