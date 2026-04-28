# Dota 2 — HSE ML Competition (2026)

Predicting Radiant team victory in Dota 2 matches from pre- and in-game features.
Course competition for the **HSE Machine Learning 1** course on [Kaggle](https://www.kaggle.com/competitions/dota-2-hse-ml-1-course-competition-2026).

> **Final result: 13th place on the private leaderboard, Gini = 0.41.**
---

## Problem

Binary classification: given match metadata (region, date, average MMR), player and hero data, in-game gold/experience advantage time series, and chat logs — predict the probability that the Radiant team wins. Evaluation metric is **Gini = 2·ROC-AUC − 1**.

## Approach

The solution is built around a **scikit-learn `LogisticRegression`** trained with **`TimeSeriesSplit` cross-validation** to respect the temporal nature of the data and avoid leakage from future matches. Features are added incrementally and validated fold-by-fold.

| Block | What it does |
|---|---|
| **Region / date features** | One-hot encoded region, day-of-week, season, weekend flag |
| **MMR** | `sqrt(avg_mmr)` transform + missing indicator |
| **Heroes** | Multi-label encoding of Radiant / Dire hero picks |
| **Gold / XP advantage** | Aggregations (mean, std, min/max, percentiles), trends (OLS slope), binarization of advantage curves |
| **Chat** | Per-match TF-IDF over tokenized Radiant / Dire chat (fit *inside* CV folds — no leakage) |
| **Tuning** | Optuna over `LogisticRegression` regularization + scaling strategy |

The final model combines all feature blocks via `scipy.sparse.hstack` and is wrapped in a reusable `DotaPipeline` class with toggle flags for each feature group, so ablations can be run with a single argument change.

## Repository layout

```
.
├── baseline.ipynb     # EDA + base features + first LogReg + Optuna tuning
├── pipeline.ipynb     # Chat TF-IDF, gold/exp aggregations & trends, final DotaPipeline
├── requirements.txt
└── README.md
```

## How to run

The notebooks expect a Kaggle environment (`kagglehub` is used to pull the competition data). To run locally:

```bash
pip install -r requirements.txt
# Place your kaggle.json in ~/.kaggle/ (chmod 600)
jupyter notebook baseline.ipynb
```

`pipeline.ipynb` additionally requires the tokenized chat artifact (`chat_tokenized.csv`), which is produced inside the notebook from the raw chat data.

## Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `SciPy` · `Optuna` · `Matplotlib` / `Seaborn` · `kagglehub`
