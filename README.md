# Freight Rate Prediction Challenge

## Overview

This repository contains an end-to-end freight rate regression solution for the Machine Learning Engineer assessment.

The development dataset contains labeled loads from January through October 2025. The final validation set contains future loads from November through December 2025.

## Approach

### 1. Data quality

- Negative `weight` values are treated as sign errors and converted with `abs()`.
- Missing `weight` and `market_index` values are imputed using statistics fitted on the development-training window only.
- Missingness indicators are retained.
- Cleaning parameters are frozen and reused for holdout/final inference.

### 2. Validation

A chronological split is used:

- Development train: January-August 2025
- Development holdout: September-October 2025

This better simulates the real task than a random split because the supplied validation data occurs later in time.

### 3. Feature engineering

Features include:

- date/calendar and cyclical date features
- pickup/delivery cities
- route
- coordinates and absolute coordinate differences
- Haversine distance
- distance circuity and log circuity
- distance, weight, equipment
- market index and quote signal
- missingness indicators

### 4. Model comparison

The training script compares:

- median-rate baseline
- distance × historical rate-per-mile baseline
- Ridge regression
- HistGradientBoosting
- log-target HistGradientBoosting
- CatBoost
- log-target CatBoost
- CatBoost log-target ablation without Haversine distance

Primary metric: **MAE**  
Secondary metric: **RMSE**  
Diagnostic: **MAPE**

The current leading model is CatBoost with a `log1p` target, based on the temporal holdout.

### 5. Final predictions

After model selection, the selected model is retrained on all labeled development data (January-October) and used to predict:

- all 12,000 rows in `validation.csv`
- all 31 December chart inputs

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run model comparison:

```bash
python -m src.train
```

This writes:

```text
outputs/model_comparison.csv
outputs/top_errors.csv
```

Generate final predictions:

```bash
python -m src.predict
```

Run the supplied scorer:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

The scorer validates the required submission structure and creates:

```text
scorer_results/candidate_december.png
```

## Important files

- `src/data.py` — data loading and train-fitted cleaning
- `src/features.py` — feature engineering
- `src/train.py` — temporal validation and model comparison
- `src/predict.py` — final training and prediction
- `score.py` — supplied submission validator/chart generator
- `validation_predictions.csv` — final 12,000-row submission
- `data/december_chart_inputs.csv` — December predictions
