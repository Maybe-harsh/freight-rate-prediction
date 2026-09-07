# Freight Rate Prediction — Technical Report

## 1. Executive Summary

### Objective

Build a machine learning regression model to predict freight `posted_rate` using the labeled development data and generate predictions for the future validation loads.

The development dataset contains 48,000 labeled loads from January through October 2025. The final validation dataset contains 12,000 future loads from November through December 2025.

### Approach

The solution follows an end-to-end workflow:

1. Inspect and validate the raw data.
2. Identify and address data-quality issues.
3. Perform exploratory data analysis.
4. Engineer temporal, geographic, route, and market features.
5. Use a chronological validation split.
6. Compare baseline, linear, tree-based, and CatBoost models.
7. Select the model using holdout MAE.
8. Retrain the selected model on all labeled development data.
9. Generate the final 12,000 validation predictions.
10. Generate the required December 2025 predictions and scorer chart.

### Final Model

The selected model is a **CatBoost Regressor trained on a `log1p`-transformed target**.

On the September–October 2025 temporal holdout:

- MAE: **$117.13**
- RMSE: **$636.27**
- MAPE: **5.18%**

The model was selected because it achieved the lowest MAE among the evaluated models.

---

## 2. Data Overview

### Development Dataset

`data/train_test.csv`

- 48,000 rows
- January–October 2025
- Target variable: `posted_rate`

### Final Validation Dataset

`data/validation.csv`

- 12,000 rows
- November–December 2025
- Unique identifier: `load_id`
- Target unavailable during prediction

### Prediction Template

`data/validation_predictions_template.csv`

Contains the required validation `load_id` values and the `predicted_rate` column.

### December Scenario

`data/december_chart_inputs.csv`

Contains 31 fixed December scenarios where the date changes from December 1 through December 31, 2025 while the lane and load characteristics remain fixed.

---

## 3. Data Quality

### Missing Values

Missing values were found in `weight` and `market_index`.

The imputation strategy is based only on the development-training window. Equipment-specific statistics are fitted on the training portion and then reused for the holdout and final inference datasets.

This prevents preprocessing from using information from the future evaluation period.

### Negative Weights

Negative `weight` values were identified in the development and validation data.

The negative values were treated as sign errors because their absolute-value distribution was consistent with the positive weight observations.

The preprocessing therefore converts:

```python
weight = abs(weight)
```
