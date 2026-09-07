# Freight Rate Prediction

An end-to-end machine learning solution for freight load rate prediction using historical load data from January through October 2025.

## Project Overview

The objective is to predict the `posted_rate` for future freight loads.

The labeled development dataset contains 48,000 loads from January through October 2025. The final validation dataset contains 12,000 future loads from November through December 2025.

The project covers:

- Data quality assessment and cleaning
- Exploratory data analysis
- Feature engineering
- Time-based model validation
- Model comparison
- Error analysis
- Final model training
- Validation prediction generation
- December 2025 prediction generation
- Submission format validation

## Data

### Development data

`data/train_test.csv`

- 48,000 labeled loads
- January–October 2025
- Target: `posted_rate`

### Final validation data

`data/validation.csv`

- 12,000 unlabeled loads
- November–December 2025
- Unique identifier: `load_id`

### Submission template

`data/validation_predictions_template.csv`

Contains the 12,000 required `load_id` values and the `predicted_rate` column to be completed.

### December scenario

`data/december_chart_inputs.csv`

Contains the fixed December scenario used by the supplied scorer. The lane and load characteristics remain fixed while the date changes across December 2025.

## Approach

### 1. Data Quality

Several data-quality issues were investigated and handled consistently.

- Negative `weight` values were treated as sign errors and converted using `abs()`.
- Missing `weight` values were imputed using equipment-specific statistics fitted on the development-training window.
- Missing `market_index` values were also imputed using training-fitted equipment-specific statistics.
- Missingness indicators were retained as model features.
- Cleaning parameters were fitted only on the training portion and reused for the holdout and final inference data.

This keeps preprocessing consistent between training and future inference.

## Exploratory Data Analysis

The main findings from the development data were:

- `distance` has a strong relationship with `posted_rate` and provides a strong simple pricing baseline.
- The target distribution is right-skewed, with a relatively small number of unusually high-rate loads.
- Rate-per-mile behavior varies across lanes and equipment types.
- `market_index` shows temporal movement between the development and future periods, making it useful as a time-varying market signal.
- The validation set contains some pickup/delivery cities that are not present in the training data, so the modeling approach must handle unseen categorical values safely.
- Geographic distance and distance circuity provide additional information beyond the supplied distance field.

The largest prediction errors are concentrated in unusually high rate-per-mile observations. These observations occur across multiple equipment types and dates, suggesting that they represent difficult spot-market pricing behavior rather than a single obvious data-entry problem.

## Validation Strategy

A chronological holdout was used instead of a random train/test split.

| Dataset portion     | Period       |   Rows |
| ------------------- | ------------ | -----: |
| Development train   | Jan–Aug 2025 | 38,477 |
| Development holdout | Sep–Oct 2025 |  9,523 |

The split is designed to reproduce the real forecasting problem: training on earlier observations and predicting a later period.

A random split would allow observations from later months to appear in the training set while evaluating on earlier observations, which would be less representative of the final November–December prediction task.

All preprocessing parameters used for the holdout were fitted only on the development-training portion.

## Feature Engineering

The model uses a combination of categorical, numerical, geographic, route, and temporal features.

### Categorical features

- Pickup city
- Delivery city
- Equipment
- Route (`pickup -> delivery`)

### Numerical features

- Distance
- Weight
- Market index
- Quote signal
- Pickup/delivery coordinates
- Latitude and longitude differences
- Haversine distance
- Distance circuity
- Log circuity
- Date/calendar features
- Cyclical date features
- Missingness indicators

CatBoost is able to handle high-cardinality categorical variables natively, which is useful for the route feature.

## Model Comparison

Several progressively more expressive models were evaluated on the September–October temporal holdout.

| Model                               |         MAE |        RMSE |      MAPE |
| ----------------------------------- | ----------: | ----------: | --------: |
| **CatBoost (log target)**           | **$117.13** |     $636.27 | **5.18%** |
| CatBoost                            |     $118.18 |     $636.34 |     5.43% |
| CatBoost (log target, no Haversine) |     $119.14 |     $636.83 |     5.24% |
| HistGradientBoosting (log target)   |     $124.99 |     $637.60 |     5.39% |
| HistGradientBoosting                |     $133.63 | **$634.56** |     6.09% |
| Ridge                               |     $175.02 |     $648.40 |     9.88% |
| Distance × rate/mile baseline       |     $256.95 |     $684.25 |    11.63% |
| Median-rate baseline                |   $1,148.92 |   $1,569.42 |    70.15% |

**Primary metric:** MAE  
**Secondary metric:** RMSE  
**Diagnostic metric:** MAPE

MAE was used as the primary selection metric because it directly measures the typical absolute dollar error and is less dominated by a small number of extreme observations.

The final selected model was **CatBoost with a `log1p` target transformation**, which achieved the lowest MAE on the temporal holdout.

The log transformation reduces the influence of the highly right-skewed target during training and improved MAE compared with the non-log CatBoost model.

The Haversine ablation produced a slightly worse MAE, indicating that the geographic feature contributed useful information, although the improvement was modest.

## Error Analysis

The largest holdout errors were primarily underpredictions of loads with unusually high rate-per-mile values.

This is important because:

- the errors occur across multiple dates;
- they occur across multiple equipment types;
- they are not isolated to one route;
- the highest-error observations have substantially higher rate-per-mile values than typical loads.

This behavior explains why RMSE remains much larger than MAE. RMSE gives substantially more weight to these extreme errors.

The project therefore does not apply aggressive special-case rules to these observations. Instead, the model is designed to learn general pricing relationships while recognizing that genuine spot-market spikes are inherently difficult to predict from the available features.

## Final Model

After model selection, the selected CatBoost model with a `log1p` target was retrained using all labeled development data from January through October 2025.

The final model was then used to generate:

- 12,000 predictions for `data/validation.csv`
- 31 predictions for the fixed December 2025 scenario

The final prediction file is:

`validation_predictions.csv`

It contains exactly:

```text
load_id,predicted_rate
```
