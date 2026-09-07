# Freight Rate Prediction Assessment — Report Outline

## 1. Executive Summary
State the problem, the chronological validation strategy, the selected model, and the final deliverables.

## 2. Data Overview
- 48,000 labeled development loads
- January-October 2025
- 12,000 future validation loads
- November-December 2025

## 3. Data Quality
Discuss:
- missing `weight`
- missing `market_index`
- negative weights
- why negative weights were converted with `abs()`
- train-fitted imputation to avoid inference-time preprocessing inconsistency

## 4. Exploratory Findings
Discuss:
- strong relationship between distance and posted rate
- target skew / high-rate tail
- equipment and rate-per-mile behavior
- temporal market-index drift
- unseen cities in validation
- circuity edge cases

## 5. Feature Engineering
Describe date, route, geographic, distance/circuity, equipment, weight and market features.

## 6. Validation Strategy
Explain the Jan-Aug / Sep-Oct temporal holdout and why random splitting would be less representative of the future prediction task.

## 7. Model Comparison
Insert the values from `outputs/model_comparison.csv`.

## 8. Error Analysis
Use `outputs/top_errors.csv` to discuss the large-error tail and why RMSE is substantially larger than MAE.

## 9. Final Model
Explain why the selected model was chosen and whether the Haversine ablation materially changed performance.

## 10. December 2025 Prediction
Insert `scorer_results/candidate_december.png` and explain that the lane is fixed while date varies.

## 11. Limitations
Mention the rare high-rate-per-mile observations and the difficulty of predicting genuine spot-market spikes.

## 12. Reproducibility
Explain how to install dependencies and run the training, prediction and scorer commands.
