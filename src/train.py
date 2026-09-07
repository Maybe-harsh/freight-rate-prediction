"""
Model comparison on a temporal holdout.

Validation strategy:
- Development training: Jan-Aug 2025
- Development holdout: Sep-Oct 2025
- Final model selection is based primarily on MAE, with RMSE as a secondary
  diagnostic because the target contains a small high-rate tail.

Run:
    python -m src.train
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import apply_cleaning, fit_cleaning, load_train
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer

SPLIT_DATE = "2025-09-01"
CATBOOST_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
SKLEARN_CATEGORICAL = ["pickup", "delivery", "equipment"]
SKLEARN_FEATURES = SKLEARN_CATEGORICAL + NUMERIC_FEATURES


def temporal_split(df: pd.DataFrame, cutoff: str = SPLIT_DATE):
    train = df[df["date"] < cutoff].reset_index(drop=True)
    holdout = df[df["date"] >= cutoff].reset_index(drop=True)
    return train, holdout


def score(y_true, y_pred, label: str) -> dict:
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-6)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Freight rates in this dataset are positive, but guard against zero.
    denom = np.maximum(np.asarray(y_true, dtype=float), 1e-6)
    mape = float(np.mean(np.abs((np.asarray(y_true) - y_pred) / denom)) * 100)
    print(f"{label:42s} MAE=${mae:8.2f}   RMSE=${rmse:8.2f}   MAPE={mape:6.2f}%")
    return {"model": label, "mae": mae, "rmse": rmse, "mape": mape}


def build_linear_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), SKLEARN_CATEGORICAL),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", Ridge(alpha=1.0)),
    ])


def build_hgb_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ), SKLEARN_CATEGORICAL),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
        )),
    ])


def fit_catboost(
    frame: pd.DataFrame,
    target: pd.Series,
    log_target: bool,
    remove_haversine: bool = False,
) -> CatBoostRegressor:
    features = [
        f for f in CATBOOST_FEATURES
        if not (remove_haversine and f == "haversine_distance")
    ]
    categorical = [f for f in CATEGORICAL_FEATURES if f in features]
    y = np.log1p(target) if log_target else target

    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    model.fit(
        Pool(frame[features], y, cat_features=categorical)
    )
    return model


def predict_catboost(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    log_target: bool,
    remove_haversine: bool = False,
) -> np.ndarray:
    features = [
        f for f in CATBOOST_FEATURES
        if not (remove_haversine and f == "haversine_distance")
    ]
    categorical = [f for f in CATEGORICAL_FEATURES if f in features]
    pred = model.predict(Pool(frame[features], cat_features=categorical))
    return np.expm1(pred) if log_target else pred


def error_analysis(
    holdout: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str = "outputs/top_errors.csv",
) -> pd.DataFrame:
    errors = holdout[
        ["load_id", "pickup", "delivery", "distance", "equipment", "weight", "date"]
    ].copy()
    errors["actual_rate"] = np.asarray(y_true)
    errors["predicted_rate"] = np.asarray(y_pred)
    errors["absolute_error"] = np.abs(errors["actual_rate"] - errors["predicted_rate"])
    errors["rate_per_mile"] = errors["actual_rate"] / errors["distance"].replace(0, np.nan)
    errors = errors.sort_values("absolute_error", ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    errors.head(50).to_csv(output_path, index=False)
    return errors


def run() -> list[dict]:
    raw = load_train()
    dev_train_raw, holdout_raw = temporal_split(raw)

    print(
        f"dev-train: {len(dev_train_raw)} rows "
        f"({dev_train_raw['date'].min().date()} - {dev_train_raw['date'].max().date()})"
    )
    print(
        f"dev-holdout: {len(holdout_raw)} rows "
        f"({holdout_raw['date'].min().date()} - {holdout_raw['date'].max().date()})"
    )

    # Fit cleaning statistics ONLY on the training window.
    params = fit_cleaning(dev_train_raw, weight_strategy="abs")
    dev_train = engineer(apply_cleaning(dev_train_raw, params))
    holdout = engineer(apply_cleaning(holdout_raw, params))

    y_train = dev_train["posted_rate"]
    y_holdout = holdout["posted_rate"]
    results: list[dict] = []

    # Baseline 1: historical median.
    results.append(
        score(
            y_holdout,
            np.full(len(y_holdout), y_train.median()),
            "Baseline: median rate",
        )
    )

    # Baseline 2: distance multiplied by historical median rate/mile.
    rpm = float((y_train / dev_train["distance"].replace(0, np.nan)).median())
    results.append(
        score(
            y_holdout,
            holdout["distance"] * rpm,
            "Baseline: distance x rate/mi",
        )
    )

    # Ridge.
    lin = build_linear_pipeline()
    lin.fit(dev_train[SKLEARN_FEATURES], y_train)
    results.append(
        score(y_holdout, lin.predict(holdout[SKLEARN_FEATURES]), "Linear (Ridge)")
    )

    # HistGradientBoosting, raw target.
    hgb = build_hgb_pipeline()
    hgb.fit(dev_train[SKLEARN_FEATURES], y_train)
    results.append(
        score(
            y_holdout,
            hgb.predict(holdout[SKLEARN_FEATURES]),
            "HistGradientBoosting",
        )
    )

    # HistGradientBoosting, log target.
    hgb_log = build_hgb_pipeline()
    hgb_log.fit(dev_train[SKLEARN_FEATURES], np.log1p(y_train))
    results.append(
        score(
            y_holdout,
            np.expm1(hgb_log.predict(holdout[SKLEARN_FEATURES])),
            "HistGradientBoosting (log target)",
        )
    )

    # CatBoost, raw target.
    cat = fit_catboost(dev_train, y_train, log_target=False)
    cat_pred = predict_catboost(cat, holdout, log_target=False)
    results.append(score(y_holdout, cat_pred, "CatBoost"))

    # CatBoost, log target.
    cat_log = fit_catboost(dev_train, y_train, log_target=True)
    cat_log_pred = predict_catboost(cat_log, holdout, log_target=True)
    results.append(score(y_holdout, cat_log_pred, "CatBoost (log target)"))

    # Ablation: remove the highly correlated Haversine distance feature.
    cat_log_no_hav = fit_catboost(
        dev_train,
        y_train,
        log_target=True,
        remove_haversine=True,
    )
    cat_log_no_hav_pred = predict_catboost(
        cat_log_no_hav,
        holdout,
        log_target=True,
        remove_haversine=True,
    )
    results.append(
        score(
            y_holdout,
            cat_log_no_hav_pred,
            "CatBoost (log target, no haversine)",
        )
    )

    # Error analysis on the leading model.
    error_analysis(holdout, y_holdout, cat_log_pred)

    results_df = pd.DataFrame(results).sort_values("mae")
    Path("outputs").mkdir(exist_ok=True)
    results_df.to_csv("outputs/model_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("Ranked by MAE (holdout = Sep-Oct 2025):")
    print(results_df.to_string(index=False))
    print("\nSaved: outputs/model_comparison.csv")
    print("Saved: outputs/top_errors.csv")

    return results


if __name__ == "__main__":
    run()
