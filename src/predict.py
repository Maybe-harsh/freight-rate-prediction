"""
Final prediction pipeline.

This script:
1. Fits cleaning statistics on ALL labeled development data.
2. Trains the selected CatBoost log-target model.
3. Predicts validation.csv.
4. Enriches and predicts december_chart_inputs.csv.
5. Writes the two required output files.

Run:
    python -m src.predict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.data import (
    apply_cleaning,
    fit_cleaning,
    load_december,
    load_train,
    load_validation,
)
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
OUTPUT_DIR = Path("outputs")


def enrich_december_inputs(
    december: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add model-required fields that are intentionally absent from the
    December chart input file.

    All lookup/default values are derived from the labeled development
    dataset only. No validation data is used.
    """
    df = december.copy()

    # ---------------------------------------------------------
    # Geographic coordinates
    # ---------------------------------------------------------
    # Build city -> coordinates lookup from the training data.
    pickup_coords = (
        train[["pickup", "pickup_lat", "pickup_lon"]]
        .drop_duplicates("pickup")
        .set_index("pickup")
    )

    delivery_coords = (
        train[["delivery", "delivery_lat", "delivery_lon"]]
        .drop_duplicates("delivery")
        .set_index("delivery")
    )

    df = df.join(
        pickup_coords,
        on="pickup",
        how="left",
    )

    df = df.join(
        delivery_coords,
        on="delivery",
        how="left",
    )

    # ---------------------------------------------------------
    # Market / quote features
    # ---------------------------------------------------------
    # December chart inputs do not provide these fields.
    # Use training-only equipment medians as frozen defaults.
    market_by_equipment = (
        train.groupby("equipment")["market_index"]
        .median()
        .to_dict()
    )

    quote_by_equipment = (
        train.groupby("equipment")["quote_signal"]
        .median()
        .to_dict()
    )

    global_market = train["market_index"].median()
    global_quote = train["quote_signal"].median()

    df["market_index"] = (
        df["equipment"]
        .map(market_by_equipment)
        .fillna(global_market)
    )

    df["quote_signal"] = (
        df["equipment"]
        .map(quote_by_equipment)
        .fillna(global_quote)
    )

    # December chart inputs already provide weight, so we preserve
    # the supplied 32,000-lb value.
    return df


def train_final_model(
    frame: pd.DataFrame,
    target: pd.Series,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        Pool(
            frame[FEATURES],
            np.log1p(target),
            cat_features=CATEGORICAL_FEATURES,
        )
    )

    return model


def predict(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
) -> np.ndarray:
    pred_log = model.predict(
        Pool(
            frame[FEATURES],
            cat_features=CATEGORICAL_FEATURES,
        )
    )

    return np.maximum(np.expm1(pred_log), 1e-6)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_train()
    validation_raw = load_validation()
    december_raw = load_december()

    # ---------------------------------------------------------
    # Fit preprocessing statistics from labeled development data
    # only.
    # ---------------------------------------------------------
    params = fit_cleaning(
        train_raw,
        weight_strategy="abs",
    )

    # ---------------------------------------------------------
    # Normal train + validation pipeline
    # ---------------------------------------------------------
    train = engineer(
        apply_cleaning(train_raw, params)
    )

    validation = engineer(
        apply_cleaning(validation_raw, params)
    )

    # ---------------------------------------------------------
    # December enrichment
    # ---------------------------------------------------------
    december_enriched = enrich_december_inputs(
        december_raw,
        train_raw,
    )

    december = engineer(
        apply_cleaning(december_enriched, params)
    )

    # ---------------------------------------------------------
    # Train final model on all labeled development data
    # ---------------------------------------------------------
    model = train_final_model(
        train,
        train["posted_rate"],
    )

    # ---------------------------------------------------------
    # Validation predictions
    # ---------------------------------------------------------
    validation_pred = predict(
        model,
        validation,
    )

    submission = pd.DataFrame({
        "load_id": validation["load_id"].astype(str),
        "predicted_rate": validation_pred,
    })

    submission.to_csv(
        "validation_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # December predictions
    # ---------------------------------------------------------
    december_out = december_raw.copy()

    december_out["predicted_rate"] = predict(
        model,
        december,
    )

    # Preserve the exact required seven-column order.
    december_out = december_out[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ]

    december_out.to_csv(
        "data/december_chart_inputs.csv",
        index=False,
    )

    print(
        f"Saved validation_predictions.csv "
        f"({len(submission):,} rows)"
    )
    print(
        "Updated data/december_chart_inputs.csv "
        "(31 rows)"
    )
    print(
        "Final model: CatBoost with log1p target"
    )


if __name__ == "__main__":
    main()