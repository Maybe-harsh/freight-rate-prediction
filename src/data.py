"""
Data loading and cleaning for the freight rate prediction task.

Design choices, and why, are documented inline next to each transform so this
file can double as the reference for the report / Loom walkthrough.

Preprocessing is fit/transform, not "clean each dataframe independently":
imputation medians are learned once from training data (`fit_cleaning`) and
then applied, frozen, to validation/december (`apply_cleaning`). This matters
in practice, not just in theory, for `market_index`: its per-equipment median
shifts ~13% between the train window (Jan-Oct) and the validation window
(Nov-Dec), consistent with it being a genuine time-varying market signal
rather than noise. Recomputing the median from validation itself would use
information a real deployed pipeline wouldn't have re-derived after the fact,
and would silently paper over that drift. `weight`'s median barely moves
train-to-validation (<1.5%), so the fix doesn't change its numbers materially
— but the pipeline is fit/transform for both, for consistency and because a
real production scorer processes one load (or a handful) at a time and can't
recompute a batch median on the fly anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

WeightStrategy = Literal["abs", "missing"]


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])


@dataclass
class CleaningParams:
    """Frozen statistics learned from training data, applied unchanged to any
    other split (validation, december, or a single new load at inference)."""

    weight_strategy: WeightStrategy
    weight_median_by_equipment: dict = field(default_factory=dict)
    market_index_median_by_equipment: dict = field(default_factory=dict)


def fit_cleaning(train: pd.DataFrame, weight_strategy: WeightStrategy = "abs") -> CleaningParams:
    df = train.copy()
    if weight_strategy == "abs":
        df["weight"] = df["weight"].abs()
    else:
        df.loc[df["weight"] < 0, "weight"] = np.nan

    return CleaningParams(
        weight_strategy=weight_strategy,
        weight_median_by_equipment=df.groupby("equipment")["weight"].median().to_dict(),
        market_index_median_by_equipment=df.groupby("equipment")["market_index"].median().to_dict(),
    )


def apply_cleaning(df: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    df = df.copy()

    # Some inference inputs (such as the December chart inputs) do not
    # contain weight or market_index. Add them using statistics learned
    # exclusively from the training data.
    if "weight" not in df.columns:
        global_weight_median = np.median(
            list(params.weight_median_by_equipment.values())
        )
        df["weight"] = df["equipment"].map(
            params.weight_median_by_equipment
        ).fillna(global_weight_median)

    if "market_index" not in df.columns:
        global_market_median = np.median(
            list(params.market_index_median_by_equipment.values())
        )
        df["market_index"] = df["equipment"].map(
            params.market_index_median_by_equipment
        ).fillna(global_market_median)

    if params.weight_strategy == "abs":
        df["weight"] = df["weight"].abs()
    else:
        df.loc[df["weight"] < 0, "weight"] = np.nan

    df["weight_was_missing"] = df["weight"].isna()
    df["market_index_was_missing"] = df["market_index"].isna()

    # Fall back to training-fitted equipment medians for missing values.
    global_weight_median = np.median(
        list(params.weight_median_by_equipment.values())
    )
    global_market_median = np.median(
        list(params.market_index_median_by_equipment.values())
    )

    df["weight"] = df.apply(
        lambda r: params.weight_median_by_equipment.get(
            r["equipment"], global_weight_median
        )
        if pd.isna(r["weight"]) else r["weight"],
        axis=1,
    )

    df["market_index"] = df.apply(
        lambda r: params.market_index_median_by_equipment.get(
            r["equipment"], global_market_median
        )
        if pd.isna(r["market_index"]) else r["market_index"],
        axis=1,
    )

    return df

def clean(df: pd.DataFrame, weight_strategy: WeightStrategy = "abs") -> pd.DataFrame:
    """Convenience wrapper for EDA / quick-look purposes ONLY: fits and applies
    cleaning on the same dataframe. Do NOT use this in the training pipeline —
    use fit_cleaning(train) + apply_cleaning(other_split, params) there so
    validation/december never influence their own imputation values."""
    params = fit_cleaning(df, weight_strategy=weight_strategy)
    return apply_cleaning(df, params)


if __name__ == "__main__":
    train = load_train()
    val = load_validation()
    dec = load_december()
    print("train:", train.shape, "| validation:", val.shape, "| december:", dec.shape)

    params = fit_cleaning(train)
    train_clean = apply_cleaning(train, params)
    val_clean = apply_cleaning(val, params)
    print("post-clean nulls (train):")
    print(train_clean[["weight", "market_index"]].isna().sum())
    print("post-clean nulls (validation, using TRAIN-fitted medians):")
    print(val_clean[["weight", "market_index"]].isna().sum())
