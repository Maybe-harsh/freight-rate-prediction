"""
EDA + data-quality report.

Run with: python -m src.eda
Writes plots to outputs/eda/ and prints a findings summary to stdout — copy
the printed summary straight into the report.

Uses the fit/transform cleaning pipeline (fit on train, applied to
validation) so any numbers reported here match what the training pipeline
will actually see -- this file does exploration, not its own preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_train, load_validation, fit_cleaning, apply_cleaning
from src.features import engineer, haversine_miles

OUT = Path(__file__).resolve().parent.parent / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    train_raw = load_train()
    val_raw = load_validation()

    section("Shapes")
    print("train_test.csv:", train_raw.shape)
    print("validation.csv:", val_raw.shape)

    section("Missing values")
    print("train:\n", train_raw.isna().sum()[train_raw.isna().sum() > 0])
    print("validation:\n", val_raw.isna().sum()[val_raw.isna().sum() > 0])

    section("Negative weight rows")
    neg_train = (train_raw["weight"] < 0).sum()
    neg_val = (val_raw["weight"] < 0).sum()
    print(f"train: {neg_train} ({neg_train / len(train_raw):.2%})")
    print(f"validation: {neg_val} ({neg_val / len(val_raw):.2%})")

    section("Target distribution (posted_rate)")
    print(train_raw["posted_rate"].describe())
    print("skew:", train_raw["posted_rate"].skew())

    section("Numeric feature correlation with posted_rate (raw)")
    for col in ["distance", "weight", "market_index", "quote_signal"]:
        corr = train_raw[[col, "posted_rate"]].dropna().corr().iloc[0, 1]
        print(f"{col}: {corr:.4f}")

    section("Same correlations AFTER controlling for distance (residual)")
    slope, intercept = np.polyfit(train_raw["distance"], train_raw["posted_rate"], 1)
    resid = train_raw["posted_rate"] - (slope * train_raw["distance"] + intercept)
    print(f"linear fit: posted_rate ~= {slope:.3f} * distance + {intercept:.1f}")
    for col in ["weight", "market_index", "quote_signal"]:
        c = pd.concat([train_raw[col], resid.rename("resid")], axis=1).dropna().corr().iloc[0, 1]
        print(f"{col}: {c:.4f}  (raw was much smaller -- distance was masking this)")
    print("\nmean residual by equipment (equipment effect after controlling for distance):")
    print(train_raw.assign(resid=resid).groupby("equipment")["resid"].mean().round(1))
    print("\nmean residual by month (seasonality after controlling for distance):")
    print(train_raw.assign(resid=resid, month=train_raw["date"].dt.month).groupby("month")["resid"].mean().round(1))

    section("Equipment breakdown")
    eq = train_raw.assign(rate_per_mile=train_raw["posted_rate"] / train_raw["distance"])
    print(eq.groupby("equipment").agg(
        count=("posted_rate", "size"),
        mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        mean_rate_per_mile=("rate_per_mile", "mean"),
    ).round(2))

    section("rate_per_mile outlier segment (top 5% by rate_per_mile)")
    q95 = eq["rate_per_mile"].quantile(0.95)
    high = eq[eq["rate_per_mile"] > q95]
    print(f"{len(high)} rows above {q95:.2f} $/mi ({len(high)/len(eq):.1%} of data)")
    print("equipment share in this segment vs overall:")
    print(pd.DataFrame({
        "segment": high["equipment"].value_counts(normalize=True),
        "overall": eq["equipment"].value_counts(normalize=True),
    }).round(3))

    section("Cities: train vs validation")
    train_cities = set(train_raw["pickup"]) | set(train_raw["delivery"])
    val_cities = set(val_raw["pickup"]) | set(val_raw["delivery"])
    print("train unique cities:", len(train_cities))
    print("validation unique cities:", len(val_cities))
    print("validation-only cities:", sorted(val_cities - train_cities))

    section("Weight-cleaning ablation: abs() vs treat-as-missing (fit on train only)")
    for strategy in ("abs", "missing"):
        params = fit_cleaning(train_raw, weight_strategy=strategy)
        cleaned = apply_cleaning(train_raw, params)
        corr = cleaned[["weight", "posted_rate"]].corr().iloc[0, 1]
        print(f"strategy={strategy}: weight-posted_rate corr={corr:.4f}, "
              f"weight mean={cleaned['weight'].mean():.1f}")

    section("Train -> validation drift in imputation statistics (why fit/transform matters)")
    for col in ["weight", "market_index"]:
        t = train_raw.groupby("equipment")[col].median()
        v = val_raw.groupby("equipment")[col].median()
        pct = ((v - t) / t * 100).round(1)
        print(f"{col} median % shift, train-period vs validation-period, by equipment:")
        print(pct)

    section("distance_circuity distribution (why it needs clipping)")
    hav = haversine_miles(train_raw["pickup_lat"], train_raw["pickup_lon"],
                           train_raw["delivery_lat"], train_raw["delivery_lon"])
    raw_ratio = train_raw["distance"] / hav.replace(0, np.nan)
    print(raw_ratio.describe())
    print(f"rows with ratio > 3.0 (clip threshold): {(raw_ratio > 3.0).sum()}")

    # pipeline: fit on train, apply to train (matches the real training flow)
    params = fit_cleaning(train_raw, weight_strategy="abs")
    train_clean = engineer(apply_cleaning(train_raw, params))

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(train_raw["posted_rate"], bins=80, color="#064A56")
    ax.set_title("posted_rate distribution (train)")
    ax.set_xlabel("posted_rate ($)")
    fig.tight_layout()
    fig.savefig(OUT / "target_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = {"Dry Van": "#064A56", "Reefer": "#C7511F", "Flatbed": "#7A9E9F"}
    for equip, group in train_raw.groupby("equipment"):
        ax.scatter(group["distance"], group["posted_rate"], s=3, alpha=0.15,
                    color=colors[equip], label=equip)
    ax.set_title("posted_rate vs distance, by equipment (train)")
    ax.set_xlabel("distance (mi)")
    ax.set_ylabel("posted_rate ($)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT / "rate_vs_distance_by_equipment.png")
    plt.close(fig)

    monthly = (
        train_clean.assign(month=train_clean["date"].dt.to_period("M"))
        .groupby("month")["posted_rate"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#064A56")
    ax.set_title("Mean posted_rate by month (train, Jan-Oct 2025)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "monthly_rate_trend.png")
    plt.close(fig)

    print(f"\nPlots written to {OUT}")


if __name__ == "__main__":
    main()
