"""
Feature engineering for the freight rate model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * np.arcsin(np.sqrt(a))


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["day_of_week"] = d.dt.dayofweek
    df["day_of_year"] = d.dt.dayofyear
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # cyclical encodings so Dec 31 and Jan 1 are treated as close, not far apart
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["haversine_distance"] = haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    # road distance vs straight-line: proxy for route circuity; also a cheap
    # sanity check against garbled lat/lon or distance values. Clipped because
    # a handful of very-short lanes (e.g. New Orleans<->Shreveport, haversine
    # ~7mi) produce ratios up to ~9.6x that reflect a tiny denominator, not a
    # genuinely inefficient route -- left unclipped this would give a few
    # rows outsized leverage. log1p keeps the raw signal too, for the model
    # to pick whichever form actually helps in validation.
    raw_circuity = df["distance"] / df["haversine_distance"].replace(0, np.nan)
    df["distance_circuity"] = raw_circuity.clip(upper=3.0)
    df["log_circuity"] = np.log1p(raw_circuity)
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["abs_lat_diff"] = df["lat_diff"].abs()
    df["abs_lon_diff"] = df["lon_diff"].abs()
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["route"] = df["pickup"] + " -> " + df["delivery"]
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_geo_features(df)
    df = add_route_features(df)
    return df


CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "route"]
NUMERIC_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_distance",
    "distance_circuity",
    "log_circuity",
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "weight_was_missing",
    "market_index_was_missing",
]
