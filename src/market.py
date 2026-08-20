from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLASSES, ODDS_COLUMNS

MARKET_FEATURES: list[str] = [f"mkt_{c}" for c in CLASSES]


def american_to_decimal(moneyline: float) -> float:
    value = float(moneyline)
    if value == 0:
        raise ValueError("moneyline odds cannot be zero")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    inverse = pd.DataFrame({c: 1.0 / df[ODDS_COLUMNS[c]] for c in CLASSES}, index=df.index)[
        list(CLASSES)
    ]
    return inverse.div(inverse.sum(axis=1), axis=0)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    probs = implied_probabilities(df)
    for c in CLASSES:
        df[f"mkt_{c}"] = probs[c]
    return df


def expected_value(probability: float, decimal_odds: float) -> float:
    return probability * decimal_odds - 1.0


def normalise_rows(probs: np.ndarray) -> np.ndarray:
    return probs / probs.sum(axis=1, keepdims=True)
