from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from config import (
    CLASSES,
    CONFIG,
    FEATURES_PATH,
    MATCHES_PATH,
    ODDS_COLUMNS,
    OUTPUT_DIR,
    set_seed,
)
from features import FEATURES, build_training_frame
from market import MARKET_FEATURES, add_market_features, implied_probabilities
from models import make_models, predict_proba_ordered

CLASS_LIST = list(CLASSES)


def load_frame(rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    matches = pd.read_parquet(MATCHES_PATH)
    df, _ = build_training_frame(matches)
    df = add_market_features(df)
    df.to_parquet(FEATURES_PATH)
    return df


def eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    floor = CONFIG.features.min_prior_matches
    return df[(df["h_matches_played"] >= floor) & (df["a_matches_played"] >= floor)]


def simulate_returns(
    probs: np.ndarray,
    test: pd.DataFrame,
    thresholds: tuple[float, ...] = CONFIG.backtest.edge_thresholds,
) -> dict[str, dict[str, Any]]:
    stake_size = CONFIG.backtest.stake
    market = implied_probabilities(test).to_numpy()
    odds = test[[ODDS_COLUMNS[c] for c in CLASSES]].to_numpy()
    actual = test["FTR"].to_numpy()

    results: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        staked = returned = 0.0
        bets = wins = 0
        for i in range(len(test)):
            if np.isnan(odds[i]).any():
                continue
            for j, outcome in enumerate(CLASSES):
                if probs[i, j] > market[i, j] * (1.0 + threshold):
                    staked += stake_size
                    bets += 1
                    if actual[i] == outcome:
                        returned += stake_size * odds[i, j]
                        wins += 1
        results[str(threshold)] = {
            "bets": bets,
            "staked": round(staked, 2),
            "returned": round(returned, 2),
            "roi_pct": round(100.0 * (returned - staked) / staked, 2) if staked else None,
            "hit_rate_pct": round(100.0 * wins / bets, 1) if bets else None,
        }
    return results


def _score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    indices = np.array([CLASS_LIST.index(c) for c in y_true])
    onehot = np.eye(len(CLASS_LIST))[indices]
    return {
        "n": len(y_true),
        "log_loss": round(float(log_loss(y_true, probs, labels=CLASS_LIST)), 4),
        "accuracy_pct": round(100.0 * float(accuracy_score(indices, probs.argmax(axis=1))), 2),
        "brier": round(float(((probs - onehot) ** 2).sum(axis=1).mean()), 4),
    }


def run(rebuild: bool = False, output: str = "backtest_report.json") -> dict[str, Any]:
    set_seed()
    df = load_frame(rebuild=rebuild)
    usable = eligible_rows(df)

    names = [*make_models(), "lgbm_market"]
    collected: dict[str, dict[str, list]] = {name: {"y": [], "p": [], "idx": []} for name in names}

    for season in CONFIG.backtest.test_seasons:
        train = usable[usable["Season"] < season]
        test = usable[(usable["Season"] == season) & (usable["Division"] == "E0")]
        if test.empty:
            continue

        for name, model in make_models().items():
            model.fit(train[FEATURES], train["FTR"])
            probs = predict_proba_ordered(model, test[FEATURES], CLASSES)
            collected[name]["y"].extend(test["FTR"])
            collected[name]["p"].append(probs)
            collected[name]["idx"].extend(test.index)

        train_mkt = train[train["mkt_H"].notna()]
        test_mkt = test[test["mkt_H"].notna()]
        if len(train_mkt) > 500 and len(test_mkt):
            columns = FEATURES + MARKET_FEATURES
            model = make_models()["lgbm"]
            model.fit(train_mkt[columns], train_mkt["FTR"])
            probs = predict_proba_ordered(model, test_mkt[columns], CLASSES)
            collected["lgbm_market"]["y"].extend(test_mkt["FTR"])
            collected["lgbm_market"]["p"].append(probs)
            collected["lgbm_market"]["idx"].extend(test_mkt.index)

    report: dict[str, Any] = {
        "seed": CONFIG.seed,
        "test_seasons": [f"{s}-{str(s + 1)[2:]}" for s in CONFIG.backtest.test_seasons],
    }

    for name, payload in collected.items():
        if not payload["p"]:
            continue
        probs = np.vstack(payload["p"])
        y_true = np.array(payload["y"])
        rows = usable.loc[payload["idx"]]
        entry = _score(y_true, probs)
        entry["roi"] = simulate_returns(probs, rows)

        has_odds = rows["OddHome"].notna()
        if has_odds.any():
            market = implied_probabilities(rows[has_odds])[CLASS_LIST].to_numpy()
            entry["market_log_loss_same_rows"] = round(
                float(log_loss(y_true[has_odds.to_numpy()], market, labels=CLASS_LIST)), 4
            )
            entry["n_with_odds"] = int(has_odds.sum())
        report[name] = entry

    probs = np.vstack(collected["lgbm"]["p"])
    y_true = np.array(collected["lgbm"]["y"])
    seasons = usable.loc[collected["lgbm"]["idx"], "Season"].to_numpy()
    report["lgbm_per_season"] = {
        f"{s}-{str(s + 1)[2:]}": _score(y_true[seasons == s], probs[seasons == s])
        for s in CONFIG.backtest.test_seasons
        if (seasons == s).any()
    }

    path = OUTPUT_DIR / output
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "lgbm_per_season"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the feature frame from matches.parquet."
    )
    parser.add_argument(
        "--output", default="backtest_report.json", help="Report filename written into output/."
    )
    args = parser.parse_args()
    run(rebuild=args.rebuild, output=args.output)


if __name__ == "__main__":
    main()
