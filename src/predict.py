from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CLASSES,
    CONFIG,
    MATCHES_PATH,
    OUTCOME_LABELS,
    OUTPUT_DIR,
    set_seed,
)
from features import FEATURES, build_training_frame, fixture_features
from market import american_to_decimal, expected_value, implied_probabilities, normalise_rows
from models import make_models, predict_proba_ordered


def load_gameweek(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config = json.loads(Path(path).read_text())
    fixtures = pd.DataFrame(config["fixtures"])
    fixtures["Date"] = pd.to_datetime(fixtures["date"])
    fixtures = fixtures.rename(columns={"home": "HomeTeam", "away": "AwayTeam"})
    for outcome, column in (("H", "ml_home"), ("D", "ml_draw"), ("A", "ml_away")):
        fixtures[f"odd_{outcome}"] = fixtures[column].map(american_to_decimal)
    return config, fixtures


def train_model(df: pd.DataFrame, name: str = "logreg") -> Any:
    floor = CONFIG.features.min_prior_matches
    usable = df[(df["h_matches_played"] >= floor) & (df["a_matches_played"] >= floor)]
    model = make_models()[name]
    model.fit(usable[FEATURES], usable["FTR"])
    return model


def apply_manual_adjustments(feats: pd.DataFrame, adjustments: dict[str, float]) -> pd.DataFrame:
    if not adjustments:
        return feats
    feats = feats.copy()
    feats["elo_home"] += feats["HomeTeam"].map(adjustments).fillna(0.0)
    feats["elo_away"] += feats["AwayTeam"].map(adjustments).fillna(0.0)
    feats["elo_diff"] = feats["elo_home"] - feats["elo_away"]
    feats["elo_xg_diff"] += feats["HomeTeam"].map(adjustments).fillna(0.0)
    feats["elo_xg_diff"] -= feats["AwayTeam"].map(adjustments).fillna(0.0)
    return feats


def build_predictions(
    fixtures: pd.DataFrame,
    feats: pd.DataFrame,
    model_probs: Any,
    market_probs: Any,
    blended: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, fixture in fixtures.iterrows():
        odds = {c: float(fixture[f"odd_{c}"]) for c in CLASSES}
        model = {c: float(model_probs[i, j]) for j, c in enumerate(CLASSES)}
        market = {c: float(market_probs[i, j]) for j, c in enumerate(CLASSES)}
        blend = {c: float(blended[i, j]) for j, c in enumerate(CLASSES)}

        divergences = []
        for c in CLASSES:
            edge = model[c] / market[c] - 1.0 if market[c] > 0 else 0.0
            ev = expected_value(model[c], odds[c])
            if edge > CONFIG.predict.value_edge and ev > 0:
                divergences.append(
                    {
                        "outcome": OUTCOME_LABELS[c],
                        "code": c,
                        "odds": round(odds[c], 2),
                        "model_prob": round(100.0 * model[c], 1),
                        "market_prob": round(100.0 * market[c], 1),
                        "edge_pct": round(100.0 * edge, 1),
                        "ev_pct": round(100.0 * ev, 1),
                    }
                )

        best = max(CLASSES, key=lambda c: blend[c])
        rows.append(
            {
                "date": fixture["date"],
                "home": fixture["HomeTeam"],
                "away": fixture["AwayTeam"],
                "elo_home": round(float(feats.loc[i, "elo_home"]), 0),
                "elo_away": round(float(feats.loc[i, "elo_away"]), 0),
                "model": {c: round(100.0 * model[c], 1) for c in CLASSES},
                "market": {c: round(100.0 * market[c], 1) for c in CLASSES},
                "blended": {c: round(100.0 * blend[c], 1) for c in CLASSES},
                "odds": {c: round(odds[c], 2) for c in CLASSES},
                "pick": OUTCOME_LABELS[best],
                "pick_code": best,
                "pick_confidence": round(100.0 * blend[best], 1),
                "value_bets": sorted(divergences, key=lambda x: -x["ev_pct"]),
            }
        )
    return rows


def run(gameweek_path: Path, output_name: str | None = None) -> dict[str, Any]:
    set_seed()
    config, fixtures = load_gameweek(gameweek_path)

    matches = pd.read_parquet(MATCHES_PATH)
    df, engine = build_training_frame(matches)
    model = train_model(df)

    feats = fixture_features(engine, fixtures[["Date", "HomeTeam", "AwayTeam"]])
    adjustments = {
        k: float(v)
        for k, v in config.get("manual_adjustments", {}).items()
        if not k.startswith("_")
    }
    feats = apply_manual_adjustments(feats, adjustments)

    model_probs = predict_proba_ordered(model, feats[FEATURES], CLASSES)
    renamed = fixtures.rename(columns={"odd_H": "OddHome", "odd_D": "OddDraw", "odd_A": "OddAway"})
    market_probs = implied_probabilities(renamed).to_numpy()
    weight = CONFIG.predict.model_weight
    blended = normalise_rows(weight * model_probs + (1.0 - weight) * market_probs)

    predictions = build_predictions(fixtures, feats, model_probs, market_probs, blended)
    payload = {
        "season": config["season"],
        "gameweek": config["gameweek"],
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "odds_source": config.get("odds_source"),
        "model_weight": weight,
        "value_edge_threshold": CONFIG.predict.value_edge,
        "seed": CONFIG.seed,
        "predictions": predictions,
    }

    name = output_name or f"predictions_gw{config['gameweek']}.json"
    (OUTPUT_DIR / name).write_text(json.dumps(payload, indent=2))

    for row in predictions:
        model_str = "/".join(f"{row['model'][c]:5.1f}" for c in "HDA")
        market_str = "/".join(f"{row['market'][c]:5.1f}" for c in "HDA")
        print(
            f"{row['home']:14s} v {row['away']:14s} "
            f"model {model_str}  market {market_str}  -> "
            f"{row['pick']} ({row['pick_confidence']}%)"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gameweek predictions.")
    parser.add_argument(
        "gameweek",
        nargs="?",
        default=str(Path(__file__).parent / "fixtures_mw1.json"),
        help="Path to the gameweek fixture JSON.",
    )
    parser.add_argument("--output", default=None, help="Output filename in output/.")
    args = parser.parse_args()
    run(Path(args.gameweek), args.output)


if __name__ == "__main__":
    main()
