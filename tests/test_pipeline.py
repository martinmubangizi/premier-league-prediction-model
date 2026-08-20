from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CLASSES, CONFIG, MATCHES_PATH, set_seed
from features import FEATURES, FeatureEngine, build_training_frame, fixture_features
from market import (
    american_to_decimal,
    expected_value,
    implied_probabilities,
    normalise_rows,
)
from models import make_models, predict_proba_ordered

TEAMS = [f"Team{i:02d}" for i in range(10)]


@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    rng = np.random.default_rng(CONFIG.seed)
    rows = []
    date = pd.Timestamp("2020-08-15")
    for season in (2020, 2021):
        for round_no in range(18):
            for i in range(0, len(TEAMS), 2):
                home, away = TEAMS[i], TEAMS[(i + 1 + round_no) % len(TEAMS)]
                if home == away:
                    continue
                hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
                rows.append(
                    {
                        "Date": date,
                        "Division": "E0",
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": hg,
                        "FTAG": ag,
                        "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                        "HS": rng.integers(5, 20),
                        "AS": rng.integers(5, 20),
                        "HST": rng.integers(1, 9),
                        "AST": rng.integers(1, 9),
                        "OddHome": 2.1,
                        "OddDraw": 3.4,
                        "OddAway": 3.6,
                        "Season": season,
                        "HomeXG": float(rng.uniform(0.4, 2.8)),
                        "AwayXG": float(rng.uniform(0.4, 2.8)),
                    }
                )
                date += pd.Timedelta(days=1)
            date += pd.Timedelta(days=6)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


@pytest.fixture(scope="module")
def built(synthetic_matches):
    return build_training_frame(synthetic_matches)


class TestMarketMath:
    def test_american_to_decimal_positive(self):
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_american_to_decimal_negative(self):
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_american_to_decimal_rejects_zero(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)

    def test_implied_probabilities_sum_to_one(self):
        df = pd.DataFrame({"OddHome": [2.0, 1.5], "OddDraw": [3.5, 4.0], "OddAway": [4.0, 7.0]})
        probs = implied_probabilities(df)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_implied_probabilities_column_order(self):
        df = pd.DataFrame({"OddHome": [1.5], "OddDraw": [4.0], "OddAway": [7.0]})
        probs = implied_probabilities(df)
        assert list(probs.columns) == list(CLASSES)
        assert probs["H"].iloc[0] > probs["D"].iloc[0] > probs["A"].iloc[0]

    def test_overround_is_removed(self):
        df = pd.DataFrame({"OddHome": [2.0], "OddDraw": [3.0], "OddAway": [4.0]})
        raw = 1 / 2.0 + 1 / 3.0 + 1 / 4.0
        assert raw > 1.0
        assert implied_probabilities(df).sum(axis=1).iloc[0] == pytest.approx(1.0)

    def test_expected_value(self):
        assert expected_value(0.5, 3.0) == pytest.approx(0.5)
        assert expected_value(0.25, 4.0) == pytest.approx(0.0)

    def test_normalise_rows(self):
        arr = np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0]])
        out = normalise_rows(arr)
        assert np.allclose(out.sum(axis=1), 1.0)


class TestFeaturePipeline:
    def test_shapes_and_columns(self, built, synthetic_matches):
        df, _ = built
        assert len(df) == len(synthetic_matches)
        missing = set(FEATURES) - set(df.columns)
        assert not missing, f"missing feature columns: {missing}"

    def test_no_feature_is_entirely_null(self, built):
        df, _ = built
        empty = [c for c in FEATURES if df[c].isna().all()]
        assert not empty, f"features that never populate: {empty}"

    def test_features_are_numeric(self, built):
        df, _ = built
        for column in FEATURES:
            assert pd.api.types.is_numeric_dtype(df[column]), column

    def test_first_match_has_no_history(self, built):
        df, _ = built
        first = df.iloc[0]
        assert pd.isna(first["h_pts5"])
        assert first["h_matches_played"] == 0

    def test_elo_is_zero_sum(self, built):
        _, engine = built
        total = sum(engine.elo.values())
        expected = CONFIG.elo.start_top_flight * len(engine.elo)
        assert total == pytest.approx(expected, abs=1e-6)

    def test_matches_played_counts_are_consistent(self, built, synthetic_matches):
        _, engine = built
        assert sum(engine.played.values()) == 2 * len(synthetic_matches)


class TestNoTargetLeakage:
    def test_replay_matches_streamed_features(self, synthetic_matches, built):
        df, _ = built
        target_index = len(synthetic_matches) // 2
        target = df.iloc[target_index]

        engine = FeatureEngine()
        for row in synthetic_matches.iloc[:target_index].itertuples():
            engine.update(
                row.Date,
                row.HomeTeam,
                row.AwayTeam,
                row.Division,
                row.Division,
                row.Season,
                row.FTHG,
                row.FTAG,
                row.FTR,
                row.HST,
                row.AST,
                row.HomeXG,
                row.AwayXG,
            )
        replay = engine.features_for(
            target["Date"],
            target["HomeTeam"],
            target["AwayTeam"],
            target["Division"],
            target["Division"],
            target["Season"],
        )
        for name in FEATURES:
            streamed, fresh = target[name], replay[name]
            if pd.isna(streamed) and pd.isna(fresh):
                continue
            assert streamed == pytest.approx(fresh), f"leak in feature: {name}"

    def test_features_precede_their_own_result(self, synthetic_matches):
        engine = FeatureEngine()
        rows = synthetic_matches.iloc[:40]
        for row in rows.itertuples():
            before = engine.features_for(
                row.Date,
                row.HomeTeam,
                row.AwayTeam,
                row.Division,
                row.Division,
                row.Season,
            )
            engine.update(
                row.Date,
                row.HomeTeam,
                row.AwayTeam,
                row.Division,
                row.Division,
                row.Season,
                row.FTHG,
                row.FTAG,
                row.FTR,
                row.HST,
                row.AST,
            )
            after = engine.features_for(
                row.Date,
                row.HomeTeam,
                row.AwayTeam,
                row.Division,
                row.Division,
                row.Season,
            )
            assert before["elo_diff"] != after["elo_diff"] or row.FTHG == row.FTAG


class TestModels:
    def test_probabilities_form_a_simplex(self, built):
        set_seed()
        df, _ = built
        usable = df.dropna(subset=["h_pts5", "a_pts5"])
        model = make_models()["logreg"]
        model.fit(usable[FEATURES], usable["FTR"])
        probs = predict_proba_ordered(model, usable[FEATURES], CLASSES)
        assert probs.shape == (len(usable), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_class_ordering_is_explicit(self, built):
        df, _ = built
        usable = df.dropna(subset=["h_pts5", "a_pts5"])
        model = make_models()["logreg"]
        model.fit(usable[FEATURES], usable["FTR"])
        ordered = predict_proba_ordered(model, usable[FEATURES].head(5), CLASSES)
        raw = model.predict_proba(usable[FEATURES].head(5))
        for j, label in enumerate(CLASSES):
            source = list(model.classes_).index(label)
            assert np.allclose(ordered[:, j], raw[:, source])

    def test_training_is_reproducible(self, built):
        df, _ = built
        usable = df.dropna(subset=["h_pts5", "a_pts5"])
        outputs = []
        for _ in range(2):
            set_seed()
            model = make_models()["lgbm"]
            model.fit(usable[FEATURES], usable["FTR"])
            outputs.append(predict_proba_ordered(model, usable[FEATURES], CLASSES))
        assert np.allclose(outputs[0], outputs[1])


class TestInference:
    def test_fixture_features_shape(self, built):
        _, engine = built
        fixtures = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2022-08-10")] * 2,
                "HomeTeam": TEAMS[:2],
                "AwayTeam": TEAMS[2:4],
            }
        )
        feats = fixture_features(engine, fixtures)
        assert len(feats) == 2
        assert not set(FEATURES) - set(feats.columns)
        assert feats["elo_home"].notna().all()
        assert feats["elo_diff"].notna().all()


@pytest.mark.skipif(not MATCHES_PATH.exists(), reason="real corpus not present")
class TestRealCorpus:
    def test_chronological_and_labelled(self):
        df = pd.read_parquet(MATCHES_PATH)
        assert df["Date"].is_monotonic_increasing
        assert set(df["FTR"].unique()) <= set(CLASSES)

    def test_results_agree_with_scores(self):
        df = pd.read_parquet(MATCHES_PATH)
        derived = np.where(
            df["FTHG"] > df["FTAG"], "H", np.where(df["FTAG"] > df["FTHG"], "A", "D")
        )
        assert (derived == df["FTR"].to_numpy()).all()

    def test_top_flight_seasons_are_complete(self):
        df = pd.read_parquet(MATCHES_PATH)
        counts = df[df["Division"] == "E0"].groupby("Season").size()
        recent = counts[counts.index >= 2021]
        assert (recent == 380).all(), recent.to_dict()
