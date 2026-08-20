# Premier League ML Prediction Engine

Week-on-week 1X2 (home / draw / away) prediction for the English Premier League,
with strict walk-forward validation, market comparison, and an accuracy + ROI
tracker that accumulates across the season.

## What the honest numbers are

Validated on **2,660 Premier League matches across 7 held-out seasons**
(2019-20 → 2025-26), training only on prior seasons at every step:

| Metric | Model | Bookmaker |
|---|---|---|
| Log loss | **0.9837** | 0.9597 |
| Winner accuracy | **53.2%** | 55.3% |
| Draw calibration | 23.5% predicted vs 23.6% actual |, |
| Flat-stake ROI on value bets | **−5.6%** |, |

**The model does not beat the closing line.** It gets close on log loss and is
well calibrated, but simulated betting against closing odds loses money, and
blending the model toward the market never improved on the market alone. This is
the normal result, closing odds are among the most efficient prediction markets
that exist. Anyone quoting you a model that "beats the bookies" on closing prices
is usually leaking future data.

Where an edge realistically lives:
1. **Early lines**, before the market converges, run the engine Monday, not Friday.
2. **Soft books** whose prices lag the sharp ones (Pinnacle/Betfair as reference).
3. **Secondary markets** (over/under, cards, corners) that get less attention.
4. **Feature edges the market underweights**, see the roadmap below.

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/premier-league-prediction-model.git
cd premier-league-prediction-model

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./scripts_download_data.sh     # fetch raw sources (~45MB)
python src/data.py             # unify into data/matches.parquet
python src/xg_data.py          # build team-level xG
python src/backtest.py         # walk-forward validation
python src/predict.py          # forecasts for the configured gameweek
python src/build_dashboard.py  # render output/dashboard.html
```

Requires Python 3.11 or 3.12.

### Training and evaluation

```bash
python src/backtest.py --rebuild            # rebuild features, then validate
python src/backtest.py --output custom.json # write elsewhere in output/
```

There is no separate "train" step to run before inference: `predict.py` refits on
the full history each time, which takes a few seconds and removes any chance of
serving a stale model.

### Inference

```bash
python src/predict.py src/fixtures_mw2.json
python src/predict.py src/fixtures_mw2.json --output week2.json
```

The gameweek file lists fixtures and American moneyline odds, and optionally
per-team Elo adjustments for team news:

```json
{
  "season": "2026-27",
  "gameweek": 2,
  "fixtures": [
    {"date": "2026-08-29", "home": "Arsenal", "away": "Everton",
     "ml_home": -400, "ml_draw": 520, "ml_away": 1100}
  ],
  "manual_adjustments": {"Everton": -40}
}
```

### Settling results and tracking

```bash
python src/track.py 2 results_gw2.json   # {"Arsenal v Everton": "2-0", ...}
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest                  # 23 tests, no data download required
ruff check src tests
```

## Layout

```
data/raw/          downloaded source CSVs (see Data sources)
data/matches.parquet   unified, cleaned match history (23,948 matches)
data/features.parquet  match history + engineered features
src/config.py      paths, seed, and every hyperparameter
src/models.py      estimator definitions
src/market.py      odds conversion and overround removal
src/data.py        loads and unifies all sources, normalises team names
src/features.py    leakage-safe feature engine (Elo, form, rest, H2H, shots)
src/backtest.py    walk-forward validation + ROI simulation
src/predict.py     generates a gameweek's predictions
src/track.py       settles results, maintains accuracy/ROI tracker
src/xg_data.py     builds match-level team xG from two sources
src/player_form.py squad availability / form features from FPL player data
src/build_dashboard.py  renders the HTML dashboard
docs/index.html    dashboard copy served by GitHub Pages
output/            predictions, backtest report, tracker, dashboard
tests/             pytest suite, runs on synthetic data
.github/workflows/ lint and test on every push
```

Dependencies flow one way: `config` into `data`/`features`/`market`/`models`,
then into `backtest` and `predict`. Evaluation code never imports from inference
code, and inference never imports from evaluation.

## Weekly workflow

```bash
# 1. Settle last week's results (once the games are played)
python src/track.py 1 results_gw1.json      # {"Arsenal v Coventry": "2-0", ...}

# 2. Create the next gameweek's fixture file (copy fixtures_mw1.json, edit
#    fixtures + odds; ml_* are American moneyline, converted automatically)
# 3. Predict
python src/predict.py src/fixtures_mw2.json

# 4. Rebuild the dashboard
python src/build_dashboard.py
```

To refresh the underlying match history mid-season, re-run `python src/data.py`.

## Features

Every feature for a match is computed **strictly from matches played before it** , 
state updates only after the row's features are recorded, so walk-forward training
can never see the future.

- **Elo** with margin-of-victory scaling and home advantage, plus an **xG-Elo**
  variant updated on the expected-goals margin. Shared across the
  Premier League and Championship so promoted teams carry a meaningful rating
  (Coventry entered 2026-27 at 1529, above Wolves and Burnley).
- **Rolling form**, points per game over the last 3 / 5 / 10 matches.
- **Venue splits**, home form for the home side, away form for the away side.
- **Goals** scored/conceded over 5 and 10 matches.
- **Shots on target** for and against over 5 matches.
- **Rest days** since each side's last match, and the differential.
- **Head-to-head**, home side's points share over the last 5 meetings.
- **Division and promotion flags**, was this side in the Championship last season.
- **Manual adjustment hook**, per-team Elo nudges in the fixture file for key
  absences (a missing first-choice keeper ≈ −40) or a new-manager bounce (≈ +25).

## Models

- **Logistic regression** (median impute → standardise → L2, C=0.1). Best log
  loss of the three; used for the shipped predictions.
- **LightGBM** multiclass, depth-limited and heavily regularised.
- **LightGBM + market features**, adds de-margined implied probabilities.
  Best accuracy (54.0%) and best ROI (−3.8%), but by construction hugs the market.

The shipped **pick** is a 25% model / 75% market blend. That weight costs almost
nothing in log loss (0.9606 vs the market's 0.9597) while keeping an independent
voice in the number.

## What happened when I added xG (a negative result worth reading)

xG data was sourced from two GitHub mirrors, Understat team-match data (2019-24)
and FPL's own Opta `expected_goals` summed to team level (2022-26), calibrated onto
a common scale. Coverage is 100% of Premier League matches from 2020-21 onward.

The data is sound: `corr(xG, goals) = 0.61`, squarely in the literature range, and
home xG exceeds away xG as it should.

**Adding xG as rolling features made the model worse**, consistently:

| Variant | Log loss |
|---|---|
| base features, all history | 0.9793 |
| control: base features, 2020+ only | 0.9894 |
| base + 13 xG features, 2020+ only | 1.0068 |
| LightGBM, base + xG, all history | 1.0126 |

Why, and this is the interesting part:

- xG **does** beat raw goal difference at predicting results (correlation with home
  points: 0.335 vs 0.286). The standard claim is correct in isolation.
- But **Elo beats both** (0.394).
- And xG is **69% collinear with Elo**. Residualise xG-difference on Elo and its
  remaining correlation with the result is **0.085**.

A margin-scaled Elo built on 25 years of results has already absorbed nearly
everything xG has to say. Adding 13 collinear features, while shrinking the
training window from 22 seasons to 5, since xG only exists from 2020, bought
variance and no signal.

**What did work**, marginally: folding xG into the Elo update itself, so the rating
moves on a blend of the scoreline and the xG margin. One extra feature, full
training window preserved, and it improved log loss from 0.9845 to 0.9837 and ROI
from −6.9% to −5.6%. Real, but small enough to be honest about.

## What happened when I added player form (a second negative result)

Squad-level features were built from FPL's player-by-gameweek data (2020-21 to
2025-26): minutes, ICT index and threat, aggregated to the eleven players with the
most minutes in each team's last five matches, which is knowable before a lineup is
announced. Features: squad availability (how much of that core actually played last
time out), squad ICT form, attacking threat, and concentration (how dependent the
team is on one player).

**These made the model worse too:**

| Variant | Log loss |
|---|---|
| control: base features, 2020+ only | 0.9890 |
| + 10 squad-form features | 1.0186 |
| + squad availability only | 1.0075 |
| + contribution-weighted availability | 1.0076 |

The diagnosis differs from the xG one, and is more interesting. Squad ICT form is
62% collinear with Elo, so it fails the same way xG did. But *availability* is
nearly **orthogonal** to Elo (r = 0.027), meaning it is genuinely new information
the ratings cannot contain. It just is not strong enough: its correlation with the
result is 0.053, and weighting it by each player's attacking contribution made it
slightly worse (0.042), not better.

The likely reason is that a minutes-based proxy cannot tell a rested player from an
injured one, and is backward-looking by construction: it reports who played last
week, not who is fit this week. No amount of feature engineering fixes that, because
the information simply is not in the historical minutes.

**Design conclusion:** team news belongs in this model as a human-supplied prior,
not a learned feature. Someone who knows "the first-choice keeper is out" on Friday
holds information no amount of historical minutes data contains. That is why the
fixture file exposes a per-team Elo adjustment (roughly -40 for a key absence, +25
for a new-manager bounce) rather than a learned availability feature.

## Known gaps (the v3 roadmap, in impact order)

1. **No squad market values in training.** Current values are carried in the
   fixture file for context, but no historical series was available, so the model
   cannot learn from them. This is the likely cause of the engine's biggest
   disagreements with the market, it rates Chelsea (€920m squad) at an Elo barely
   above Fulham's (€361m) because Elo only sees results, not the summer window.
   **Treat large model-vs-market gaps involving heavily-invested squads as a model
   blind spot, not as value.**
2. **No injury/lineup data.** Wire the manual adjustment hook, or a paid API.
3. **No real injury or lineup feed.** The minutes proxy failed; an actual team-news
   feed would test the availability hypothesis properly.
4. **No European fixture congestion flag**, no referee profiles, no weather.
5. **No Poisson / Dixon-Coles goal model** as an ensemble component. This is the
   most promising item on the list: a different functional form rather than another
   correlated feature.
6. **Season-boundary staleness**, rolling form carries across the summer, so
   matchweek 1 is the model's weakest week of the entire season.

## Data sources

- [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
 , E0 + E1, 2000–2025, with 1X2 odds, shots and cards (football-data.co.uk derived).
- [datasets/football-datasets](https://github.com/datasets/football-datasets)
 , Premier League 2025-26 results.
- [openfootball/england](https://github.com/openfootball/england)
 , Championship 2025-26 results (for the promoted clubs).
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
 , Understat team-match xG, and FPL's Opta `expected_goals` for the current season.

## Model card

Inputs, outputs, metrics, limitations and ethical considerations are documented in
[MODEL_CARD.md](MODEL_CARD.md).

## Reproducibility

Seeds are fixed at 42 across `random`, `numpy` and LightGBM, which runs with
`deterministic=True` and `n_jobs=1`. Dependencies are pinned exactly. Validation is
walk-forward with no shuffling, so re-running `backtest.py` on the pinned set
reproduces the published metrics.

## Licence

MIT. See [LICENSE](LICENSE).

This is an independent research project, not affiliated with or endorsed by the
Premier League or any club. It is not a betting product, and the backtest says
staking on it loses money.
