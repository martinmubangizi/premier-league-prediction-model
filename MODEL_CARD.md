# Model Card: Premier League 1X2 Outcome Model

Version 1.1. Last updated 20 August 2026.

## Model details

**Developed by:** Martin Mubangizi
**Model type:** Multinomial classifier over three outcomes (home win, draw, away win)
**Shipped estimator:** Multinomial logistic regression (median imputation, standardisation, L2 at C = 0.1)
**Alternatives evaluated:** LightGBM; LightGBM with market-derived features
**Training data:** 23,948 English Premier League and Championship matches, 2000 to 2026
**Licence:** MIT
**Seed:** 42, set across `random`, `numpy`, and LightGBM (`deterministic=True`, `n_jobs=1`)

## Intended use

**Primary use:** A public forecasting experiment. It publishes calibrated probabilities before kick-off and scores itself against outcomes, as a demonstration of honest model evaluation.

**Intended users:** People interested in sports analytics, forecast evaluation, or reproducible ML practice.

**Out of scope:**

- **Betting or staking decisions.** Simulated flat-stake returns against market odds are negative (-5.6% at a 5% edge threshold). The model loses money in backtest. It is not a betting product and must not be sold or presented as tipping advice.
- Leagues other than the English top two tiers. Ratings and home-advantage constants are fitted to this population.
- In-play or live prediction. Every feature is pre-match.
- Individual player outcomes, scorelines, or any market other than 1X2.

## Inputs

34 numeric features per fixture, all computed strictly from matches played before it:

| Group | Features |
|---|---|
| Ratings | `elo_home`, `elo_away`, `elo_diff`, `elo_xg_diff` |
| Form | Points per game over last 3, 5, 10 matches, per side |
| Venue form | Home form for the home side, away form for the away side (5-match window) |
| Goals | Scored and conceded over 5 and 10 matches, per side |
| Shots | Shots on target for and against, 5-match window |
| Scheduling | Days of rest per side (capped at 30) and the differential |
| History | Home side's points share over the last 5 meetings |
| Status | Division indicator and promoted-this-season flag, per side |
| Guards | Matches played per side, used to exclude thin-history rows |

**Optional human input:** a per-team Elo adjustment supplied in the gameweek fixture file, intended for team news (approximately -40 for a key absence, +25 for a new-manager bounce). This exists because availability could not be learned reliably from historical data (see Limitations).

## Outputs

A probability distribution over `{home win, draw, away win}` summing to 1. The published figure is a blend of 25% model and 75% overround-free market probability. The raw model output is also reported so the two can be compared.

## Training and evaluation

**Protocol:** Walk-forward by season. For each held-out season the model is refit from scratch on all prior seasons. No shuffling, no k-fold, no early stopping against the test fold.

**Held out:** Seven seasons, 2019-20 through 2025-26, 2,660 matches.

**Exclusions:** Matches where either side has fewer than eight prior matches, so rolling windows are populated rather than imputed.

## Metrics

| Metric | Model | Market benchmark |
|---|---|---|
| Log loss (primary) | 0.9837 | 0.9597 |
| Accuracy | 53.2% | 55.3% |
| Brier score | 0.596 | not computed |
| Simulated ROI, 5% edge threshold | -5.6% | not applicable |

**Why log loss is primary:** accuracy is degenerate here. A model that never predicts a draw can score well on accuracy while being badly calibrated, and draws are roughly 24% of matches.

**Calibration** across the seven held-out seasons:

| Bucket | Predicted | Observed | n |
|---|---|---|---|
| Home win 10-20% | 0.154 | 0.149 | 215 |
| Home win 30-40% | 0.352 | 0.334 | 467 |
| Home win 50-65% | 0.571 | 0.544 | 618 |
| Home win 65%+ | 0.735 | 0.708 | 456 |
| Draw 20-30% | 0.248 | 0.250 | 2115 |
| Away win 20-30% | 0.248 | 0.264 | 571 |

Aggregate draw rate: 23.5% predicted against 23.6% observed. The model is mildly overconfident in the top bucket, the standard shrinkage signature, which the market blend corrects.

## Limitations

1. **It does not beat the market.** The betting market is better calibrated on the same matches. Blending toward the market never improved on the market alone.

2. **No squad valuation.** The model rates teams on results, so it cannot see transfer spending. Its largest disagreements with the market cluster on expensively assembled squads that have not yet converted spending into results, and these are model errors rather than market inefficiencies.

3. **Expected goals adds almost nothing.** xG is 69% collinear with Elo; residualised on Elo its correlation with the outcome is 0.085. Adding rolling xG features degraded log loss. Only folding xG into the Elo update helped, and only by 0.0008.

4. **Player availability could not be learned.** Squad availability derived from historical minutes is nearly orthogonal to Elo (r = 0.027), so it is genuinely new information, but its correlation with the outcome is only 0.053 and adding it degraded performance. A minutes-based proxy cannot separate a rested player from an injured one, and reports who played last week rather than who is fit this week. Team news therefore enters as a human-supplied prior.

5. **Season-boundary staleness.** Rolling form carries across the summer, so the opening matchweek is the model's weakest week of the season.

6. **Promoted clubs.** These carry Elo from a lower division. The cross-division scale is maintained by promoted and relegated clubs transporting ratings, which is reasonable but noisier than within-division comparisons.

7. **Draw modelling.** Draws are never the argmax prediction, which is expected for 1X2 models but means accuracy understates the model's usefulness on that class.

8. **Data provenance.** Odds are from a mirrored aggregate rather than a single book's verified closing line, so the market benchmark is approximate.

## Ethical considerations

This model touches gambling, an activity with real potential for harm. Three deliberate choices follow from that:

- The negative backtest is published as prominently as any other result. Selective reporting here is not a presentational issue but a safety one.
- The dashboard opens with a notice stating it is not a betting product.
- The project is not monetised and tips are not sold. The backtest says staking on it loses money, and that is stated publicly rather than buried.

Anyone claiming a model that reliably beats closing odds should be assumed to have leaked future information until they show a walk-forward backtest.

## Reproducing these numbers

```bash
pip install -r requirements.txt
./scripts_download_data.sh
python src/data.py
python src/xg_data.py
python src/backtest.py --rebuild
```

Expect `output/backtest_report.json` to match the metrics above. Seeds are fixed and LightGBM runs single-threaded and deterministic, so results are bit-reproducible on the pinned dependency set.
