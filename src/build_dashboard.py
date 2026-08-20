import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

SER = {"H": ("--home", "Home win"), "D": ("--draw", "Draw"), "A": ("--away", "Away win")}


def bar(probs, label, muted=False):
    segs = []
    for c in ["H", "D", "A"]:
        v = probs[c]
        txt = f"{v:.0f}" if v >= 9 else ""
        segs.append(
            f'<div class="sg" style="width:{v}%;background:var({SER[c][0]})" '
            f'title="{SER[c][1]} {v}%"><span>{txt}</span></div>'
        )
    cls = "bar muted" if muted else "bar"
    return (
        f'<div class="brow"><span class="blab">{label}</span>'
        f'<div class="{cls}">{"".join(segs)}</div></div>'
    )


def card(p):
    gaps = ""
    if p["value_bets"]:
        top = p["value_bets"][0]
        gaps = (
            f'<div class="gap"><span class="gtag">Model leans</span> '
            f"{top['outcome']}: model {top['model_prob']}% "
            f"vs market {top['market_prob']}%</div>"
        )
    return f"""
      <article class="fx">
        <div class="fxtop">
          <div class="tm">{p["home"]}</div>
          <div class="vs">v</div>
          <div class="tm ta">{p["away"]}</div>
        </div>
        <div class="fxmeta">{p["date"]} &middot; rating {p["elo_home"]:.0f} &ndash; {p["elo_away"]:.0f}</div>
        {bar(p["model"], "MODEL")}
        {bar(p["market"], "MARKET", muted=True)}
        <div class="fxfoot">Most likely: <b>{p["pick"]}</b> &middot; {p["pick_confidence"]}%</div>
        {gaps}
      </article>"""


def rows(preds):
    out = []
    for p in preds:
        out.append(
            f"<tr><td>{p['home']} v {p['away']}</td>"
            + "".join(f"<td>{p['model'][c]}%</td>" for c in "HDA")
            + "".join(f"<td class='dim'>{p['market'][c]}%</td>" for c in "HDA")
            + f"<td>{p['pick']}</td></tr>"
        )
    return "".join(out)


def tracker_block(tr):
    if not tr or not tr.get("gameweeks"):
        return """<div class="pend"><div class="pendk">Awaiting results</div>
        <p>Once matchweek 1 has been played, every prediction on this page is scored
        against what actually happened. Accuracy, probability quality and the running
        record then appear here and build up week on week, so the model's claims stay
        checkable rather than remembered selectively.</p></div>"""
    c = tr["cumulative"]
    rr = "".join(
        f"<tr><td>GW{g['gameweek']}</td><td>{g['matches']}</td>"
        f"<td>{g['accuracy_pct']}%</td><td>{g['log_loss']}</td></tr>"
        for g in tr["gameweeks"]
    )
    return f"""
      <div class="kpis">
        <div class="kpi"><div class="kn">{c["accuracy_pct"]}%</div><div class="kl">Accuracy to date</div></div>
        <div class="kpi"><div class="kn">{c["log_loss"]}</div><div class="kl">Log loss</div></div>
        <div class="kpi"><div class="kn">{c["matches"]}</div><div class="kl">Matches scored</div></div>
      </div>
      <table><thead><tr><th>Week</th><th>Matches</th><th>Accuracy</th><th>Log loss</th></tr></thead>
      <tbody>{rr}</tbody></table>"""


def tech_section():
    return """
  <h2>Technical appendix</h2>
  <div class="tech">

    <h3>Preventing target leakage</h3>
    <p>Features are produced by a streaming engine rather than by grouped
    <code>rolling()</code> calls over the full table. The engine walks matches in
    chronological order and, for each fixture, emits features from accumulated state
    <em>before</em> writing that fixture's result back into state. A row therefore
    cannot see its own outcome, and no season-level aggregate can bleed backwards
    into earlier matches. This is enforced by a regression test that rebuilds a
    mid-season fixture's features from a fresh engine fed only strictly-earlier
    matches and asserts an exact match on all 34 features.</p>

    <h3>Ratings</h3>
    <p>Elo with a margin-of-victory multiplier, updated as</p>
    <p class="eq">E<sub>h</sub> = 1 / (1 + 10<sup>-((R<sub>h</sub> + 60) - R<sub>a</sub>)/400</sup>)
    &nbsp;&nbsp; R'<sub>h</sub> = R<sub>h</sub> + K&middot;&mu;(&delta;)&middot;(S<sub>h</sub> - E<sub>h</sub>)</p>
    <p>with K = 20, a 60-point home-field constant, and &mu;(&delta;) scaling the
    update by goal margin (1.0 for one goal, 1.5 for two, (11+&delta;)/8 beyond).
    Premier League and Championship share a single rating pool, so promoted and
    relegated clubs transport their ratings across divisions and the two tiers stay
    on a common scale. Ratings are never reset between seasons.</p>
    <p>A second rating stream, <b>xG-Elo</b>, runs the identical update but replaces
    the match score with a blend of the scoreline and the expected-goals margin,
    S = 0.5&middot;clip(0.5 + (xG<sub>h</sub> - xG<sub>a</sub>)/4, 0, 1) + 0.5&middot;S<sub>actual</sub>.
    Where xG is unavailable it degrades gracefully to the plain update.</p>

    <h3>Validation protocol</h3>
    <p>Walk-forward by season: for each held-out season <i>s</i>, the model is refit
    from scratch on every match with season &lt; <i>s</i> and evaluated on <i>s</i>.
    No shuffling, no k-fold, no early stopping against the test fold. Seven seasons
    are held out in total. Rows are excluded where either side has fewer than eight
    prior matches, so the rolling windows are populated rather than imputed.</p>
    <p>Log loss is the primary metric. Accuracy is reported but deliberately
    de-emphasised: it is degenerate for this problem, since a model that never
    predicts a draw can score well on it while being badly calibrated. The
    bookmakers' de-margined implied probabilities are evaluated on the identical
    rows as a benchmark.</p>

    <h3>Ablations</h3>
    <p>Each row refits the whole pipeline. The training-window control is the row
    that matters: expected goals and player data only exist from 2020, so any
    variant using them trains on a shorter history, and that cost must be separated
    from the feature's own contribution. Test window 2022-23 to 2025-26,
    1,520 matches.</p>
    <table class="abl"><thead><tr><th>Variant</th><th>Log loss</th><th>vs control</th></tr></thead><tbody>
      <tr><td>Bookmaker benchmark</td><td>0.9485</td><td class="dim">reference</td></tr>
      <tr class="hl"><td>Shipped model, full history</td><td>0.9786</td><td class="dim">baseline</td></tr>
      <tr><td>Control: same features, 2020+ training only</td><td>0.9890</td><td class="dim">control</td></tr>
      <tr><td>+ 13 rolling xG / xGA features</td><td>1.0068</td><td class="neg">-0.0178</td></tr>
      <tr><td>+ xG, LightGBM, full history, NaN-aware</td><td>1.0126</td><td class="neg">-0.0236</td></tr>
      <tr><td>+ 10 squad-form features</td><td>1.0186</td><td class="neg">-0.0296</td></tr>
      <tr><td>+ squad availability only</td><td>1.0075</td><td class="neg">-0.0185</td></tr>
      <tr><td>+ contribution-weighted availability</td><td>1.0076</td><td class="neg">-0.0186</td></tr>
      <tr class="hl"><td>xG folded into the Elo update instead</td><td>0.9780</td><td class="pos">+0.0006</td></tr>
    </tbody></table>

    <h3>Why the rich features failed</h3>
    <p>Both expected goals and player form carry genuine signal in isolation.
    xG-difference predicts the home result better than goal-difference does
    (r = 0.335 against 0.286). But Elo predicts it better than either (r = 0.394),
    and xG-difference is 69% collinear with Elo. Residualising xG-difference on Elo
    leaves r = 0.085 against the outcome. A margin-scaled rating fitted over 25
    seasons has already absorbed most of what xG contributes, so the features add
    parameters without adding information, and the shortened training window makes
    the trade strictly negative.</p>
    <p>Squad availability behaves differently and is the more interesting failure. It
    is close to <em>orthogonal</em> to Elo (r = 0.027), so it is genuinely new
    information, but its correlation with the result is only 0.053, and a
    contribution-weighted version scored no better (0.042). The likely cause is that
    a minutes-based proxy cannot separate a rested player from an injured one, and
    is backward-looking by construction: it reports who played last week, not who is
    fit this week. The design conclusion is that team news belongs in the model as a
    human-supplied prior, which is why the fixture file exposes a per-team Elo
    adjustment rather than a learned availability feature.</p>

    <h3>Calibration</h3>
    <p>Reliability across seven held-out seasons, predicted against observed:</p>
    <table class="abl"><thead><tr><th>Bucket</th><th>Predicted</th><th>Observed</th><th>n</th></tr></thead><tbody>
      <tr><td>Home win 10-20%</td><td>0.154</td><td>0.149</td><td>215</td></tr>
      <tr><td>Home win 30-40%</td><td>0.352</td><td>0.334</td><td>467</td></tr>
      <tr><td>Home win 50-65%</td><td>0.571</td><td>0.544</td><td>618</td></tr>
      <tr><td>Home win 65%+</td><td>0.735</td><td>0.708</td><td>456</td></tr>
      <tr><td>Draw 20-30%</td><td>0.248</td><td>0.250</td><td>2115</td></tr>
      <tr><td>Away win 20-30%</td><td>0.248</td><td>0.264</td><td>571</td></tr>
    </tbody></table>
    <p>Aggregate draw rate is predicted at 23.5% against 23.6% observed. The model is
    mildly overconfident in the top bucket and mildly underconfident at the bottom,
    the standard shrinkage signature, which is why the published figure blends
    toward the market rather than shipping the raw output.</p>

    <h3>Models and configuration</h3>
    <p>The shipped estimator is multinomial logistic regression: median imputation,
    then standardisation, then an L2 penalty at C = 0.1. on this feature count and sample size
    it beats gradient boosting on log loss, which is the expected ordering for a
    low-signal tabular problem. LightGBM (500 trees, learning rate 0.03, 15 leaves,
    minimum 60 samples per leaf, 0.8 subsample and column sample, L2 = 5) is
    retained for comparison and for a variant that consumes de-margined market
    probabilities as features, which reaches the best accuracy at 53.9% while
    tracking the market by construction.</p>

    <h3>What would actually move the number</h3>
    <p>A Dixon-Coles bivariate Poisson goal model, fitted with time decay and a
    low-score correlation term, would supply a genuinely different functional form
    rather than another correlated feature, and would open the goals markets where
    prices are softer. Historical squad market values would address the failure mode
    visible in the fixtures above, where the model rates a heavily-invested squad on
    results alone. Real injury and lineup feeds, as opposed to a minutes proxy, would
    test the availability hypothesis properly. Absent those, the honest read is that
    this model is near the ceiling of what public box-score data supports.</p>

  </div>"""


def build():
    preds = json.loads((OUT / "predictions_gw1.json").read_text())
    bt = json.loads((OUT / "backtest_report.json").read_text())
    tr = json.loads((OUT / "tracker.json").read_text()) if (OUT / "tracker.json").exists() else None
    lg = bt["logreg"]
    mk = lg.get("market_log_loss_same_rows")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Centre &middot; Prediction Model &middot; {preds["season"]} Week {preds["gameweek"]}</title>
<style>
  :root {{
    --ink:#37003c; --ink-2:#26002a; --ink-3:#4a0f50;
    --line:rgba(255,255,255,.13); --line-2:rgba(255,255,255,.07);
    --fg:#ffffff; --fg-2:#cbb6cf; --fg-3:#9b83a1;
    --home:#0e93b0; --draw:#b8801c; --away:#cf4478;
    --acc:#00e0b8;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    background:var(--ink-2); color:var(--fg);
    font:15px/1.55 "Inter","Segoe UI",system-ui,-apple-system,Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }}
  .shell{{max-width:1120px;margin:0 auto;padding:0 20px 72px}}

  /* ---- masthead ---- */
  .mast{{
    background:linear-gradient(103deg,var(--ink) 0%,var(--ink-3) 62%,#5c1a63 100%);
    margin:0 -20px; padding:34px 20px 30px; position:relative; overflow:hidden;
    border-bottom:3px solid var(--acc);
  }}
  .mast::after{{content:"";position:absolute;top:-40%;right:-6%;width:320px;height:200%;
    background:rgba(255,255,255,.045);transform:rotate(14deg)}}
  .mast::before{{content:"";position:absolute;top:-40%;right:14%;width:90px;height:200%;
    background:rgba(0,224,184,.10);transform:rotate(14deg)}}
  .mwrap{{max-width:1080px;margin:0 auto;position:relative;z-index:1}}
  .eyebrow{{font-size:11px;letter-spacing:.22em;text-transform:uppercase;
    color:var(--acc);font-weight:700;margin-bottom:9px}}
  h1{{font-size:clamp(27px,4.6vw,42px);line-height:1.03;letter-spacing:-.028em;font-weight:800}}
  .mast p{{color:var(--fg-2);font-size:13.5px;margin-top:10px;max-width:64ch}}

  h2{{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--acc);
    font-weight:700;margin:44px 0 16px;padding-bottom:9px;border-bottom:1px solid var(--line)}}

  /* ---- notice ---- */
  .notice{{background:rgba(255,255,255,.045);border:1px solid var(--line);
    border-left:3px solid var(--acc);border-radius:5px;padding:17px 19px;margin-top:26px}}
  .notice h3{{font-size:13px;letter-spacing:.05em;text-transform:uppercase;margin-bottom:7px}}
  .notice p{{font-size:13.5px;color:var(--fg-2)}}
  .notice p + p{{margin-top:9px}}
  .notice b{{color:var(--fg)}}

  /* ---- glossary ---- */
  .gloss{{display:grid;grid-template-columns:repeat(3,1fr);gap:11px}}
  @media (max-width:820px){{ .gloss{{grid-template-columns:repeat(2,1fr)}} }}
  @media (max-width:520px){{ .gloss{{grid-template-columns:1fr}} }}
  .gi{{background:rgba(255,255,255,.04);border:1px solid var(--line-2);border-radius:5px;padding:14px 16px}}
  .gk{{font-size:12.5px;font-weight:700;letter-spacing:.03em;margin-bottom:5px;
    display:flex;align-items:center;gap:8px}}
  .gk i{{width:9px;height:9px;border-radius:2px;display:inline-block;flex:none}}
  .gi p{{font-size:12.5px;color:var(--fg-2);line-height:1.5}}

  /* ---- kpis ---- */
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:11px}}
  .kpi{{background:rgba(255,255,255,.04);border:1px solid var(--line-2);border-radius:5px;padding:15px 17px}}
  .kn{{font-size:27px;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums}}
  .kl{{font-size:11.5px;color:var(--fg-3);margin-top:2px;letter-spacing:.02em}}

  /* ---- fixtures ---- */
  .fxs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:11px}}
  .fx{{background:linear-gradient(160deg,rgba(255,255,255,.062),rgba(255,255,255,.028));
    border:1px solid var(--line);border-radius:6px;padding:17px 18px 15px;
    transition:border-color .16s,transform .16s}}
  .fx:hover{{border-color:rgba(0,224,184,.45);transform:translateY(-1px)}}
  .fxtop{{display:flex;align-items:baseline;gap:10px}}
  .tm{{font-size:15px;font-weight:750;letter-spacing:-.012em;flex:1;text-transform:uppercase}}
  .tm.ta{{text-align:right}}
  .vs{{font-size:10.5px;color:var(--fg-3);flex:none;letter-spacing:.1em}}
  .fxmeta{{font-size:11px;color:var(--fg-3);margin:4px 0 13px;
    font-variant-numeric:tabular-nums;letter-spacing:.02em}}
  .brow{{display:flex;align-items:center;gap:9px;margin-bottom:5px}}
  .blab{{font-size:9.5px;letter-spacing:.13em;color:var(--fg-3);width:50px;flex:none;
    text-align:right;font-weight:650}}
  .bar{{display:flex;height:23px;flex:1;gap:2px;border-radius:3px;overflow:hidden}}
  .bar.muted{{height:15px;opacity:.5}}
  .sg{{display:flex;align-items:center;justify-content:center;min-width:0;transition:opacity .14s}}
  .sg span{{font-size:10.5px;font-weight:750;color:#fff;font-variant-numeric:tabular-nums;
    text-shadow:0 1px 2px rgba(0,0,0,.4)}}
  .bar:hover .sg{{opacity:.45}} .bar .sg:hover{{opacity:1}}
  .fxfoot{{font-size:12px;color:var(--fg-2);margin-top:11px;padding-top:10px;
    border-top:1px solid var(--line-2)}}
  .fxfoot b{{color:var(--fg);font-weight:700}}
  .gap{{font-size:11.5px;color:var(--fg-3);margin-top:8px;line-height:1.45}}
  .gtag{{color:var(--acc);font-weight:700;letter-spacing:.05em;font-size:10px;
    text-transform:uppercase;margin-right:4px}}

  /* ---- pending ---- */
  .pend{{background:rgba(255,255,255,.04);border:1px solid var(--line-2);
    border-radius:5px;padding:19px 21px}}
  .pendk{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--acc);font-weight:700;margin-bottom:8px}}
  .pend p{{font-size:13.5px;color:var(--fg-2);max-width:74ch}}

  /* ---- table ---- */
  details{{margin-top:15px}}
  summary{{cursor:pointer;font-size:12.5px;color:var(--fg-3);padding:7px 0;
    letter-spacing:.03em;list-style:none}}
  summary::before{{content:"+ ";color:var(--acc);font-weight:700}}
  details[open] summary::before{{content:"\\2212 "}}
  table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:9px;
    font-variant-numeric:tabular-nums}}
  th{{text-align:left;padding:8px 10px;font-size:10px;letter-spacing:.11em;
    text-transform:uppercase;color:var(--fg-3);border-bottom:1px solid var(--line);font-weight:650}}
  td{{padding:8px 10px;border-bottom:1px solid var(--line-2);color:var(--fg-2)}}
  td:first-child{{color:var(--fg);font-weight:550}}
  td.dim{{color:var(--fg-3)}}


  .tech{{max-width:none}}
  .tech h3{{font-size:13.5px;font-weight:750;margin:22px 0 7px;letter-spacing:-.005em}}
  .tech h3:first-child{{margin-top:0}}
  .tech p{{font-size:13px;color:var(--fg-2);max-width:88ch;margin-bottom:8px;line-height:1.62}}
  .tech b{{color:var(--fg)}}
  .tech em{{color:var(--fg);font-style:italic}}
  .tech code{{background:rgba(255,255,255,.07);padding:1px 5px;border-radius:3px;
    font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
  .eq{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
    background:rgba(255,255,255,.05);border:1px solid var(--line-2);border-radius:4px;
    padding:11px 14px;color:var(--fg)!important;overflow-x:auto;white-space:nowrap}}
  table.abl{{max-width:660px;margin:11px 0 4px}}
  table.abl td:first-child{{font-weight:450;color:var(--fg-2)}}
  tr.hl td{{color:var(--fg)!important;font-weight:650}}
  td.pos{{color:var(--acc);font-weight:650}}
  td.neg{{color:#e0708f;font-weight:650}}
  footer{{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);
    font-size:11.5px;color:var(--fg-3);line-height:1.65}}
  footer b{{color:var(--fg-2)}}
  @media (max-width:560px){{ .blab{{width:44px}} .tm{{font-size:13.5px}} }}
</style></head>
<body>

<div class="mast"><div class="mwrap">
  <div class="eyebrow">Match Centre &middot; {preds["season"]} &middot; Week {preds["gameweek"]}</div>
  <h1>Premier League<br>Prediction Model</h1>
  <p>A statistical model trained on 23,948 English league matches since 2000. It publishes a
  probability for every result before kick-off, then scores itself against what actually happens.</p>
</div></div>

<div class="shell">

  <div class="notice">
    <h3>This is not a betting product</h3>
    <p>Nothing here is a tip, a recommendation, or advice to stake money. This is a
    <b>forecasting experiment published in the open</b>. The point is to see whether a model
    built from public data can produce honest probabilities, and to be held to them publicly.</p>
    <p>The measured answer so far: <b>it does not beat the betting market.</b> Over seven
    held-out seasons the model's probabilities scored slightly worse than the bookmakers' own
    ({lg["log_loss"]} against {mk}, where lower is better), and a simulated staking test came out
    <b>losing money</b>. Anyone who tells you their model reliably beats closing odds is usually
    fooling themselves with leaked data. Please read the numbers below as a case study in
    forecasting, not as a way to make money.</p>
  </div>

  <h2>How to read a match card</h2>
  <div class="gloss">
    <div class="gi"><div class="gk"><i style="background:var(--home)"></i>Home win</div>
      <p>Chance the home side wins. In the table this is the <b>H</b> column.</p></div>
    <div class="gi"><div class="gk"><i style="background:var(--draw)"></i>Draw</div>
      <p>Chance the match ends level. The <b>D</b> column.</p></div>
    <div class="gi"><div class="gk"><i style="background:var(--away)"></i>Away win</div>
      <p>Chance the away side wins. The <b>A</b> column.</p></div>
    <div class="gi"><div class="gk">MODEL, the top bar</div>
      <p>What this model thinks, from form, ratings and match history alone. It has never
      seen the odds. Written <b>Mdl H / Mdl D / Mdl A</b> in the table.</p></div>
    <div class="gi"><div class="gk">MARKET, the faded bar</div>
      <p>What bookmakers' odds imply, with their profit margin stripped out, so the three
      add to 100%. Written <b>Mkt H / Mkt D / Mkt A</b>. Treat this as the score to beat.</p></div>
    <div class="gi"><div class="gk">Where the bars differ</div>
      <p>A gap means the model and the market disagree. Usually the market is right and the
      gap is showing you a blind spot in the model, which is the interesting part.</p></div>
  </div>

  <h2>How the model was tested</h2>
  <div class="kpis">
    <div class="kpi"><div class="kn">{lg["n"]}</div><div class="kl">Matches held out</div></div>
    <div class="kpi"><div class="kn">{lg["accuracy_pct"]}%</div><div class="kl">Correct result</div></div>
    <div class="kpi"><div class="kn">{lg["log_loss"]}</div><div class="kl">Model log loss</div></div>
    <div class="kpi"><div class="kn">{mk}</div><div class="kl">Market log loss</div></div>
  </div>
  <div class="notice" style="border-left-color:var(--draw)">
    <h3>Why week 1 is the model's hardest week</h3>
    <p>Form is carried over from last May, so the model is reading a season that no longer
    exists, with a summer of transfers in between. Two of the three promoted
    clubs have no top-flight record at all. Expect these particular forecasts to be the least
    reliable of the entire season, and judge the model in November, not this weekend.</p>
  </div>

  <h2>Week {preds["gameweek"]} forecasts</h2>
  <div class="fxs">{"".join(card(p) for p in preds["predictions"])}</div>

  <details><summary>Show all numbers as a table</summary>
    <table><thead><tr><th>Match</th>
      <th>Mdl H</th><th>Mdl D</th><th>Mdl A</th>
      <th>Mkt H</th><th>Mkt D</th><th>Mkt A</th><th>Most likely</th></tr></thead>
      <tbody>{rows(preds["predictions"])}</tbody></table>
  </details>

  {tech_section()}

  <h2>Scorecard</h2>
  {tracker_block(tr)}

  <footer>
    <b>Method.</b> Elo ratings (including a variant driven by expected goals), rolling form over
    3/5/10 matches, separate home and away form, goals and shots on target, days of rest,
    head-to-head record and promotion status. Logistic regression and gradient boosting,
    validated by training only on seasons before each season tested, so the model never sees
    its own future.
    Expected-goals data covers 2020 onward.<br><br>
    <b>Independent project.</b> Not affiliated with, endorsed by, or connected to the Premier League
    or any club. Built from publicly available data for research and demonstration. Odds shown are
    published market prices used as a benchmark, not an invitation to bet. Generated
    {preds["generated"]}.
  </footer>

</div></body></html>"""
    path = OUT / "dashboard.html"
    path.write_text(html)
    print(f"wrote {path} ({len(html)} bytes)")
    return path


if __name__ == "__main__":
    build()
