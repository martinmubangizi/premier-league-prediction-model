from collections import defaultdict, deque

import numpy as np
import pandas as pd

ELO_START_E0 = 1500.0
ELO_START_E1 = 1380.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0

FEATURES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_xg_diff",
    "h_pts3",
    "h_pts5",
    "h_pts10",
    "a_pts3",
    "a_pts5",
    "a_pts10",
    "h_venue_pts5",
    "a_venue_pts5",
    "h_gf5",
    "h_ga5",
    "a_gf5",
    "a_ga5",
    "h_gf10",
    "h_ga10",
    "a_gf10",
    "a_ga10",
    "h_sot_for5",
    "h_sot_ag5",
    "a_sot_for5",
    "a_sot_ag5",
    "h_rest",
    "a_rest",
    "rest_diff",
    "h_div",
    "a_div",
    "h_promoted",
    "a_promoted",
    "h2h_home_share",
    "h_matches_played",
    "a_matches_played",
]

XG_FEATURES = [
    "h_xg5",
    "h_xga5",
    "a_xg5",
    "a_xga5",
    "h_xg10",
    "h_xga10",
    "a_xg10",
    "a_xga10",
    "h_xgd5",
    "a_xgd5",
    "xgd_diff",
    "h_xg_overperf5",
    "a_xg_overperf5",
]


def _pts(res: str, side: str) -> int:
    if res == "D":
        return 1
    return 3 if res == side else 0


def _avg(dq, n):
    if not dq:
        return np.nan
    items = list(dq)[-n:]
    return float(np.mean(items)) if items else np.nan


def _avg_notna(dq, n):
    items = [x for x in list(dq)[-n:] if x == x]
    return float(np.mean(items)) if items else np.nan


class FeatureEngine:
    def __init__(self):
        self.elo = {}
        self.elo_xg = {}
        self.pts = defaultdict(lambda: deque(maxlen=10))
        self.venue_pts = defaultdict(lambda: deque(maxlen=5))
        self.gf = defaultdict(lambda: deque(maxlen=10))
        self.ga = defaultdict(lambda: deque(maxlen=10))
        self.sot_for = defaultdict(lambda: deque(maxlen=5))
        self.sot_ag = defaultdict(lambda: deque(maxlen=5))
        self.last_date = {}
        self.division = {}
        self.prev_season_div = {}
        self.cur_season_div = {}
        self.cur_season = None
        self.h2h = defaultdict(lambda: deque(maxlen=5))
        self.played = defaultdict(int)
        self.xg = defaultdict(lambda: deque(maxlen=10))
        self.xga = defaultdict(lambda: deque(maxlen=10))
        self.xg_overperf = defaultdict(lambda: deque(maxlen=5))

    def _get_elo(self, team, div):
        if team not in self.elo:
            self.elo[team] = ELO_START_E0 if div == "E0" else ELO_START_E1
        return self.elo[team]

    def _get_elo_xg(self, team, div):
        if team not in self.elo_xg:
            self.elo_xg[team] = ELO_START_E0 if div == "E0" else ELO_START_E1
        return self.elo_xg[team]

    def _season_roll(self, season):
        if self.cur_season is None:
            self.cur_season = season
        elif season != self.cur_season:
            self.prev_season_div = dict(self.cur_season_div)
            self.cur_season_div = {}
            self.cur_season = season

    def features_for(self, date, home, away, div_h, div_a, season):
        eh = self._get_elo(home, div_h)
        ea = self._get_elo(away, div_a)
        xh = self._get_elo_xg(home, div_h)
        xa = self._get_elo_xg(away, div_a)
        f = {
            "elo_home": eh,
            "elo_away": ea,
            "elo_diff": eh - ea,
            "elo_xg_diff": xh - xa,
            "h_pts3": _avg(self.pts[home], 3),
            "h_pts5": _avg(self.pts[home], 5),
            "h_pts10": _avg(self.pts[home], 10),
            "a_pts3": _avg(self.pts[away], 3),
            "a_pts5": _avg(self.pts[away], 5),
            "a_pts10": _avg(self.pts[away], 10),
            "h_venue_pts5": _avg(self.venue_pts[(home, "H")], 5),
            "a_venue_pts5": _avg(self.venue_pts[(away, "A")], 5),
            "h_gf5": _avg(self.gf[home], 5),
            "h_ga5": _avg(self.ga[home], 5),
            "a_gf5": _avg(self.gf[away], 5),
            "a_ga5": _avg(self.ga[away], 5),
            "h_gf10": _avg(self.gf[home], 10),
            "h_ga10": _avg(self.ga[home], 10),
            "a_gf10": _avg(self.gf[away], 10),
            "a_ga10": _avg(self.ga[away], 10),
            "h_sot_for5": _avg_notna(self.sot_for[home], 5),
            "h_sot_ag5": _avg_notna(self.sot_ag[home], 5),
            "a_sot_for5": _avg_notna(self.sot_for[away], 5),
            "a_sot_ag5": _avg_notna(self.sot_ag[away], 5),
            "h_div": 1.0 if div_h == "E0" else 0.0,
            "a_div": 1.0 if div_a == "E0" else 0.0,
            "h_promoted": 1.0
            if (div_h == "E0" and self.prev_season_div.get(home) == "E1")
            else 0.0,
            "a_promoted": 1.0
            if (div_a == "E0" and self.prev_season_div.get(away) == "E1")
            else 0.0,
            "h_matches_played": float(self.played[home]),
            "a_matches_played": float(self.played[away]),
        }
        for side, team in (("h", home), ("a", away)):
            ld = self.last_date.get(team)
            rest = (date - ld).days if ld is not None else np.nan
            f[f"{side}_rest"] = min(rest, 30.0) if rest == rest else np.nan
        f["rest_diff"] = (
            (f["h_rest"] - f["a_rest"])
            if (f["h_rest"] == f["h_rest"] and f["a_rest"] == f["a_rest"])
            else np.nan
        )
        key = tuple(sorted((home, away)))
        meetings = self.h2h[key]
        if meetings:
            pts_home = sum(
                p_a if a == home else p_b for a, p_a, p_b in [(m[0], m[1], m[2]) for m in meetings]
            )
            f["h2h_home_share"] = pts_home / (3.0 * len(meetings))
        else:
            f["h2h_home_share"] = np.nan

        for side, team in (("h", home), ("a", away)):
            f[f"{side}_xg5"] = _avg_notna(self.xg[team], 5)
            f[f"{side}_xga5"] = _avg_notna(self.xga[team], 5)
            f[f"{side}_xg10"] = _avg_notna(self.xg[team], 10)
            f[f"{side}_xga10"] = _avg_notna(self.xga[team], 10)
            xg5, xga5 = f[f"{side}_xg5"], f[f"{side}_xga5"]
            f[f"{side}_xgd5"] = (xg5 - xga5) if (xg5 == xg5 and xga5 == xga5) else np.nan
            f[f"{side}_xg_overperf5"] = _avg_notna(self.xg_overperf[team], 5)
        hd, ad = f["h_xgd5"], f["a_xgd5"]
        f["xgd_diff"] = (hd - ad) if (hd == hd and ad == ad) else np.nan
        return f

    def update(
        self,
        date,
        home,
        away,
        div_h,
        div_a,
        season,
        fthg,
        ftag,
        ftr,
        hst=np.nan,
        ast=np.nan,
        hxg=np.nan,
        axg=np.nan,
    ):
        self._season_roll(season)
        self.cur_season_div[home] = div_h
        self.cur_season_div[away] = div_a
        eh, ea = self._get_elo(home, div_h), self._get_elo(away, div_a)
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + ELO_HOME_ADV) - ea) / 400.0))
        score_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        margin = abs(fthg - ftag)
        mult = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8.0)
        delta = ELO_K * mult * (score_h - exp_h)
        self.elo[home] = eh + delta
        self.elo[away] = ea - delta
        xh, xa = self._get_elo_xg(home, div_h), self._get_elo_xg(away, div_a)
        exp_xh = 1.0 / (1.0 + 10 ** (-((xh + ELO_HOME_ADV) - xa) / 400.0))
        if hxg == hxg and axg == axg:
            s_xg = min(max(0.5 + (hxg - axg) / 4.0, 0.0), 1.0)
            s = 0.5 * s_xg + 0.5 * score_h
        else:
            s = score_h
        dxg = ELO_K * mult * (s - exp_xh)
        self.elo_xg[home] = xh + dxg
        self.elo_xg[away] = xa - dxg
        self.pts[home].append(_pts(ftr, "H"))
        self.pts[away].append(_pts(ftr, "A"))
        self.venue_pts[(home, "H")].append(_pts(ftr, "H"))
        self.venue_pts[(away, "A")].append(_pts(ftr, "A"))
        self.gf[home].append(fthg)
        self.ga[home].append(ftag)
        self.gf[away].append(ftag)
        self.ga[away].append(fthg)
        self.sot_for[home].append(hst)
        self.sot_ag[home].append(ast)
        self.sot_for[away].append(ast)
        self.sot_ag[away].append(hst)
        self.last_date[home] = date
        self.last_date[away] = date
        self.played[home] += 1
        self.played[away] += 1
        key = tuple(sorted((home, away)))
        self.h2h[key].append((home, _pts(ftr, "H"), _pts(ftr, "A")))
        self.xg[home].append(hxg)
        self.xga[home].append(axg)
        self.xg[away].append(axg)
        self.xga[away].append(hxg)
        if hxg == hxg:
            self.xg_overperf[home].append(fthg - hxg)
        if axg == axg:
            self.xg_overperf[away].append(ftag - axg)


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, FeatureEngine]:
    eng = FeatureEngine()
    rows = []
    has_xg = "HomeXG" in matches.columns
    for r in matches.itertuples():
        f = eng.features_for(r.Date, r.HomeTeam, r.AwayTeam, r.Division, r.Division, r.Season)
        rows.append(f)
        eng.update(
            r.Date,
            r.HomeTeam,
            r.AwayTeam,
            r.Division,
            r.Division,
            r.Season,
            r.FTHG,
            r.FTAG,
            r.FTR,
            r.HST,
            r.AST,
            getattr(r, "HomeXG", np.nan) if has_xg else np.nan,
            getattr(r, "AwayXG", np.nan) if has_xg else np.nan,
        )
    feats = pd.DataFrame(rows, index=matches.index)
    out = pd.concat([matches, feats], axis=1)
    return out, eng


def fixture_features(eng: FeatureEngine, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in fixtures.itertuples():
        season = r.Date.year - (1 if r.Date.month < 7 else 0)
        eng._season_roll(season)
        f = eng.features_for(r.Date, r.HomeTeam, r.AwayTeam, "E0", "E0", season)
        f["h_promoted"] = 1.0 if eng.prev_season_div.get(r.HomeTeam) == "E1" else 0.0
        f["a_promoted"] = 1.0 if eng.prev_season_div.get(r.AwayTeam) == "E1" else 0.0
        rows.append(f)
    return pd.concat([fixtures.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
