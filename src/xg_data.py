import io
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

UNDERSTAT_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
FPL_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

US_TEAMS = {
    "Arsenal": "Arsenal",
    "Aston_Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "Crystal_Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester_City": "Man City",
    "Manchester_United": "Man United",
    "Newcastle_United": "Newcastle",
    "Norwich": "Norwich",
    "Nottingham_Forest": "Nott'm Forest",
    "Sheffield_United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "Watford": "Watford",
    "West_Bromwich_Albion": "West Brom",
    "West_Ham": "West Ham",
    "Wolverhampton_Wanderers": "Wolves",
}

FPL_TEAMS = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}


def fetch_understat() -> pd.DataFrame:
    rows = []
    sess = requests.Session()
    for season in UNDERSTAT_SEASONS:
        for fname, short in US_TEAMS.items():
            url = f"{BASE}/{season}/understat/understat_{fname}.csv"
            try:
                r = sess.get(url, timeout=20)
                if r.status_code != 200 or not r.text.startswith("h_a"):
                    continue
                d = pd.read_csv(io.StringIO(r.text))
            except Exception:
                continue
            d["team"] = short
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[
        [
            "team",
            "date",
            "h_a",
            "xG",
            "xGA",
            "npxG",
            "npxGA",
            "deep",
            "deep_allowed",
            "scored",
            "missed",
        ]
    ]
    df = df.drop_duplicates(subset=["team", "date"], keep="first")
    df["source"] = "understat"
    return df


def fetch_fpl() -> pd.DataFrame:
    out = []
    for season in FPL_SEASONS:
        p = RAW / "fpl" / f"merged_{season}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        d["date"] = pd.to_datetime(d["kickoff_time"]).dt.tz_localize(None).dt.normalize()
        d["team_s"] = d["team"].map(FPL_TEAMS)
        g = d.groupby(["team_s", "date", "was_home"], as_index=False).agg(
            xG=("expected_goals", "sum"), scored=("goals_scored", "sum")
        )
        g = g.rename(columns={"team_s": "team"})
        g["h_a"] = g["was_home"].map({True: "h", False: "a"})
        out.append(g[["team", "date", "h_a", "xG", "scored"]])
    df = pd.concat(out, ignore_index=True)
    df = df.dropna(subset=["team"])
    df = df.drop_duplicates(subset=["team", "date"], keep="first")
    df["source"] = "fpl"
    return df


def attach_fpl_xga(fpl: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[["Date", "HomeTeam", "AwayTeam"]].copy()
    m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
    lookup = fpl.set_index(["team", "date"])["xG"].to_dict()
    recs = []
    for r in m.itertuples():
        hx = lookup.get((r.HomeTeam, r.Date))
        ax = lookup.get((r.AwayTeam, r.Date))
        if hx is not None:
            recs.append((r.HomeTeam, r.Date, "h", hx, ax))
        if ax is not None:
            recs.append((r.AwayTeam, r.Date, "a", ax, hx))
    d = pd.DataFrame(recs, columns=["team", "date", "h_a", "xG", "xGA"])
    d["source"] = "fpl"
    return d.dropna(subset=["xG", "xGA"])


def build(matches_path=None):
    matches = pd.read_parquet(matches_path or (RAW.parent / "matches.parquet"))
    us = fetch_understat()
    fpl_raw = fetch_fpl()
    fpl = attach_fpl_xga(fpl_raw, matches)

    ov = us.merge(fpl, on=["team", "date"], suffixes=("_us", "_fpl"))
    if len(ov) > 100:
        print(f"OVERLAP CHECK  n={len(ov)}")
        print(
            f"  xG   understat mean {ov.xG_us.mean():.3f} | fpl mean {ov.xG_fpl.mean():.3f} "
            f"| corr {ov.xG_us.corr(ov.xG_fpl):.3f}"
        )
        print(
            f"  xGA  understat mean {ov.xGA_us.mean():.3f} | fpl mean {ov.xGA_fpl.mean():.3f} "
            f"| corr {ov.xGA_us.corr(ov.xGA_fpl):.3f}"
        )

    fpl = fpl.copy()
    if len(ov) > 100:
        import numpy as np

        for c in ["xG", "xGA"]:
            b, a = np.polyfit(ov[f"{c}_fpl"], ov[f"{c}_us"], 1)
            fpl[c] = a + b * fpl[c]
            print(f"  calibrated fpl {c}: {a:.3f} + {b:.3f} * raw")

    keep = ["team", "date", "h_a", "xG", "xGA", "source"]
    combined = pd.concat([us[keep], fpl[keep]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["team", "date"], keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)
    out = RAW.parent / "team_xg.parquet"
    combined.to_parquet(out)
    print(
        f"\n{len(combined)} team-match xG rows "
        f"({combined.date.min().date()} -> {combined.date.max().date()}) -> {out}"
    )
    by_year = combined.date.dt.year.astype(str) + "-" + combined.source
    print(combined.groupby(by_year).size().tail(12))
    return combined


if __name__ == "__main__":
    build()
