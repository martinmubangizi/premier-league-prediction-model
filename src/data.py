import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

NAME_FIX = {
    "Nottm Forest": "Nott'm Forest",
    "Middlesboro": "Middlesbrough",
}

OPENFOOTBALL_NAMES = {
    "Birmingham City FC": "Birmingham",
    "Ipswich Town FC": "Ipswich",
    "Charlton Athletic FC": "Charlton",
    "Watford FC": "Watford",
    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull",
    "Southampton FC": "Southampton",
    "Wrexham AFC": "Wrexham",
    "Middlesbrough FC": "Middlesbrough",
    "Swansea City AFC": "Swansea",
    "Swansea City FC": "Swansea",
    "Norwich City FC": "Norwich",
    "Millwall FC": "Millwall",
    "Oxford United FC": "Oxford",
    "Portsmouth FC": "Portsmouth",
    "Queens Park Rangers FC": "QPR",
    "Preston North End FC": "Preston",
    "Stoke City FC": "Stoke",
    "Derby County FC": "Derby",
    "West Bromwich Albion FC": "West Brom",
    "Blackburn Rovers FC": "Blackburn",
    "Sheffield United FC": "Sheffield United",
    "Bristol City FC": "Bristol City",
    "Leicester City FC": "Leicester",
    "Sheffield Wednesday FC": "Sheffield Weds",
}

MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


def _fix(name: str) -> str:
    return NAME_FIX.get(name, name)


def load_xgabora() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Matches_all.csv", low_memory=False)
    df = df[df["Division"].isin(["E0", "E1"])].copy()
    df = df.rename(
        columns={
            "MatchDate": "Date",
            "FTHome": "FTHG",
            "FTAway": "FTAG",
            "FTResult": "FTR",
            "HomeShots": "HS",
            "AwayShots": "AS",
            "HomeTarget": "HST",
            "AwayTarget": "AST",
        }
    )
    keep = [
        "Date",
        "Division",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "OddHome",
        "OddDraw",
        "OddAway",
    ]
    df = df[keep]
    df["HomeTeam"] = df["HomeTeam"].map(_fix)
    df["AwayTeam"] = df["AwayTeam"].map(_fix)
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df


def load_pl_mirror(season_file: str = "premier-league_2526.csv") -> pd.DataFrame:
    df = pd.read_csv(RAW / season_file)
    df["Division"] = "E0"
    for c in ["OddHome", "OddDraw", "OddAway"]:
        df[c] = float("nan")
    keep = [
        "Date",
        "Division",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "OddHome",
        "OddDraw",
        "OddAway",
    ]
    df["HomeTeam"] = df["HomeTeam"].map(_fix)
    df["AwayTeam"] = df["AwayTeam"].map(_fix)
    return df[keep]


def load_openfootball_championship(fname: str = "champ_2526.txt") -> pd.DataFrame:
    text = (RAW / fname).read_text()
    lines = text.splitlines()
    hd = re.search(r"(\d{4})/\d{2}", text)
    start_year = int(hd.group(1))
    rows, month, day = [], None, None
    in_playoffs = False
    date_re = re.compile(
        r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
    )
    match_re = re.compile(r"^\s*(?:\d{1,2}:\d{2}\s+)?(.{2,40}?)\s+v\s+(.{2,40}?)\s+(\d+)-(\d+)")
    for ln in lines:
        if ln.strip().startswith("#"):
            continue
        if re.search(r"Playoff", ln, re.I):
            in_playoffs = True
        dm = date_re.match(ln)
        if dm:
            month = MONTHS[dm.group(1)]
            day = int(dm.group(2))
            continue
        mm = match_re.match(ln)
        if mm and not in_playoffs and month:
            year = start_year if month >= 7 else start_year + 1
            home = OPENFOOTBALL_NAMES.get(mm.group(1).strip(), mm.group(1).strip())
            away = OPENFOOTBALL_NAMES.get(mm.group(2).strip(), mm.group(2).strip())
            hg, ag = int(mm.group(3)), int(mm.group(4))
            date = f"{year:04d}-{month:02d}-{day:02d}"
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append((date, "E1", home, away, hg, ag, ftr))
    df = pd.DataFrame(
        rows, columns=["Date", "Division", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    )
    for c in ["HS", "AS", "HST", "AST", "OddHome", "OddDraw", "OddAway"]:
        df[c] = float("nan")
    return df


def load_all() -> pd.DataFrame:
    base = load_xgabora()
    pl26 = load_pl_mirror()
    ch26 = load_openfootball_championship()
    cutoff = base["Date"].max()
    pl26 = pl26[pl26["Date"] > cutoff]
    ch26 = ch26[ch26["Date"] > cutoff]
    df = pd.concat([base, pl26, ch26], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Season"] = df["Date"].dt.year - (df["Date"].dt.month < 8).astype(int)
    return df


def attach_xg(df: pd.DataFrame) -> pd.DataFrame:
    p = RAW.parent / "team_xg.parquet"
    if not p.exists():
        df["HomeXG"] = float("nan")
        df["AwayXG"] = float("nan")
        return df
    xg = pd.read_parquet(p)
    lut = xg.set_index(["team", "date"])["xG"].to_dict()
    d = pd.to_datetime(df["Date"]).dt.normalize()
    df["HomeXG"] = [
        lut.get((t, dt), float("nan")) for t, dt in zip(df["HomeTeam"], d, strict=False)
    ]
    df["AwayXG"] = [
        lut.get((t, dt), float("nan")) for t, dt in zip(df["AwayTeam"], d, strict=False)
    ]
    return df


if __name__ == "__main__":
    df = load_all()
    df = attach_xg(df)
    print(
        f"xG coverage: {df['HomeXG'].notna().mean():.1%} of all matches; "
        f"{df[df.Division == 'E0']['HomeXG'].notna().mean():.1%} of PL matches"
    )
    out = RAW.parent / "matches.parquet"
    df.to_parquet(out)
    print(df.groupby(["Season", "Division"]).size().unstack(fill_value=0).tail(6))
    print(f"{len(df)} matches -> {out}")
