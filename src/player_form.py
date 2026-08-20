from pathlib import Path

import numpy as np
import pandas as pd

from xg_data import FPL_TEAMS

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
WINDOW = 5
CORE = 11

PLAYER_FEATURES = [
    "h_sq_avail",
    "a_sq_avail",
    "sq_avail_diff",
    "h_sq_ict",
    "a_sq_ict",
    "sq_ict_diff",
    "h_sq_threat",
    "a_sq_threat",
    "h_sq_conc",
    "a_sq_conc",
]


def load_players() -> pd.DataFrame:
    out = []
    for s in SEASONS:
        p = RAW / "fpl" / f"merged_{s}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        if "team" not in d.columns:
            continue
        d["date"] = pd.to_datetime(d["kickoff_time"]).dt.tz_localize(None).dt.normalize()
        d["team_s"] = d["team"].map(FPL_TEAMS)
        d["pid"] = s + "_" + d["element"].astype(str)
        for c in ["minutes", "ict_index", "threat"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        out.append(d[["pid", "team_s", "date", "minutes", "ict_index", "threat"]])
    df = pd.concat(out, ignore_index=True).dropna(subset=["team_s"])
    df = df.groupby(["team_s", "date", "pid"], as_index=False).agg(
        minutes=("minutes", "sum"), ict_index=("ict_index", "sum"), threat=("threat", "sum")
    )
    return df.rename(columns={"team_s": "team"})


def build() -> pd.DataFrame:
    pl = load_players()
    rows = []
    for team, g in pl.groupby("team", sort=False):
        dates = sorted(g["date"].unique())
        by_date = {d: sub for d, sub in g.groupby("date")}
        for i, d in enumerate(dates):
            prev = dates[max(0, i - WINDOW) : i]
            if len(prev) < 3:
                rows.append((team, d, np.nan, np.nan, np.nan, np.nan))
                continue
            hist = pd.concat([by_date[p] for p in prev])
            tot = hist.groupby("pid").agg(
                mins=("minutes", "sum"), ict=("ict_index", "sum"), thr=("threat", "sum")
            )
            core = tot.nlargest(CORE, "mins")
            last = by_date[prev[-1]].set_index("pid")["minutes"]
            avail = last.reindex(core.index).fillna(0.0).sum() / (CORE * 90.0)
            ict = core["ict"].sum() / len(prev)
            thr = core["thr"].sum() / len(prev)
            conc = (core["thr"].max() / core["thr"].sum()) if core["thr"].sum() > 0 else np.nan
            rows.append((team, d, float(avail), float(ict), float(thr), conc))
    out = pd.DataFrame(rows, columns=["team", "date", "sq_avail", "sq_ict", "sq_threat", "sq_conc"])
    path = RAW.parent / "squad_form.parquet"
    out.to_parquet(path)
    print(
        f"{len(out)} team-match squad rows "
        f"({out.date.min().date()} -> {out.date.max().date()}) -> {path}"
    )
    print(out[["sq_avail", "sq_ict", "sq_threat", "sq_conc"]].describe().round(3).to_string())
    return out


def attach(df: pd.DataFrame) -> pd.DataFrame:
    p = RAW.parent / "squad_form.parquet"
    if not p.exists():
        for c in PLAYER_FEATURES:
            df[c] = np.nan
        return df
    sq = pd.read_parquet(p)
    lut = {(t, d): (a, i, th, c) for t, d, a, i, th, c in sq.itertuples(index=False)}
    dts = pd.to_datetime(df["Date"]).dt.normalize()
    blank = (np.nan,) * 4
    H = [lut.get((t, d), blank) for t, d in zip(df["HomeTeam"], dts, strict=False)]
    A = [lut.get((t, d), blank) for t, d in zip(df["AwayTeam"], dts, strict=False)]
    for idx, name in enumerate(["avail", "ict", "threat", "conc"]):
        df[f"h_sq_{name}"] = [x[idx] for x in H]
        df[f"a_sq_{name}"] = [x[idx] for x in A]
    df["sq_avail_diff"] = df["h_sq_avail"] - df["a_sq_avail"]
    df["sq_ict_diff"] = df["h_sq_ict"] - df["a_sq_ict"]
    return df


if __name__ == "__main__":
    build()
