import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "output" / "tracker.json"
CLASSES = ["A", "D", "H"]
CODE = {"Home win": "H", "Draw": "D", "Away win": "A"}
STAKE = 10.0


def _load():
    if TRACKER.exists():
        return json.loads(TRACKER.read_text())
    return {"gameweeks": [], "cumulative": {}}


def settle(gameweek: int, results: dict, season="2026-27"):
    preds = json.loads((ROOT / "output" / f"predictions_gw{gameweek}.json").read_text())
    tr = _load()
    tr["gameweeks"] = [
        g for g in tr["gameweeks"] if not (g["gameweek"] == gameweek and g["season"] == season)
    ]

    rows, lls, correct, n = [], [], 0, 0
    staked = returned = nbets = 0.0
    for p in preds["predictions"]:
        key = f"{p['home']} v {p['away']}"
        if key not in results:
            continue
        hg, ag = (int(x) for x in results[key].split("-"))
        actual = "H" if hg > ag else ("A" if ag > hg else "D")
        prob = p["blended"][actual] / 100.0
        lls.append(-np.log(max(prob, 1e-9)))
        hit = p["pick_code"] == actual
        correct += hit
        n += 1
        bets = []
        for vb in p["value_bets"]:
            staked += STAKE
            nbets += 1
            win = vb["code"] == actual
            ret = STAKE * vb["odds"] if win else 0.0
            returned += ret
            bets.append(
                {
                    "outcome": vb["outcome"],
                    "odds": vb["odds"],
                    "won": win,
                    "profit": round(ret - STAKE, 2),
                }
            )
        rows.append(
            {
                "match": key,
                "score": results[key],
                "actual": actual,
                "pick": p["pick"],
                "correct": bool(hit),
                "prob_assigned": round(100 * prob, 1),
                "bets": bets,
            }
        )

    gw = {
        "season": season,
        "gameweek": gameweek,
        "matches": n,
        "accuracy_pct": round(100 * correct / n, 1) if n else None,
        "log_loss": round(float(np.mean(lls)), 4) if lls else None,
        "bets": int(nbets),
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "roi_pct": round(100 * (returned - staked) / staked, 2) if staked else None,
        "results": rows,
    }
    tr["gameweeks"].append(gw)
    tr["gameweeks"].sort(key=lambda g: (g["season"], g["gameweek"]))

    tot_n = sum(g["matches"] for g in tr["gameweeks"])
    tot_c = sum(g["matches"] * g["accuracy_pct"] / 100 for g in tr["gameweeks"])
    tot_s = sum(g["staked"] for g in tr["gameweeks"])
    tot_r = sum(g["returned"] for g in tr["gameweeks"])
    ll_all = [g["log_loss"] * g["matches"] for g in tr["gameweeks"] if g["log_loss"]]
    tr["cumulative"] = {
        "matches": tot_n,
        "accuracy_pct": round(100 * tot_c / tot_n, 1) if tot_n else None,
        "log_loss": round(sum(ll_all) / tot_n, 4) if tot_n else None,
        "bets": sum(g["bets"] for g in tr["gameweeks"]),
        "staked": round(tot_s, 2),
        "returned": round(tot_r, 2),
        "roi_pct": round(100 * (tot_r - tot_s) / tot_s, 2) if tot_s else None,
        "profit": round(tot_r - tot_s, 2),
    }
    TRACKER.write_text(json.dumps(tr, indent=2))
    print(json.dumps(gw, indent=2)[:800])
    print("CUMULATIVE:", json.dumps(tr["cumulative"]))
    return tr


if __name__ == "__main__":
    import sys

    gw = int(sys.argv[1])
    res = json.loads(Path(sys.argv[2]).read_text())
    settle(gw, res)
