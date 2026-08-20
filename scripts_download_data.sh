#!/usr/bin/env bash
# Refresh raw source data. Run from the repo root.
set -e
mkdir -p data/raw
curl -sL -o data/raw/Matches_all.csv \
  "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
for s in 2425 2526 2627; do
  curl -sfL -o "data/raw/premier-league_${s}.csv" \
    "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league/season-${s}.csv" || true
done
for y in 2025-26 2026-27; do
  curl -sfL -o "data/raw/champ_${y}.txt" \
    "https://raw.githubusercontent.com/openfootball/england/master/${y}/2-championship.txt" || true
done
echo "done"
