from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
OUTPUT_DIR: Path = ROOT / "output"
DOCS_DIR: Path = ROOT / "docs"

MATCHES_PATH: Path = DATA_DIR / "matches.parquet"
FEATURES_PATH: Path = DATA_DIR / "features.parquet"
TEAM_XG_PATH: Path = DATA_DIR / "team_xg.parquet"
SQUAD_FORM_PATH: Path = DATA_DIR / "squad_form.parquet"

SEED: int = 42

CLASSES: tuple[str, ...] = ("A", "D", "H")
ODDS_COLUMNS: dict[str, str] = {"H": "OddHome", "D": "OddDraw", "A": "OddAway"}
OUTCOME_LABELS: dict[str, str] = {"H": "Home win", "D": "Draw", "A": "Away win"}


@dataclass(frozen=True)
class EloConfig:
    k_factor: float = 20.0
    home_advantage: float = 60.0
    start_top_flight: float = 1500.0
    start_second_tier: float = 1380.0
    xg_margin_divisor: float = 4.0
    xg_blend_weight: float = 0.5


@dataclass(frozen=True)
class FeatureConfig:
    form_windows: tuple[int, ...] = (3, 5, 10)
    venue_window: int = 5
    h2h_window: int = 5
    max_rest_days: float = 30.0
    min_prior_matches: int = 8


@dataclass(frozen=True)
class ModelConfig:
    logreg_c: float = 0.1
    logreg_max_iter: int = 2000
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 15
    lgbm_min_child_samples: int = 60
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 5.0


@dataclass(frozen=True)
class BacktestConfig:
    test_seasons: tuple[int, ...] = tuple(range(2019, 2026))
    ablation_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025)
    stake: float = 10.0
    edge_thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10)


@dataclass(frozen=True)
class PredictConfig:
    model_weight: float = 0.25
    value_edge: float = 0.05


@dataclass(frozen=True)
class Config:
    elo: EloConfig = field(default_factory=EloConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    seed: int = SEED


CONFIG = Config()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, OUTPUT_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
