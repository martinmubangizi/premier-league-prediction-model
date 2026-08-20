from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG, SEED

DEFAULT_MODEL = "logreg"


def build_logreg(cfg: Any = CONFIG.model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg.logreg_max_iter,
                    C=cfg.logreg_c,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_lgbm(cfg: Any = CONFIG.model) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        min_child_samples=cfg.lgbm_min_child_samples,
        subsample=cfg.lgbm_subsample,
        subsample_freq=1,
        colsample_bytree=cfg.lgbm_colsample_bytree,
        reg_lambda=cfg.lgbm_reg_lambda,
        objective="multiclass",
        num_class=3,
        verbose=-1,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
    )


def make_models() -> dict[str, BaseEstimator]:
    return {"logreg": build_logreg(), "lgbm": build_lgbm()}


def predict_proba_ordered(
    model: BaseEstimator, features: Any, classes: tuple[str, ...]
) -> np.ndarray:
    order = list(model.classes_)
    index = [order.index(c) for c in classes]
    return model.predict_proba(features)[:, index]
