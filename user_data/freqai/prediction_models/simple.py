from __future__ import annotations

import joblib
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LinearRegression


class simple:
    """
    A minimal FreqAI-compatible regression model wrapper using
    sklearn's LinearRegression. This is a simple starting point
    to learn how custom FreqAI models are structured.

    Expected usage (Freqtrade): pass `--freqaimodel simple` (or set in config)
    and place this file under `user_data/freqai/prediction_models/simple.py`.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.model: LinearRegression = LinearRegression(**kwargs)

    # --- Basic estimator API ---
    def fit(self, X: np.ndarray, y: np.ndarray) -> "simple":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any) -> "simple":
        self.model.set_params(**params)
        return self

    # --- Persistence helpers (used by FreqAI) ---
    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "simple":
        self.model = joblib.load(file_path)
        return self

    # --- Introspection helpers ---
    @property
    def is_classifier(self) -> bool:
        return False

    @property
    def is_regressor(self) -> bool:
        return True


# Optional CamelCase alias - some loaders prefer this naming style
Simple = simple

