import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class Normalizer:
    """
    Normalizador OHLCV robusto e estável.
    - Fit no preprocess
    - Transform na inferência e backtest
    - Suporte total GPU-friendly (numpy only)
    """

    def __init__(self, scaler_path: Path):
        self.scaler_path = scaler_path
        self.mean_ = None
        self.std_ = None

    # -------------------------------------------------------
    # Save / load
    # -------------------------------------------------------
    def save(self):
        payload = {"mean": self.mean_, "std": self.std_}
        joblib.dump(payload, self.scaler_path)

    def load(self):
        payload = joblib.load(self.scaler_path)
        self.mean_ = payload["mean"]
        self.std_ = payload["std"]

    # -------------------------------------------------------
    # Fit (preprocess)
    # -------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        data = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1e-8

        self.save()

    # -------------------------------------------------------
    # Transform (inference / backtest)
    # -------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None:
            self.load()

        data = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        scaled = (data - self.mean_) / self.std_

        out = df.copy()
        (
            out["open"],
            out["high"],
            out["low"],
            out["close"],
            out["volume"],
        ) = scaled.T

        out.replace([np.inf, -np.inf], 0, inplace=True)
        out.fillna(0, inplace=True)

        return out

    # -------------------------------------------------------
    # Inverse close (para previsões)
    # -------------------------------------------------------
    def inverse_close(self, norm_val: float) -> float:
        idx = 3  # close
        return float(norm_val * self.std_[idx] + self.mean_[idx])
