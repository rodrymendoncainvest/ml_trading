import numpy as np
import pandas as pd


class FeatureGenerator:
    """
    Feature engineering industrial e determinística.
    Produz features sem NaN, sem infinite e ordem fixa.
    """

    @staticmethod
    def _safe_div(a, b):
        b = b.replace(0, 1e-9)
        out = a / b
        return out.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean().fillna(0)

    # ----------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        close = df["close"]

        # Retornos
        out["ret_1"] = close.pct_change().fillna(0)
        out["ret_5"] = close.pct_change(5).fillna(0)

        # EMAs
        out["ema_10"] = self._ema(close, 10)
        out["ema_20"] = self._ema(close, 20)

        # RSI
        delta = close.diff().fillna(0)
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        rs = self._safe_div(up.rolling(14).mean(), down.rolling(14).mean())
        out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

        # Volume features
        out["vol_chg"] = self._safe_div(df["volume"], df["volume"].shift(1))

        out.replace([np.inf, -np.inf], 0, inplace=True)
        out.fillna(0, inplace=True)
        return out
