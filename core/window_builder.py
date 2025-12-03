import numpy as np
import pandas as pd


class WindowBuilder:
    """
    Cria janelas para modelos TCN / LSTM / Transformer.
    """

    def __init__(self, window, horizon, feature_cols, target_col="close"):
        self.window = window
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target_col = target_col

    # ----------------------------------------------------------
    def build(self, df):
        data = df[self.feature_cols].values.astype(np.float32)
        target = df[self.target_col].values.astype(np.float32)

        X, y = [], []
        limit = len(df) - self.window - self.horizon + 1

        for i in range(limit):
            X.append(data[i : i + self.window])
            y.append(target[i + self.window : i + self.window + self.horizon])

        return np.array(X), np.array(y)

    # ----------------------------------------------------------
    def build_single(self, df):
        data = df[self.feature_cols].values.astype(np.float32)
        return data[-self.window :]
