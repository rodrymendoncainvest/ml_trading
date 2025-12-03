import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Backtest industrial ultra simples:
        - previsão rolling
        - estratégia: buy se previsão > close, sell se previsão < close
    """

    def __init__(self, fee=0.0005):
        self.fee = fee

    # ------------------------------------------------------------
    # SINAIS A PARTIR DAS PREVISÕES
    # ------------------------------------------------------------
    def build_signals(self, df):
        df = df.copy()
        df["signal"] = 0

        df.loc[df["prediction"] > df["close"], "signal"] = 1
        df.loc[df["prediction"] < df["close"], "signal"] = -1

        return df

    # ------------------------------------------------------------
    # BACKTEST CORE
    # ------------------------------------------------------------
    def run(self, df):
        df = df.copy()

        if "signal" not in df.columns:
            df = self.build_signals(df)

        df["position"] = df["signal"].shift(1).fillna(0)

        df["return"] = df["close"].pct_change().fillna(0)
        df["strategy_return"] = df["position"] * df["return"]

        # custos ao mudar de posição
        df["trade_cost"] = (df["position"].diff().abs() * self.fee).fillna(0)

        df["strategy_return"] -= df["trade_cost"]

        df["equity"] = (1 + df["strategy_return"]).cumprod()

        # drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

        return df
