import numpy as np
import pandas as pd

class BacktestMetrics:

    @staticmethod
    def compute(df):
        result = {}

        strat = df["strategy_return"]

        result["total_return_pct"] = (df["equity"].iloc[-1] - 1) * 100
        result["max_drawdown_pct"] = df["drawdown"].min() * 100

        # sharpe simples (close-to-close hourly)
        if strat.std() > 0:
            result["sharpe"] = (strat.mean() / strat.std()) * np.sqrt(24 * 365)
        else:
            result["sharpe"] = 0.0

        # winrate
        wins = (strat > 0).sum()
        total = (strat != 0).sum()
        result["winrate_pct"] = (wins / total * 100) if total > 0 else 0

        return result
