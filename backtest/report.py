import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class BacktestReport:

    def __init__(self, ticker):
        self.ticker = ticker
        self.base = Path("reports") / ticker
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # SALVAR MÉTRICAS
    # ------------------------------------------------------------
    def save_metrics(self, metrics):
        out = self.base / "metrics.json"
        out.write_text(json.dumps(metrics, indent=4))

    # ------------------------------------------------------------
    # EQUITY CURVE
    # ------------------------------------------------------------
    def plot_equity(self, df):
        plt.figure(figsize=(12,5))
        plt.plot(df["timestamp"], df["equity"])
        plt.title(f"Equity Curve — {self.ticker}")
        plt.tight_layout()
        plt.savefig(self.base / "equity.png")
        plt.close()

    # ------------------------------------------------------------
    # DRAWDOWN
    # ------------------------------------------------------------
    def plot_drawdown(self, df):
        plt.figure(figsize=(12,4))
        plt.plot(df["timestamp"], df["drawdown"])
        plt.title(f"Drawdown — {self.ticker}")
        plt.tight_layout()
        plt.savefig(self.base / "drawdown.png")
        plt.close()

    # ------------------------------------------------------------
    # RELATÓRIO COMPLETO
    # ------------------------------------------------------------
    def generate(self, df, metrics):
        self.save_metrics(metrics)
        self.plot_equity(df)
        self.plot_drawdown(df)
        print(f"✔ Relatório guardado em {self.base}")
