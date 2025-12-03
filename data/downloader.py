import yfinance as yf
import pandas as pd
from pathlib import Path


class DataDownloader:
    def download(self, ticker: str, out: Path, period="730d"):
        df = yf.download(ticker, interval="1h", period=period, progress=False)

        if df.empty:
            raise ValueError(f"Sem dados para {ticker}")

        df = df.reset_index().rename(columns={"Datetime": "timestamp"})
        df.to_csv(out, index=False)
        return df
