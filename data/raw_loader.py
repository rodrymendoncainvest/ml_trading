import pandas as pd
from pathlib import Path


class RawLoader:
    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
