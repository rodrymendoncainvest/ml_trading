import torch
import pandas as pd
import numpy as np
from pathlib import Path

from core.device import device
from data.ingest import load_raw_series
from data.normalizer import Normalizer
from data.features import FeatureGenerator
from data.feature_scaler import FeatureScaler
from data.window import WindowBuilder
from models.loader import load_model
from data.prices import PriceDenormalizer


class RollingForecaster:
    """
    Inference industrial:
        faz rolling forecast 1-step-ahead sobre toda a série.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.paths = self._paths()

    def _paths(self):
        base = Path("storage") / self.ticker
        return {
            "raw": Path("raw") / f"{self.ticker}_1H.csv",
            "scaler": base / "scaler.pkl",
            "feat_scaler": base / "features_scaler.pkl",
            "dataset_meta": base / "dataset_meta.json",
            "model": base / "best_model.pth",
            "pred": base / "predictions" / f"{self.ticker}_rolling.parquet",
        }

    # ------------------------------------------------------------------
    # ROLLING FORECAST
    # ------------------------------------------------------------------
    def run(self):

        # ------------------------------------------------------------
        # 1. CARREGAR SÉRIE RAW (OHLCV reais)
        # ------------------------------------------------------------
        df_raw = load_raw_series(self.paths["raw"])

        # ------------------------------------------------------------
        # 2. NORMALIZAR (carregando o scaler gravado no preprocess)
        # ------------------------------------------------------------
        norm = Normalizer(self.ticker)
        df_norm = norm.transform(df_raw)

        denorm = PriceDenormalizer(norm.scaler)

        # ------------------------------------------------------------
        # 3. FEATURES
        # ------------------------------------------------------------
        feat = FeatureGenerator().generate(df_norm)

        # ------------------------------------------------------------
        # 4. FEATURE SCALING
        # ------------------------------------------------------------
        fs = FeatureScaler(self.paths["feat_scaler"])
        feat_scaled = fs.transform(feat.drop(columns=["timestamp"]))
        feat_scaled["timestamp"] = feat["timestamp"].values

        # ------------------------------------------------------------
        # 5. METADATA DO DATASET (window / horizon / features)
        # ------------------------------------------------------------
        meta = pd.read_json(self.paths["dataset_meta"])
        window = int(meta["window_size"][0])
        horizon = int(meta["horizon"][0])
        feature_cols = meta["feature_cols"][0]

        wb = WindowBuilder(window, horizon, feature_cols)

        # ------------------------------------------------------------
        # 6. CARREGAR MODELO TREINADO
        # ------------------------------------------------------------
        model = load_model(
            self.paths["model"],
            input_dim=meta["n_features"][0],
            window=window,
            horizon=horizon,
        )

        model.to(device)
        model.eval()

        # ------------------------------------------------------------
        # 7. ROLLING FORECAST
        # ------------------------------------------------------------
        predictions = []
        timestamps = []

        for t in range(window, len(feat_scaled)):
            window_df = feat_scaled.iloc[t - window : t]

            X = wb.build_single(window_df)
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(X)[0].cpu().numpy()

            pred_next = pred_norm[0]      # 1-step-ahead
            pred_real = denorm.inverse(pred_next)

            predictions.append(pred_real)
            timestamps.append(feat_scaled["timestamp"].iloc[t])

        # ------------------------------------------------------------
        # 8. GUARDAR RESULTADO
        # ------------------------------------------------------------
        out = pd.DataFrame({
            "timestamp": timestamps,
            "prediction": predictions,
        })

        self.paths["pred"].parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.paths["pred"], index=False)

        print(f"✔ Rolling forecast concluído → {self.paths['pred']}")
        return out
