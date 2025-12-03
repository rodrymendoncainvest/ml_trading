import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
from settings import Settings
from core import Normalizer, FeatureGenerator, WindowBuilder
from core.device import device
from data.downloader import DataDownloader
from data.raw_loader import RawLoader

from models import ModelRegistry
from trainer import Trainer

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.report import BacktestReport


# -----------------------------------------------------------
# 1) LOAD RAW CSV
# -----------------------------------------------------------

def load_raw(ticker):
    fpath = Path("data") / f"{ticker}_1H.csv"
    fpath.parent.mkdir(parents=True, exist_ok=True)

    if not fpath.exists():
        print(f"Dados não encontrados em {fpath}, a descarregar via yfinance...")
        DataDownloader().download(ticker=ticker, out=fpath)

    loader = RawLoader()
    return loader.load(fpath)


# -----------------------------------------------------------
# 2) PREPROCESS
# -----------------------------------------------------------

def preprocess(ticker, settings):

    print(f"\n[PREPROCESS] {ticker}")

    raw = load_raw(ticker)

    out = Path("storage") / ticker
    out.mkdir(parents=True, exist_ok=True)

    # 1) Normalização OHLCV
    norm = Normalizer(out / "scaler.joblib")
    norm.fit(raw)
    df_norm = norm.transform(raw)

    # 2) Feature engineering
    feat = FeatureGenerator().generate(df_norm)

    # 3) Window builder
    wb = WindowBuilder(
        window=settings.window,
        horizon=settings.horizon,
        feature_cols=feat.columns.drop("timestamp").tolist(),
    )

    X, y = wb.build(feat)

    # guardar dataset
    np.save(out / "X.npy", X.astype(np.float32))
    np.save(out / "y.npy", y.astype(np.float32))

    meta = {
        "ticker": ticker,
        "window": settings.window,
        "horizon": settings.horizon,
        "n_features": X.shape[2],
        "feature_cols": wb.feature_cols
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=4))

    print("✔ Preprocess concluído")
    return X, y, wb.feature_cols


# -----------------------------------------------------------
# 3) TRAIN
# -----------------------------------------------------------

def train(ticker, settings):
    print(f"\n[TRAIN] {ticker}")

    out = Path("storage") / ticker
    X = np.load(out / "X.npy")
    y = np.load(out / "y.npy")
    meta = json.loads((out / "meta.json").read_text())

    n_feat = meta["n_features"]
    window = meta["window"]
    horizon = meta["horizon"]

    # split simples
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    results = {}

    for model_name in settings.models:
        print(f"\n🔥 Treinar modelo: {model_name}")

        model = ModelRegistry.create(
            name=model_name,
            input_dim=n_feat,
            horizon=horizon,
            window=window,
        ).to(device)

        trainer = Trainer(
            model=model,
            lr=settings.lr,
            batch_size=settings.batch,
            patience=settings.patience,
            save_path=out / f"{model_name}.pth",
        )

        _, best_loss = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=settings.epochs,
        )

        results[model_name] = best_loss

    # escolher o melhor
    best = min(results, key=results.get)
    (out / f"{best}.pth").rename(out / "best_model.pth")

    print(f"\n✔ Melhor modelo: {best}")
    (out / "best_model.txt").write_text(best)

    return best


# -----------------------------------------------------------
# 4) INFERENCE — ROLLING FORECAST
# -----------------------------------------------------------

def inference(ticker, settings):
    print(f"\n[INFERENCE] ROLLING → {ticker}")

    out = Path("storage") / ticker
    meta = json.loads((out / "meta.json").read_text())
    best = (out / "best_model.txt").read_text().strip()

    # reload raw
    df_raw = load_raw(ticker)

    # normalize with existing scaler
    norm = Normalizer(out / "scaler.joblib")
    df_norm = norm.transform(df_raw)

    # features
    feat = FeatureGenerator().generate(df_norm)

    # window builder
    wb = WindowBuilder(
        window=meta["window"],
        horizon=meta["horizon"],
        feature_cols=meta["feature_cols"]
    )

    # load model
    model = ModelRegistry.create(
        name=best,
        input_dim=meta["n_features"],
        horizon=meta["horizon"],
        window=meta["window"],
    ).to(device)

    state = torch.load(out / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    ts = feat["timestamp"].tolist()

    # Rolling 1-step ahead
    for t in range(meta["window"], len(feat)):
        window_df = feat.iloc[t - meta["window"] : t]
        window = wb.build_single(window_df)

        w = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(w)[0].cpu().numpy()

        pred_real = float(norm.inverse_close(pred_norm[0]))
        preds.append(pred_real)

    # criar dataframe
    pred_df = pd.DataFrame({
        "timestamp": ts[meta["window"]:],
        "prediction": preds
    })

    pred_df.to_csv(out / "rolling_predictions.csv", index=False)

    print("✔ Rolling predictions concluídas")
    return pred_df


# -----------------------------------------------------------
# 5) BACKTEST
# -----------------------------------------------------------

def backtest(ticker, settings):
    print(f"\n[BACKTEST] {ticker}")

    out = Path("storage") / ticker

    df_raw = load_raw(ticker)
    df_pred = pd.read_csv(out / "rolling_predictions.csv")
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], utc=True)

    df = df_raw.merge(df_pred, on="timestamp", how="left")
    df["prediction"] = df["prediction"].ffill().fillna(0)

    engine = BacktestEngine()
    results = engine.run(df)

    metrics = BacktestMetrics.compute(results)

    rep = BacktestReport(ticker)
    rep.generate(results, metrics)

    print("✔ Backtest concluído")
    return results, metrics


# -----------------------------------------------------------
# 6) PIPELINE COMPLETA
# -----------------------------------------------------------

def run_pipeline(ticker):
    settings = Settings()

    preprocess(ticker, settings)
    train(ticker, settings)
    inference(ticker, settings)
    backtest(ticker, settings)

    print("\n✔ Pipeline COMPLETA TERMINADA ✔\n")


if __name__ == "__main__":
    t = input("Ticker: ").strip().upper()
    run_pipeline(t)
