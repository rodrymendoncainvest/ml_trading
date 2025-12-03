import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
from settings import Settings
from core.normalizer import Normalizer
from core.features import FeatureGenerator
from core.window import WindowBuilder

from models.factory import ModelFactory

from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.report import BacktestReport


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# 1) LOAD RAW CSV
# -----------------------------------------------------------

def load_raw(ticker):
    fpath = Path("data") / f"{ticker}_1H.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {fpath}")

    df = pd.read_csv(fpath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -----------------------------------------------------------
# 2) PREPROCESS
# -----------------------------------------------------------

def preprocess(ticker, settings):

    print(f"\n[PREPROCESS] {ticker}")

    raw = load_raw(ticker)

    # 1) Normalização OHLCV
    norm = Normalizer(ticker=ticker)
    df_norm = norm.fit_transform(raw)

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
    out = Path("storage") / ticker
    out.mkdir(parents=True, exist_ok=True)

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

        model = ModelFactory.create(
            name=model_name,
            input_dim=n_feat,
            window=window,
            horizon=horizon
        ).to(DEVICE)

        best_loss = ModelFactory.train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=DEVICE,
            lr=settings.lr,
            epochs=settings.epochs,
            patience=settings.patience,
            save_path=out / f"{model_name}.pth"
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
    norm = Normalizer(ticker=ticker)
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
    model = ModelFactory.create(
        name=best,
        input_dim=meta["n_features"],
        window=meta["window"],
        horizon=meta["horizon"]
    ).to(DEVICE)

    state = torch.load(out / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = []
    ts = feat["timestamp"].tolist()

    # Rolling 1-step ahead
    for t in range(meta["window"], len(feat)):
        window_df = feat.iloc[t - meta["window"] : t]
        window = wb.build_single(window_df)

        w = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)

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
    df["prediction"] = df["prediction"].fillna(method="ffill").fillna(0)

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
