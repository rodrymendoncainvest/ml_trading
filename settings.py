# settings.py

class Settings:
    """
    Configurações industriais para todo o pipeline.
    """

    # ----------------------------------------------------------
    # Modelo
    # ----------------------------------------------------------
    models = [
        "TCN",
        "LSTM",
        "TRANSFORMER"
    ]

    # ----------------------------------------------------------
    # Treino
    # ----------------------------------------------------------
    epochs = 60
    patience = 10
    lr = 1e-4
    batch = 64

    # ----------------------------------------------------------
    # Dados
    # ----------------------------------------------------------
    window = 64          # número de candles por janela
    horizon = 1          # prevemos apenas 1 hora à frente (mais robusto)

    # ----------------------------------------------------------
    # Backtest (parâmetros futuros aqui)
    # ----------------------------------------------------------
    fee = 0.0004
    position_size = 1.0
    threshold = 0.001
