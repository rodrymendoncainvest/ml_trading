from .tcn import TCNModel
from .lstm import LSTMModel
from .transformer import TransformerModel


class ModelRegistry:
    """
    Criador de modelos por string.
    """

    @staticmethod
    def create(name: str, input_dim: int, horizon: int, window: int):
        name = name.upper()

        if name == "TCN":
            return TCNModel(input_dim, horizon, window)

        if name == "LSTM":
            return LSTMModel(input_dim, horizon)

        if name == "TRANSFORMER":
            return TransformerModel(input_dim, horizon)

        raise ValueError(f"Modelo desconhecido: {name}")
