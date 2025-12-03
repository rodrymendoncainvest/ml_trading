import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from core.device import device
import numpy as np
from pathlib import Path


# =====================================================================
# DATASET INDUSTRIAL
# =====================================================================

class SequenceDataset(Dataset):
    """
    Dataset para forecasting:
        X → (window, features)
        y → (horizon)
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# TREINADOR INDUSTRIAL
# =====================================================================

class Trainer:
    """
    Treinador universal:
      - suporta TCN, LSTM, Transformer
      - early stopping
      - checkpoints
      - data loader
      - treino rápido em GPU
      - logging clean industrial
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        batch_size: int = 64,
        patience: int = 20,
        save_path: str = "models/best_model.pth",
    ):
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.save_path = Path(save_path)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # FIT
    # --------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val, epochs=200):

        train_ds = SequenceDataset(X_train, y_train)
        val_ds = SequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_loss = float("inf")
        patience_counter = 0

        history = {"train_loss": [], "val_loss": []}

        print("\n========== TREINO INICIADO ==========\n")
        print(f"Device: {device}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for Xb, yb in train_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)

                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train = total_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:03d} | Train: {avg_train:.6f} | Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1

                if patience_counter >= self.patience:
                    print("\nEARLY STOPPING ativado.\n")
                    break

        print("\n========== TREINO TERMINADO ==========\n")
        print(f"Melhor modelo guardado em: {self.save_path}")

        return history, best_loss

    # --------------------------------------------------------------------
    # AVALIAÇÃO
    # --------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total = 0.0

        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = self.model(Xb)
            total += self.loss_fn(pred, yb).item()

        return total / len(loader)

    # --------------------------------------------------------------------
    # CARREGAR MODELO TREINADO
    # --------------------------------------------------------------------
    def load_best(self):
        if self.save_path.exists():
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=device)
            )
            print(f"Modelo carregado: {self.save_path}")
        else:
            print("Nenhum modelo guardado encontrado.")
