from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FraudMLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(1)


class MLPTrainer:
    def __init__(
        self,
        seed: int = 42,
        epochs: int = 30,
        batch_size: int = 8192,
        patience: int = 5,
    ):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.scaler = StandardScaler()
        self.model: FraudMLP | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> "MLPTrainer":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x_train = self.scaler.fit_transform(x_train).astype(np.float32)
        x_validation = self.scaler.transform(x_validation).astype(np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_validation = np.asarray(y_validation, dtype=np.int64)
        positives = int(y_train.sum())
        negatives = len(y_train) - positives
        if positives == 0:
            raise ValueError("MLP training partition has no positive examples")

        self.model = FraudMLP(x_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        best_score = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        validation_tensor = torch.from_numpy(x_validation)
        for _ in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(self.model(validation_tensor)).numpy()
            score = float(average_precision_score(y_validation, scores))
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPTrainer must be fit before prediction")
        transformed = self.scaler.transform(values).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.from_numpy(transformed))).numpy()
