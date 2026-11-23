"""Deep learning for tabular data: a residual MLP in PyTorch with an sklearn-compatible API.

Architecture (per block): LayerNorm -> Linear -> GELU -> Dropout -> Linear, plus a skip
connection. Trained with AdamW, cosine LR decay, label smoothing / class weights, gradient
clipping and early stopping on a validation split. Falls back to sklearn's MLP when torch is
not installed so the rest of the system keeps working.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:  # optional dependency
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, x):
            return x + self.net(x)

    class TabularResNet(nn.Module):
        def __init__(self, n_in: int, n_out: int, hidden: int, n_blocks: int, dropout: float) -> None:
            super().__init__()
            self.stem = nn.Linear(n_in, hidden)
            self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, n_out))

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))


class _BaseDeep(BaseEstimator):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 128,
        patience: int = 10,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    # subclasses define: _n_out, _loss, _prepare_y, _stratify
    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        strat = y if self._stratify else None
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, random_state=self.random_state, stratify=strat
            )
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TabularResNet(X.shape[1], self._n_out, self.hidden_dim, self.n_blocks, self.dropout).to(device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        loss_fn = self._loss(y_tr).to(device)
        Xt, yt = torch.tensor(X_tr, device=device), torch.tensor(y_tr, device=device)
        Xv, yv = torch.tensor(X_va, device=device), torch.tensor(y_va, device=device)
        best, best_state, bad = float("inf"), None, 0
        self.history_: list[dict[str, float]] = []
        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), device=device)
            total = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self._forward_out(self.model_(Xt[idx])), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            self.model_.eval()
            with torch.no_grad():
                val = float(loss_fn(self._forward_out(self.model_(Xv)), yv))
            self.history_.append({"epoch": epoch + 1, "train_loss": total / len(Xt), "val_loss": val})
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu").eval()
        self.best_val_loss_ = best
        self.n_epochs_ = len(self.history_)

    def _forward_out(self, out):
        return out

    def _raw_predict(self, X) -> np.ndarray:
        with torch.no_grad():
            return self.model_(torch.tensor(np.asarray(X, dtype=np.float32))).numpy()


class DeepTabularClassifier(ClassifierMixin, _BaseDeep):
    _stratify = True

    def _loss(self, y_tr):
        counts = np.bincount(y_tr, minlength=self._n_out).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1) / self._n_out
        return nn.CrossEntropyLoss(weight=torch.tensor(weights), label_smoothing=0.05)

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        y_enc = self.encoder_.transform(y).astype(np.int64)
        self._n_out = len(self.classes_)
        if TORCH_AVAILABLE:
            self._fit_torch(X, y_enc)
        else:  # pragma: no cover
            from sklearn.neural_network import MLPClassifier

            self.fallback_ = MLPClassifier((self.hidden_dim, self.hidden_dim), early_stopping=True,
                                           max_iter=self.epochs * 3, random_state=self.random_state).fit(X, y_enc)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self, "fallback_"):  # pragma: no cover
            return self.fallback_.predict_proba(X)
        logits = self._raw_predict(X)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.encoder_.inverse_transform(self.predict_proba(X).argmax(axis=1))


class DeepTabularRegressor(RegressorMixin, _BaseDeep):
    _stratify = False
    _n_out = 1

    def _loss(self, y_tr):
        return nn.SmoothL1Loss()

    def _forward_out(self, out):
        return out.squeeze(-1)

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)
        if TORCH_AVAILABLE:
            self._fit_torch(X, y_scaled)
        else:  # pragma: no cover
            from sklearn.neural_network import MLPRegressor

            self.fallback_ = MLPRegressor((self.hidden_dim, self.hidden_dim), early_stopping=True,
                                          max_iter=self.epochs * 3, random_state=self.random_state).fit(X, y_scaled)
        return self

    def predict(self, X) -> np.ndarray:
        if hasattr(self, "fallback_"):  # pragma: no cover
            raw = self.fallback_.predict(X)
        else:
            raw = self._raw_predict(X).squeeze(-1)
        return raw * self.y_std_ + self.y_mean_


def deep_info() -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        return {"backend": "sklearn-mlp (install torch for the residual network)"}
    return {"backend": f"torch {torch.__version__}", "cuda": torch.cuda.is_available()}
