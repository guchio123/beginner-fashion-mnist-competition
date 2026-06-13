from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    output_size: int = 10
    conv_channels: tuple[int, ...] = (32, 64, 128, 256)
    fc_hidden: int = 512
    learning_rate: float = 0.0018
    batch_size: int = 256
    label_smoothing: float = 0.05
    seed: int = 42


class _ResBlock(nn.Module):
    """Conv→BN→ReLU→Conv→BN + skip connection (1×1 conv if channels change)"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class _Net(nn.Module):
    """
    Stem → ResBlock stages → Pool → GAP → FC(hidden) → FC(10)
    """

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        channels = list(config.conv_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.layers.append(nn.Sequential(_ResBlock(in_ch, out_ch), nn.MaxPool2d(2)))
            in_ch = out_ch
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(config.fc_hidden, config.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class SimpleCNN:
    def __init__(self, config: CNNConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self._net = _Net(config).to(self.device)
        self._optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        self._criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def _np_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """numpy (N,28,28) or (N,784) → tensor (N,1,28,28) on device."""
        return torch.from_numpy(x.astype(np.float32).reshape(-1, 1, 28, 28)).to(self.device)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int, lr: float | None = None) -> float:
        if lr is not None:
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr

        self._net.train()
        rng = np.random.default_rng(self.config.seed + epoch)

        # ---- エポック先頭で1回だけ tensor 変換（バッチ毎の変換コストを排除） ----
        X = self._np_to_tensor(x)                                          # (N,1,28,28)
        Y = torch.from_numpy(y.astype(np.int64)).to(self.device)           # (N,)
        idx = torch.from_numpy(rng.permutation(len(x)).astype(np.int64))   # CPU でシャッフル

        total_loss, steps = 0.0, 0
        bs = self.config.batch_size
        arange28 = torch.arange(28, device=self.device)

        for start in range(0, len(x), bs):
            xb = X[idx[start : start + bs]].clone()   # (N, 1, 28, 28)
            yb = Y[idx[start : start + bs]]
            N = xb.shape[0]

            # ---- Augmentation をすべて torch 演算で実行（numpy ↔ tensor 変換ゼロ） ----
            # 水平反転
            flip = torch.rand(N, device=self.device) < 0.5
            xb[flip] = xb[flip].flip(-1)
            # ランダムクロップ: 2px パディング → 28×28 クロップ
            xb = F.pad(xb, (2, 2, 2, 2))                                   # (N,1,32,32)
            r_off = torch.randint(0, 5, (N,), device=self.device)
            c_off = torch.randint(0, 5, (N,), device=self.device)
            ri = (r_off[:, None] + arange28)                               # (N,28)
            ci = (c_off[:, None] + arange28)                               # (N,28)
            ni = torch.arange(N, device=self.device)
            xb = xb[:, 0][
                ni[:, None, None].expand(N, 28, 28),
                ri[:, :, None].expand(N, 28, 28),
                ci[:, None, :].expand(N, 28, 28),
            ].unsqueeze(1)                                                  # (N,1,28,28)

            self._optimizer.zero_grad()
            loss = self._criterion(self._net(xb), yb)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def predict(self, x: np.ndarray, tta: bool = True) -> np.ndarray:
        self._net.eval()
        X = self._np_to_tensor(x)
        preds = []
        bs = self.config.batch_size
        with torch.no_grad():
            for i in range(0, len(x), bs):
                xb = X[i : i + bs]
                logits = self._net(xb)
                if tta:
                    logits_flip = self._net(xb.flip(-1))
                    logits = (logits + logits_flip) * 0.5
                preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def to_state(self) -> dict[str, Any]:
        return {
            "model_type": "SimpleCNN",
            "config": {
                "output_size": self.config.output_size,
                "conv_channels": list(self.config.conv_channels),
                "fc_hidden": self.config.fc_hidden,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "label_smoothing": self.config.label_smoothing,
                "seed": self.config.seed,
            },
            "state_dict": {k: v.cpu() for k, v in self._net.state_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SimpleCNN":
        cfg = state.get("config")
        if not isinstance(cfg, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config = CNNConfig(
            output_size=int(cfg["output_size"]),
            conv_channels=tuple(int(c) for c in cfg["conv_channels"]),
            fc_hidden=int(cfg["fc_hidden"]),
            learning_rate=float(cfg.get("learning_rate", 0.003)),
            batch_size=int(cfg.get("batch_size", 128)),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            seed=int(cfg.get("seed", 42)),
        )
        model = cls(config)
        state_dict = state.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid state: missing 'state_dict'")
        model._net.load_state_dict({k: v.to(model.device) for k, v in state_dict.items()})
        return model
