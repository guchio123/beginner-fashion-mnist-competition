from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_sizes: tuple[int, ...] = (256, 128)
    output_size: int = 10
    learning_rate: float = 0.001
    batch_size: int = 256
    seed: int = 42


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)

        sizes = [config.input_size, *config.hidden_sizes, config.output_size]
        self.num_layers = len(sizes) - 1
        self.params: dict[str, np.ndarray] = {}

        for i in range(self.num_layers):
            fan_in = sizes[i]
            # He initialization for hidden layers, Xavier for output
            std = np.sqrt(2.0 / fan_in) if i < self.num_layers - 1 else np.sqrt(1.0 / fan_in)
            self.params[f"W{i+1}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * std).astype(np.float32)
            self.params[f"b{i+1}"] = np.zeros(sizes[i + 1], dtype=np.float32)

        # Adam optimizer state
        self._t = 0
        self._m: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        linears: list[np.ndarray] = []
        activations: list[np.ndarray] = [x]
        a = x
        for i in range(1, self.num_layers + 1):
            z = np.dot(a, self.params[f"W{i}"]) + self.params[f"b{i}"]
            linears.append(z)
            a = np.maximum(0.0, z) if i < self.num_layers else _softmax(z)
            activations.append(a)
        return linears, activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, activations = self._forward(x)
        return activations[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        total = x.shape[0]
        batch_size = self.config.batch_size
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            pred = self.predict(x_batch)
            correct += int(np.sum(pred == y_batch))
        return float(correct) / float(total)

    def _adam_update(self, key: str, grad: np.ndarray, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._m[key] = beta1 * self._m[key] + (1.0 - beta1) * grad
        self._v[key] = beta2 * self._v[key] + (1.0 - beta2) * grad ** 2
        m_hat = self._m[key] / (1.0 - beta1 ** self._t)
        v_hat = self._v[key] / (1.0 - beta2 ** self._t)
        self.params[key] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size
        lr = self.config.learning_rate

        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            linears, activations = self._forward(x_batch)
            probs = activations[-1]

            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            self._t += 1
            d_a = (probs - y_one_hot) / x_batch.shape[0]

            for i in range(self.num_layers, 0, -1):
                a_prev = activations[i - 1]
                dW = np.dot(a_prev.T, d_a)
                db = np.sum(d_a, axis=0)

                # Compute gradient for previous layer BEFORE updating W{i}
                if i > 1:
                    d_a_prev = np.dot(d_a, self.params[f"W{i}"].T) * (linears[i - 2] > 0).astype(np.float32)

                self._adam_update(f"W{i}", dW, lr)
                self._adam_update(f"b{i}", db, lr)

                if i > 1:
                    d_a = d_a_prev

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_sizes": list(self.config.hidden_sizes),
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config_dict: dict[str, Any] = config_obj

        # Backward compatibility: old format used single `hidden_size`
        raw_hidden = config_dict.get("hidden_sizes")
        if raw_hidden is not None:
            hidden_sizes = tuple(int(h) for h in raw_hidden)
        else:
            hidden_sizes = (int(config_dict.get("hidden_size", 256)),)

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_sizes=hidden_sizes,
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.001)),
            batch_size=int(config_dict.get("batch_size", 256)),
            seed=int(config_dict.get("seed", 42)),
        )

        params_obj = state.get("params")
        if not isinstance(params_obj, dict):
            raise ValueError("Invalid state: 'params' must be a dict")
        params: dict[str, np.ndarray] = {}
        for key, value in params_obj.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray):
                raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
            params[key] = value

        model = cls(config)
        model.params = params
        return model
