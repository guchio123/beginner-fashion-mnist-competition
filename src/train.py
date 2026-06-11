# uv run src/train.py

import math
import pickle
from pathlib import Path

from load_fashion_mnist import load_train_data
from network import CNNConfig, SimpleCNN

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 20
CONV_CHANNELS = (32, 64)
FC_HIDDEN = 128
LEARNING_RATE = 0.003   # cosine decay: 0.003 → ~0 over EPOCHS
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 256
SEED = 42


def cosine_lr(epoch: int, epochs: int, lr_max: float) -> float:
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))


def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    model = SimpleCNN(
        CNNConfig(
            output_size=10,
            conv_channels=CONV_CHANNELS,
            fc_hidden=FC_HIDDEN,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            label_smoothing=LABEL_SMOOTHING,
            seed=SEED,
        )
    )

    for epoch in range(1, EPOCHS + 1):
        lr = cosine_lr(epoch, EPOCHS, LEARNING_RATE)
        loss = model.train_epoch(x_train, t_train, epoch=epoch, lr=lr)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} lr={lr:.5f} "
            f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
