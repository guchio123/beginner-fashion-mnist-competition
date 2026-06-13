# uv run src/train.py

import math
import pickle
from pathlib import Path

from load_fashion_mnist import load_full_train_data, load_train_data
from network import CNNConfig, SimpleCNN

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 50
CONV_CHANNELS = (32, 64, 128, 256)
FC_HIDDEN = 512
LEARNING_RATE = 0.0018  # cosine decay: 0.0018 → ~0 over EPOCHS
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 256
SEED = 42
FULL_TRAIN_EPOCHS = 8
FULL_TRAIN_LR = 0.0002


def cosine_lr(epoch: int, epochs: int, lr_max: float) -> float:
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))


def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    x_full, t_full = load_full_train_data()

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

    best_state = None
    best_valid_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        lr = cosine_lr(epoch, EPOCHS, LEARNING_RATE)
        loss = model.train_epoch(x_train, t_train, epoch=epoch, lr=lr)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} lr={lr:.5f} "
            f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = model.to_state()

    if best_state is None:
        best_state = model.to_state()

    model._net.load_state_dict(best_state["state_dict"])
    for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
        lr = FULL_TRAIN_LR
        loss = model.train_epoch(x_full, t_full, epoch=epoch + EPOCHS, lr=lr)
        print(f"FineTune {epoch:02d}/{FULL_TRAIN_EPOCHS} lr={lr:.5f} loss={loss:.4f}")

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Best valid_acc={best_valid_acc:.4f}")
    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
