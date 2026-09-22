import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from cached_dataset import CachedTrajectoryDataset
from models.tcn import SepsisTCN


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

CHECKPOINT_PATH = "models/best_tcn_y12.pt"


def evaluate(model, loader, device):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy())

            labels.extend(y.cpu().numpy())

    probabilities = np.array(probabilities)

    labels = np.array(labels)

    predictions = (probabilities >= 0.5).astype(int)

    auroc = roc_auc_score(labels, probabilities)

    auprc = average_precision_score(labels, probabilities)

    recall = recall_score(labels, predictions)

    precision = precision_score(labels, predictions, zero_division=0)

    f1 = f1_score(labels, predictions, zero_division=0)

    print("\nValidation Results")
    print("==================")

    print("AUROC:", auroc)

    print("AUPRC:", auprc)

    print("Recall:", recall)

    print("Precision:", precision)

    print("F1:", f1)

    return auprc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # -------------------------
    # Cached datasets
    # -------------------------

    train_dataset = CachedTrajectoryDataset(split="train")

    val_dataset = CachedTrajectoryDataset(split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # -------------------------
    # Class imbalance
    # -------------------------

    train_labels = np.load("data/cache/train_y.npy", mmap_mode="r")

    positive = train_labels.sum()

    negative = len(train_labels) - positive

    pos_weight = torch.tensor([negative / positive], dtype=torch.float32).to(device)

    print("Positive samples:", int(positive))

    print("Negative samples:", int(negative))

    print("pos_weight:", pos_weight.item())

    # -------------------------
    # Model
    # -------------------------

    model = SepsisTCN(input_channels=69).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_auprc = 0.0

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)

            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        print("Loss:", avg_loss)

        auprc = evaluate(model, val_loader, device)

        # -------------------------
        # Save best model
        # -------------------------

        if auprc > best_auprc:
            best_auprc = auprc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auprc": best_auprc,
                },
                CHECKPOINT_PATH,
            )

            print("✓ Saved best model")

            print("Best AUPRC:", best_auprc)


if __name__ == "__main__":
    main()
