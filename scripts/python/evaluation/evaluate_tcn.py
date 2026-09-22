import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from cached_dataset import CachedTrajectoryDataset
from models.tcn import SepsisTCN


BATCH_SIZE = 128

CHECKPOINT_PATH = "models/best_tcn_y12.pt"


def evaluate(model, loader, device):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)

            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy())

            labels.extend(y.numpy())

    probabilities = np.array(probabilities)

    labels = np.array(labels)

    predictions = (probabilities >= 0.5).astype(int)

    print("\nTest Results")
    print("==================")

    print("AUROC:", roc_auc_score(labels, probabilities))

    print("AUPRC:", average_precision_score(labels, probabilities))

    print("Recall:", recall_score(labels, predictions))

    print("Precision:", precision_score(labels, predictions, zero_division=0))

    print("F1:", f1_score(labels, predictions, zero_division=0))

    print("\nConfusion Matrix")

    print(confusion_matrix(labels, predictions))


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # -------------------------
    # Test dataset
    # -------------------------

    test_dataset = CachedTrajectoryDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # -------------------------
    # Load model
    # -------------------------

    model = SepsisTCN(input_channels=69).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])

    print("Validation best AUPRC:", checkpoint["best_auprc"])

    # -------------------------
    # Test evaluation
    # -------------------------

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
