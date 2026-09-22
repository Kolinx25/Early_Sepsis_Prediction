import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import SepsisTrajectoryDataset


CACHE_DIR = "data/cache"

BATCH_SIZE = 512


def export_split(split):

    print(f"\nLoading {split} dataset...")

    dataset = SepsisTrajectoryDataset(split=split, target="y_12")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    X_list = []
    y_list = []

    for i, (X, y) in enumerate(loader):
        X_list.append(X.numpy().astype(np.float32))

        y_list.append(y.numpy().astype(np.float32))

        if i % 50 == 0:
            print("Processed batches:", i)

    X = np.concatenate(X_list, axis=0)

    y = np.concatenate(y_list, axis=0)

    print(split, "X:", X.shape)

    print(split, "y:", y.shape)

    np.save(f"{CACHE_DIR}/{split}_X.npy", X)

    np.save(f"{CACHE_DIR}/{split}_y.npy", y)

    print(f"{split} saved.")


def main():

    os.makedirs(CACHE_DIR, exist_ok=True)

    for split in ["train", "validation", "test"]:
        export_split(split)

    print("\nCache creation complete.")


if __name__ == "__main__":
    main()
