import numpy as np

from torch.utils.data import DataLoader

from dataset import SepsisTrajectoryDataset


dataset = SepsisTrajectoryDataset(split="train", target="y_12")


loader = DataLoader(dataset, batch_size=4096, shuffle=False)


total_missing_combinations = 0
total_combinations = 0

feature_missing_counts = np.zeros(69)


for X, y in loader:
    X = X.numpy()

    # Shape:
    # (batch, 12, 69)

    all_missing_feature = np.all(np.isnan(X), axis=1)

    total_missing_combinations += all_missing_feature.sum()

    total_combinations += all_missing_feature.size

    feature_missing_counts += all_missing_feature.sum(axis=0)


print(
    "Sample-feature combinations missing across all 12 hours:",
    total_missing_combinations,
)

print("Total sample-feature combinations:", total_combinations)

print("Percentage:", total_missing_combinations / total_combinations * 100)


print("\nFeatures with all 12 hours missing:")

for i, count in enumerate(feature_missing_counts):
    if count > 0:
        print(i, count)
