from torch.utils.data import DataLoader

from dataset import SepsisTrajectoryDataset


dataset = SepsisTrajectoryDataset(split="train", target="y_12")


loader = DataLoader(dataset, batch_size=32, shuffle=True)


X_batch, y_batch = next(iter(loader))


print("X batch shape:", X_batch.shape)
print("y batch shape:", y_batch.shape)
print("Positive labels in batch:", y_batch.sum().item())
