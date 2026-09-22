from dataset import SepsisTrajectoryDataset


dataset = SepsisTrajectoryDataset(split="train", target="y_12")


print("Length:", len(dataset))


X, y = dataset[0]


print("X shape:", X.shape)
print("y:", y)
