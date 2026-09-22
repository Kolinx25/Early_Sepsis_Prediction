from cached_dataset import CachedTrajectoryDataset


dataset = CachedTrajectoryDataset(split="train")


print("Length:", len(dataset))


X, y = dataset[0]


print("X:", X.shape)


print("y:", y)
