import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import DataLoader

from dataset import SepsisTrajectoryDataset


def aggregate_sequence(X):
    """
    Convert:

        (samples, 12, 69)

    into:

        (samples, 138)

    using:

        mean over time
        last observation

    """

    with np.errstate(all="ignore"):
        mean_features = np.nanmean(X, axis=1)

    last_features = X[:, -1, :]

    return np.concatenate([mean_features, last_features], axis=1)


def load_dataset(split):

    dataset = SepsisTrajectoryDataset(split=split, target="y_12")

    loader = DataLoader(dataset, batch_size=4096, shuffle=False)

    X_batches = []
    y_batches = []

    for batch_X, batch_y in loader:
        # batch_X:
        # (batch, 12, 69)

        batch_X = batch_X.numpy()

        # Immediately reduce memory:
        #
        # (batch,12,69)
        #       ↓
        # (batch,138)

        batch_X = aggregate_sequence(batch_X)

        X_batches.append(batch_X)

        y_batches.append(batch_y.numpy())

    X = np.concatenate(X_batches, axis=0)

    y = np.concatenate(y_batches, axis=0)

    return X, y


def main():

    print("Loading training data...")

    X_train, y_train = load_dataset("train")

    print("Loading validation data...")

    X_val, y_val = load_dataset("validation")

    print("Train:", X_train.shape)

    print("Validation:", X_val.shape)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    print("Training baseline...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    print("\nBaseline Results")

    print("================")

    print("AUROC:", roc_auc_score(y_val, probabilities))

    print("AUPRC:", average_precision_score(y_val, probabilities))

    print("Recall:", recall_score(y_val, predictions))

    print("Precision:", precision_score(y_val, predictions))

    print("F1:", f1_score(y_val, predictions))


if __name__ == "__main__":
    main()
