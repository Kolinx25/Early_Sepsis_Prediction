import torch

from torch.utils.data import Dataset

from database import Database


class SepsisTrajectoryDataset(Dataset):
    """
    PostgreSQL-backed trajectory dataset.

    Returns:

        X:
            Tensor shape (12, 69)

        y:
            Future sepsis label

    Feature order:

        34 clinical features
        34 missingness indicators
        1 temporal feature (iculos)

    """

    def __init__(self, split="train", target="y_12"):

        self.split = split
        self.target = target

        db = Database()

        try:
            db.connect()

            query = f"""

            SELECT

                tw.x_sequence,
                tw.{target}

            FROM trajectory_windows tw


            JOIN patient_splits ps

            ON tw.patient_id = ps.patient_id


            WHERE ps.split = '{split}'


            ORDER BY

                tw.patient_id,

                tw.prediction_iculos;

            """

            self.data = db.fetch_dataframe(query)

        finally:
            db.disconnect()

        print(f"{split} dataset loaded:", len(self.data))

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        sequence = row["x_sequence"]

        # PostgreSQL JSONB returns Python lists.
        #
        # TCN cannot process NaN values.
        #
        # Missing clinical values become 0.0.
        #
        # Missingness channels remain available
        # as separate binary features.

        sequence = [
            [0.0 if value is None else float(value) for value in timestep]
            for timestep in sequence
        ]

        X = torch.tensor(sequence, dtype=torch.float32)

        y = torch.tensor(row[self.target], dtype=torch.float32)

        return X, y
