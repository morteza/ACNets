import keras
import math


class Julia2018Dataset(keras.utils.PyDataset):

    def __init__(self, X: list, y: list, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Return the number of batches in the dataset.

        Returns:
            int: Number of batches in the dataset.
        """

        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        """Return a batch of data, indexed by idx.

        Args:
            idx (int): Index of the batch to return. 0-indexed.
        """

        # Return x, y for batch idx.
        low = idx * self.batch_size
        # the last batch may be smaller if num of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.X))
        X_batch = self.X[low:high]
        y_batch = self.y[low:high]

        # TODO read X and y from disk or other storage
        return X_batch, y_batch
