import numpy as np
from keras.utils import Sequence
from utils.Dataset import Dataset


class PatchSequence(Sequence):
    def __init__(self, df, batch_size=32, patch_size=(299, 299), shuffle=True, preprocess_fn=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X, y = [], []
        for _, row in batch_df.iterrows():
            img = self.df.to_3_channel_patch(row['path'], *self.patch_size, preprocess_fn=self.preprocess_fn)
            X.append(img)
            y.append(row['findings'])

        return np.array(X), np.array(y)
