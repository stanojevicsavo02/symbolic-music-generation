"""
Dataset and collate_fn: the bridge between .pkl chunks and the model.
- Dataset: loads chunks, does random-crop to context length, builds (input, target) shift-by-one.
- collate_fn: assembles a batch, pads (PAD) up to the longest sequence IN THAT batch.
"""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MidiChunkDataset(Dataset):
    def __init__(self, pkl_path, context_len=512, pad_id=0, crop=True):
        with open(pkl_path, "rb") as f:
            self.chunks = pickle.load(f)      # list of np.int16 arrays, varying lengths
        self.context_len = context_len
        self.pad_id = pad_id
        self.crop = crop

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx].astype(np.int64)   # int64 because torch embedding expects long

        # random-crop: if the chunk is longer than the context, take a random window
        if len(chunk) > self.context_len:
            if self.crop:
                start = np.random.randint(0, len(chunk) - self.context_len + 1)
            else:
                start = 0
            chunk = chunk[start : start + self.context_len]

        # shift-by-one: input = all but the last, target = all but the first
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def collate_fn(batch, pad_id=0):
    """Batch is a list of (x, y) pairs of DIFFERENT lengths. Pad them up to the longest in the batch."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    x_padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    y_padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, : len(x)] = x
        y_padded[i, : len(y)] = y

    return x_padded, y_padded


if __name__ == "__main__":
    # quick check that the Dataset works as intended
    ds = MidiChunkDataset(PROJECT_ROOT / "data" / "chunks_train.pkl",
                          context_len=512, pad_id=0, crop=True)
    print(f"Number of chunks in dataset: {len(ds)}")

    x, y = ds[0]
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")
    print(f"Same length? {x.shape == y.shape}")

    print(f"\nx[1:5] = {x[1:5].tolist()}")
    print(f"y[0:4] = {y[0:4].tolist()}")
    print(f"Matches (y = x shifted left)? {torch.equal(x[1:5], y[0:4])}")

    # bonus: check that random-crop yields <= context_len
    lens = [len(ds[i][0]) for i in range(min(100, len(ds)))]
    print(f"\nx lengths across 100 samples: min={min(lens)}, max={max(lens)}")
    print(f"All <= 511 (context_len-1)? {max(lens) <= 511}")