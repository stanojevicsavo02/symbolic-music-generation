"""
Preparation phase: MIDI corpus -> tokens -> chunks by bars (~512) -> disk.
Runs ONCE. Training later reads the ready-made chunks.
"""
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

from src.tokenizer import build_tokenizer

# --- configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "pop1k7"
OUT_FILE = PROJECT_ROOT / "data" / "chunks_train.pkl"
VAL_FILE = PROJECT_ROOT / "data" / "chunks_val.pkl"

TARGET_LEN = 512       # soft target: chunk_by_bars cuts at the first Bar AFTER this
MAX_LEN = 768          # hard limit: drop only chunks longer than this (true outliers)
MIN_LEN = 100          # ~2 bars
VAL_RATIO = 0.05       # 5% of songs go to validation
SEED = 42


def chunk_by_bars(token_ids, bar_token_id, max_len):
    """Cut a single sequence into chunks <= max_len, but only at bar boundaries."""
    chunks = []
    current = []
    for tid in token_ids:
        # if we would exceed max_len AND the current token starts a new bar,
        # close the chunk BEFORE this Bar token
        if tid == bar_token_id and len(current) >= max_len:
            chunks.append(np.array(current, dtype=np.int16))
            current = []
        current.append(tid)
    if len(current) > 1:               # leftover, if not empty
        chunks.append(np.array(current, dtype=np.int16))
    return chunks


def main():
    tokenizer = build_tokenizer()
    bar_token_id = tokenizer.vocab["Bar_None"]

    midi_files = sorted(DATA_DIR.rglob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files.")

    # train/val split AT THE SONG LEVEL (not chunk level)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(midi_files))
    n_val = int(len(midi_files) * VAL_RATIO)
    val_idx = set(indices[:n_val].tolist())

    train_chunks, val_chunks = [], []
    skipped = 0
    n_too_short = 0
    n_too_long = 0
    total_before = 0

    for i, path in enumerate(tqdm(midi_files, desc="Tokenizing")):
        try:
            tokens = tokenizer(path)
            ids = tokens[0].ids
        except Exception:
            skipped += 1
            continue

        chunks = chunk_by_bars(ids, bar_token_id, TARGET_LEN)

        for c in chunks:
            total_before += 1
            if len(c) < MIN_LEN:
                n_too_short += 1
                continue
            if len(c) > MAX_LEN:
                n_too_long += 1
                continue
            if i in val_idx:
                val_chunks.append(c)
            else:
                train_chunks.append(c)

    print(f"\nSkipped (read error): {skipped}")
    print(f"\n--- filter ---")
    print(f"Chunks before filter: {total_before}")
    print(f"Too short (<{MIN_LEN}):  {n_too_short}  "
          f"({100*n_too_short/total_before:.1f}%)")
    print(f"Too long (>{MAX_LEN}): {n_too_long}  "
          f"({100*n_too_long/total_before:.1f}%)")
    print(f"Kept total: {len(train_chunks) + len(val_chunks)}")
    print(f"\nTrain chunks: {len(train_chunks)}")
    print(f"Val chunks:   {len(val_chunks)}")

    lens = [len(c) for c in train_chunks]
    print(f"Chunk length: min={min(lens)}, max={max(lens)}, "
          f"mean={sum(lens)/len(lens):.0f}")

    with open(OUT_FILE, "wb") as f:
        pickle.dump(train_chunks, f)
    with open(VAL_FILE, "wb") as f:
        pickle.dump(val_chunks, f)
    print(f"\nSaved to {OUT_FILE.name} and {VAL_FILE.name}")


if __name__ == "__main__":
    main()