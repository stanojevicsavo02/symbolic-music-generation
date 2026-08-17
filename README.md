# Symbolic Music Generation with Relative Self-Attention

A decoder-only Transformer that generates pop-piano music in symbolic (MIDI) form,
comparing **absolute** vs **relative** positional attention. Course project for a
Neural Networks (MSc) module; designed to extend into a master's thesis (emotion
conditioning + diffusion).

The core result: **relative self-attention** (Music Transformer, Huang et al. 2018)
outperforms absolute positional encoding on symbolic music, lowering validation
perplexity from **5.96 → 5.30** under identical training conditions.

## Overview

Music is tokenized with **REMI** (Bar / Position / Pitch / Velocity / Duration),
turning MIDI into discrete sequences a Transformer can model. A small (~5–6M parameter)
decoder-only Transformer is trained on next-token prediction over the **Pop1k7** corpus
(1747 pop-piano performances). The same model runs in two modes via a single switch —
absolute sinusoidal positional encoding, or relative-position self-attention — enabling
a clean ablation where the *only* difference is the attention mechanism.

## Results

| Model | Attention | Parameters | Val perplexity | s/epoch |
|-------|-----------|-----------:|---------------:|--------:|
| M2    | absolute  | 4,867,835  | 5.96           | ~67     |
| M3    | relative  | 6,440,699  | 5.30           | ~89     |

- **Perplexity:** relative attention is ~11% better; the gap widened throughout training
  and relative passed absolute's *final* score by epoch 27 of 50.
- **Embedding analysis:** trained only on next-token prediction (with arbitrary token IDs),
  the model independently learned **pitch height** (cosine similarity decreases with pitch
  distance) and **octave equivalence** (similarity spikes at 12/24 semitones).
- **Validation metrics:** generated music resembles real Pop1k7 in pitch-class entropy and
  scale consistency, but relative attention trends toward higher note density — a trade-off
  suggesting lower perplexity does not guarantee greater perceptual realism.

## Project structure

```
symbolic-music-generation/
├── data/                       # Pop1k7 MIDI + cached chunks (gitignored)
│   └── pop1k7/                  # midi_synchronized files
├── src/
│   ├── tokenizer.py            # REMI tokenizer configuration
│   ├── dataset.py              # Dataset + collate_fn (crop, shift-by-one, padding)
│   ├── model.py                # Transformer; absolute/relative attention switch
│   └── generate.py             # autoregressive sampling + detokenize to MIDI
├── scripts/
│   ├── prepare_data.py         # tokenize corpus -> chunks by bars -> .pkl
│   ├── test_skew.py            # verifies the relative-attention skewing trick
│   ├── analyze_embeddings.py   # embedding structure analysis (pitch/harmony)
│   ├── validation_metrics.py   # metrics: entropy, scale consistency, note density
│   └── final_pianoroll.py      # absolute vs relative piano-roll figure
├── checkpoints/                # trained models (gitignored)
├── figures/                    # generated figures
├── notes/theory.md             # design decisions & theory (report source)
└── requirements.txt
```

## Setup

Requires an NVIDIA GPU with recent drivers (developed on an RTX 5070 / Blackwell,
which needs CUDA 12.8+).

```bash
conda create -n music python=3.11 -y
conda activate music

# PyTorch with CUDA 12.8 (Blackwell) — install FIRST
pip install torch --index-url https://download.pytorch.org/whl/cu128

# project dependencies
pip install miditok pretty_midi music21 numpy pandas matplotlib tqdm tensorboard scikit-learn
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

Download **Pop1k7** from Zenodo (https://zenodo.org/records/13167761), unzip, and place
the `midi_synchronized` files under `data/pop1k7/`. The `midi_synchronized` version is
required (it carries the beat/bar grid REMI needs); the raw `midi_transcribed` version
lacks it.

## Usage

All commands run from the project root.

**1. Prepare data** (tokenize corpus → chunks by bars → cached `.pkl`, runs once):

```bash
python -m scripts.prepare_data
```

**2. Train** — set `ATTENTION_TYPE` in `src/train.py` to `"absolute"` or `"relative"`,
then:

```bash
python -m src.train
```

Trains for 50 epochs (~55 min absolute / ~75 min relative on an RTX 5070), saving the
best checkpoint to `checkpoints/{attention}_best.pt` and TensorBoard logs to `runs/`.

**3. Generate music:**

```bash
python -m src.generate --attention relative --out generated.mid
```

Options: `--temperature` (default 1.0), `--top_p` (default 0.95), `--tokens` (default 512),
`--tempo` (default 100 BPM, applied at render since the model works in metric space).
Open the resulting `.mid` in any MIDI player.

**4. Evaluate & visualize:**

```bash
python -m scripts.analyze_embeddings     # embedding structure (pitch height, harmony)
python -m scripts.validation_metrics     # entropy, scale consistency, note density
python -m scripts.final_pianoroll        # absolute vs relative piano-roll figure
```

## Method notes

- **REMI over MIDI-like tokenization:** encodes time as a metric address (bar + position)
  rather than time-shifts, so the model gets the pulse "for free" and `Duration` tokens
  avoid dangling note-offs.
- **Chunking by bars:** songs (~3–5k tokens) are cut at bar boundaries into ≤512-token
  chunks, keeping each chunk grammatically whole. Train/val split is at the **song** level
  to avoid data leakage.
- **Relative attention** adds a learned distance term `q · r` to the attention score,
  computed efficiently with the skewing trick (verified against a naive implementation in
  `scripts/test_skew.py`).
- The **only** difference between the two models is `attention_type`; the parameter-count
  difference is a consequence of the learned relative-position tables, not a separate knob.

See `notes/theory.md` for the full reasoning behind every design decision.

## References

- Huang et al., *Music Transformer: Generating Music with Long-Term Structure* (2018) —
  https://arxiv.org/abs/1809.04281
- Hsiao et al., *Compound Word Transformer* (2021) — source of the Pop1k7 dataset.