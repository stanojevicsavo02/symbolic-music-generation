"""
M5: generate pieces from both models and plot piano-rolls.
Piano-roll = time (x) vs pitch (y), each note a horizontal bar.
Lets you eyeball structure (chords, melody, rhythm) and pick examples for the report.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer
from src.generate import generate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def score_to_pianoroll(score, ax, title, color):
    """Draw one score as a piano-roll on the given axis."""
    notes = score.tracks[0].notes if score.tracks else []
    tpq = score.ticks_per_quarter
    segments, pitches = [], []
    for n in notes:
        start = n.time / tpq              # in beats
        end = (n.time + n.duration) / tpq
        segments.append([(start, n.pitch), (end, n.pitch)])
        pitches.append(n.pitch)

    lc = LineCollection(segments, colors=color, linewidths=3)
    ax.add_collection(lc)
    if pitches:
        ax.set_xlim(0, max(s[1][0] for s in segments) + 1)
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time (beats)")
    ax.set_ylabel("MIDI pitch")
    ax.grid(alpha=0.2)


def generate_and_decode(model, tokenizer, tokens=384, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids = generate(model, tokenizer, max_new_tokens=tokens, top_p=0.95)
    from miditok import TokSequence
    tokseq = TokSequence(ids=ids)
    tokenizer.complete_sequence(tokseq)
    return tokenizer.decode([tokseq])


def load_model(attention_type, ckpt_name):
    model = MusicTransformer(vocab_size=VOCAB_SIZE, attention_type=attention_type).to(DEVICE)
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    tokenizer = build_tokenizer()
    out_dir = PROJECT_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    midi_dir = PROJECT_ROOT / "generated_examples"
    midi_dir.mkdir(exist_ok=True)

    rel_model = load_model("relative", "relative_best.pt")

    # generate a few candidates from the relative model, each with a fixed seed
    # so you can reproduce the exact one you like
    n_candidates = 4
    fig, axes = plt.subplots(n_candidates, 1, figsize=(11, 2.6 * n_candidates))
    for i in range(n_candidates):
        seed = 100 + i
        score = generate_and_decode(rel_model, tokenizer, tokens=384, seed=seed)
        score_to_pianoroll(score, axes[i], f"relative — seed {seed}", "#2a6f97")
        # save the MIDI too, so you can listen and keep the good ones
        from symusic import Tempo
        score.tempos = [Tempo(time=0, qpm=100)]
        score.dump_midi(midi_dir / f"relative_seed{seed}.mid")

    plt.tight_layout()
    out = out_dir / "pianoroll_candidates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"MIDI files in {midi_dir}")
    plt.show()


if __name__ == "__main__":
    main()