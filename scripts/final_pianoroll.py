"""
Final report figure: absolute vs relative piano-roll on the same seed.
Visually supports the ablation — reader compares the two models' output structure directly.
"""
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer
from src.generate import generate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 101          # your chosen example
TOKENS = 384


def score_to_pianoroll(score, ax, title, color):
    notes = score.tracks[0].notes if score.tracks else []
    tpq = score.ticks_per_quarter
    segments, pitches = [], []
    for n in notes:
        start, end = n.time / tpq, (n.time + n.duration) / tpq
        segments.append([(start, n.pitch), (end, n.pitch)])
        pitches.append(n.pitch)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=3))
    if pitches:
        ax.set_xlim(0, max(s[1][0] for s in segments) + 1)
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("MIDI pitch")
    ax.grid(alpha=0.2)


def gen(model, tokenizer, seed):
    torch.manual_seed(seed)
    ids = generate(model, tokenizer, max_new_tokens=TOKENS, top_p=0.95)
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

    rel = load_model("relative", "relative_best.pt")
    abs_ = load_model("absolute", "absolute_best.pt")

    score_rel = gen(rel, tokenizer, SEED)
    score_abs = gen(abs_, tokenizer, SEED)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    score_to_pianoroll(score_rel, axes[0], "Relative attention", "#2a6f97")
    score_to_pianoroll(score_abs, axes[1], "Absolute attention", "#c1272d")
    axes[1].set_xlabel("time (beats)")

    plt.suptitle(f"Generated piano-rolls (same seed = {SEED})", fontsize=12, y=1.0)
    plt.tight_layout()

    out = PROJECT_ROOT / "figures" / "pianoroll_comparison.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()