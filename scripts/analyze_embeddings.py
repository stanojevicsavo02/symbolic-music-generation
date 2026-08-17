"""
M5 deeper analysis: does the pitch-embedding space encode musical structure?
Three angles:
  1. How much variance do the first 2 PCA dims actually capture? (Is 2D even fair?)
  2. Color by pitch CLASS (C, C#, ... mod 12) instead of absolute height —
     maybe the model groups by harmonic function, not raw pitch.
  3. Cosine-similarity matrix — are pitch-close notes numerically more similar?
     (This is quantitative, doesn't depend on a 2D squeeze.)
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def plot_similarity_vs_distance(dist_sim, out_path):
    """Line plot: cosine similarity vs pitch distance. Shows BOTH findings at once —
    the decreasing trend (pitch height) and the octave bumps at 12/24 (harmony)."""
    distances = sorted(d for d in dist_sim if d <= 30)
    means = [np.mean(dist_sim[d]) for d in distances]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(distances, means, "-o", color="#3a6ea5", markersize=4, linewidth=1.5,
            label="mean cosine similarity")

    # highlight octaves (12, 24) — where harmony breaks the decreasing trend
    for octave in [12, 24]:
        if octave in dist_sim:
            ax.scatter([octave], [np.mean(dist_sim[octave])], color="#c1272d",
                       s=120, zorder=5, marker="o",
                       label="octave" if octave == 12 else None)
            ax.annotate(f"octave\n({octave} st)",
                        (octave, np.mean(dist_sim[octave])),
                        textcoords="offset points", xytext=(0, 18),
                        ha="center", fontsize=9, color="#c1272d")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("pitch distance (semitones)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("Embedding similarity vs pitch distance\n"
                 "(decreasing trend = pitch height; octave bumps = harmony)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


def main():
    tokenizer = build_tokenizer()
    model = MusicTransformer(vocab_size=VOCAB_SIZE, attention_type="relative")
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "relative_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    emb = model.token_embedding.embedding.weight.detach().numpy()

    # gather pitch tokens sorted by pitch value
    pitch_ids, pitch_values = [], []
    for token_str, tid in tokenizer.vocab.items():
        if token_str.startswith("Pitch_"):
            pitch_ids.append(tid)
            pitch_values.append(int(token_str.split("_")[1]))
    order = np.argsort(pitch_values)
    pitch_ids = np.array(pitch_ids)[order]
    pitch_values = np.array(pitch_values)[order]
    pitch_emb = emb[pitch_ids]                          # (n, d), sorted low->high

    # --- TEST 1: PCA explained variance ---
    pca = PCA(n_components=10).fit(pitch_emb)
    ev = pca.explained_variance_ratio_
    print("TEST 1 — PCA explained variance:")
    print(f"  first 2 dims capture: {ev[:2].sum()*100:.1f}%")
    print(f"  first 5 dims capture: {ev[:5].sum()*100:.1f}%")
    print(f"  per-dim: {[f'{v*100:.1f}%' for v in ev[:6]]}")

    # --- TEST 2: correlation of PC1 with pitch height ---
    pc = PCA(n_components=2).fit_transform(pitch_emb)
    corr1 = np.corrcoef(pc[:, 0], pitch_values)[0, 1]
    corr2 = np.corrcoef(pc[:, 1], pitch_values)[0, 1]
    print("\nTEST 2 — correlation of PCA dims with pitch height:")
    print(f"  PC1 vs pitch: r = {corr1:+.3f}")
    print(f"  PC2 vs pitch: r = {corr2:+.3f}")
    print("  (|r| near 1 = strong ordering; near 0 = none)")

    # --- TEST 3: cosine similarity vs pitch distance ---
    norm = pitch_emb / np.linalg.norm(pitch_emb, axis=1, keepdims=True)
    cos = norm @ norm.T                                 # (n, n) cosine similarity
    # average similarity for each pitch-distance
    n = len(pitch_values)
    dist_sim = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = abs(int(pitch_values[i]) - int(pitch_values[j]))
                dist_sim.setdefault(d, []).append(cos[i, j])
    print("\nTEST 3 — avg cosine similarity by pitch distance:")
    for d in [1, 2, 3, 5, 7, 12, 24]:
        if d in dist_sim:
            print(f"  distance {d:2d} semitones: {np.mean(dist_sim[d]):+.3f}")
    print("  (if close pitches are more similar, small distances > large distances)")

    # --- TEST 4: octave / harmonic similarity ---
    # do notes an octave apart (same pitch class) stay similar despite the distance?
    print("\nTEST 4 — harmonic structure (octaves = same pitch class):")

    # average similarity for octave multiples vs their neighbors
    for d in [11, 12, 13, 23, 24, 25]:
        if d in dist_sim:
            marker = "  <- octave" if d in (12, 24) else ""
            print(f"  distance {d:2d} semitones: {np.mean(dist_sim[d]):+.3f}{marker}")

    # direct test: average similarity of same-pitch-class pairs vs different-pitch-class
    same_pc, diff_pc = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                if pitch_values[i] % 12 == pitch_values[j] % 12:
                    same_pc.append(cos[i, j])
                else:
                    diff_pc.append(cos[i, j])
    print(f"\n  same pitch class (C-C, D-D...):  {np.mean(same_pc):+.3f}")
    print(f"  different pitch class:           {np.mean(diff_pc):+.3f}")
    print("  (if harmony is encoded, same-class > different-class)")

    # --- FIGURE: PCA colored by pitch class + similarity heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pitch_class = pitch_values % 12
    sc = axes[0].scatter(pc[:, 0], pc[:, 1], c=pitch_class, cmap="hsv", s=60)
    axes[0].set_title("PCA — colored by pitch class (C..B)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    cb = plt.colorbar(sc, ax=axes[0], ticks=range(12))
    cb.ax.set_yticklabels(PITCH_CLASSES); cb.set_label("pitch class")

    im = axes[1].imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Cosine similarity (pitches sorted low→high)")
    axes[1].set_xlabel("pitch idx"); axes[1].set_ylabel("pitch idx")
    plt.colorbar(im, ax=axes[1], label="cosine similarity")

    plt.tight_layout()
    out = PROJECT_ROOT / "figures" / "embedding_analysis.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out}")
    plt.show()
    # new figure: similarity vs distance (pitch height + harmony in one plot)
    fig_out = PROJECT_ROOT / "figures" / "similarity_vs_distance.png"
    fig_out.parent.mkdir(exist_ok=True)
    plot_similarity_vs_distance(dist_sim, fig_out)


if __name__ == "__main__":
    main()