"""
M5: visualize learned token embeddings. Extract the embedding table from a
trained model, project pitch-token vectors to 2D, and check if they order by pitch.
Ordering = the model learned musical structure (IDs are arbitrary, so it can only
come from training).
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251


def main():
    tokenizer = build_tokenizer()

    # load the relative model (our best)
    model = MusicTransformer(vocab_size=VOCAB_SIZE, attention_type="relative")
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "relative_best.pt",
                      map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # extract the embedding table: (vocab_size, d_model)
    emb = model.token_embedding.embedding.weight.detach().numpy()
    print(f"Embedding table shape: {emb.shape}")

    # find the Pitch tokens and their pitch values
    pitch_ids, pitch_values = [], []
    for token_str, tid in tokenizer.vocab.items():
        if token_str.startswith("Pitch_"):
            pitch_ids.append(tid)
            pitch_values.append(int(token_str.split("_")[1]))
    pitch_ids = np.array(pitch_ids)
    pitch_values = np.array(pitch_values)
    print(f"Found {len(pitch_ids)} Pitch tokens, range {pitch_values.min()}-{pitch_values.max()}")

    pitch_emb = emb[pitch_ids]                      # (n_pitches, d_model)

    # two projections
    pca = PCA(n_components=2).fit_transform(pitch_emb)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(pitch_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, proj, title in [(axes[0], pca, "PCA"), (axes[1], tsne, "t-SNE")]:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=pitch_values,
                        cmap="viridis", s=60)
        ax.set_title(f"Pitch embeddings — {title}")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        plt.colorbar(sc, ax=ax, label="MIDI pitch (low → high)")

    plt.tight_layout()
    out = PROJECT_ROOT / "figures" / "pitch_embeddings.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out}")
    plt.show()


if __name__ == "__main__":
    main()