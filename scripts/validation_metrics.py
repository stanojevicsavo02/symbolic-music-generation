"""
M5 validation metrics: do generated pieces statistically resemble real music?
Three length-normalized metrics on (a) real Pop1k7, (b) generated from each model:
  - pitch-class entropy (tonal variety)
  - scale consistency (% notes in best-fit major scale)
  - note density (notes per bar)
Compared as distributions (mean ± std), not single values.
"""
from pathlib import Path
import numpy as np
import torch
from symusic import Score
from tqdm import tqdm

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer
from src.generate import generate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# major scale intervals (semitones from root)
MAJOR = [0, 2, 4, 5, 7, 9, 11]


def pitch_class_entropy(pitches):
    """Shannon entropy over the 12 pitch classes. Length-independent."""
    if len(pitches) == 0:
        return 0.0
    pc = np.array([p % 12 for p in pitches])
    hist = np.bincount(pc, minlength=12).astype(float)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())     # 0..log2(12)=3.58


def scale_consistency(pitches):
    """Fraction of notes falling in the best-fit major scale. Length-independent."""
    if len(pitches) == 0:
        return 0.0
    pc = np.array([p % 12 for p in pitches])
    best = 0.0
    for root in range(12):                            # try all 12 major scales
        scale = {(root + i) % 12 for i in MAJOR}
        frac = np.mean([p in scale for p in pc])
        best = max(best, frac)
    return float(best)


def note_density(score):
    """Notes per bar. Length-normalized by construction."""
    notes = score.tracks[0].notes if score.tracks else []
    if len(notes) == 0:
        return 0.0
    # estimate bars from total time / ticks-per-bar (4/4: 4 beats per bar)
    tpq = score.ticks_per_quarter
    ticks_per_bar = tpq * 4
    end = max(n.time + n.duration for n in notes)
    n_bars = max(1, end / ticks_per_bar)
    return len(notes) / n_bars


def pitches_from_score(score):
    notes = score.tracks[0].notes if score.tracks else []
    return [n.pitch for n in notes]


def metrics_for_score(score):
    pitches = pitches_from_score(score)
    return {
        "pc_entropy": pitch_class_entropy(pitches),
        "scale_consistency": scale_consistency(pitches),
        "note_density": note_density(score),
    }


def summarize(name, metric_dicts):
    print(f"\n{name} (n={len(metric_dicts)}):")
    for key in ["pc_entropy", "scale_consistency", "note_density"]:
        vals = [m[key] for m in metric_dicts]
        print(f"  {key:20s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


def collect_real(tokenizer, n_samples=100):
    """Metrics on a sample of real Pop1k7 songs."""
    midi_files = sorted((PROJECT_ROOT / "data" / "pop1k7").rglob("*.mid"))
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(midi_files), size=min(n_samples, len(midi_files)), replace=False)
    out = []
    for idx in tqdm(chosen, desc="real"):
        try:
            score = Score(midi_files[idx])
            out.append(metrics_for_score(score))
        except Exception:
            continue
    return out


def collect_generated(model, tokenizer, n_samples=100, tokens=512):
    """Metrics on pieces generated from a model."""
    out = []
    for _ in tqdm(range(n_samples), desc="generated"):
        ids = generate(model, tokenizer, max_new_tokens=tokens, top_p=0.95)
        from miditok import TokSequence
        tokseq = TokSequence(ids=ids)
        tokenizer.complete_sequence(tokseq)
        score = tokenizer.decode([tokseq])
        out.append(metrics_for_score(score))
    return out


def load_model(attention_type, ckpt_name):
    model = MusicTransformer(vocab_size=VOCAB_SIZE, attention_type=attention_type).to(DEVICE)
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    tokenizer = build_tokenizer()

    real = collect_real(tokenizer, n_samples=100)
    summarize("REAL Pop1k7", real)

    abs_model = load_model("absolute", "absolute_best.pt")
    gen_abs = collect_generated(abs_model, tokenizer, n_samples=100)
    summarize("GENERATED (absolute)", gen_abs)

    rel_model = load_model("relative", "relative_best.pt")
    gen_rel = collect_generated(rel_model, tokenizer, n_samples=100)
    summarize("GENERATED (relative)", gen_rel)


if __name__ == "__main__":
    main()