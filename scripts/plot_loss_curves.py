"""Plot validation perplexity for both models from TensorBoard logs."""
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_scalar(run_dir, tag):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def main():
    runs = {
        "Absolute": PROJECT_ROOT / "runs" / "m2_absolute",
        "Relative": PROJECT_ROOT / "runs" / "m3_relative",
    }
    colors = {"Absolute": "#c1272d", "Relative": "#2a6f97"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, run_dir in runs.items():
        try:
            steps, vals = load_scalar(run_dir, "val/perplexity")
            ax.plot(range(len(vals)), vals, "-o", markersize=3,
                    label=name, color=colors[name])
        except Exception as e:
            print(f"Could not load {name}: {e}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Validation perplexity: relative vs absolute attention")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = PROJECT_ROOT / "figures" / "loss_curves.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()

if __name__ == "__main__":
    main()