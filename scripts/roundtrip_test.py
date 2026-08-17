from pathlib import Path
from symusic import Score
from src.tokenizer import build_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "pop1k7"
OUT = PROJECT_ROOT / "roundtrip_out.mid"

tokenizer = build_tokenizer()
midi_files = sorted(DATA_DIR.rglob("*.mid"))

original = Score(midi_files[0])
tokens = tokenizer(original)
reconstructed = tokenizer(tokens)      # detokenizacija: tokeni -> Score
reconstructed.dump_midi(OUT)

orig_notes = original.tracks[0].notes
rec_notes = reconstructed.tracks[0].notes

print(f"Original:      {len(orig_notes)} nota")
print(f"Rekonstruisan: {len(rec_notes)} nota")

for i in range(5):
    o, r = orig_notes[i], rec_notes[i]
    print(f"{i}: pitch {o.pitch}->{r.pitch} | start {o.time}->{r.time} | dur {o.duration}->{r.duration} | vel {o.velocity}->{r.velocity}")