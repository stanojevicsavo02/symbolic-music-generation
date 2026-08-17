from pathlib import Path
from symusic import Score
from src.tokenizer import build_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "pop1k7"

tokenizer = build_tokenizer()

midi_files = sorted(DATA_DIR.rglob("*.mid"))
print(f"Broj MIDI fajlova: {len(midi_files)}")

score = Score(midi_files[0])
print(f"\nFajl: {midi_files[0].name}")
print(f"Ticks per quarter: {score.ticks_per_quarter}")
print(f"Tempo promene: {len(score.tempos)}  -> prvi: {score.tempos[0] if score.tempos else None}")
print(f"Time signature: {score.time_signatures[0] if score.time_signatures else None}")
print(f"Broj trakova: {len(score.tracks)}")

notes = score.tracks[0].notes
print(f"Broj nota: {len(notes)}")
print(f"Pitch opseg: {min(n.pitch for n in notes)} - {max(n.pitch for n in notes)}")

tokens = tokenizer(score)
seq = tokens[0]
print(f"\nDuzina token sekvence: {len(seq.ids)}")
print(f"Velicina vokabulara: {len(tokenizer)}")
print("\nPrvih 30 tokena:")
for tok in seq.tokens[:30]:
    print("  ", tok)