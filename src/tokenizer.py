from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

DATA_DIR = Path("data/pop1k7")

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
}

def build_tokenizer():
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    return REMI(config)