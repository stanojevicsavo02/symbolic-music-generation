"""
Autoregressive generation (M4): sample token-by-token from a trained model,
then detokenize to MIDI. Works for both absolute and relative checkpoints.
"""
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

from src.model import MusicTransformer
from src.tokenizer import build_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 251
PAD_ID = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def generate(model, tokenizer, max_new_tokens=512, temperature=1.0, top_k=None,
             top_p=None, context_len=512):
    model.eval()

    bos_id = tokenizer.vocab["BOS_None"]
    eos_id = tokenizer.vocab.get("EOS_None", None)

    # start with just the BOS token
    seq = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)   # (1, 1)

    for _ in range(max_new_tokens):
        # feed only the last context_len tokens (model can't see more)
        seq_cond = seq[:, -context_len:]
        logits = model(seq_cond)                 # (1, T, vocab)
        logits = logits[:, -1, :]                # last position only -> (1, vocab)

        # temperature: <1 sharpens (safer), >1 flattens (wilder)
        logits = logits / temperature

        # optional top-k: keep only the k most likely tokens
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        # optional top-p (nucleus): keep smallest set whose prob mass >= p
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()   # keep at least one
            remove[:, 0] = False
            idx_to_remove = remove.scatter(1, sorted_idx, remove)
            logits[idx_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)

        seq = torch.cat([seq, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return seq[0].tolist()


def load_model(ckpt_path, attention_type):
    model = MusicTransformer(vocab_size=VOCAB_SIZE, pad_id=PAD_ID,
                             attention_type=attention_type).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {attention_type} model (epoch {ckpt['epoch']}, "
          f"val_loss {ckpt['val_loss']:.4f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", default="relative", choices=["absolute", "relative"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default="generated.mid")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tempo", type=int, default=100)
    args = parser.parse_args()

    ckpt_path = args.ckpt or (PROJECT_ROOT / "checkpoints" / f"{args.attention}_best.pt")

    tokenizer = build_tokenizer()
    model = load_model(ckpt_path, args.attention)

    token_ids = generate(model, tokenizer,
                         max_new_tokens=args.tokens,
                         temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p)
    print(f"Generated {len(token_ids)} tokens")

    # detokenize -> MIDI
    from miditok import TokSequence
    tokseq = TokSequence(ids=token_ids)
    tokenizer.complete_sequence(tokseq)  # fill in .tokens from .ids
    score = tokenizer.decode([tokseq])  # decode expects a list of sequences (one per track)

    # set a fixed tempo for listening (model has no tempo tokens)
    from symusic import Tempo
    score.tempos = [Tempo(time=0, qpm=args.tempo)]
    out_path = PROJECT_ROOT / args.out
    score.dump_midi(out_path)
    print(f"Saved MIDI to {out_path}")


if __name__ == "__main__":
    main()