"""
Training (M2). Full training loop over the corpus with validation,
LR scheduling (warmup + cosine decay), checkpointing and TensorBoard logging.
The overfit_one_chunk sanity check is kept for future debugging.
"""
from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import partial

from src.dataset import MidiChunkDataset, collate_fn
from src.model import MusicTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- config ---
VOCAB_SIZE = 251
PAD_ID = 0
CONTEXT_LEN = 512
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42                          # <-- add: same seed for both models (fair ablation)
ATTENTION_TYPE = "relative"        # <-- the ONE switch: "absolute" (M2) or "relative" (M3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# outputs derived from ATTENTION_TYPE so the two runs never overwrite each other
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_NAME = f"{ATTENTION_TYPE}_best.pt"
LOG_DIR = PROJECT_ROOT / "runs" / f"m3_{ATTENTION_TYPE}" if ATTENTION_TYPE == "relative" \
    else PROJECT_ROOT / "runs" / f"m2_{ATTENTION_TYPE}"

def overfit_one_chunk(steps=800, lr=3e-4):
    # one chunk, no cropping (we want the SAME example every step)
    ds = MidiChunkDataset(PROJECT_ROOT / "data" / "chunks_train.pkl",
                          context_len=512, pad_id=PAD_ID, crop=False)
    x, y = ds[0]
    x = x.unsqueeze(0).to(DEVICE)     # (1, T) -> add batch dim
    y = y.unsqueeze(0).to(DEVICE)
    print(f"Chunk length: {x.size(1)} tokens")

    model = MusicTransformer(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, dropout=0.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    model.train()
    for step in range(steps):
        logits = model(x)                        # (1, T, vocab)

        # flatten for cross-entropy: (T, vocab) vs (T,)
        loss = loss_fn(
            logits.view(-1, VOCAB_SIZE),         # (1*T, vocab)
            y.view(-1),                          # (1*T,)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")


def make_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup for warmup_steps, then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)          # 0 -> 1 linearly
        # cosine decay from 1 -> 0 over the remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    """Validation loss — measurement only, no backprop."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        # weight by number of non-pad tokens so the average is correct
        n_tokens = (y != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(1, total_tokens)


def train():
    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(exist_ok=True)

    train_ds = MidiChunkDataset(PROJECT_ROOT / "data" / "chunks_train.pkl",
                                context_len=CONTEXT_LEN, pad_id=PAD_ID, crop=True)
    val_ds = MidiChunkDataset(PROJECT_ROOT / "data" / "chunks_val.pkl",
                              context_len=CONTEXT_LEN, pad_id=PAD_ID, crop=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=partial(collate_fn, pad_id=PAD_ID),
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=partial(collate_fn, pad_id=PAD_ID),
                            num_workers=4, pin_memory=True)

    model = MusicTransformer(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, dropout=0.1, attention_type=ATTENTION_TYPE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    total_steps = EPOCHS * len(train_loader)
    scheduler = make_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    writer = SummaryWriter(LOG_DIR)

    print(f"Attention: {ATTENTION_TYPE} | Device: {DEVICE} | "
          f"train batches/epoch: {len(train_loader)} | total steps: {total_steps}")

    global_step = 0
    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        val_loss = evaluate(model, val_loader, loss_fn)
        val_ppl = math.exp(val_loss)
        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/perplexity", val_ppl, global_step)
        print(f"epoch {epoch} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "attention_type": ATTENTION_TYPE},  # <-- record which model
                       CKPT_DIR / CKPT_NAME)
            print(f"  saved new best (val_loss {val_loss:.4f})")

    writer.close()


if __name__ == "__main__":
    train()