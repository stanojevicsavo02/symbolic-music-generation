"""
Transformer language model for symbolic music.
Decoder-only, causal self-attention, switchable absolute (M2) or relative (M3) position.
Built bottom-up: embeddings -> positions -> attention blocks -> output head.
"""
import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Maps token IDs (int64) to learned vectors of size d_model."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len) of int64 token IDs
        # scale by sqrt(d_model): standard trick so embedding magnitude
        # matches the positional encoding we add next
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Adds fixed (non-learned) sinusoidal position information to embeddings."""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        # precompute a (max_len, d_model) table of sinusoids, once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                          # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        # register as buffer: part of the model, moves to GPU, but NOT trained
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention (absolute-position baseline, M2)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads          # dimension per head

        # one linear layer produces Q, K, V for all heads at once (3 * d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)   # output projection after heads merge
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape                         # batch, seq_len, d_model

        # project to Q, K, V and split into heads
        qkv = self.qkv(x)                         # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)            # each (B, T, C)

        # reshape (B, T, C) -> (B, n_heads, T, d_head): each head works independently
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attention scores: how much each position attends to every other
        # (B, nh, T, d_head) @ (B, nh, d_head, T) -> (B, nh, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal mask: position i may not look at positions > i (the future)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))

        # softmax over the last dim -> attention weights that sum to 1
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum of values
        out = attn @ v                            # (B, nh, T, d_head)

        # merge heads back: (B, nh, T, d_head) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class RelativeMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with relative position (Music Transformer, 2018)."""
    def __init__(self, d_model, n_heads, max_len=1024, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # learned relative position embeddings: one vector per possible distance.
        # for causal attention we only need distances 0..max_len-1 (looking back).
        # shape: (n_heads, max_len, d_head) — each head learns its own table.
        self.rel_emb = nn.Parameter(torch.randn(n_heads, max_len, self.d_head) * 0.02)

    def _skew(self, qe):
        """
        Skewing (Huang et al. 2018). qe rows hold q_i · rel over distances
        ordered T-1..0. Pad + reshape shifts each row into (i, j) alignment.
        """
        B, H, T, _ = qe.shape
        qe = torch.nn.functional.pad(qe, (1, 0))        # (B,H,T,T+1)
        qe = qe.reshape(B, H, T + 1, T)                 # shift
        qe = qe[:, :, 1:, :]                            # (B,H,T,T)
        return qe

    def forward(self, x):
        B, T, C = x.shape

        assert T <= self.max_len, f"seq_len {T} exceeds max_len {self.max_len}"

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # (B, nh, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # content term: q · k  (same as absolute attention)
        content = q @ k.transpose(-2, -1)                             # (B, nh, T, T)

        # relative term: q · r
        # take the last T relative vectors (distances 0..T-1), per head
        rel = self.rel_emb[:, :T, :]  # (nh, T, d_head), distance 0..T-1
        rel = torch.flip(rel, dims=[1])  # now ordered distance T-1..0
        qe = torch.einsum("bhid,hjd->bhij", q, rel)  # (B, nh, T, T)
        rel_scores = self._skew(qe)                                  # align to (i, j)

        scores = (content + rel_scores) / math.sqrt(self.d_head)

        # causal mask (identical to absolute version)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward: expand -> nonlinearity -> contract."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # expand (typically d_ff = 4 * d_model)
            nn.GELU(),                   # nonlinearity
            nn.Linear(d_ff, d_model),    # contract back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One decoder block: attention + feed-forward, each with residual + pre-norm."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 attention_type="absolute", max_len=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if attention_type == "absolute":
            self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        elif attention_type == "relative":
            self.attn = RelativeMultiHeadSelfAttention(d_model, n_heads, max_len, dropout)
        else:
            raise ValueError(f"unknown attention_type: {attention_type}")
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # pre-norm: normalize BEFORE the sublayer, then add residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """Decoder-only Transformer LM for symbolic music. Absolute or relative attention."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, max_len=1024, dropout=0.1, pad_id=0,
                 attention_type="absolute"):
        super().__init__()
        self.attention_type = attention_type
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # absolute positional encoding is only used in absolute mode;
        # in relative mode, position lives inside attention (no input PE)
        self.pos_encoding = PositionalEncoding(d_model, max_len) \
            if attention_type == "absolute" else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             attention_type=attention_type, max_len=max_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id

    def forward(self, x):
        # x: (B, T) token IDs
        x = self.token_embedding(x)          # (B, T, d_model)
        if self.pos_encoding is not None:    # absolute mode only
            x = self.pos_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.head(x)                # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    vocab_size = 251
    x = torch.randint(0, vocab_size, (2, 100))
    expected = (2, 100, vocab_size)

    for attn_type in ["absolute", "relative"]:
        model = MusicTransformer(vocab_size=vocab_size, attention_type=attn_type)
        n_params = sum(p.numel() for p in model.parameters())
        logits = model(x)
        print(f"[{attn_type}] params: {n_params:,} | "
              f"output {tuple(logits.shape)} | match: {tuple(logits.shape) == expected}")