"""
Verify the skewing trick produces the same result as a naive, obviously-correct
relative-score computation. If they match, skewing is trustworthy.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

B, H, T, D = 1, 1, 4, 8      # tiny: 1 batch, 1 head, 4 positions, head-dim 8

q = torch.randn(B, H, T, D)
rel = torch.randn(H, T, D)   # rel[h, dist, :] = vector for distance `dist`


def naive_rel_scores(q, rel):
    """
    Obviously-correct version: for each (i, j), distance d = i - j (j <= i),
    score = q_i · rel[d]. Positions j > i (future) are left as 0 (masked later).
    """
    B, H, T, D = q.shape
    out = torch.zeros(B, H, T, T)
    for i in range(T):
        for j in range(T):
            d = i - j
            if d >= 0:                       # only look back (causal)
                out[:, :, i, j] = (q[:, :, i, :] * rel[:, d, :]).sum(-1)
    return out


def skew(qe):
    """The skewing trick from the model."""
    B, H, T, _ = qe.shape
    qe = F.pad(qe, (1, 0))
    qe = qe.reshape(B, H, T + 1, T)
    qe = qe[:, :, 1:, :]
    return qe


# naive path FIRST — uses the ORIGINAL rel (this is the golden standard)
naive = naive_rel_scores(q, rel)

# skew path: reverse distance order into a SEPARATE variable, don't touch rel
rel_flipped = torch.flip(rel, dims=[1])       # reverse distance order: T-1..0
qe = torch.einsum("bhid,hjd->bhij", q, rel_flipped)   # (B, H, T, T)
skewed = skew(qe)

# compare only the causal (lower-triangular) part — that's what attention uses
mask = torch.tril(torch.ones(T, T)).bool()
skewed_masked = skewed.masked_fill(~mask, 0.0)
naive_masked = naive.masked_fill(~mask, 0.0)

print("Max difference (causal part):", (skewed_masked - naive_masked).abs().max().item())
print("Match?", torch.allclose(skewed_masked, naive_masked, atol=1e-5))

# DIAGNOSTIC: print both matrices for the tiny case to see the misalignment
print("\nqe (before skew), indexed (i, distance):")
print(qe[0, 0].round(decimals=2))

print("\nskewed (i, j) — what the trick produces:")
print(skewed[0, 0].round(decimals=2))

print("\nnaive (i, j) — the correct target:")
print(naive[0, 0].round(decimals=2))

print("\ncausal mask:")
print(torch.tril(torch.ones(T, T)))