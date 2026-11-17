"""Sanity tests for transformer.py and the set-mode encoder. Run: python test.py"""
import random
import sys

import torch
import torch.nn.functional as F

from transformer import (XRanTransformer, PrototypeStore, EventVocab, windowize,
                         multitask_loss, integrated_gradients, score_run)
from deploy import encode_events

torch.manual_seed(0)
random.seed(0)
results = []


def check(name, ok, info=""):
    print(("PASS" if ok else "FAIL"), "-", name, info)
    results.append(bool(ok))


# 1. windowing --------------------------------------------------------------
w = windowize(list(range(3, 503)), W=128, S=32)
check("windowize: every window has length 127", all(len(x) == 127 for x in w), f"(n_windows={len(w)})")
short = windowize([3, 4, 5], W=128, S=32)
check("windowize: short run gives one padded window",
      len(short) == 1 and short[0][:3] == [3, 4, 5] and set(short[0][3:]) == {0})

# 2. forward pass -----------------------------------------------------------
model = XRanTransformer(vocab_size=50, n_families=4, d_model=64, n_heads=4,
                        n_layers=2, d_ff=128, max_len=128, dropout=0.0)
x = torch.randint(3, 50, (8, 127))
x[:, 100:] = 0                                   # simulate padding
det, fam, z = model(x)
check("forward: output shapes", det.shape == (8, 3) and fam.shape == (8, 4) and z.shape == (8, 64),
      f"(det={tuple(det.shape)}, fam={tuple(fam.shape)}, z={tuple(z.shape)})")
check("forward: no NaN/Inf with padding", all(torch.isfinite(t).all() for t in (det, fam, z)))

# 3. loss masking -----------------------------------------------------------
y_none = torch.zeros(8, dtype=torch.long)
yf_none = torch.full((8,), -1)
check("loss: family term ignored when no ransomware in batch",
      torch.allclose(multitask_loss(det, fam, y_none, yf_none), F.cross_entropy(det, y_none)))
y_r = torch.full((8,), 2)
yf_r = torch.randint(0, 4, (8,))
check("loss: family term added for ransomware windows",
      multitask_loss(det, fam, y_r, yf_r).item() > F.cross_entropy(det, y_r).item())

# 4. can it overfit a tiny batch? (catches wiring bugs) ----------------------
xb = torch.randint(3, 50, (16, 127))
yb = torch.randint(0, 3, (16,))
yfb = torch.where(yb == 2, torch.randint(0, 4, (16,)), torch.full((16,), -1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
first = None
for step in range(200):
    d, f, _ = model(xb)
    loss = multitask_loss(d, f, yb, yfb)
    first = loss.item() if first is None else first
    opt.zero_grad(); loss.backward(); opt.step()
check("overfit: loss falls by at least half on a 16-sample batch", loss.item() < 0.5 * first,
      f"(start={first:.3f}, end={loss.item():.3f})")

# 5. Integrated Gradients completeness --------------------------------------
model.eval()
xi = xb[:1].clone()
attr = integrated_gradients(model, xi, target_class=2, steps=64)
pad = xi.eq(0)
with torch.no_grad():
    emb = model.embed(xi)
    f_x = model.detection_head(model.encode_embeds(emb, pad))[0, 2]
    f_b = model.detection_head(model.encode_embeds(torch.zeros_like(emb), pad))[0, 2]
gap = abs(attr.sum().item() - (f_x - f_b).item())
tol = 0.1 * max(1.0, abs((f_x - f_b).item()))
check("IG: attributions sum to F(x) - F(baseline) (completeness)", gap < tol,
      f"(gap={gap:.4f}, tol={tol:.4f}; if it fails, raise steps)")

# 6. prototype store: novelty logic -----------------------------------------
centers = torch.randn(4, 64) * 5
fam_ids = torch.arange(4).repeat_interleave(10)
z_known = centers[fam_ids] + 0.1 * torch.randn(40, 64)
store = PrototypeStore()
store.fit(z_known, fam_ids)
store.calibrate(z_known, q=0.95)
novel, _, _ = store.is_novel(z_known)
check("novelty: known-family samples are mostly not flagged", novel.float().mean().item() <= 0.1,
      f"(flagged {novel.float().mean().item():.0%})")
new_center = torch.randn(64) * 5
z_new = new_center + 0.1 * torch.randn(10, 64)
novel, _, _ = store.is_novel(z_new)
check("novelty: samples from an unseen cluster are flagged", novel.float().mean().item() >= 0.9,
      f"(flagged {novel.float().mean().item():.0%})")
store.update(z_new, family_id=99)               # analyst confirms new family
novel, _, nearest = store.is_novel(z_new + 0.0)
check("incremental update: new family is no longer novel and is the nearest prototype",
      novel.float().mean().item() <= 0.2 and all(n == 99 for n in nearest),
      f"(flagged {novel.float().mean().item():.0%})")

# 7. real-time scoring ------------------------------------------------------
p = score_run(model, torch.randint(3, 50, (5, 127)))
check("score_run: returns 3 probabilities in [0, 1]", p.shape == (3,) and (p >= 0).all() and (p <= 1).all())

# 8. set mode: encoding ignores order and repeats ----------------------------
vocab = EventVocab(min_freq=1).fit([["API:a", "API:b", "API:c", "API:d"]])
w1, _ = encode_events(vocab, ["API:a", "API:b", "API:b", "API:c"], W=16, S=8, mode="set")
w2, _ = encode_events(vocab, ["API:c", "API:b", "API:a", "API:a", "API:zzz"], W=16, S=8, mode="set")
check("set mode: same behaviours in any order/with repeats give the same window; unknown tokens are ignored",
      w1 == w2 and len(w1) == 1 and len(w1[0]) == 15)

# 9. set-mode model really is permutation-invariant ---------------------------
sm = XRanTransformer(vocab_size=30, n_families=3, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                     max_len=16, dropout=0.0, use_positions=False).eval()
a = torch.tensor([[5, 9, 12, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
b = torch.tensor([[20, 5, 12, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
with torch.no_grad():
    check("set-mode model: shuffling the tokens does not change the output",
          torch.allclose(sm(a)[0], sm(b)[0], atol=1e-5))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
