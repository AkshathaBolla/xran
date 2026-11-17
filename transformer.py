"""
XRan: explainable ransomware / malware / benign detection with a custom transformer.

Two ways to read a run of behaviour events (chosen by `mode`, stored inside every saved model):
  sequence  sliding windows over the ordered events; the encoder uses learned position embeddings
  set       one window holding the DISTINCT known tokens in a fixed order; position embeddings are OFF,
            so the model is permutation-invariant (the order of tokens inside the window means nothing)

Maps to the diagram:
  (b) EventVocab + windowize()           -> tokenization + sliding window (W, S)
  (c) XRanTransformer.encode()           -> [CLS] + N x encoder blocks -> pooled z
  (d) detection_head / family_head       -> 3-class softmax / family classifier
      PrototypeStore                     -> novelty check (distance to family prototypes)
  (e) integrated_gradients()             -> token attributions on the embedding layer

Input format (one JSON object per sandbox run) you need to produce from your dataset:
  {"sha": "...", "label": 0|1|2, "family": "lockbit" | null,
   "events": ["API:CreateFileW", "DLL:kernel32.dll", "MUTEX:<generic>", "NET:dns", ...]}
  label: 0 = benign, 1 = malware, 2 = ransomware
"""
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
BENIGN, MALWARE, RANSOMWARE = 0, 1, 2


# ---------------------------------------------------------------- (b) preprocessing
class EventVocab:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.stoi = {PAD: 0, UNK: 1, CLS: 2}

    def fit(self, sequences):
        counts = Counter(e for seq in sequences for e in seq)
        for tok, c in counts.items():
            if c >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        return self

    def encode(self, events):
        return [self.stoi.get(e, 1) for e in events]

    def __len__(self):
        return len(self.stoi)


def windowize(ids, W=128, S=32):
    """Sliding windows over one run. Pads the last window. Reserves 1 slot for [CLS]."""
    L = W - 1
    if len(ids) <= L:
        return [ids + [0] * (L - len(ids))]
    out = []
    for start in range(0, max(1, len(ids) - L + S), S):
        w = ids[start:start + L]
        if len(w) < L:
            w = w + [0] * (L - len(w))
        out.append(w)
    return out


# ---------------------------------------------------------------- (c) model
class XRanTransformer(nn.Module):
    def __init__(self, vocab_size, n_families, d_model=128, n_heads=4,
                 n_layers=3, d_ff=256, max_len=128, dropout=0.1, use_positions=True):
        super().__init__()
        self.d_model = d_model
        self.use_positions = use_positions                  # False for set mode (order carries no meaning)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)       # learned positional encoding
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.detection_head = nn.Linear(d_model, 3)         # benign / malware / ransomware
        self.family_head = nn.Linear(d_model, n_families)   # used only for ransomware

    def embed(self, ids):
        """ids: (B, L) without [CLS]. Returns token embeddings (B, L, d)."""
        return self.tok_emb(ids)

    def encode_embeds(self, tok_e, pad_mask):
        """tok_e: (B, L, d) token embeddings, pad_mask: (B, L) True where PAD.
        Prepends [CLS], adds positions, runs the encoder, returns pooled z (B, d)."""
        B = tok_e.size(0)
        cls = self.tok_emb.weight[2].expand(B, 1, -1)       # <cls> id = 2
        x = torch.cat([cls, tok_e], dim=1)
        if self.use_positions:
            pos = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_emb(pos)[None]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], 1)
        h = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(h[:, 0])                            # pooled embedding z

    def forward(self, ids):
        pad_mask = ids.eq(0)
        z = self.encode_embeds(self.embed(ids), pad_mask)
        return self.detection_head(z), self.family_head(z), z


# ---------------------------------------------------------------- (d) novelty / prototypes
class PrototypeStore:
    """One prototype (mean embedding) per known family. Novel = far from all of them."""
    def __init__(self):
        self.protos = {}      # family_id -> (vector, count)
        self.threshold = None

    @torch.no_grad()
    def fit(self, z, family_ids):
        for f in family_ids.unique().tolist():
            self.protos[f] = (z[family_ids == f].mean(0), int((family_ids == f).sum()))

    @torch.no_grad()
    def update(self, z_new, family_id):
        """Incremental update from analyst-confirmed samples (running mean).
        Use a NEW family_id for a new family."""
        n_new = z_new.size(0)
        if family_id in self.protos:
            v, n = self.protos[family_id]
            self.protos[family_id] = ((v * n + z_new.sum(0)) / (n + n_new), n + n_new)
        else:
            self.protos[family_id] = (z_new.mean(0), n_new)

    @torch.no_grad()
    def distance(self, z):
        P = torch.stack([v for v, _ in self.protos.values()])      # (F, d)
        d = 1 - F.normalize(z, dim=-1) @ F.normalize(P, dim=-1).T  # cosine distance
        dmin, idx = d.min(1)
        ids = list(self.protos.keys())
        return dmin, [ids[i] for i in idx.tolist()]

    @torch.no_grad()
    def calibrate(self, z_val, q=0.95):
        """Set threshold from VALIDATION ransomware samples of known families.
        Better: pick it on a validation split with held-out families."""
        d, _ = self.distance(z_val)
        self.threshold = torch.quantile(d, q).item()

    @torch.no_grad()
    def is_novel(self, z):
        d, nearest = self.distance(z)
        return d > self.threshold, d, nearest


# ---------------------------------------------------------------- (e) explainability
def integrated_gradients(model, ids, target_class, steps=32):
    """IG on the embedding layer. ids: (1, L). Returns one attribution per token."""
    model.eval()
    pad_mask = ids.eq(0)
    emb = model.embed(ids).detach()
    base = torch.zeros_like(emb)                 # zero-embedding baseline
    total = torch.zeros_like(emb)
    for a in torch.linspace(0, 1, steps):
        x = (base + a * (emb - base)).requires_grad_(True)
        logits = model.detection_head(model.encode_embeds(x, pad_mask))
        grad, = torch.autograd.grad(logits[0, target_class], x)
        total += grad
    attr = ((emb - base) * total / steps).sum(-1)[0]    # (L,)
    return attr.masked_fill(pad_mask[0], 0.0)


def top_tokens(vocab, ids, attr, k=10):
    """Top-k tokens by attribution. Then map them to behavior groups
    (encryption, backup deletion, persistence...) with a hand-written dict."""
    order = attr.abs().argsort(descending=True)[:k]
    return [(vocab.itos[ids[0, i].item()], round(attr[i].item(), 4)) for i in order]


# ---------------------------------------------------------------- training
def multitask_loss(det_logits, fam_logits, y3, y_fam, w_fam=0.5):
    """Detection loss on every window; family loss only where label == ransomware."""
    loss = F.cross_entropy(det_logits, y3)
    m = (y3 == RANSOMWARE) & (y_fam >= 0)
    if m.any():
        loss = loss + w_fam * F.cross_entropy(fam_logits[m], y_fam[m])
    return loss


@torch.no_grad()
def score_run(model, windows, agg="topk", k=3):
    """Real-time scoring: run each window, aggregate window probabilities.
    Early windows of ransomware may look benign, so 'mean' dilutes the signal;
    top-k mean of the ransomware/malware probability is a safer default."""
    model.eval()
    p = F.softmax(model(windows)[0], dim=-1)             # (n_windows, 3)
    if agg == "mean":
        return p.mean(0)
    kk = min(k, p.size(0))
    return p.topk(kk, dim=0).values.mean(0)


# ---------------------------------------------------------------- smoke test (synthetic)
def _synthetic(n=300, n_fam=4):
    """Fake runs where each class has its own 'signature' tokens. Replace with real data."""
    base = [f"API:Call{i}" for i in range(40)]
    sig = {0: ["API:RegQueryValue"], 1: ["API:WriteProcessMemory", "NET:http"],
           2: ["API:CryptEncrypt", "FILE:rename_.locked", "API:DeleteShadowCopy"]}
    runs = []
    for _ in range(n):
        y = random.choice([0, 1, 2])
        fam = random.randrange(n_fam) if y == 2 else -1
        ev = [random.choice(base) for _ in range(random.randint(100, 400))]
        for s in sig[y] + ([f"MUTEX:fam{fam}"] if y == 2 else []):
            for _ in range(5):
                ev.insert(random.randrange(len(ev)), s)
        runs.append((ev, y, fam))
    return runs


if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0)
    runs = _synthetic()
    vocab = EventVocab(min_freq=2).fit([r[0] for r in runs])
    X, Y3, YF = [], [], []
    for ev, y, fam in runs:                      # NOTE: split by run/family BEFORE windowing
        for w in windowize(vocab.encode(ev), W=128, S=64):
            X.append(w); Y3.append(y); YF.append(fam)
    X, Y3, YF = torch.tensor(X), torch.tensor(Y3), torch.tensor(YF)

    model = XRanTransformer(len(vocab), n_families=4, max_len=128)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for epoch in range(5):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            det, fam, _ = model(X[b])
            loss = multitask_loss(det, fam, Y3[b], YF[b])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch}  loss {loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        det, _, z = model(X)
    print("window accuracy:", (det.argmax(1) == Y3).float().mean().item())

    store = PrototypeStore()
    r = Y3 == RANSOMWARE
    store.fit(z[r], YF[r]); store.calibrate(z[r])
    print("novelty threshold:", round(store.threshold, 4))

    i = (Y3 == RANSOMWARE).nonzero()[0].item()
    attr = integrated_gradients(model, X[i:i + 1], target_class=RANSOMWARE)
    print("top tokens:", top_tokens(vocab, X[i:i + 1], attr, k=5))
