"""
train.py - train the XRan transformer on a real dataset and save a checkpoint for deploy.py.

Input: a JSONL file, one run per line:
  {"sha": "...", "label": 0|1|2, "family": "lockbit" or null, "events": ["API:CreateFileW", ...],
   "first_seen": "2023-05-01"   <- only needed for --split time}
  label: 0 = benign, 1 = malware, 2 = ransomware

Examples:
  python train.py --make-demo-jsonl demo_runs.jsonl                         # synthetic data, to test this script
  python train.py --data runs.jsonl --out xran.pt
  python train.py --data runs.jsonl --holdout-families 3 --out xran.pt     # hold out 3 families as "novel"
  python train.py --data runs.jsonl --split time --out xran.pt              # train on old, test on new

Leak-free by construction: splits are by RUN (never by window), the vocabulary is fitted on training
runs only, and held-out families never appear in training.
"""
import argparse
import copy
import json
import random
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from deploy import normalize_event, save_checkpoint, LABELS, RANSOMWARE, encode_events

try:
    from transformer import EventVocab, XRanTransformer, PrototypeStore, windowize
except ImportError:
    from xran_transformer import EventVocab, XRanTransformer, PrototypeStore, windowize


# ---------------------------------------------------------------- data
def load(path):
    runs, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("events") or r.get("label") not in (0, 1, 2):
                dropped += 1
                continue
            r.setdefault("family", None)
            runs.append(r)
    if dropped:
        print(f"warning: dropped {dropped} runs with empty events or a label outside 0/1/2")
    return runs


def split_runs(runs, mode, n_holdout, seed):
    rng = random.Random(seed)
    held = set()
    if n_holdout:
        fams = sorted({r["family"] for r in runs if r["label"] == RANSOMWARE and r["family"]})
        rng.shuffle(fams)
        held = set(fams[:n_holdout])
    novel = [r for r in runs if r["label"] == RANSOMWARE and r["family"] in held]
    pool = [r for r in runs if not (r["label"] == RANSOMWARE and r["family"] in held)]

    if mode == "time":
        if any("first_seen" not in r for r in pool):
            raise SystemExit("--split time needs a 'first_seen' field on every run")
        pool.sort(key=lambda r: str(r["first_seen"]))
        a, b = int(0.70 * len(pool)), int(0.85 * len(pool))
        train, val, test = pool[:a], pool[a:b], pool[b:]
    else:
        groups = defaultdict(list)                       # stratify by class (and family for ransomware)
        for r in pool:
            groups[(r["label"], r["family"] if r["label"] == RANSOMWARE else None)].append(r)
        train, val, test = [], [], []
        for g in groups.values():
            rng.shuffle(g)
            if len(g) < 3:                               # tiny groups: keep them all in training
                train += g
                continue
            a, b = int(0.70 * len(g)), int(0.85 * len(g))
            train += g[:a]; val += g[a:b]; test += g[b:]
    train_fams = {r["family"] for r in train if r["label"] == RANSOMWARE and r["family"]}
    # ransomware in the test split from families never seen in training also counts as "novel"
    late_novel = [r for r in test if r["label"] == RANSOMWARE and r["family"] and r["family"] not in train_fams]
    late_ids = {id(r) for r in late_novel}
    test = [r for r in test if id(r) not in late_ids]
    return train, val, test, novel + late_novel


def cut(r, n):
    return r["events"][:n]


def make_windows(runs, vocab, fam2id, W, S, max_w, max_events, mode="sequence"):
    X, Y, YF = [], [], []
    for r in runs:
        ws, _ = encode_events(vocab, cut(r, max_events), W, S, mode)
        if len(ws) > max_w:                              # keep evenly spaced windows so long runs don't dominate
            ws = [ws[int(i * (len(ws) - 1) / (max_w - 1))] for i in range(max_w)]
        fid = fam2id.get(r["family"], -1) if r["label"] == RANSOMWARE else -1
        for w in ws:
            X.append(w); Y.append(r["label"]); YF.append(fid)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(YF)


# ---------------------------------------------------------------- scoring / metrics
@torch.no_grad()
def run_outputs(model, vocab, events, W, S, device, k=3, mode="sequence"):
    wins_list, _ = encode_events(vocab, events, W, S, mode)
    wins = torch.tensor(wins_list).to(device)
    det, fam, z = model(wins)
    p = F.softmax(det, dim=-1)
    agg = p.topk(min(k, p.size(0)), dim=0).values.mean(0)
    peak = int(p[:, RANSOMWARE].argmax())
    return agg.cpu(), fam[peak].cpu(), z[peak:peak + 1].cpu()


def evaluate(model, vocab, runs, fam2id, args, device, trunc=None):
    out = []
    n = trunc or args.max_events
    if args.mode == "set":                                # one window per run, so whole batches can go through at once
        wins = [encode_events(vocab, r["events"][:n], args.W, args.S, "set")[0][0] for r in runs]
        with torch.no_grad():
            for i in range(0, len(wins), 512):
                det, fam, z = model(torch.tensor(wins[i:i + 512]).to(device))
                p = F.softmax(det, dim=-1).cpu()
                for j in range(p.size(0)):
                    r = runs[i + j]
                    out.append({"y": r["label"], "pred": int(p[j].argmax()), "fam": fam2id.get(r["family"], -1),
                                "fam_pred": int(fam[j].argmax()), "z": z[j:j + 1].cpu(),
                                "source": r.get("source", "?")})
        return out
    for r in runs:
        agg, fam_logits, z = run_outputs(model, vocab, r["events"][:n], args.W, args.S, device, mode=args.mode)
        out.append({"y": r["label"], "pred": int(agg.argmax()), "fam": fam2id.get(r["family"], -1),
                    "fam_pred": int(fam_logits.argmax()), "z": z,
                    "source": r.get("source", "?")})
    return out


def prf(res, verbose=False):
    cm = [[0] * 3 for _ in range(3)]
    for r in res:
        cm[r["y"]][r["pred"]] += 1
    f1s = []
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[i][c] for i in range(3)) - tp
        fn = sum(cm[c]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        if sum(cm[c]):
            f1s.append(f1)
        if verbose:
            print(f"  {LABELS[c]:<11} precision {p:.3f}  recall {rc:.3f}  F1 {f1:.3f}  (n={sum(cm[c])})")
    if verbose:
        print("  confusion matrix (rows = true, cols = predicted: benign, malware, ransomware)")
        for c in range(3):
            print("   ", cm[c])
    return sum(f1s) / len(f1s) if f1s else 0.0


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def make_demo(path, n=600):
    """Write a synthetic JSONL so you can test this script end to end before using real data."""
    try:
        from transformer import _synthetic
    except ImportError:
        from xran_transformer import _synthetic
    random.seed(0)
    with open(path, "w", encoding="utf-8") as f:
        for i, (ev, y, fam) in enumerate(_synthetic(n=n)):
            f.write(json.dumps({"sha": f"demo{i}", "label": y, "family": f"fam{fam}" if y == 2 else None,
                                "events": ev, "first_seen": f"2024-{1 + i % 12:02d}-01"}) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-demo-jsonl", metavar="PATH", help="write a synthetic dataset to PATH and exit")
    ap.add_argument("--data")
    ap.add_argument("--out", default="xran.pt")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--holdout-families", type=int, default=0)
    ap.add_argument("--mode", choices=["sequence", "set"], default="sequence",
                    help="sequence: sliding windows over ordered events. set: one window of distinct tokens (no order)")
    ap.add_argument("--exclude-from-train", metavar="SOURCE:CLASS,...",
                    help="hold out whole source/class groups (e.g. malbehavd:0) and report how the model treats them")
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--max-events", type=int, default=5000, help="cut each run to this many events")
    ap.add_argument("--max-windows", type=int, default=20, help="max training windows per run")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.make_demo_jsonl:
        make_demo(args.make_demo_jsonl)
        return
    if not args.data:
        ap.error("--data is required (or use --make-demo-jsonl PATH)")

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = load(args.data)
    train, val, test, novel = split_runs(runs, args.split, args.holdout_families, args.seed)

    extra = []
    if args.exclude_from_train:
        pairs = {(g.split(":")[0], int(g.split(":")[1])) for g in args.exclude_from_train.split(",")}
        in_pairs = lambda r: (r.get("source", "?"), r["label"]) in pairs
        extra = [r for r in train + val + test if in_pairs(r)]
        train, val, test = ([r for r in part if not in_pairs(r)] for part in (train, val, test))
        print(f"left out of training entirely: {len(extra)} runs from {sorted(pairs)}")

    print("runs per class (all):", {LABELS[k]: v for k, v in sorted(Counter(r["label"] for r in runs).items())})
    print(f"split: train {len(train)} | val {len(val)} | test(known) {len(test)} | novel-family test {len(novel)}")
    missing = [LABELS[c] for c in range(3) if not any(r["label"] == c for r in train)]
    if missing:
        print(f"WARNING: no training runs for class(es): {missing}. That class can never be predicted.")

    src_cls = Counter((r.get("source", "?"), r["label"]) for r in train)
    sources = sorted({sc for sc, _ in src_cls})
    print("training runs by source and class:")
    for sc in sources:
        print(f"  {sc:<14}", {LABELS[c]: src_cls[(sc, c)] for c in range(3) if src_cls[(sc, c)]})
    if len(sources) > 1:
        for c in range(3):
            srcs_c = {sc for (sc, k) in src_cls if k == c}
            if len(srcs_c) == 1:
                print(f"WARNING: every '{LABELS[c]}' training run comes from one source ({next(iter(srcs_c))}). "
                      f"The model may learn to recognise the source instead of the behaviour.")
    fams = sorted({r["family"] for r in train if r["label"] == RANSOMWARE and r["family"]})
    fam2id = {f: i for i, f in enumerate(fams)}
    print(f"ransomware families in training: {len(fams)}")

    if len(sources) > 1:
        toks = defaultdict(set)
        for r in train:
            toks[r.get("source", "?")].update(normalize_event(e) for e in cut(r, args.max_events))
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                x, y = toks[sources[i]], toks[sources[j]]
                print(f"token overlap {sources[i]} vs {sources[j]}: {len(x & y)} shared "
                      f"({len(x)} vs {len(y)} distinct; Jaccard {len(x & y) / max(1, len(x | y)):.0%})")
    vocab = EventVocab(args.min_freq).fit([[normalize_event(e) for e in cut(r, args.max_events)] for r in train])
    print(f"vocabulary size: {len(vocab)} (fitted on training runs only)")
    if args.mode == "set":
        args.S = min(args.S, 16)                         # in set mode S only sets how often a live stream is rescored
        args.W = max(args.W, len(vocab) - 2)             # one window must hold every known token
        print(f"set mode: one window per run, W = {args.W}")
    X, Y, YF = make_windows(train, vocab, fam2id, args.W, args.S, max(2, args.max_windows), args.max_events, args.mode)
    print(f"training windows: {len(X)}")

    counts = torch.bincount(Y, minlength=3).float()
    cw = torch.where(counts > 0, counts.sum() / (3 * counts.clamp(min=1)), torch.zeros(3)).to(device)

    model = XRanTransformer(len(vocab), n_families=max(1, len(fams)), d_model=args.d_model, n_heads=args.heads,
                            n_layers=args.layers, d_ff=args.d_ff, max_len=args.W,
                            use_positions=(args.mode != "set")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            x, y, yf = X[b].to(device), Y[b].to(device), YF[b].to(device)
            det, fam, _ = model(x)
            loss = F.cross_entropy(det, y, weight=cw)
            m = (y == RANSOMWARE) & (yf >= 0)
            if m.any():
                loss = loss + 0.5 * F.cross_entropy(fam[m], yf[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        vf1 = prf(evaluate(model, vocab, val, fam2id, args, device)) if val else 0.0
        print(f"epoch {epoch:2d}  train loss {total / len(X):.4f}  val macro-F1 {vf1:.3f}")
        if not val or vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    model.eval()

    # prototypes from windows the model itself calls ransomware, threshold from validation runs
    with torch.no_grad():
        zs, ps = [], []
        for i in range(0, len(X), 256):
            det, _, z = model(X[i:i + 256].to(device))
            zs.append(z.cpu()); ps.append(F.softmax(det, -1)[:, RANSOMWARE].cpu())
        Z, P = torch.cat(zs), torch.cat(ps)
    mask = (Y == RANSOMWARE) & (YF >= 0) & (P > 0.5)
    if mask.sum() == 0:
        mask = (Y == RANSOMWARE) & (YF >= 0)
    store = PrototypeStore()
    if mask.sum() > 0:
        store.fit(Z[mask], YF[mask])
        val_r = [r for r in evaluate(model, vocab, [x for x in val if x["label"] == RANSOMWARE], fam2id, args, device)]
        if val_r:
            store.calibrate(torch.cat([r["z"] for r in val_r]), q=0.95)

    # ---------------- final report on held-out data
    print("\n=== TEST (known families, never used for training or model selection) ===")
    res = evaluate(model, vocab, test, fam2id, args, device)
    prf(res, verbose=True)
    test_sources = sorted({r["source"] for r in res})
    if len(test_sources) > 1:
        print("  per source (accuracy and per-class recall):")
        for sc in test_sources:
            rs = [r for r in res if r["source"] == sc]
            rec = {LABELS[c]: round(sum(r["pred"] == c for r in rs if r["y"] == c) / sum(r["y"] == c for r in rs), 3)
                   for c in range(3) if any(r["y"] == c for r in rs)}
            print(f"    {sc:<14} n={len(rs):<5} accuracy {sum(r['y'] == r['pred'] for r in rs) / len(rs):.3f}  recall {rec}")
    rk = [r for r in res if r["y"] == RANSOMWARE and r["fam"] >= 0]
    if rk:
        print(f"  family accuracy on known-family ransomware runs: "
              f"{sum(r['fam_pred'] == r['fam'] for r in rk) / len(rk):.3f}  (n={len(rk)})")

    if novel and store.protos and store.threshold is not None:
        print(f"\n=== NOVELTY (novel-family ransomware runs: {len(novel)}) ===")
        nres = evaluate(model, vocab, novel, fam2id, args, device)
        d_known = [float(store.distance(r["z"])[0][0]) for r in rk]
        d_novel = [float(store.distance(r["z"])[0][0]) for r in nres]
        print(f"  AUROC (novel scored higher than known): {auroc(d_novel, d_known):.3f}")
        print(f"  flagged as novel at threshold {store.threshold:.4f}: known {sum(d > store.threshold for d in d_known) / max(1, len(d_known)):.1%}"
              f" | novel {sum(d > store.threshold for d in d_novel) / max(1, len(d_novel)):.1%}")
        print(f"  novel runs detected as ransomware at all: {sum(r['pred'] == RANSOMWARE for r in nres) / len(nres):.1%}")
    elif novel:
        print("\n(novel runs exist but no prototypes/threshold could be built; skipping novelty report)")

    if extra:
        print("\n=== LEFT-OUT GROUPS (never seen in training) ===")
        grp = defaultdict(list)
        for r in evaluate(model, vocab, extra, fam2id, args, device):
            grp[(r["source"], r["y"])].append(r["pred"])
        for (sc, c), preds in sorted(grp.items()):
            cnt = Counter(preds)
            print(f"  {sc}:{LABELS[c]:<10} n={len(preds):<5} predicted as " +
                  ", ".join(f"{LABELS[k]} {cnt[k] / len(preds):.1%}" for k in range(3)))

    print("\n=== EARLY DETECTION (only the first N events of each test run) ===")
    ordered_test = [r for r in test if r.get("ordered", True)]
    if len(ordered_test) < len(test):
        print(f"  (using only the {len(ordered_test)} of {len(test)} test runs whose events are in recorded order)")
    for n in (32, 64, 127, 256, 512, None):
        if not ordered_test:
            print("  no ordered runs available; skipped")
            break
        r_ = evaluate(model, vocab, ordered_test, fam2id, args, device, trunc=n)

        def recall(c):
            xs = [r for r in r_ if r["y"] == c]
            return "n/a" if not xs else f"{sum(r['pred'] == c for r in xs) / len(xs):.3f}"
        ben = [r for r in r_ if r["y"] == 0]
        fpr = sum(r["pred"] == RANSOMWARE for r in ben) / max(1, len(ben))
        print(f"  first {str(n or 'all'):>4} events: recall benign {recall(0)} | malware {recall(1)} | "
              f"ransomware {recall(2)} | benign flagged as ransomware {fpr:.3f}")
    if not any(r["label"] == RANSOMWARE for r in ordered_test):
        print("  NOTE: no ransomware run in the test set has recorded order, so early detection of ransomware "
              "cannot be measured with this data.")

    model.cpu()
    cfg = dict(n_families=max(1, len(fams)), d_model=args.d_model, n_heads=args.heads, n_layers=args.layers,
               d_ff=args.d_ff, W=args.W, S=args.S, mode=args.mode)
    save_checkpoint(args.out, model, vocab, store, cfg, {i: f for f, i in fam2id.items()})
    print("\nsaved", args.out, "- load it with deploy.Detector")


if __name__ == "__main__":
    main()
