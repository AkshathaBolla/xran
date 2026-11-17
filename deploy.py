"""
deploy.py - save/load a trained XRan model and score event data coming from your application.

Quick plumbing test (trains on SYNTHETIC data, so verdicts mean nothing for real software):
    python deploy.py --demo
    python deploy.py --ckpt xran_demo.pt --events sample_events.jsonl
    Get-Content sample_stream.txt | python deploy.py --ckpt xran_demo.pt --stream     (PowerShell)
    cat sample_stream.txt | python deploy.py --ckpt xran_demo.pt --stream             (Linux/macOS)

From your own Python application:
    from deploy import Detector
    det = Detector("xran_demo.pt")
    result = det.predict(list_of_event_strings)            # one finished run
    # or live:
    from deploy import StreamDetector
    stream = StreamDetector(det)
    r = stream.push("API:CreateFileW")                     # returns a dict every S events, else None

IMPORTANT: the same normalize_event() must be used when you build the TRAINING data and
when you score your application's data. Otherwise tokens will not match the vocabulary.
"""
import argparse
import json
import random
import re
import sys
from collections import deque

import torch
import torch.nn.functional as F

try:
    from transformer import (EventVocab, XRanTransformer, PrototypeStore, windowize,
                             multitask_loss, integrated_gradients, top_tokens, _synthetic)
except ImportError:
    from xran_transformer import (EventVocab, XRanTransformer, PrototypeStore, windowize,
                                  multitask_loss, integrated_gradients, top_tokens, _synthetic)

LABELS = ["benign", "malware", "ransomware"]
RANSOMWARE = 2

# Hand-written grouping of tokens into behaviours a human analyst recognises (diagram block e).
# These are DESCRIPTIVE labels I chose for the 58 API tokens of the shipped model, not learned or validated
# categories. Edit freely. The first four groups also cover tokens from richer models/recordings.
BEHAVIOR_GROUPS = {
    "encryption": ["crypt", ".locked"],
    "backup deletion": ["shadowcopy", "vssadmin", "wbadmin", "bcdedit"],
    "memory / process manipulation": ["allocatevirtualmemory", "protectvirtualmemory", "writeprocessmemory",
                                      "getcontextthread", "setcontextthread", "setwindowshookex",
                                      "createremotethread", "openprocess", "createprocessinternal"],
    "persistence / services": ["openscmanager", "openservice", "regcreatekeyex", "regsetvalueex", "regdeletekey",
                               "createservice", "schtasks"],
    "file system activity": ["querydirectoryfile", "setinformationfile", "queryinformationfile", "writefile",
                             "createdirectory", "removedirectory", "getfileattributesex", "deviceiocontrol",
                             "getfileinformationbyhandle"],
    "registry enumeration": ["querykey", "regenumkey", "regenumvalue", "regqueryinfokey"],
    "system discovery": ["getcomputername", "getsystemdirectory", "getsystemwindowsdirectory", "globalmemorystatus",
                         "getnativesysteminfo", "createtoolhelp32snapshot", "process32", "getdiskfreespace",
                         "getcursorpos", "getsystemmetrics", "findwindow", "shgetfolderpath"],
    "evasion / timing": ["delayexecution", "addvectoredexceptionhandler"],
}
_HEXLIKE = re.compile(r"^[0-9a-fA-F\-{}]{16,}$")


# ---------------------------------------------------------------- tokens
def normalize_event(e: str) -> str:
    """Turn one raw event string into a vocabulary token. Keep this identical for training and scoring."""
    e = e.strip()
    if e.startswith("API:"):                       # make names from different tools comparable
        name = e[4:]
        if len(name) > 3 and name[:2] in ("Nt", "Zw") and name[2].isupper():
            name = name[2:]                        # NtCreateFile -> CreateFile
        if len(name) > 3 and name[-1] in "AW" and name[-2].islower():
            name = name[:-1]                       # CreateFileW -> CreateFile
        return "API:" + name.lower()
    if e.startswith("DLL:"):
        return "DLL:" + e[4:].lower()
    if e.startswith("FILE:"):                      # keep only the extension, drop user-specific paths
        name = e[5:].replace("\\", "/").rsplit("/", 1)[-1]
        return "FILE:." + name.rsplit(".", 1)[-1].lower() if "." in name else "FILE:"
    if e.startswith("MUTEX:") and _HEXLIKE.match(e[6:]):
        return "MUTEX:<hex>"                       # random-looking mutex names would bloat the vocab
    return e


def encode_events(vocab, events, W, S, mode="sequence"):
    """Raw event strings -> list of token-id windows (each W-1 ids, 0 = padding) and the unknown-token rate.
    sequence mode: sliding windows over the ordered events (length W-1, stride S).
    set mode: ONE window holding the distinct known tokens in a fixed order (presence only, no order, no repeats).
    The same function is used for training and scoring, so the two can never drift apart."""
    ids = vocab.encode([normalize_event(e) for e in events])
    unk = ids.count(1) / len(ids) if ids else 0.0
    if mode == "set":
        s_ = sorted({i for i in ids if i > 2})[:W - 1]
        return [s_ + [0] * (W - 1 - len(s_))], unk
    return windowize(ids, W, S), unk


def group_behaviors(tokens):
    out = {}
    for g, keys in BEHAVIOR_GROUPS.items():
        hit = [t for t in tokens if any(k in t.lower() for k in keys)]
        if hit:
            out[g] = hit
    return out


def events_from_cuckoo_report(path):
    """UNVERIFIED against your Cuckoo/CAPE version: reads behavior.processes[].calls[].api in file order.
    Check the keys against one real report before relying on this. Must match how you built training data."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    beh = r.get("behavior", {})
    summ = beh.get("summary", {})
    ev = []
    for m in summ.get("mutexes", summ.get("mutex", [])):
        ev.append("MUTEX:" + str(m))
    for d in summ.get("dll_loaded", []):
        ev.append("DLL:" + str(d))
    for p in beh.get("processes", []):
        for c in p.get("calls", []):
            ev.append("API:" + str(c.get("api", "unknown")))
    return ev


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(path, model, vocab, store, cfg, family_names=None):
    torch.save({
        "state_dict": model.state_dict(),
        "stoi": vocab.stoi,
        "cfg": cfg,
        "protos": {k: (v.cpu(), n) for k, (v, n) in store.protos.items()},
        "threshold": store.threshold,
        "family_names": family_names or {},
    }, path)


# ---------------------------------------------------------------- inference
class Detector:
    def __init__(self, ckpt_path, k=3):
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ck["cfg"]
        self.W, self.S, self.k = cfg["W"], cfg["S"], k
        self.mode = cfg.get("mode", "sequence")
        self.vocab = EventVocab()
        self.vocab.stoi = ck["stoi"]
        self.vocab.itos = {i: t for t, i in self.vocab.stoi.items()}
        self.model = XRanTransformer(len(self.vocab.stoi), n_families=cfg["n_families"],
                                     d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                     n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
                                     max_len=cfg["W"], dropout=0.0,
                                     use_positions=(self.mode != "set"))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.store = PrototypeStore()
        self.store.protos = {int(k_): (v, int(n)) for k_, (v, n) in ck["protos"].items()}
        self.store.threshold = ck["threshold"]
        self.family_names = ck.get("family_names", {})

    def predict(self, events, explain=True):
        if not events:
            return {"verdict": "no_data"}
        wins_list, unk_rate = encode_events(self.vocab, events, self.W, self.S, self.mode)
        wins = torch.tensor(wins_list)
        with torch.no_grad():
            det, fam, z = self.model(wins)
        p = F.softmax(det, dim=-1)                                   # (n_windows, 3)
        agg = p.topk(min(self.k, p.size(0)), dim=0).values.mean(0)   # per-class top-k mean over windows
        label = int(agg.argmax())
        peak = int(p[:, label].argmax())                             # most suspicious window for that label
        out = {
            "verdict": LABELS[label],
            "scores": {LABELS[i]: round(float(agg[i]), 3) for i in range(3)},
            "windows": int(p.size(0)),
            "peak_window": peak,
            "mode": self.mode,
        }
        if self.mode == "set":
            known = int((wins[0] > 2).sum())
            out["known_behaviors"] = f"{known} of {len(self.vocab.stoi) - 3}"
            if known < 10:                                   # training runs typically show 12-40 of the 58
                out["warning"] = (f"Only {known} of the {len(self.vocab.stoi) - 3} behaviors this model knows were observed; "
                                  f"the verdict is unreliable. Record longer or exercise more of the program.")
        else:
            out["unknown_token_rate"] = round(unk_rate, 3)
            if unk_rate > 0.3:
                out["warning"] = "Over 30% of events are not in the training vocabulary; output is unreliable."
        if label == RANSOMWARE:
            fp = F.softmax(fam[peak], dim=-1)
            fid = int(fp.argmax())
            out["family"] = self.family_names.get(fid, f"family_{fid}")
            out["family_confidence"] = round(float(fp[fid]), 3)
            if self.store.protos and self.store.threshold is not None:
                novel, d, near = self.store.is_novel(z[peak:peak + 1])
                out["novel"] = bool(novel[0])
                out["novelty_distance"] = round(float(d[0]), 4)
                out["novelty_threshold"] = round(float(self.store.threshold), 4)
                out["nearest_family"] = self.family_names.get(near[0], f"family_{near[0]}")
        if explain:
            x = wins[peak:peak + 1]
            attr = integrated_gradients(self.model, x, target_class=label)
            top = [(t, a) for t, a in top_tokens(self.vocab, x, attr, k=10) if t != "<pad>"]
            out["top_tokens"] = top
            out["behaviors"] = group_behaviors([t for t, _ in top])
        return out


class StreamDetector:
    """Feed events one at a time. Sequence models score the latest window every S events once W-1 events have
    arrived. Set models score everything seen so far every S events."""
    def __init__(self, detector, alert_at=0.8, k=3):
        self.d, self.alert_at, self.k = detector, alert_at, k
        self.buf = deque(maxlen=(detector.W - 1) if detector.mode == "sequence" else 100000)
        self.hist = deque(maxlen=50)
        self.count = 0
        self.since = 0

    def push(self, event):
        self.buf.append(event)
        self.count += 1
        self.since += 1
        if self.since < self.d.S or (self.d.mode == "sequence" and len(self.buf) < self.d.W - 1):
            return None
        self.since = 0
        window = list(self.buf)
        r = self.d.predict(window, explain=False)
        self.hist.append(r["scores"]["ransomware"])
        top = sorted(self.hist, reverse=True)[:self.k]
        risk = sum(top) / len(top)
        alert = risk >= self.alert_at
        if alert:
            r = self.d.predict(window, explain=True)       # explanation only when it is needed
        r["events_seen"] = self.count
        r["risk"] = round(risk, 3)
        r["alert"] = alert
        return r


# ---------------------------------------------------------------- demo (synthetic data)
def train_demo(path="xran_demo.pt", epochs=6, W=128, S=64):
    torch.manual_seed(0)
    random.seed(0)
    runs = _synthetic()
    vocab = EventVocab(min_freq=2).fit([[normalize_event(e) for e in ev] for ev, _, _ in runs])
    X, Y3, YF = [], [], []
    for ev, y, fam in runs:
        ids = vocab.encode([normalize_event(e) for e in ev])
        for w in windowize(ids, W, S):
            X.append(w); Y3.append(y); YF.append(fam)
    X, Y3, YF = torch.tensor(X), torch.tensor(Y3), torch.tensor(YF)
    cfg = dict(n_families=4, d_model=128, n_heads=4, n_layers=3, d_ff=256, W=W, S=S, mode="sequence")
    model = XRanTransformer(len(vocab), n_families=cfg["n_families"], d_model=cfg["d_model"],
                            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
                            max_len=W)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            det, fam, _ = model(X[b])
            loss = multitask_loss(det, fam, Y3[b], YF[b])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch}  loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        _, _, z = model(X)
    store = PrototypeStore()
    r = Y3 == RANSOMWARE
    store.fit(z[r], YF[r])
    store.calibrate(z[r])
    save_checkpoint(path, model, vocab, store, cfg, {i: f"fam{i}" for i in range(4)})
    print("saved", path)

    # sample input files for the plumbing test: one run per class, in raw (un-normalized) form
    samples, seen = [], set()
    for ev, y, fam in _synthetic(n=60):
        if y not in seen:
            seen.add(y)
            samples.append({"expected": LABELS[y], "events": ev})
    with open("sample_events.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    ransom = next(s for s in samples if s["expected"] == "ransomware")
    with open("sample_stream.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ransom["events"]) + "\n")
    print("wrote sample_events.jsonl and sample_stream.txt")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="train on synthetic data and write sample files")
    ap.add_argument("--ckpt", default="xran_demo.pt")
    ap.add_argument("--events", help="JSONL file; each line is a list of events or {'events': [...]}")
    ap.add_argument("--stream", action="store_true", help="read one event per line from stdin")
    ap.add_argument("--alert-at", type=float, default=0.8)
    a = ap.parse_args()

    if a.demo:
        train_demo(a.ckpt)
        return
    det = Detector(a.ckpt)
    if a.events:
        with open(a.events, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                events = obj["events"] if isinstance(obj, dict) else obj
                res = det.predict(events)
                if isinstance(obj, dict) and "expected" in obj:
                    res["expected"] = obj["expected"]
                print(f"--- run {n}")
                print(json.dumps(res, indent=2))
    elif a.stream:
        sd = StreamDetector(det, alert_at=a.alert_at)
        for line in sys.stdin:
            ev = line.strip()
            if not ev:
                continue
            r = sd.push(ev)
            if r:
                print(f"events={r['events_seen']} verdict={r['verdict']} risk={r['risk']} alert={r['alert']}")
                if r["alert"]:
                    print(json.dumps(r, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
