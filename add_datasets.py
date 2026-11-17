"""
add_datasets.py - download two extra public datasets from GitHub and append them to runs.jsonl
(same format and the same 58-token vocabulary as build_dataset.py).

    pip install openpyxl
    python add_datasets.py --runs runs.jsonl --vocab runs.vocab.json --out runs_v2.jsonl

Sources (both recorded in Cuckoo sandboxes, both on GitHub; check their licences yourself):
  winpe      drzehra14/Windows-PE-Malware-API-dataset   439 goodware + 582 malware, API call counts per sample.
             Paper: Applied Sciences 2024, 14(3), 1015 (Dynamic Malware Classification and API Categorisation
             of Windows PE Files). threat_categories.xlsx labels 122 of the malware as ransomware, so benign,
             malware AND ransomware come from ONE sandbox - the thing the current data lacks.
  ransomset  gabrielolivs/RansomSet   6 ransomware families (Conti, WannaCry, Ryuk, CryptoLocker, LockBit,
             Sodinokibi) + "Normal" processes, one row per process with API call counts.

Both are count tables, so runs are sets (ordered=false), like MLRan. Counts > 0 become "token present".
Rows with fewer than 3 known tokens and exact duplicate token sets (within source+class) are dropped.
"""
import argparse, csv, io, json, os, random, sys, urllib.parse, urllib.request
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
RAW = "https://raw.githubusercontent.com"
WINPE = f"{RAW}/drzehra14/Windows-PE-Malware-API-dataset/main/"
RANSOMSET = f"{RAW}/gabrielolivs/RansomSet/main/Dataset/ransomset-multiclass-dataset.csv"
FAMILY_FIX = {"sodinikibi": "revil", "sodinokibi": "revil"}   # same family; MLRan calls it revil
MIN_TOKENS = 3


def normalize(name):
    """Same rules as deploy.normalize_event for API names (kept here so this script needs no torch)."""
    if len(name) > 3 and name[:2] in ("Nt", "Zw") and name[2].isupper():
        name = name[2:]
    if len(name) > 3 and name[-1] in "AW" and name[-2].islower():
        name = name[:-1]
    return "API:" + name.lower()


def get(url, cache):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(urllib.parse.unquote(url)))
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        print("downloading", url)
        urllib.request.urlretrieve(url.replace(" ", "%20"), path)
    return path


def xlsx_rows(path):
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(it)]
    return header, it


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs.jsonl")
    ap.add_argument("--vocab", default="runs.vocab.json")
    ap.add_argument("--out", default="runs_v2.jsonl")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vocab = set(json.load(open(a.vocab, encoding="utf-8")))

    runs = [json.loads(l) for l in open(a.runs, encoding="utf-8")]
    seen = {(r["source"], r["label"], tuple(sorted(set(r["events"])))) for r in runs}
    base = len(runs)
    stats = Counter()

    def add(src, sha, label, toks, family=None):
        toks = sorted(set(toks))
        if len(toks) < MIN_TOKENS:
            stats[(src, label, "dropped: fewer than 3 known tokens")] += 1
            return
        key = (src, label, tuple(toks))
        if key in seen:
            stats[(src, label, "dropped: duplicate token set")] += 1
            return
        seen.add(key)
        runs.append({"sha": str(sha), "label": label, "family": family, "events": toks,
                     "source": src, "ordered": False})
        stats[(src, label, "kept")] += 1

    # ---- Windows-PE-Malware-API-dataset
    cat = {}
    _, it = xlsx_rows(get(WINPE + "threat_categories.xlsx", a.cache))
    for sha, c in it:
        cat[str(sha).strip().lower()] = str(c).strip().lower()
    header, it = xlsx_rows(get(WINPE + "Goodware and malware dataset.xlsx", a.cache))
    lab = header.index("malware")                      # the sheet has a stray empty last column
    cols = [(i, normalize(h)) for i, h in enumerate(header) if 0 < i < lab and normalize(h) in vocab]
    print(f"winpe: {len(cols)} of {lab - 1} API columns map into the vocabulary")
    for row in it:
        if row[0] is None:
            continue
        sha = str(row[0]).strip().lower()
        is_mal = float(row[lab] or 0) == 1.0
        label = 0 if not is_mal else (2 if cat.get(sha) == "ransomware" else 1)
        add("winpe", sha, label, [t for i, t in cols if row[i] not in (None, 0, 0.0, "0")])

    # ---- RansomSet
    path = get(RANSOMSET, a.cache)
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        cols = [(i, normalize(h)) for i, h in enumerate(header[2:-1], start=2) if normalize(h) in vocab]
        print(f"ransomset: {len(cols)} of {len(header) - 3} API columns map into the vocabulary")
        for n, row in enumerate(rd):
            fam = row[-1].strip().lower()
            toks = [t for i, t in cols if row[i] not in ("", "0", "0.0")]
            if fam == "normal":
                add("ransomset", f"ransomset-{n}", 0, toks)
            else:
                add("ransomset", f"ransomset-{n}", 2, toks, FAMILY_FIX.get(fam, fam))

    random.Random(a.seed).shuffle(runs)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    names = ["benign", "malware", "ransomware"]
    print(f"\nwrote {len(runs)} runs ({base} existing + {len(runs) - base} new) to {a.out}")
    for (s, c, what), n in sorted(stats.items()):
        print(f"  {s:<10} {names[c]:<11} {n:>6}  {what}")
    cnt = Counter((r["source"], r["label"]) for r in runs)
    print("\nclass totals:", {names[c]: sum(n for (s, k), n in cnt.items() if k == c) for c in range(3)})
    print("sources per class:", {names[c]: sorted({s for (s, k) in cnt if k == c}) for c in range(3)})


if __name__ == "__main__":
    main()
