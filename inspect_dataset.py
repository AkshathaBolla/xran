"""
inspect_dataset.py - print the structure of a dataset file so you know how to convert it.

    python inspect_dataset.py path\\to\\file_or_folder      (.csv .tsv .txt .jsonl .json .xlsx .zip, or a folder)

Standard library only. Paste the output back so the right adapter can be written.
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
MAX_ROWS = 20000
HINTS = ("label", "class", "family", "type", "categor", "malware", "ransom", "benign", "target", "name")
SEQ_HEADER = re.compile(r"^(t|api|call|step|seq)[_\-]?\d+$", re.I)


def short(v, n=60):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def inspect_table(path, delim):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f, delimiter=delim)
        header = next(rd)
        ncol = len(header)
        print(f"columns: {ncol}")
        print("first 15:", [short(h, 30) for h in header[:15]])
        print("last 8:  ", [short(h, 30) for h in header[-8:]])
        seq_cols = sum(1 for h in header if SEQ_HEADER.match(h.strip()))
        look = sorted(set(list(range(min(ncol, 60))) + list(range(max(60, ncol - 20), ncol))
                          + [i for i, h in enumerate(header) if any(k in h.lower() for k in HINTS)][:60]))
        vals = {i: Counter() for i in look}
        bin_ok = {i: True for i in range(min(ncol, 200))}
        long_cell = None
        n = 0
        first_row = None
        for row in rd:
            n += 1
            if first_row is None:
                first_row = row
            for i in look:
                if i < len(row) and len(vals[i]) <= 300:
                    vals[i][row[i]] += 1
            for i in bin_ok:
                if bin_ok[i] and i < len(row) and row[i] not in ("0", "1", "", "0.0", "1.0"):
                    bin_ok[i] = False
            if long_cell is None and n <= 50:
                for i, c in enumerate(row[:200]):
                    if len(c) > 200 and re.search(r"[ ,;|]", c):
                        long_cell = (header[i] if i < ncol else i, short(c, 100))
                        break
            if n >= MAX_ROWS:
                break
    print(f"rows scanned: {n}" + (" (capped)" if n >= MAX_ROWS else ""))
    if first_row:
        print("first row (first 12 values):", [short(v, 25) for v in first_row[:12]])
    binary_frac = sum(bin_ok.values()) / max(1, len(bin_ok))
    print(f"\nfraction of first {len(bin_ok)} columns that are only 0/1: {binary_frac:.2f}")
    if binary_frac > 0.8:
        print("  -> mostly presence/absence features: NO event order is available in this file.")
    if seq_cols >= 20 and binary_frac <= 0.5:
        print(f"  -> {seq_cols} columns are named like steps (t_0, t_1, ...) and hold non-binary values: "
              f"likely one event ID per step, so ORDER IS AVAILABLE.")
    elif seq_cols >= 20:
        print(f"  -> {seq_cols} columns have step-like names but hold 0/1 values: these are feature indicators, not order.")
    if long_cell:
        print(f"  -> column {long_cell[0]!r} holds long text that may be a log/sequence: {long_cell[1]!r}")
    print("\ncandidate label/metadata columns:")
    shown = 0
    for i in look:
        c = vals[i]
        name = header[i] if i < ncol else str(i)
        text = any(not is_num(k) for k in c if k != "")
        hinted = any(k in name.lower() for k in HINTS)
        if (text and 2 <= len(c) <= 200) or (hinted and len(c) <= 200):
            print(f"  [{i}] {short(name, 40)}: {len(c)} distinct; top: {c.most_common(8)}")
            shown += 1
            if shown >= 12:
                break
    if shown == 0:
        print("  (none found in the columns checked)")


def inspect_json(path):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        if path.lower().endswith(".jsonl"):
            for i, line in enumerate(f):
                if line.strip():
                    recs.append(json.loads(line))
                if i >= 2000:
                    break
        else:
            data = json.load(f)                       # loads the whole file; fine for files up to a few hundred MB
            recs = data if isinstance(data, list) else [data]
            recs = recs[:2000]
    print(f"records inspected: {len(recs)}")
    if not recs or not isinstance(recs[0], dict):
        print("first record:", short(recs[0] if recs else None, 300))
        return
    keys = list(recs[0].keys())
    print(f"keys in first record ({len(keys)}):")
    for k in keys[:40]:
        col = [r.get(k) for r in recs if isinstance(r, dict)]
        v = col[0]
        if isinstance(v, list):
            lens = [len(x) for x in col if isinstance(x, list)]
            print(f"  {short(k, 40)}: list, length min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}; "
                  f"first items: {[short(x, 25) for x in v[:5]]}")
        elif isinstance(v, dict):
            print(f"  {short(k, 40)}: dict with keys {list(v.keys())[:10]}")
        else:
            c = Counter(str(x) for x in col)
            tail = f"top: {c.most_common(6)}" if len(c) <= 50 else f"e.g. {short(v, 50)!r}"
            print(f"  {short(k, 40)}: {type(v).__name__}, {len(c)} distinct; {tail}")


def inspect_txt(path):
    """Plain text: one run per line is common for API-call datasets."""
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip():
                lines.append(line.rstrip("\n"))
            if i >= 2000:
                break
    print(f"non-empty lines read: {len(lines)} (capped at ~2000)")
    for ln in lines[:3]:
        print("  line:", short(ln, 200))
    toks = [re.split(r"[\s,;|]+", ln.strip()) for ln in lines]
    lens = [len(t) for t in toks]
    if lens:
        print(f"tokens per line: min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}")
        vocab = Counter(t for row in toks for t in row if t)
        print(f"distinct tokens: {len(vocab)}; most common: {vocab.most_common(8)}")
        if all(is_num(t) for t, _ in vocab.most_common(20)):
            print("  -> tokens are numbers: API calls stored as integer IDs. You need the ID-to-name mapping for readable explanations.")


def inspect_xlsx(path):
    import xml.etree.ElementTree as ET
    import zipfile
    m = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"m": m}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % m)))
        sheets = sorted(n for n in names if n.startswith("xl/worksheets/sheet"))
        print("worksheets:", sheets)
        for sh in sheets[:2]:
            rows = ET.fromstring(z.read(sh)).findall(".//m:row", ns)
            print(f"{sh}: {len(rows)} rows; first 6:")
            for r in rows[:6]:
                vals = []
                for c in r.findall("m:c", ns):
                    v = c.find("m:v", ns)
                    if v is None:
                        is_ = c.find("m:is", ns)
                        vals.append("".join(t.text or "" for t in is_.iter("{%s}t" % m)) if is_ is not None else "")
                    else:
                        vals.append(shared[int(v.text)] if c.get("t") == "s" else v.text)
                print("  ", [short(x, 40) for x in vals[:8]])


def inspect_path(path, depth=0):
    ext = os.path.splitext(path)[1].lower()
    if os.path.isdir(path):
        counts, total, files = Counter(), 0, []
        for root, _, fs in os.walk(path):
            for fn in fs:
                fp = os.path.join(root, fn)
                counts[os.path.splitext(fn)[1].lower()] += 1
                total += os.path.getsize(fp)
                files.append(fp)
        print(f"folder: {len(files)} files, {total / 1e6:.1f} MB total; by extension: {dict(counts)}")
        print("sample file names:", [os.path.relpath(f, path) for f in files[:8]])
        for fp in [f for f in files if f.lower().endswith((".csv", ".txt", ".tsv", ".jsonl"))][:2]:
            print(f"\n--- sample file: {os.path.relpath(fp, path)}")
            inspect_path(fp, depth + 1)
    elif ext == ".zip":
        import tempfile
        import zipfile
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            print(f"zip: {len(infos)} members, {sum(i.file_size for i in infos) / 1e6:.1f} MB uncompressed")
            for i in infos[:15]:
                print(f"  {i.filename}  ({i.file_size / 1e3:.1f} KB)")
            with tempfile.TemporaryDirectory() as tmp:
                shown = 0
                for i in infos:
                    if shown < 2 and i.filename.lower().endswith((".csv", ".txt", ".tsv", ".jsonl")) and not i.is_dir():
                        out = os.path.join(tmp, os.path.basename(i.filename))
                        with open(out, "wb") as fo:
                            fo.write(z.read(i))
                        print(f"\n--- sample member: {i.filename}")
                        inspect_path(out, depth + 1)
                        shown += 1
    elif ext in (".csv", ".tsv"):
        inspect_table(path, "\t" if ext == ".tsv" else ",")
    elif ext == ".txt":
        inspect_txt(path)
    elif ext in (".json", ".jsonl"):
        inspect_json(path)
    elif ext == ".xlsx":
        inspect_xlsx(path)
    else:
        print("Unsupported extension. Convert to .csv/.txt/.jsonl, or tell me the format.")



def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) \
        else os.path.getsize(path)
    print(f"path: {path}  ({size / 1e6:.1f} MB)")
    inspect_path(path)


if __name__ == "__main__":
    main()
