"""
convert.py - turn dataset files into runs.jsonl for train.py (standard library only).

Classes: 0 = benign, 1 = malware, 2 = ransomware. YOU say which raw label means what with --map or
--all-label, because a flipped label would still train "successfully" and silently invert every result.

Modes:
  table   one run per row of a CSV/TSV: id column, label column, and API-call columns (or one text cell)
  txt     a text file with one run per line, plus a separate labels file in the same line order
  folder  a folder or .zip of per-run CSV/TXT files, all with one fixed label

Examples (check column names and label values with inspect_dataset.py first):
  python convert.py table --file MalBehavD-V1-dataset.csv --label-col labels --map 0=0,1=1 --source malbehavd --out runs.jsonl
  python convert.py txt --file all_analysis_data.txt --labels labels.csv --all-label 1 --source malapi --append --out runs.jsonl
  python convert.py folder --dir ransomware_csvs --events-col API --label 2 --source mendeley --append --out runs.jsonl

Use --append to add a second or third dataset to the same runs.jsonl. Duplicate event sequences are
dropped across datasets (so the same sample can't end up in both train and test).
Map a raw label to -1 to drop those rows (for example --map 0=0,1=-1).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import sys
import zipfile
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
SPLIT = re.compile(r"[\s,;|]+")
LABELS = ["benign", "malware", "ransomware"]


# ---------------------------------------------------------------- helpers
def parse_map(s):
    m = {}
    for part in s.split(","):
        k, v = part.rsplit("=", 1)
        m[k.strip()] = int(v)
    return m


def tok(x, prefix):
    x = str(x).strip()
    if not x or x.lower() in ("nan", "none", "null"):
        return None
    if re.fullmatch(r"\d+\.0", x):
        x = x[:-2]
    return f"{prefix}:{x}" if prefix else x


def collapse(ev):
    return [e for i, e in enumerate(ev) if i == 0 or e != ev[i - 1]]


def col_index(header, ref):
    if ref is None:
        return None
    if ref in header:
        return header.index(ref)
    if ref.isdigit() and int(ref) < len(header):
        return int(ref)
    raise SystemExit(f"column {ref!r} not found. Columns start with: {header[:12]}")


class Out:
    def __init__(self, a):
        self.a = a
        self.seen = set()
        if a.append and os.path.exists(a.out):
            with open(a.out, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.seen.add(self._h(json.loads(line)["events"]))
        self.f = open(a.out, "a" if a.append else "w", encoding="utf-8")
        self.stats, self.lens, self.vocab = Counter(), [], Counter()
        self.dups = self.short = self.n = self.dropped = 0

    @staticmethod
    def _h(ev):
        return hashlib.md5("\x1f".join(ev).encode("utf-8", "ignore")).hexdigest()

    def add(self, rid, label, events, raw):
        if label is None or label < 0:
            self.dropped += 1
            return
        if self.a.collapse_repeats:
            events = collapse(events)
        if len(events) < self.a.min_events:
            self.short += 1
            return
        events = events[:self.a.max_events]
        h = self._h(events)
        if not self.a.no_dedup and h in self.seen:
            self.dups += 1
            return
        self.seen.add(h)
        self.f.write(json.dumps({"sha": str(rid), "label": label, "family": None, "events": events,
                                 "source": self.a.source, "raw_label": str(raw)}) + "\n")
        self.n += 1
        self.stats[label] += 1
        self.lens.append(len(events))
        self.vocab.update(set(events))

    def report(self):
        self.f.close()
        print(f"\nwrote {self.n} runs to {self.a.out} (source = {self.a.source})")
        print("per class:", {LABELS[k]: v for k, v in sorted(self.stats.items())})
        print(f"skipped: {self.short} with fewer than {self.a.min_events} events, {self.dups} duplicates, "
              f"{self.dropped} dropped by label mapping")
        if self.lens:
            print(f"events per run: min/median/max = {min(self.lens)}/{int(statistics.median(self.lens))}/{max(self.lens)}")
        print(f"distinct tokens in this source: {len(self.vocab)}; common: {self.vocab.most_common(8)}")
        if self.a.expect and abs(self.n - self.a.expect) > 0.05 * self.a.expect:
            print(f"WARNING: the dataset description says about {self.a.expect} runs but {self.n} were written. "
                  f"Check the label mapping, skipped columns and the duplicate count.")


def label_for(raw, a, unmapped):
    if a.all_label is not None:
        return a.all_label
    if raw in a.map:
        return a.map[raw]
    unmapped[raw] += 1
    return None


# ---------------------------------------------------------------- modes
def run_table(a, out):
    with open(a.file, newline="", encoding="utf-8", errors="replace") as f:
        head = f.readline()
        f.seek(0)
        rd = csv.reader(f, delimiter="\t" if head.count("\t") > head.count(",") else ",")
        header = next(rd)
        li, ii = col_index(header, a.label_col), col_index(header, a.id_col)
        skip = {col_index(header, c) for c in (a.skip_cols or [])}
        seq = [i for i in range(len(header)) if i not in skip and i != li and i != ii]
        print(f"label column: {header[li]!r} | id column: {header[ii] if ii is not None else '(row number)'} | "
              f"sequence columns: {len(seq)}")
        seen_raw, unmapped = Counter(), Counter()
        for n, row in enumerate(rd):
            if li >= len(row):
                continue
            raw = row[li].strip()
            seen_raw[raw] += 1
            cells = [row[i] for i in seq if i < len(row)]
            parts = SPLIT.split(cells[0]) if len(seq) == 1 and cells else cells
            events = [t for t in (tok(c, a.prefix) for c in parts) if t]
            out.add(row[ii] if ii is not None and ii < len(row) else f"{a.source}-{n}",
                    label_for(raw, a, unmapped), events, raw)
    print("raw label values seen:", dict(seen_raw))
    if unmapped:
        print("WARNING: raw labels with no mapping (rows skipped):", dict(unmapped))


def run_txt(a, out):
    with open(a.file, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    with open(a.labels, newline="", encoding="utf-8", errors="replace") as f:
        labs = [r[a.labels_col].strip() if len(r) > a.labels_col else "" for r in csv.reader(f)]
    if a.labels_header:
        labs = labs[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    while labs and not labs[-1]:
        labs.pop()
    if len(lines) != len(labs):
        raise SystemExit(f"{len(lines)} sequence lines but {len(labs)} labels: the files are not line-aligned. "
                         f"Check --labels-header and --labels-col.")
    seen_raw, unmapped = Counter(), Counter()
    for i, (line, raw) in enumerate(zip(lines, labs)):
        seen_raw[raw] += 1
        events = [t for t in (tok(c, a.prefix) for c in SPLIT.split(line.strip())) if t]
        out.add(f"{a.source}-{i}", label_for(raw, a, unmapped), events, raw)
    print("raw label values seen:", dict(seen_raw))
    if unmapped:
        print("WARNING: raw labels with no mapping (rows skipped):", dict(unmapped))


def iter_folder(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir() and info.filename.lower().endswith((".csv", ".tsv", ".txt", ".log")):
                    yield info.filename, z.read(info).decode("utf-8", "replace")
    else:
        for root, _, fs in os.walk(path):
            for fn in sorted(fs):
                if fn.lower().endswith((".csv", ".tsv", ".txt", ".log")):
                    fp = os.path.join(root, fn)
                    with open(fp, encoding="utf-8", errors="replace") as f:
                        yield os.path.relpath(fp, path), f.read()


def run_folder(a, out):
    n = 0
    for name, text in iter_folder(a.dir):
        if a.limit and n >= a.limit:
            break
        if name.lower().endswith((".csv", ".tsv")):
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if name.lower().endswith(".tsv") else ","))
            if not rows:
                continue
            if a.events_col is None:
                raise SystemExit(f"--events-col is required for CSV files. {name} starts with columns: {rows[0][:12]}")
            header, body = rows[0], rows[1:]
            ci = col_index(header, a.events_col)
            events = [t for t in (tok(r[ci], a.prefix) for r in body if ci < len(r)) if t]
        else:
            events = [t for t in (tok(c, a.prefix) for c in SPLIT.split(text.strip())) if t]
        out.add(os.path.splitext(name)[0], a.label, events, a.label)
        n += 1
    print(f"files read: {n}")


# ---------------------------------------------------------------- CLI
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True, help="short dataset name, stored on every run")
    common.add_argument("--out", default="runs.jsonl")
    common.add_argument("--append", action="store_true", help="add to an existing runs.jsonl")
    common.add_argument("--prefix", default="API", help="token prefix; use '' for none")
    common.add_argument("--min-events", type=int, default=5)
    common.add_argument("--max-events", type=int, default=5000)
    common.add_argument("--collapse-repeats", action="store_true", help="merge consecutive identical events")
    common.add_argument("--no-dedup", action="store_true")
    common.add_argument("--expect", type=int, help="run count the dataset description promises, for a sanity check")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("table", parents=[common])
    t.add_argument("--file", required=True)
    t.add_argument("--label-col", required=True)
    t.add_argument("--id-col")
    t.add_argument("--skip-cols", nargs="*", help="columns to ignore, e.g. an unnamed index column")

    x = sub.add_parser("txt", parents=[common])
    x.add_argument("--file", required=True)
    x.add_argument("--labels", required=True)
    x.add_argument("--labels-col", type=int, default=0)
    x.add_argument("--labels-header", action="store_true", help="labels file has a header line")

    d = sub.add_parser("folder", parents=[common])
    d.add_argument("--dir", required=True, help="folder or .zip")
    d.add_argument("--events-col", help="CSV column holding the API/system-call name")
    d.add_argument("--label", type=int, required=True, choices=[0, 1, 2])
    d.add_argument("--limit", type=int, help="read only the first N files (for testing)")

    for p in (t, x):
        p.add_argument("--map", type=parse_map, default={}, help="raw=class pairs, e.g. 0=0,1=1")
        p.add_argument("--all-label", type=int, choices=[0, 1, 2], help="give every row this class")
    a = ap.parse_args()
    if a.mode in ("table", "txt") and a.all_label is None and not a.map:
        ap.error("give --map raw=class,... or --all-label C")
    if a.mode == "folder":
        a.all_label, a.map = a.label, {}

    out = Out(a)
    {"table": run_table, "txt": run_txt, "folder": run_folder}[a.mode](a, out)
    out.report()


if __name__ == "__main__":
    main()
