"""
build_dataset.py - download three public datasets from GitHub and build runs.jsonl for train.py.

    python build_dataset.py                       # downloads into .\\data and writes runs.jsonl + runs.vocab.json
    python build_dataset.py --cache D:\\datasets --malapi-max 3000 --out runs.jsonl

Sources (all recorded in Cuckoo sandboxes; all free; licences as listed on their GitHub pages, check them yourself):
  mlran      MLRan, feature-selected release: 4,880 runs, 2,330 ransomware (63 families) + 2,550 goodware.
             https://github.com/faithfulco/mlran   paper: arXiv 2505.18613 "MLRan: A Behavioural Dataset for
             Ransomware Analysis and Detection"
  malbehavd  MalbehavD-V1: 1,285 benign + 1,285 malware, ordered API-call sequences.
             https://github.com/mpasco/MalbehavD-V1
  malapi     Mal-API-2019: 7,107 malware (8 types), ordered API-call sequences, ~2 GB uncompressed.
             https://github.com/ocatak/malware_api_class   paper: arXiv 1905.01999 "A Benchmark API Call Dataset
             for Windows PE Malware Classification"
Please cite the papers if you publish results.

Classes: 0 benign, 1 malware, 2 ransomware.

REPRESENTATION. MLRan only publishes 483 selected presence/absence features, of which 63 are API calls
(58 distinct after name normalisation). Those 58 API tokens are the only behaviour all three datasets
share, so every run is reduced to them. Using the same token universe for all sources prevents the
easiest shortcut (one source having tokens the others lack). Sequences from MalbehavD and Mal-API keep
their order (flag "ordered": true) so early detection can be measured; MLRan runs are sets ("ordered": false).

KNOWN LIMIT. Ransomware comes only from MLRan and malware only from MalbehavD/Mal-API, so the
ransomware-vs-malware boundary is partly confounded with the data source. train.py reports per-source
results and can hold source/class groups out to measure that.

Runs with fewer than 3 distinct known tokens are dropped (they carry no usable information), and exact
duplicate token sets within the same source and class are dropped.
"""
import argparse
import csv
import json
import os
import random
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
from deploy import normalize_event  # noqa: E402  (shared token normalisation)

RAW = "https://raw.githubusercontent.com"
FILES = {
    "mlran_train": f"{RAW}/faithfulco/mlran/main/6_experiments/FS_MLRan_Datasets/MLRan_X_train_RFE.csv",
    "mlran_test": f"{RAW}/faithfulco/mlran/main/6_experiments/FS_MLRan_Datasets/MLRan_X_test_RFE.csv",
    "mlran_names": f"{RAW}/faithfulco/mlran/main/6_experiments/FS_MLRan_Datasets/RFE_selected_feature_names_dic.json",
    "mlran_meta": f"{RAW}/faithfulco/mlran/main/6_experiments/FS_MLRan_Datasets/mlran_dataset_metadata.csv",
    "malbehavd": f"{RAW}/mpasco/MalbehavD-V1/main/MalBehavD-V1-dataset.csv",
    "malapi_zip": f"{RAW}/ocatak/malware_api_class/master/mal-api-2019.zip",
    "malapi_labels": f"{RAW}/ocatak/malware_api_class/master/labels.csv",
}
LOCAL = {k: os.path.basename(v) for k, v in FILES.items()}
MIN_TOKENS = 3


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    for key, url in FILES.items():
        path = os.path.join(cache, LOCAL[key])
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return {k: os.path.join(cache, v) for k, v in LOCAL.items()}


def collapse(seq):
    return [t for i, t in enumerate(seq) if i == 0 or t != seq[i - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data", help="where downloads are kept")
    ap.add_argument("--out", default="runs.jsonl")
    ap.add_argument("--malapi-max", type=int, default=3000, help="malware runs to take from Mal-API-2019 (balanced over its types)")
    ap.add_argument("--max-events", type=int, default=400, help="ordered events kept per sequence run")
    ap.add_argument("--max-raw", type=int, default=5000, help="raw calls read per Mal-API-2019 run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    P = fetch(a.cache)

    # ---- shared vocabulary: the API features MLRan kept
    names = json.load(open(P["mlran_names"], encoding="utf-8"))
    api_cols = {k: v for k, v in names.items() if v.startswith("API:")}
    canon = {k: normalize_event(v) for k, v in api_cols.items()}
    vocab = sorted(set(canon.values()))
    lower2canon = {v[4:].lower(): normalize_event(v) for v in api_cols.values()}
    print(f"shared vocabulary: {len(api_cols)} MLRan API features -> {len(vocab)} normalised tokens")

    runs, seen, dropped = [], set(), Counter()

    def add(src, sha, label, tokens, ordered, family=None, first_seen=None, extra=None):
        if len(set(tokens)) < MIN_TOKENS:
            dropped[(src, label, "fewer than 3 known tokens")] += 1
            return
        key = (src, label, tuple(sorted(set(tokens))))
        if key in seen:
            dropped[(src, label, "duplicate token set")] += 1
            return
        seen.add(key)
        r = {"sha": str(sha), "label": label, "family": family, "events": tokens, "source": src, "ordered": ordered}
        if first_seen:
            r["first_seen"] = str(first_seen)
        if extra:
            r.update(extra)
        runs.append(r)

    # ---- MLRan
    meta = {m["sample_id"]: m for m in csv.DictReader(open(P["mlran_meta"], encoding="utf-8"))}
    for part in ("mlran_train", "mlran_test"):
        with open(P[part], encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd)
            cols = [(i, canon[c]) for i, c in enumerate(header) if c in canon]
            for row in rd:
                toks = sorted({t for i, t in cols if row[i] not in ("0", "0.0")})
                m = meta.get(row[0], {})
                ransom = row[1] == "1"
                add("mlran", m.get("sha256", row[0]), 2 if ransom else 0, toks, False,
                    family=(m.get("ransomware_family") or None) if ransom else None,
                    first_seen=m.get("Year"))

    # ---- MalbehavD-V1
    with open(P["malbehavd"], encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            seq = [lower2canon[x.lower()] for x in row[2:] if x and x.lower() in lower2canon]
            add("malbehavd", row[0], 1 if row[1].strip() == "1" else 0, collapse(seq)[:a.max_events], True,
                first_seen="2021")

    # ---- Mal-API-2019 (all malware; choose a type-balanced subset, then read only those lines)
    labels = [l.strip() for l in open(P["malapi_labels"], encoding="utf-8") if l.strip()]
    by_type = defaultdict(list)
    for i, t in enumerate(labels):
        by_type[t].append(i)
    per_type = max(1, a.malapi_max // len(by_type))
    chosen = {}
    for t, idxs in sorted(by_type.items()):
        rng.shuffle(idxs)
        for i in idxs[:per_type]:
            chosen[i] = t
    print(f"Mal-API-2019: using {len(chosen)} of {len(labels)} runs ({per_type} per type); reading the zip...")
    with zipfile.ZipFile(P["malapi_zip"]) as z, z.open("all_analysis_data.txt") as f:
        for i, line in enumerate(f):
            if i not in chosen:
                continue
            words = line.decode("utf-8", "ignore").split(None, a.max_raw)[:a.max_raw]
            seq = [lower2canon[w] for w in words if w in lower2canon]
            add("malapi", f"malapi-{i}", 1, collapse(seq)[:a.max_events], True, extra={"malware_type": chosen[i]})

    rng.shuffle(runs)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    vp = os.path.splitext(a.out)[0] + ".vocab.json"
    json.dump(vocab, open(vp, "w", encoding="utf-8"), indent=1)

    print(f"\nwrote {len(runs)} runs to {a.out} and the {len(vocab)}-token vocabulary to {vp}")
    cnt = Counter((r["source"], r["label"]) for r in runs)
    names3 = ["benign", "malware", "ransomware"]
    for (s, c), n in sorted(cnt.items()):
        print(f"  {s:<10} {names3[c]:<11} {n}")
    for (s, c, why), n in sorted(dropped.items()):
        print(f"  dropped {n:>5} {s}/{names3[c]}: {why}")
    fams = Counter(r["family"] for r in runs if r["label"] == 2)
    print(f"ransomware families: {len(fams)}; smallest: {sorted(fams.items(), key=lambda x: x[1])[:5]}")


if __name__ == "__main__":
    main()
