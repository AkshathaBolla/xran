"""
events_to_run.py - turn recorder logs (from record_events.py) into transformer input.

    python events_to_run.py --in logs\\app_run1.jsonl --out runs_app.jsonl
    python events_to_run.py --in logs\\ --out runs_app.jsonl                     # every .jsonl in a folder
    python events_to_run.py --in logs\\app_run1.jsonl --out runs_app.jsonl --events-txt stream.txt
    python events_to_run.py --in logs\\benign\\ --out runs.jsonl --label 0 --source myrecorder --append   # training data

Score it:
    python deploy.py --ckpt xran.pt --events runs_app.jsonl
    Get-Content stream.txt | python deploy.py --ckpt xran.pt --stream         (PowerShell)

Output: one JSON object per recorded run: {"sha", "family", "events": [tokens...], "source", ...}
and "label" (0 benign / 1 malware / 2 ransomware) only if you pass --label.

Token types (--token-types, comma separated, default: api):
    api    API:<function name>                    one per recorded call
    dll    DLL:<file name>                        every module loaded (initial and later)
    mutex  MUTEX:<name>                           mutexes created or opened
    file   FILE:<path>                            paths touched (train/deploy keep only the extension)
    reg    REG:run_key | service | policy | other registry key class
    cmd    CMD:shadow_delete | bcdedit_recovery | ...   known-risky command lines
IMPORTANT: use only the token types your model was TRAINED on. A model trained on API-only datasets
(Mendeley, MalbehavD, Mal-API-2019) has never seen DLL/MUTEX/FILE/REG/CMD tokens; they would all be
treated as unknown. Keep the default (api) for those models.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

CMD_RULES = [
    (re.compile(r"vssadmin.*delete\s+shadows", re.I), "shadow_delete"),
    (re.compile(r"wmic.*shadowcopy.*delete", re.I), "shadow_delete"),
    (re.compile(r"wbadmin.*delete", re.I), "backup_delete"),
    (re.compile(r"bcdedit.*(recoveryenabled|bootstatuspolicy)", re.I), "bcdedit_recovery"),
    (re.compile(r"cipher(\.exe)?\s+/w", re.I), "cipher_wipe"),
    (re.compile(r"schtasks", re.I), "schtasks"),
    (re.compile(r"powershell", re.I), "powershell"),
]


def classify_cmd(cmd):
    for rx, name in CMD_RULES:
        if rx.search(cmd):
            return name
    return "other"


def classify_key(key):
    k = key.lower().replace("/", "\\")
    if k.endswith("\\run") or "\\run\\" in k or k.endswith("\\runonce"):
        return "run_key"
    if "\\services\\" in k or k.endswith("\\services"):
        return "service"
    if "\\policies\\" in k:
        return "policy"
    return "other"


def base(path):
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def tokens_for(ev, types):
    api, args = ev.get("api", ""), ev.get("args") or {}
    out = []
    if "api" in types and api != "__module__":
        out.append("API:" + api)
    if "dll" in types and args.get("dll"):
        out.append("DLL:" + base(args["dll"]).lower())
    if "mutex" in types and args.get("mutex"):
        out.append("MUTEX:" + args["mutex"])
    if "file" in types:
        for k in ("path", "path2"):
            if args.get(k):
                out.append("FILE:" + args[k])
    if "reg" in types and args.get("key"):
        out.append("REG:" + classify_key(args["key"]))
    if "cmd" in types and args.get("cmd"):
        out.append("CMD:" + classify_cmd(args["cmd"]))
    return out


def load_log(path):
    meta, events = {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "meta" and "recorder" in r:
                meta = r
            elif r.get("type") == "ev":
                events.append(r)
    events.sort(key=lambda e: (e.get("ts", 0), e.get("pid", 0), e.get("seq", 0)))
    return meta, events


def collapse(ev):
    return [e for i, e in enumerate(ev) if i == 0 or e != ev[i - 1]]


def summarize(events):
    apis = Counter(e["api"] for e in events if e["api"] != "__module__")
    dlls = Counter(base((e.get("args") or {}).get("dll", "")).lower() for e in events if (e.get("args") or {}).get("dll"))
    mutexes = sorted({(e.get("args") or {}).get("mutex") for e in events if (e.get("args") or {}).get("mutex")})
    exts = Counter()
    for e in events:
        for k in ("path", "path2"):
            p = (e.get("args") or {}).get(k)
            if p and "." in base(p):
                exts["." + base(p).rsplit(".", 1)[-1].lower()] += 1
    cmds = Counter(classify_cmd(e["args"]["cmd"]) for e in events if (e.get("args") or {}).get("cmd"))
    keys = Counter(classify_key(e["args"]["key"]) for e in events if (e.get("args") or {}).get("key"))
    print(f"  raw events: {len(events)} from {len({e.get('pid') for e in events})} process(es)")
    print("  top APIs:", apis.most_common(10))
    print(f"  DLLs seen: {len(dlls)}; mutexes: {mutexes[:10]}")
    print("  file extensions touched:", exts.most_common(8))
    print("  registry key classes:", dict(keys), "| command classes:", dict(cmds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="log files or folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--token-types", default="api", help="comma list of: api,dll,mutex,file,reg,cmd (default api)")
    ap.add_argument("--label", type=int, choices=[0, 1, 2], help="class of these runs, for training data")
    ap.add_argument("--source", default="xran_recorder")
    ap.add_argument("--collapse-repeats", action="store_true")
    ap.add_argument("--max-events", type=int, default=5000)
    ap.add_argument("--events-txt", help="also write the first run's tokens, one per line (for deploy.py --stream)")
    ap.add_argument("--quiet", action="store_true", help="skip the per-run summary")
    a = ap.parse_args()

    types = {t.strip() for t in a.token_types.split(",") if t.strip()}
    bad = types - {"api", "dll", "mutex", "file", "reg", "cmd"}
    if bad:
        sys.exit(f"unknown token types: {sorted(bad)}")

    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".jsonl"))
        else:
            files.append(p)
    if not files:
        sys.exit("no input logs found")

    written, first_tokens = 0, None
    with open(a.out, "a" if a.append else "w", encoding="utf-8") as out:
        for path in files:
            meta, events = load_log(path)
            if not events:
                print(f"skipping {path}: no events recorded")
                continue
            toks = [t for ev in events for t in tokens_for(ev, types)]
            if a.collapse_repeats:
                toks = collapse(toks)
            toks = toks[:a.max_events]
            rec = {"sha": meta.get("sha256") or os.path.splitext(os.path.basename(path))[0],
                   "family": None, "events": toks, "source": a.source,
                   "log": os.path.basename(path), "raw_events": len(events)}
            if a.label is not None:
                rec["label"] = a.label
            out.write(json.dumps(rec) + "\n")
            written += 1
            if first_tokens is None:
                first_tokens = toks
            print(f"{path}: {len(events)} raw events -> {len(toks)} tokens ({', '.join(sorted(types))})")
            if len(toks) < 127:
                print(f"  note: fewer than 127 tokens; the model will see one padded window, so treat the verdict with care")
            if not a.quiet:
                summarize(events)
    print(f"\nwrote {written} run(s) to {a.out}")
    if a.events_txt and first_tokens is not None:
        with open(a.events_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(first_tokens) + "\n")
        print(f"wrote the first run's tokens to {a.events_txt}")


if __name__ == "__main__":
    main()
