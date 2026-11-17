"""
analyze_app.py - give it an application of your own; it records the behavior, converts the log into
transformer input, and scores it with your trained model.

    python analyze_app.py                                   # opens a window: browse to your application
    python analyze_app.py --app "C:\\path\\myapp.exe" --duration 60
    python analyze_app.py --app myscript.py --args="--input data.txt" --duration 30
    python analyze_app.py --log analysis\\myapp_20261111_101500.jsonl      # re-score an earlier recording
    python analyze_app.py --app "C:\\path\\goodapp.exe" --label 0 --append-to runs.jsonl --no-score   # collect training data

Supported: .exe .com .py .pyw .bat .cmd .ps1 .jar  (installers: run the installer's .exe directly)

Needs record_events.py, events_to_run.py and deploy.py in the same folder, plus `pip install frida frida-tools`.
Without a trained model (--ckpt, default xran.pt) it still records and converts, and tells you so.

SAFETY: the application is really started on this computer. Use programs you trust, or run unknown
samples only inside an isolated virtual machine.
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SUPPORTED = (".exe", ".com", ".py", ".pyw", ".bat", ".cmd", ".ps1", ".jar", "")


def build_command(app, args):
    """Return (program to start, argument list) for the kind of file the user picked."""
    app = os.path.abspath(app)
    ext = os.path.splitext(app)[1].lower()
    extra = shlex.split(args, posix=False) if args else []
    extra = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in extra]
    if ext in ("", ".exe", ".com"):
        return app, extra
    if ext in (".py", ".pyw"):
        return sys.executable, [app] + extra
    if ext in (".bat", ".cmd"):
        return os.environ.get("COMSPEC", shutil.which("cmd.exe") or "cmd.exe"), ["/c", app] + extra
    if ext == ".ps1":
        ps = shutil.which("powershell.exe") or shutil.which("pwsh")
        if not ps:
            raise RuntimeError("PowerShell was not found on PATH")
        return ps, ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", app] + extra
    if ext == ".jar":
        java = shutil.which("java")
        if not java:
            raise RuntimeError("Java was not found on PATH")
        return java, ["-jar", app] + extra
    raise RuntimeError(f"unsupported file type {ext!r}. Supported: {', '.join(e for e in SUPPORTED if e)}")


def analyze(app=None, args="", duration=60, ckpt="xran.pt", workdir="analysis", follow_children=True,
            kill=True, token_types="api", log=None, emit=print, label=None, append_to=None, score=True):
    """Record (unless log is given), convert, score. Returns a report dict. All text goes through emit()."""
    from events_to_run import load_log, tokens_for, summarize, collapse
    if append_to and label is None:
        raise RuntimeError("--append-to needs --label (0 benign, 1 malware, 2 ransomware)")
    os.makedirs(workdir, exist_ok=True)
    types = {t.strip() for t in token_types.split(",") if t.strip()}

    if log is None:
        if not app or not os.path.exists(app):
            raise RuntimeError(f"application not found: {app!r}")
        spawn, spawn_args = build_command(app, args)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = os.path.join(workdir, f"{os.path.splitext(os.path.basename(app))[0]}_{stamp}.jsonl")
        emit(f"Starting: {spawn} {subprocess.list2cmdline(spawn_args)}")
        emit(f"Recording for up to {duration:g} seconds -> {log}")
        cmd = [sys.executable, os.path.join(HERE, "record_events.py"), "--spawn", spawn,
               "--args=" + subprocess.list2cmdline(spawn_args), "--out", log, "--duration", str(duration)]
        if follow_children:
            cmd.append("--follow-children")
        if kill:
            cmd.append("--kill")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            emit("  " + line.rstrip())
        p.wait()
        if p.returncode != 0 or not os.path.exists(log):
            raise RuntimeError("recording failed (see the messages above)")
    else:
        emit(f"Using existing recording: {log}")

    meta, events = load_log(log)
    if not events:
        raise RuntimeError("the recording contains no events. The program may have exited immediately, "
                           "or security software blocked Frida.")
    tokens = [t for ev in events for t in tokens_for(ev, types)]
    tokens = tokens[:200000]
    emit(f"\nRecorded {len(events)} raw events -> {len(tokens)} tokens ({', '.join(sorted(types))})")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize(events)
    for line in buf.getvalue().splitlines():
        emit(line)

    report = {"app": app, "log": log, "raw_events": len(events), "tokens": len(tokens),
              "token_types": sorted(types), "result": None}
    run_path = os.path.splitext(log)[0] + ".run.jsonl"
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"sha": meta.get("sha256") or os.path.basename(log), "family": None,
                            "events": tokens, "source": "xran_recorder"}) + "\n")
    report["run_file"] = run_path

    if append_to:
        if label is None:
            raise RuntimeError("--append-to needs --label (0 benign, 1 malware, 2 ransomware)")
        with open(append_to, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sha": meta.get("sha256") or os.path.basename(log), "label": label, "family": None,
                                "events": tokens, "source": "xran_recorder",
                                "log": os.path.basename(log)}) + "\n")
        emit(f"\nAdded this run to {append_to} as class {label} for training.")

    if not score:
        emit("\nScoring skipped (--no-score).")
    elif not os.path.exists(ckpt):
        emit(f"\nNo trained model at {ckpt!r}, so nothing was scored. Train one (train.py), then run:")
        emit(f"  python deploy.py --ckpt {ckpt} --events {run_path}")
    else:
        try:
            from deploy import Detector
        except ImportError as e:
            raise RuntimeError(f"could not load the model code: {e}")
        emit("\nScoring with the model...")
        res = Detector(ckpt).predict(tokens)
        report["result"] = res
        emit("\n=== RESULT ===")
        emit(f"verdict: {res.get('verdict')}   scores: {res.get('scores')}")
        if res.get("mode") == "set":
            emit(f"behaviors recognised: {res.get('known_behaviors')}  (how many of the behaviours this model knows were seen)")
        else:
            emit(f"unknown token rate: {res.get('unknown_token_rate')}  (share of events the model never saw in training)")
        if res.get("warning"):
            emit("WARNING: " + res["warning"])
        if res.get("mode") != "set" and len(tokens) < 127:
            emit("NOTE: fewer than 127 tokens were recorded, so the model saw one padded window. Record longer.")
        if res.get("family"):
            emit(f"closest ransomware family: {res['family']} (confidence {res.get('family_confidence')})")
        if res.get("novel") is not None:
            emit(f"looks new/unseen: {res['novel']} (distance {res.get('novelty_distance')} vs threshold "
                 f"{res.get('novelty_threshold')}, nearest {res.get('nearest_family')})")
        if res.get("top_tokens"):
            emit("tokens that drove the verdict: " + ", ".join(f"{t} ({a:+.2f})" for t, a in res["top_tokens"][:8]))
        if res.get("behaviors"):
            emit("behavior groups: " + "; ".join(f"{g}: {', '.join(ts)}" for g, ts in res["behaviors"].items()))
    rp = os.path.splitext(log)[0] + ".report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    emit(f"\nSaved: {log}\n       {run_path}\n       {rp}")
    return report


def gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("XRan - analyze an application")
    app_v, args_v, dur_v = tk.StringVar(), tk.StringVar(), tk.StringVar(value="60")
    ckpt_v = tk.StringVar(value="xran.pt")
    follow_v, kill_v = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
    q = queue.Queue()

    def row(r, label, var, browse=None):
        tk.Label(root, text=label, anchor="w").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        tk.Entry(root, textvariable=var, width=70).grid(row=r, column=1, padx=6)
        if browse:
            tk.Button(root, text="Browse...", command=browse).grid(row=r, column=2, padx=6)

    def pick_app():
        p = filedialog.askopenfilename(title="Choose your application", filetypes=[
            ("Programs", "*.exe *.py *.pyw *.bat *.cmd *.ps1 *.jar"), ("All files", "*.*")])
        if p:
            app_v.set(p)

    def pick_ckpt():
        p = filedialog.askopenfilename(title="Choose the trained model", filetypes=[("Model", "*.pt"), ("All", "*.*")])
        if p:
            ckpt_v.set(p)

    row(0, "Application", app_v, pick_app)
    row(1, "Arguments (optional)", args_v)
    row(2, "Record for (seconds)", dur_v)
    row(3, "Trained model (.pt)", ckpt_v, pick_ckpt)
    tk.Checkbutton(root, text="Also record programs it launches", variable=follow_v).grid(row=4, column=1, sticky="w")
    tk.Checkbutton(root, text="Close the application when recording ends", variable=kill_v).grid(row=5, column=1, sticky="w")
    tk.Label(root, fg="#a00", wraplength=620, justify="left",
             text="The application will really run on this computer. Use programs you trust, or a virtual machine.").grid(
        row=6, column=0, columnspan=3, padx=6, pady=4, sticky="w")
    btn = tk.Button(root, text="Analyze")
    btn.grid(row=7, column=1, sticky="w", padx=6, pady=4)
    out = scrolledtext.ScrolledText(root, width=110, height=28)
    out.grid(row=8, column=0, columnspan=3, padx=6, pady=6)

    def poll():
        try:
            while True:
                kind, msg = q.get_nowait()
                if kind == "line":
                    out.insert("end", msg + "\n")
                    out.see("end")
                else:
                    btn.config(state="normal")
        except queue.Empty:
            pass
        root.after(100, poll)

    def work(args):
        try:
            analyze(emit=lambda s: q.put(("line", s)), **args)
        except Exception as e:
            q.put(("line", f"\nERROR: {e}"))
        q.put(("done", ""))

    def go():
        if not app_v.get().strip():
            messagebox.showinfo("XRan", "Choose an application first.")
            return
        if not messagebox.askokcancel("Run this program?", f"This will start:\n\n{app_v.get()}\n\non this computer."):
            return
        try:
            dur = float(dur_v.get())
        except ValueError:
            messagebox.showinfo("XRan", "Seconds must be a number.")
            return
        out.delete("1.0", "end")
        btn.config(state="disabled")
        threading.Thread(target=work, daemon=True, args=(dict(
            app=app_v.get().strip(), args=args_v.get(), duration=dur, ckpt=ckpt_v.get().strip(),
            follow_children=follow_v.get(), kill=kill_v.get()),)).start()

    btn.config(command=go)
    poll()
    root.mainloop()


def main():
    if len(sys.argv) == 1:
        return gui()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui", action="store_true", help="open the file-picker window")
    ap.add_argument("--app", help="path of your application or script")
    ap.add_argument("--args", default="", help='arguments for it; write --args="--flag value"')
    ap.add_argument("--duration", type=float, default=60, help="seconds to record (default 60)")
    ap.add_argument("--ckpt", default="xran.pt", help="trained model from train.py")
    ap.add_argument("--workdir", default="analysis", help="where logs and reports are saved")
    ap.add_argument("--token-types", default="api", help="api (default). Others only if your model was trained on them")
    ap.add_argument("--no-follow", action="store_true", help="do not record programs it launches")
    ap.add_argument("--keep-running", action="store_true", help="leave the application running afterwards")
    ap.add_argument("--log", help="score an existing recording instead of running anything")
    ap.add_argument("--label", type=int, choices=[0, 1, 2], help="class of this application: 0 benign, 1 malware, 2 ransomware")
    ap.add_argument("--append-to", help="add this labeled run to a training file (runs.jsonl) for train.py")
    ap.add_argument("--no-score", action="store_true", help="record and convert only")
    a = ap.parse_args()
    if a.gui:
        return gui()
    if not a.app and not a.log:
        ap.error("give --app (or --log, or run with no arguments for the window)")
    try:
        analyze(app=a.app, args=a.args, duration=a.duration, ckpt=a.ckpt, workdir=a.workdir,
                follow_children=not a.no_follow, kill=not a.keep_running, token_types=a.token_types, log=a.log,
                label=a.label, append_to=a.append_to, score=not a.no_score)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
