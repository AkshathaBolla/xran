"""
record_events.py - record the behavior of a Windows program into a JSONL log.

Records, in order: Win32 API calls, DLL loads, mutex creation/opening, file, registry, process,
memory, crypto, service and network calls, plus the arguments that matter (DLL name, mutex name,
file path, registry key, command line, host). Then use events_to_run.py to turn the log into
transformer input.

Setup (Windows, normal PowerShell):
    pip install frida frida-tools

Usage:
    # start a program under observation for 120 seconds (also follow programs it launches)
    python record_events.py --spawn "C:\\path\\to\\app.exe" --duration 120 --follow-children --out logs\\app_run1.jsonl
    # same, but score events live with a trained checkpoint
    python record_events.py --spawn "C:\\path\\to\\app.exe" --duration 120 --follow-children --live-ckpt tmp.pt --out logs\\app_live.jsonl
    # with arguments for the program
    python record_events.py --spawn "C:\\path\\to\\app.exe" --args="--input data.txt" --out logs\\app_run2.jsonl
    # attach to something already running (only calls made after attaching are recorded)
    python record_events.py --name notepad.exe --duration 60 --out logs\\notepad.jsonl
    python record_events.py --pid 1234 --out logs\\pid1234.jsonl

SAFETY: only run programs you trust on your own machine. Run unknown or suspicious samples ONLY in
an isolated virtual machine with a snapshot and no network access to anything you care about.
Press Ctrl-C to stop early; the log is flushed first.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shlex
import sys
import threading
import time

try:
    import frida
except ImportError:
    sys.exit("Frida is not installed. Run: pip install frida frida-tools")

RECORDER_VERSION = "xran-frida-3-live"

# (module, function, [(argument index, kind, field), ...])
# kind: w = UTF-16 string pointer, a = ANSI string pointer, u = UNICODE_STRING pointer
# field names are interpreted by events_to_run.py: dll, mutex, path, path2, key, cmd, host, url, name
HOOKS = [
    # ---- files
    ("kernel32.dll", "CreateFileW", [(0, "w", "path")]),
    ("kernel32.dll", "CreateFileA", [(0, "a", "path")]),
    ("kernel32.dll", "ReadFile", []),
    ("kernel32.dll", "WriteFile", []),
    ("kernel32.dll", "DeleteFileW", [(0, "w", "path")]),
    ("kernel32.dll", "MoveFileExW", [(0, "w", "path"), (1, "w", "path2")]),
    ("kernel32.dll", "CopyFileW", [(0, "w", "path"), (1, "w", "path2")]),
    ("kernel32.dll", "SetFileAttributesW", [(0, "w", "path")]),
    ("kernel32.dll", "FindFirstFileW", [(0, "w", "path")]),
    ("kernel32.dll", "FindNextFileW", []),
    ("kernel32.dll", "GetLogicalDrives", []),
    ("kernel32.dll", "GetDriveTypeW", [(0, "w", "path")]),
    # ---- mutexes
    ("kernel32.dll", "CreateMutexW", [(2, "w", "mutex")]),
    ("kernel32.dll", "CreateMutexA", [(2, "a", "mutex")]),
    ("kernel32.dll", "CreateMutexExW", [(1, "w", "mutex")]),
    ("kernel32.dll", "CreateMutexExA", [(1, "a", "mutex")]),
    ("kernel32.dll", "OpenMutexW", [(2, "w", "mutex")]),
    ("kernel32.dll", "OpenMutexA", [(2, "a", "mutex")]),
    # ---- DLL loading
    ("kernel32.dll", "LoadLibraryW", [(0, "w", "dll")]),
    ("kernel32.dll", "LoadLibraryA", [(0, "a", "dll")]),
    ("kernel32.dll", "LoadLibraryExW", [(0, "w", "dll")]),
    ("kernel32.dll", "LoadLibraryExA", [(0, "a", "dll")]),
    ("ntdll.dll", "LdrLoadDll", [(2, "u", "dll")]),
    # ---- registry
    ("advapi32.dll", "RegOpenKeyExW", [(1, "w", "key")]),
    ("advapi32.dll", "RegCreateKeyExW", [(1, "w", "key")]),
    ("advapi32.dll", "RegSetValueExW", [(1, "w", "key")]),
    ("advapi32.dll", "RegDeleteKeyW", [(1, "w", "key")]),
    ("advapi32.dll", "RegDeleteValueW", [(1, "w", "key")]),
    ("advapi32.dll", "RegQueryValueExW", [(1, "w", "key")]),
    # ---- processes, injection, memory
    ("kernel32.dll", "CreateProcessW", [(0, "w", "path"), (1, "w", "cmd")]),
    ("kernel32.dll", "CreateProcessA", [(0, "a", "path"), (1, "a", "cmd")]),
    ("kernel32.dll", "WinExec", [(0, "a", "cmd")]),
    ("shell32.dll", "ShellExecuteW", [(2, "w", "path"), (3, "w", "cmd")]),
    ("kernel32.dll", "OpenProcess", []),
    ("kernel32.dll", "CreateRemoteThread", []),
    ("kernel32.dll", "WriteProcessMemory", []),
    ("kernel32.dll", "VirtualAllocEx", []),
    ("kernel32.dll", "VirtualAlloc", []),
    ("kernel32.dll", "VirtualProtect", []),
    ("user32.dll", "SetWindowsHookExW", []),
    ("kernel32.dll", "IsDebuggerPresent", []),
    # ---- crypto
    ("advapi32.dll", "CryptAcquireContextW", []),
    ("advapi32.dll", "CryptGenKey", []),
    ("advapi32.dll", "CryptImportKey", []),
    ("advapi32.dll", "CryptEncrypt", []),
    ("bcrypt.dll", "BCryptGenerateSymmetricKey", []),
    ("bcrypt.dll", "BCryptEncrypt", []),
    # ---- services
    ("advapi32.dll", "OpenSCManagerW", []),
    ("advapi32.dll", "CreateServiceW", [(1, "w", "name")]),
    ("advapi32.dll", "StartServiceW", []),
    # ---- network
    ("wininet.dll", "InternetOpenW", []),
    ("wininet.dll", "InternetConnectW", [(1, "w", "host")]),
    ("wininet.dll", "InternetOpenUrlW", [(1, "w", "url")]),
    ("wininet.dll", "HttpSendRequestW", []),
    ("ws2_32.dll", "WSAStartup", []),
    ("ws2_32.dll", "socket", []),
    ("ws2_32.dll", "connect", []),
    ("ws2_32.dll", "send", []),
    ("ws2_32.dll", "recv", []),
    ("ws2_32.dll", "getaddrinfo", [(0, "a", "host")]),
    ("dnsapi.dll", "DnsQuery_W", [(0, "w", "host")]),
]

# The 58 API tokens the trained model knows (see runs.vocab.json). Each is hooked so that only its FIRST call
# is recorded (the model looks at presence, not counts), which keeps recordings small and the program fast.
# Module/function names are my best mapping from the Cuckoo names used in the training data; a function that
# does not exist on your Windows version is skipped silently.
MODEL_HOOKS = (
    [("ntdll.dll", f) for f in (
        "NtAllocateVirtualMemory", "NtProtectVirtualMemory", "NtDelayExecution", "NtGetContextThread",
        "NtSetContextThread", "NtOpenProcess", "NtQueryDirectoryFile", "NtQueryInformationFile", "NtQueryKey",
        "NtSetInformationFile", "NtWriteFile", "LdrGetProcedureAddress", "RtlAddVectoredExceptionHandler")]
    + [("kernel32.dll", f) for f in (
        "CreateActCtxW", "CreateDirectoryW", "CreateProcessInternalW", "CreateToolhelp32Snapshot", "DeviceIoControl",
        "FindResourceA", "FindResourceExW", "GetComputerNameA", "GetComputerNameW", "GetDiskFreeSpaceExW",
        "GetFileAttributesExW", "GetFileInformationByHandle", "GetNativeSystemInfo", "GetSystemDirectoryA",
        "GetSystemDirectoryW", "GetSystemWindowsDirectoryW", "GetTempPathW", "GlobalMemoryStatus", "LoadResource",
        "Process32FirstW", "Process32NextW", "RemoveDirectoryA", "RemoveDirectoryW", "WriteConsoleW")]
    + [("user32.dll", f) for f in (
        "DrawTextExW", "FindWindowW", "GetCursorPos", "GetSystemMetrics", "LoadStringA", "SendNotifyMessageW")]
    + [("advapi32.dll", f) for f in (
        "OpenServiceW", "RegCreateKeyExA", "RegEnumKeyExW", "RegEnumValueW", "RegQueryInfoKeyW", "RegSetValueExA")]
    + [("ole32.dll", f) for f in ("CoCreateInstance", "CoUninitialize")]
    + [("shell32.dll", f) for f in ("SHGetFolderPathW", "SHGetSpecialFolderLocation")]
    + [("version.dll", f) for f in ("GetFileVersionInfoSizeW", "GetFileVersionInfoW")]
    + [("rpcrt4.dll", "UuidCreate"), ("ws2_32.dll", "ioctlsocket")]
)


def all_specs():
    """[module, function, args, once] for every hook. 'once' = record only the first call."""
    have = {(m, f) for m, f, _ in HOOKS}
    specs = [[m, f, [[i, k, fld] for i, k, fld in a], False] for m, f, a in HOOKS]
    specs += [[m, f, [], True] for m, f in MODEL_HOOKS if (m, f) not in have]
    return specs


AGENT_JS = r"""
'use strict';
const HOOKS = __HOOKS__;
const MAX_STR = 300;
let seq = 0;
let buf = [];
const done = {};
let pending = HOOKS.slice();
let tries = 0;

function flush() {
  if (buf.length) { send({type: 'batch', pid: Process.id, events: buf}); buf = []; }
}
function emit(api, mod, args) {
  buf.push({seq: seq++, ts: Date.now(), tid: Process.getCurrentThreadId(), api: api, mod: mod, args: args});
  if (buf.length >= 500) flush();
}
function readArg(p, kind) {
  try {
    if (p.isNull()) return null;
    if (kind === 'w') return p.readUtf16String(MAX_STR);
    if (kind === 'a') return p.readUtf8String(MAX_STR);
    if (kind === 'u') {
      const len = p.readU16();
      const sp = p.add(Process.pointerSize).readPointer();
      if (sp.isNull()) return null;
      return sp.readUtf16String(Math.min(len / 2, MAX_STR));
    }
  } catch (e) {}
  return null;
}
function hook(spec) {                      // returns true when finished (hooked, or nothing to retry)
  const key = spec[0] + '!' + spec[1];
  if (done[key]) return true;
  const m = Process.findModuleByName(spec[0]);
  if (!m) return false;
  const addr = m.findExportByName(spec[1]);
  if (!addr) { done[key] = true; return true; }
  try {
    const listener = Interceptor.attach(addr, {
      onEnter: function (args) {
        const out = {};
        for (const [i, kind, field] of spec[2]) {
          const v = readArg(args[i], kind);
          if (v !== null && v !== undefined) out[field] = v;
        }
        emit(spec[1], spec[0], out);
        if (spec[3]) { try { listener.detach(); } catch (e) {} }   // first call only
      }
    });
  } catch (e) { send({type: 'warn', text: 'hook failed ' + key + ': ' + e}); }
  done[key] = true;
  return true;
}
function tryHooks() { pending = pending.filter(function (s) { return !hook(s); }); }

Process.enumerateModules().forEach(function (m) {
  emit('__module__', m.name, {dll: m.name.toLowerCase()});
});
tryHooks();
send({type: 'ready', pid: Process.id, hooked: Object.keys(done).length, waiting_for_modules: pending.length});
const timer = setInterval(function () {    // modules loaded later (wininet, ws2_32, ...) get hooked when they appear
  tryHooks(); tries++;
  if (pending.length === 0 || tries > 120) clearInterval(timer);
}, 500);
setInterval(flush, 200);
rpc.exports = { flush: function () { flush(); return true; } };
"""


def agent_source():
    return AGENT_JS.replace("__HOOKS__", json.dumps(all_specs()))


class Recorder:
    def __init__(self, device, out_file, follow_children, max_events, live_stream=None):
        self.device = device
        self.f = out_file
        self.follow = follow_children
        self.max_events = max_events
        self.live_stream = live_stream
        self.lock = threading.Lock()
        self.n = 0
        self.stop = threading.Event()
        self.stop_reason = None
        self.sessions, self.scripts, self.alive = {}, {}, set()

    def write(self, obj):
        with self.lock:
            self.f.write(json.dumps(obj) + "\n")

    def attach(self, pid):
        session = self.device.attach(pid)
        if self.follow:
            session.enable_child_gating()
        script = session.create_script(agent_source())
        script.on("message", lambda message, data, pid=pid: self.on_message(pid, message))
        session.on("detached", lambda reason, *rest, pid=pid: self.on_detached(pid, reason))
        self.sessions[pid], self.scripts[pid] = session, script
        self.alive.add(pid)
        script.load()

    def on_message(self, pid, message):
        if message.get("type") == "error":
            print(f"[agent error pid {pid}] {message.get('description')}", file=sys.stderr)
            return
        p = message.get("payload") or {}
        kind = p.get("type")
        if kind == "batch":
            accepted = []
            with self.lock:
                for ev in p["events"]:
                    if self.n >= self.max_events:
                        self.stop_reason = self.stop_reason or "max-events"
                        self.stop.set()
                        break
                    rec = {"type": "ev", "pid": p["pid"]}
                    rec.update(ev)
                    self.f.write(json.dumps(rec) + "\n")
                    self.n += 1
                    accepted.append(rec)
            # Keep inference outside the file lock so scoring never blocks log writes.
            if self.live_stream is not None and accepted:
                from events_to_run import tokens_for
                for ev in accepted:
                    for token in tokens_for(ev, {"api"}):
                        result = self.live_stream.push(token)
                        if result is not None:
                            scores = result.get("scores", {})
                            status = "ALERT" if result.get("alert") else "monitor"
                            print(
                                f"\n[LIVE {status}] events={result.get('events_seen')} "
                                f"risk={result.get('risk')} scores={scores} "
                                f"verdict={result.get('verdict')}"
                            , flush=True)
                            if result.get("warning"):
                                print("[LIVE warning] " + result["warning"], flush=True)
                            if result.get("alert"):
                                print("[LIVE explanation] " + str(result.get("top_tokens", [])[:8]), flush=True)
        elif kind == "ready":
            print(f"[pid {p['pid']}] hooks installed: {p['hooked']}; waiting for modules: {p['waiting_for_modules']}")
        elif kind == "warn":
            print(f"[pid {pid}] {p.get('text')}", file=sys.stderr)

    def on_detached(self, pid, reason):
        self.write({"type": "meta", "event": "detached", "pid": pid, "reason": str(reason)})
        self.alive.discard(pid)
        if not self.alive:
            self.stop_reason = self.stop_reason or "process-exited"
            self.stop.set()

    def on_child_added(self, child):
        try:
            self.write({"type": "meta", "event": "child", "pid": child.pid,
                        "parent_pid": getattr(child, "parent_pid", None), "path": getattr(child, "path", None)})
            self.attach(child.pid)
        except Exception as e:                       # never leave a gated child suspended
            print(f"could not attach to child {child.pid}: {e}", file=sys.stderr)
        finally:
            self.device.resume(child.pid)

    def flush_and_close(self):
        for pid, script in list(self.scripts.items()):
            try:
                try:
                    script.exports_sync.flush()
                except AttributeError:
                    script.exports.flush()
            except Exception:
                pass
        time.sleep(0.4)                              # let the final batches arrive
        for pid in list(self.scripts):
            try:
                self.scripts[pid].unload()
            except Exception:
                pass
            try:
                self.sessions[pid].detach()
            except Exception:
                pass


def split_args(text):
    """Split an argument string the Windows way and drop the quotes around quoted tokens."""
    toks = shlex.split(text, posix=False) if text else []
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]


def sha256_of(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spawn", help="path of a program to start under observation")
    g.add_argument("--pid", type=int, help="attach to a running process id")
    g.add_argument("--name", help="attach to a running process by name, e.g. notepad.exe")
    ap.add_argument("--args", default="", help="arguments for --spawn, as one string")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=120, help="seconds to record (default 120)")
    ap.add_argument("--follow-children", action="store_true", help="also record programs it launches")
    ap.add_argument("--max-events", type=int, default=500000)
    ap.add_argument("--kill", action="store_true", help="terminate a --spawn'ed program when recording ends")
    ap.add_argument("--live-ckpt", help="checkpoint to score incoming API events live (e.g. tmp.pt)")
    ap.add_argument("--alert-at", type=float, default=0.8, help="live ransomware-risk threshold (default 0.8)")
    a = ap.parse_args()

    if platform.system() != "Windows":
        print("warning: the hooks target Windows APIs; on this OS most will not be found.", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    device = frida.get_local_device()

    spawned, target_path, pid = False, None, None
    if a.spawn:
        target_path = a.spawn
        pid = device.spawn([a.spawn] + split_args(a.args))
        spawned = True
    elif a.pid:
        pid = a.pid
    else:
        matches = [p for p in device.enumerate_processes() if p.name.lower() == a.name.lower()]
        if not matches:
            sys.exit(f"no running process named {a.name!r}")
        pid = matches[0].pid
        if len(matches) > 1:
            print(f"{len(matches)} processes match; using pid {pid}", file=sys.stderr)

    live_stream = None
    if a.live_ckpt:
        from deploy import Detector, StreamDetector
        detector = Detector(a.live_ckpt)
        live_stream = StreamDetector(detector, alert_at=a.alert_at)
        print(f"[LIVE] loaded {a.live_ckpt}; mode={detector.mode}; scoring API tokens every {detector.S} events")
        if detector.mode == "set":
            print("[LIVE warning] This checkpoint is set-mode: it ignores event order. Live alerts are preliminary, not temporal early-detection evidence.")

    start = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        rec = Recorder(device, f, a.follow_children, a.max_events, live_stream=live_stream)
        rec.write({"type": "meta", "recorder": RECORDER_VERSION, "started": datetime.datetime.now().isoformat(),
                   "target": target_path or a.name or a.pid, "sha256": sha256_of(target_path) if target_path else None,
                   "root_pid": pid, "frida": getattr(frida, "__version__", "?"), "hooks": len(all_specs()),
                   "platform": platform.platform()})
        if a.follow_children:
            device.on("child_added", rec.on_child_added)
        try:
            rec.attach(pid)
            if spawned:
                device.resume(pid)
            while not rec.stop.is_set() and time.time() - start < a.duration:
                time.sleep(0.2)
            rec.stop_reason = rec.stop_reason or ("duration" if time.time() - start >= a.duration else "interrupted")
        except KeyboardInterrupt:
            rec.stop_reason = "interrupted"
        finally:
            rec.flush_and_close()
            if spawned and a.kill:
                try:
                    device.kill(pid)
                except Exception:
                    pass
            rec.write({"type": "meta", "event": "end", "events": rec.n, "seconds": round(time.time() - start, 1),
                       "stopped_because": rec.stop_reason})
    print(f"\nrecorded {rec.n} events in {time.time() - start:.1f}s (stopped: {rec.stop_reason}) -> {a.out}")
    print("next: python events_to_run.py --in", a.out, "--out runs_app.jsonl")


if __name__ == "__main__":
    main()
