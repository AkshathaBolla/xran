import json
from collections import Counter,defaultdict
from deploy import normalize_event

missing = {
    "API:createfile",
    "API:createmutex",
    "API:createmutexex",
    "API:getdrivetype",
    "API:isdebuggerpresent",
    "API:ldrloaddll",
    "API:loadlibrary",
    "API:loadlibraryex",
    "API:openmutex",
    "API:readfile",
    "API:regopenkeyex",
    "API:regqueryvalueex",
    "API:virtualalloc",
    "API:virtualprotect",
    "API:shellexecute"
}

event_counts = Counter()
run_counts = Counter()
class_counts = defaultdict(Counter)
source_counts = defaultdict(Counter)

with open("runs_v2.jsonl",encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        found = set()

        for event in data.get("events",[]):
            token = normalize_event(event)
            if token in missing:
                event_counts[token] += 1
                found.add(token)

        label = str(data.get("label","unknown"))
        source = str(data.get("source","unknown"))

        for token in found:
            run_counts[token] += 1
            class_counts[token][label] += 1
            source_counts[token][source] += 1

print("MISSING API FREQUENCY IN TRAINING DATA")
for token in sorted(missing):
    print("\n",token)
    print("Event occurrences:",event_counts[token])
    print("Runs containing API:",run_counts[token])
    print("Class distribution:",dict(class_counts[token]))
    print("Source distribution:",dict(source_counts[token]))