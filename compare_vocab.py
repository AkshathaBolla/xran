import json
import torch
from deploy import normalize_event

ckpt = torch.load("tmp.pt", map_location="cpu", weights_only=False)
vocab = set(ckpt["stoi"])

files = [
    "runs_app.jsonl",
    "calculator_run.jsonl",
    "explorer_run.jsonl"
]

print("MODEL VOCABULARY:", len(vocab))

for filename in files:
    with open(filename, encoding="utf-8") as f:
        data = json.loads(f.readline())

    raw_events = set(data["events"])
    normalized = {normalize_event(e) for e in raw_events}
    known = normalized & vocab
    unknown = normalized - vocab

    print("\nFILE:", filename)
    print("Unique raw APIs:", len(raw_events))
    print("Unique normalized APIs:", len(normalized))
    print("Known APIs:", len(known))
    print("Unknown APIs:", len(unknown))
    print("Coverage: {:.2f}%".format(
        len(known) / len(normalized) * 100 if normalized else 0
    ))
    print("Unknown tokens:", sorted(unknown))