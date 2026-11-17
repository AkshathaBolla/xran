# XRan – explainable ransomware / malware / benign detection

One transformer, three answers: **benign**, **malware** or **ransomware** (plus the closest ransomware
family, a "looks new" flag, and which behaviours drove the verdict).

```
 your program ──► record_events.py ──► events_to_run.py ──► deploy.Detector (xran.pt) ──► verdict
                  (Frida hooks)         (log → tokens)        (transformer.py)             + explanation
                         └───────────── analyze_app.py does all three steps in one command ─────────────┘

 public datasets ──► build_dataset.py ──► runs.jsonl ──► train.py ──► xran.pt
```

## What each file does

| File | Role |
|---|---|
| **transformer.py** | The model itself: `XRanTransformer` (token embeddings, [CLS], encoder layers, 3-class head, family head), `PrototypeStore` (new-family check), `integrated_gradients` (explanations). In *set* mode position embeddings are off, so the model reads a run as an unordered set of behaviours. |
| **deploy.py** | Loads `xran.pt` and scores runs: `Detector` (one finished run), `StreamDetector` (live events). Also holds `normalize_event`, the one function that cleans every token for both training and scoring. |
| **train.py** | Trains the transformer from `runs.jsonl`, with leak-free splits, and prints the full evaluation (per class, per source, held-out families, early detection). Saves `xran.pt`. |
| **build_dataset.py** | Downloads MLRan, MalbehavD-V1 and Mal-API-2019 from GitHub and builds `runs.jsonl` + `runs.vocab.json` (58 shared API tokens). |
| **runs.jsonl / runs.vocab.json** | The ready-made training data (4,716 runs) and its token list. Rebuild any time with `build_dataset.py`. |
| **baseline.py** | Random forest / logistic regression on the same split, for an honest comparison (needs `pip install scikit-learn`). |
| **record_events.py** | Starts or attaches to a Windows program with Frida and logs its API calls, DLL loads, mutexes, files, registry, process and network activity. |
| **events_to_run.py** | Turns a recorder log into model input (`--token-types api` for this model). |
| **analyze_app.py** | Pick any application → record → convert → score, with a file-picker window or one command. |
| **inspect_dataset.py / convert.py** | For adding other datasets (Mendeley etc.): look at a file's structure, then convert it into `runs.jsonl` format. |
| **test.py** | 14 unit tests for the model. |
| **xran.pt** | The trained model. |
| **MODEL_CARD.md** | What the model was trained on, measured results, and what it can and cannot claim. **Read this before trusting a verdict.** |

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch frida frida-tools
pip install scikit-learn          # only for baseline.py
python test.py                    # expect: 14/14 checks passed
```

## Use the trained model on your own application

```powershell
python analyze_app.py                                   # window: browse to your program, click Analyze
python analyze_app.py --app "C:\path\myapp.exe" --duration 60
```

It starts the program for real, so use programs you trust, or a virtual machine for anything unknown.
Results are saved in `analysis\` (the raw log, the model input, and a JSON report). Re-score a saved log with
`python analyze_app.py --log analysis\myapp_....jsonl`.

## Retrain from scratch

```powershell
python build_dataset.py                                   # downloads about 20 MB, writes runs.jsonl
python train.py --data runs.jsonl --mode set --W 64 --d-model 64 --heads 4 --layers 2 --d-ff 128 --epochs 40 --lr 1e-3 --patience 6 --out xran.pt
```

Diagnostics (each is a separate training run):

```powershell
python train.py --data runs.jsonl --mode set --W 64 --holdout-families 8 --out tmp.pt          # does the new-family check work?
python train.py --data runs.jsonl --mode set --W 64 --exclude-from-train malbehavd:0 --out tmp.pt   # how is an unseen benign source treated?
python train.py --data runs.jsonl --mode set --W 64 --exclude-from-train mlran:0 --out tmp.pt
python baseline.py --data runs.jsonl
```

## Add your own labeled recordings (the best way to improve it)

```powershell
python analyze_app.py --app "C:\path\goodapp.exe" --label 0 --append-to my_runs.jsonl --no-score
```

`--label` is 0 benign, 1 malware, 2 ransomware (record malware or ransomware only inside an isolated VM).
Then combine `runs.jsonl` and `my_runs.jsonl` and retrain. Runs recorded by this recorder match what
`analyze_app.py` feeds the model, which the public datasets (recorded with Cuckoo) do not.

## Other datasets (for example Mendeley)

```powershell
python inspect_dataset.py path\to\file_or_folder          # see the structure
python convert.py ...                                     # see the docstring in convert.py for the three layouts
```
Keep only tokens from `runs.vocab.json`, otherwise the new source brings tokens the others lack and the model
can learn the source instead of the behaviour.
