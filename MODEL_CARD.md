# Model card – `xran.pt`

**What it is.** A small custom transformer (79,362 parameters) that reads a program run as an *unordered set of
58 API-call behaviours* and answers **benign / malware / ransomware**, plus the closest of 63 ransomware
families, a "looks new" flag and the behaviours that drove the verdict (Integrated Gradients).

**Bottom line.** It is a working, honestly evaluated *prototype*, not a dependable detector. On held-out runs from
the training data it reaches a macro-F1 of about 0.79. Two checks below show that its "benign" and
"ransomware vs malware" decisions lean on **which dataset a run came from**, so its verdicts on a program
recorded by your own tools are not yet trustworthy. Record your own labeled runs and retrain (see README) before
relying on it.

## Training data (`runs.jsonl`, built by `build_dataset.py`)

| Source | Class | Runs used |
|---|---|---|
| MLRan (feature-selected release) | benign | 1,033 |
| MLRan | ransomware (63 families) | 889 |
| MalbehavD-V1 | benign | 779 |
| MalbehavD-V1 | malware | 265 |
| Mal-API-2019 (3,000 of 7,107 taken, balanced over 8 types) | malware | 1,750 |
| **Total** | | **4,716** |

- Every run is reduced to the **58 API tokens** that MLRan's published features share with the other two datasets.
- Runs with fewer than 3 known tokens were dropped, and exact duplicate token sets within a source and class were
  dropped (this removed most of MalbehavD's malware: 1,285 → 265, and 2,330 → 889 MLRan ransomware runs).
- MLRan runs are sets (no call order); MalbehavD and Mal-API-2019 runs keep call order.

## Model

Set mode: one window of the distinct known tokens, **no position embeddings** (verified permutation-invariant by
`test.py`). 2 encoder layers, 4 heads, width 64, feed-forward 128, [CLS] pooling, 3-class head, 63-family head
(loss only on ransomware runs), class-weighted loss, AdamW 1e-3, early stopping on validation macro-F1
(stopped after epoch 28 of 40). Split: 70/15/15 by run, stratified by class and ransomware family.

## Results (test split, 736 runs never used for training or model selection; single seed, no confidence intervals)

| Class | Precision | Recall | F1 | Runs |
|---|---|---|---|---|
| benign | 0.825 | 0.886 | 0.855 | 272 |
| malware | 0.817 | 0.782 | 0.799 | 303 |
| ransomware | 0.727 | 0.696 | 0.711 | 161 |

Macro-F1 **0.788**. Confusion matrix (rows true, columns predicted: benign, malware, ransomware):
`[241, 22, 9]`, `[33, 237, 33]`, `[18, 31, 112]`.

**Per source** (accuracy; recall per class): Mal-API malware 0.850; MalbehavD benign 0.843 but **malware only 0.372**;
MLRan benign 0.921, ransomware 0.696.

**Baselines on the identical split** (`baseline.py`, trained on train+val): logistic regression macro-F1 0.749;
**random forest 0.815** (benign 0.859, malware 0.826, ransomware 0.760). The random forest is slightly ahead of the
transformer on this data. With 58 binary tokens and about 4,000 training runs that is plausible; I did not test
whether the gap is statistically meaningful.

**Family classification:** 0.453 accuracy on the 161 known-family test ransomware runs (58 families present;
always guessing the most common family would score 0.056). The smallest training families have 1–2 runs, so
rare families are essentially unlearned.

**New-family check** (separate run holding out 8 ransomware families entirely, 84 novel-family test runs):
AUROC 0.709 for "novel scores higher than known". At the calibrated threshold it flags only 6.0% of novel-family
runs (3.5% of known ones). 63.1% of the unseen-family runs were still detected as ransomware (70.8% for known
families). **The novelty flag is a weak signal at best.**

**Early detection:** measurable only on ordered runs, which contain no ransomware (MLRan has no call order). On the
424 ordered test runs (benign and malware only) recall is flat from the first 32 recognised events (benign 0.84,
malware 0.79) because these sequences are short and show most of their distinct behaviours early. **Early
detection of ransomware could not be measured.** One illustrative stream (an MLRan ransomware run diluted 9:1 with
unrecognised events) turned "ransomware" after 112 events; that is a single example, not a measurement.

## The checks that expose the weaknesses

Ransomware comes only from MLRan; malware only from MalbehavD and Mal-API-2019. To see how much the model relies on
the source, whole benign groups were left out of training in two extra runs:

| Left out of training | Predicted benign | malware | ransomware |
|---|---|---|---|
| MalbehavD benign (779 runs) | **27.0%** | 44.2% | 28.9% |
| MLRan benign (1,033 runs) | 75.8% | 12.5% | 11.7% |

So "benign" does not transfer across sources: benign programs from an unfamiliar source are mostly called malware
or ransomware. Also, MLRan goodware uses a median of 39 distinct tokens against 15 for MalbehavD benign, so token
count alone separates sources. A program recorded by `record_events.py` is a *new source*.

## A real check on a program recorded by this project's recorder

A Notepad recording made with the earlier recorder (61 hooks, only 5 of the model's 58 behaviours observed) was
scored **"malware" (0.57 vs benign 0.42)**: a false positive on a benign program. This is the source problem in
practice. The recorder now hooks all 58 model APIs, but I could not re-record on Windows, so I do not know how much
that helps. The model now warns whenever fewer than 10 of the 58 behaviours were observed (training runs typically
show 12 to 40).

## What you can and cannot claim

- **Can claim:** a transformer-based, explainable classifier prototype; benign vs ransomware and malware vs benign
  inside the training sources; real family labels used for a family head; a full record, convert, score pipeline.
- **Cannot claim:** that it generalises to other sandboxes or to programs recorded with your recorder; that
  ransomware is separated from malware by behaviour (it may be separated by source); early/real-time detection of
  ransomware; reliable family or new-family detection.

## Other limits (not verified)

- **MalbehavD label direction.** The files do not say which label is malware. I assumed 1 = malware; the only
  evidence is that a model trained on MLRan alone scores label 1 as more ransomware-like (0.765) than label 0 (0.587).
  Check a few hashes on VirusTotal.
- **MLRan feature selection.** The 483 features were chosen by MLRan's authors; I did not verify that the selection
  used training data only, so the MLRan ransomware numbers may be optimistic.
- **Recorder coverage.** `record_events.py` now hooks the 58 model APIs (first call only), using my best guess of
  module and function names. It has never been run on real Windows with real Frida; names missing on your Windows
  version are skipped silently, so check `behaviors recognised` in every result.
- **Behaviour groups** in explanations are hand-written descriptive labels, not learned or validated categories.
- **Statistics:** one split, one seed, no confidence intervals.
- **Licences/citations:** MLRan and MalbehavD list MIT on their GitHub pages; I did not check Mal-API-2019's licence.
  Cite: arXiv 2505.18613 (MLRan), arXiv 1905.01999 (Mal-API-2019), and the MalbehavD-V1 repository.

## How to improve it (in order of value)

1. Record **your own labeled runs** with `analyze_app.py --label ... --append-to ...` (malware and ransomware only in
   an isolated VM) and retrain, so training and scoring use the same recorder.
2. Add ransomware from a second sandbox (for example the Mendeley dataset, which I could not download from my
   environment), filtered to `runs.vocab.json` tokens, so ransomware no longer comes from one source.
3. Look up MalbehavD's hashes on VirusTotal or MalwareBazaar to find its ransomware (lookup limits and terms not
   checked), which puts ransomware and malware in the same sandbox.

## Reproduce

```
python build_dataset.py
python train.py --data runs.jsonl --mode set --W 64 --d-model 64 --heads 4 --layers 2 --d-ff 128 --epochs 40 --lr 1e-3 --patience 6 --out xran.pt
```
Logs of every run reported here are in `reports/`.
