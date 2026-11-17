"""
baseline.py - simple non-transformer baselines on the SAME split as train.py, for an honest comparison.
Needs: pip install scikit-learn

    python baseline.py --data runs.jsonl
"""
import argparse
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from deploy import normalize_event, LABELS
from train import load, split_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--holdout-families", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = load(a.data)
    train, val, test, novel = split_runs(runs, a.split, a.holdout_families, a.seed)
    vocab = sorted({normalize_event(e) for r in train for e in r["events"]})
    idx = {t: i for i, t in enumerate(vocab)}

    def X(rs):
        m = np.zeros((len(rs), len(vocab)), dtype=np.float32)
        for i, r in enumerate(rs):
            for e in r["events"]:
                j = idx.get(normalize_event(e))
                if j is not None:
                    m[i, j] = 1
        return m

    Xtr, ytr = X(train + val), np.array([r["label"] for r in train + val])
    Xte, yte = X(test), np.array([r["label"] for r in test])
    print(f"train {len(Xtr)} (train+val) | test {len(Xte)} | tokens {len(vocab)}")
    for name, clf in [("logistic regression", LogisticRegression(max_iter=3000, class_weight="balanced")),
                      ("random forest", RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0))]:
        pred = clf.fit(Xtr, ytr).predict(Xte)
        f1s = []
        print(f"\n== {name}")
        for c in range(3):
            tp = int(((pred == c) & (yte == c)).sum()); fp = int(((pred == c) & (yte != c)).sum()); fn = int(((pred != c) & (yte == c)).sum())
            p_ = tp / (tp + fp) if tp + fp else 0.0; r_ = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
            f1s.append(f1)
            print(f"  {LABELS[c]:<11} precision {p_:.3f}  recall {r_:.3f}  F1 {f1:.3f}  (n={int((yte == c).sum())})")
        print(f"  macro-F1 {np.mean(f1s):.3f}")
        src = np.array([r.get("source", "?") for r in test])
        for s_ in sorted(set(src)):
            m = src == s_
            rec = {LABELS[c]: round(float((pred[m & (yte == c)] == c).mean()), 3) for c in range(3) if (m & (yte == c)).any()}
            print(f"    {s_:<10} n={int(m.sum()):<5} accuracy {float((pred[m] == yte[m]).mean()):.3f}  recall {rec}")


if __name__ == "__main__":
    main()
