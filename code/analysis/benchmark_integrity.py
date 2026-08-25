"""Reproduces README finding 4 — the DAVIS benchmark is partly self-contradictory.

DAVIS lists 442 target entries but far fewer unique sequences: the point mutations were
never applied to the sequence strings, so e.g. `BRAF` and `BRAF(V600E)` are byte-identical
model inputs that carry different affinity labels. This script measures the consequences:

  * how many target keys collapse into shared sequences
  * what fraction of TRAINING pairs are mutually contradictory (same input, different label)
  * the resulting irreducible floor on training MSE — no architecture can go below it
  * what fraction of cold-protein TEST pairs are unlearnable by construction, and how much
    test MSE that accounts for

Read-only. The split is regenerated in memory from data.csv, so this never touches the
CSVs on disk and is safe to run while a training job is in progress. CPU only.

    python code/analysis/benchmark_integrity.py                  # seed 41 (default)
    python code/analysis/benchmark_integrity.py --seed 42
    python code/analysis/benchmark_integrity.py --check-maps     # also verify contact maps
                                                                 # collide (loads a 1.3 GB pkl)
"""
import argparse
import collections
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cold_split import create_fold_setting_cold  # noqa: E402
from hyperparameter import HyperParameter        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=41, help='split seed to analyse')
    ap.add_argument('--dataset', default='davis')
    ap.add_argument('--check-maps', action='store_true',
                    help='also confirm colliding keys share an identical AF2 contact map')
    args = ap.parse_args()

    hp = HyperParameter()
    hp.set_dataset(args.dataset)
    prots = pd.read_csv(hp.prots_dir)
    df = pd.read_csv(os.path.join(hp.data_root, args.dataset, 'data.csv'))

    seq = dict(zip(prots.target_key.astype(str), prots.target_sequence))
    h = {k: hashlib.md5(v.encode()).hexdigest() for k, v in seq.items()}

    # ---------------------------------------------------------------- 1. collisions
    by_hash = collections.defaultdict(list)
    for k, v in h.items():
        by_hash[v].append(k)
    groups = [ks for ks in by_hash.values() if len(ks) > 1]
    dup = sum(len(g) - 1 for g in groups)

    print("=" * 74)
    print(f"  DAVIS BENCHMARK INTEGRITY  —  {args.dataset}, split seed {args.seed}")
    print("=" * 74)
    print("\n[1] Sequence collisions")
    print(f"    target keys                 : {len(prots)}")
    print(f"    UNIQUE sequences            : {prots.target_sequence.nunique()}")
    print(f"    keys sharing a sequence     : {sum(len(g) for g in groups)} "
          f"across {len(groups)} groups ({dup} redundant)")
    for g in sorted(groups, key=lambda x: -len(x))[:6]:
        print(f"      {len(g):>2} keys | {', '.join(sorted(g))[:88]}")

    # ------------------------------------------------- 2. contradictory training pairs
    fold = create_fold_setting_cold(df, args.seed, [0.8, 0.1, 0.1], ['target_key'])
    tr, te = fold['train'], fold['test']
    tr = tr.assign(h=tr.target_key.astype(str).map(h))
    te = te.assign(h=te.target_key.astype(str).map(h))

    grp = tr.groupby(['h', 'drug_key']).affinity
    size = grp.transform('size')
    within = (tr.affinity - grp.transform('mean')) ** 2   # best any model could do

    print("\n[2] Contradictory TRAINING pairs (identical input, different label)")
    print(f"    train pairs                 : {len(tr)}")
    print(f"    in a colliding group        : {int((size > 1).sum())} "
          f"({100 * (size > 1).mean():.1f}%)")
    print(f"    IRREDUCIBLE train-MSE floor : {within.mean():.4f}  <- no model can beat this")

    # ------------------------------------------------------ 3. unlearnable test pairs
    lut = tr.groupby(['h', 'drug_key']).affinity.mean()
    key = list(zip(te.h, te.drug_key))
    pred = np.array([lut.get(k, np.nan) for k in key])
    m = ~np.isnan(pred)
    err = (te.affinity.values[m] - pred[m]) ** 2

    print("\n[3] Unlearnable cold-protein TEST pairs")
    print(f"    test pairs                  : {len(te)}")
    print(f"    with an IDENTICAL (sequence, drug) in train : {int(m.sum())} "
          f"({100 * m.mean():.1f}%)")
    if m.sum():
        print(f"    best possible MSE on those  : {err.mean():.4f}  "
              f"(copy the training label)")
        print(f"    contribution to test MSE    : {err.sum() / len(te):.4f}  "
              f"<- unfixable by ANY architecture")
        affected = sorted(te[m].target_key.astype(str).unique())
        print(f"    affected test proteins ({len(affected)}): {', '.join(affected)}")

    # ------------------------------------------- 4. how different are the labels really
    piv = df.pivot_table(index='target_key', columns='drug_key', values='affinity')
    rows = []
    for g in groups:
        g = [k for k in g if k in piv.index]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = piv.loc[g[i]], piv.loc[g[j]]
                ok = a.notna() & b.notna()
                if ok.sum() >= 20:
                    rows.append(((a[ok] - b[ok]) ** 2).mean())
    if rows:
        print("\n[4] Label disagreement between identical-input pairs")
        print(f"    ALL within-group key-pairs  : {len(rows)}")
        print(f"      median MSE                : {np.median(rows):.4f}")
        print(f"      mean   MSE                : {np.mean(rows):.4f}")

    # EXPERIMENTS.md quotes the wildtype-vs-mutant subset specifically, which is a
    # narrower (and lower) figure than the all-pairs number above — mutant-vs-mutant
    # comparisons are excluded. Both are reported so either can be checked.
    import re
    wt_rows = []
    for k in piv.index:
        base = re.split(r'[(\-]', str(k))[0].strip()
        if base == str(k) or base not in piv.index:
            continue
        a, b = piv.loc[base], piv.loc[k]
        ok = a.notna() & b.notna()
        if ok.sum() >= 20 and h.get(base) == h.get(str(k)):
            wt_rows.append(((a[ok] - b[ok]) ** 2).mean())
    if wt_rows:
        print(f"    WILDTYPE-vs-mutant subset   : {len(wt_rows)} pairs")
        print(f"      median MSE                : {np.median(wt_rows):.4f}")
        print(f"      mean   MSE                : {np.mean(wt_rows):.4f}  "
              f"<- the figure quoted in EXPERIMENTS.md")

    # ------------------------------------------------------- 5. optional map check
    if args.check_maps:
        import pickle
        print("\n[5] Do colliding keys also share an AF2 contact map?")
        cm = pickle.load(open(hp.contact_map, 'rb'))['contact_map']
        same = diff = 0
        for g in groups:
            g = [k for k in g if k in cm]
            for k in g[1:]:
                a, b = np.asarray(cm[g[0]]), np.asarray(cm[k])
                if a.shape == b.shape and (a == b).all():
                    same += 1
                else:
                    diff += 1
        print(f"    identical contact map       : {same}")
        print(f"    different                   : {diff}")
        print("    -> identical maps mean the model receives byte-identical input")

    print("\n" + "=" * 74)
    print("  Every published result on this benchmark is subject to these limits.")
    print("=" * 74)


if __name__ == '__main__':
    main()
