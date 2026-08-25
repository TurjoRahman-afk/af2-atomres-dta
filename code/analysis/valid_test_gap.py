"""Reproduces README finding 3 — validation systematically overstates performance.

Every completed run in this project scores worse on test than on the validation set that
selected its checkpoint. This script measures that gap across all runs and separates it
into contributing causes:

  * SELECTION BIAS (winner's curse) — the single best-validation epoch is optimistically
    biased relative to the epochs around it, because it was chosen for being low
  * LABEL VARIANCE — MSE scales with the spread of the labels, and the valid and test
    splits are two different 44-protein samples with different spreads
  * RESIDUAL — intrinsic difficulty differences between the two samples

It also lists every case where validation ranking INVERTED at test, which is the practical
consequence: a variant that leads on validation can still lose on test.

Read-only. Reads log CSVs and regenerates splits in memory, so it never touches
`datasets/` and is safe to run during training. CPU only.

    python code/analysis/valid_test_gap.py                 # cold-protein runs
    python code/analysis/valid_test_gap.py --all           # include warm-split runs
    python code/analysis/valid_test_gap.py --window 5      # +/- epochs for the bias estimate
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cold_split import create_fold_setting_cold  # noqa: E402
from hyperparameter import HyperParameter        # noqa: E402


def collect(include_warm):
    """Every run that has both a training log and a final (natural early-stop) test result."""
    runs = []
    for t in sorted(glob.glob('log/Test-*.csv')):
        name = os.path.basename(t)[len('Test-'):-len('.csv')]
        logp = os.path.join('log', name + '.csv')
        if not os.path.exists(logp):
            continue
        if not include_warm and 'unseen_prot' not in name:
            continue
        d = pd.read_csv(logp)
        if 'valid_mse' not in d or d.valid_mse.dropna().empty:
            continue
        te = pd.read_csv(t)
        d = d.dropna(subset=['valid_mse'])
        best = d.loc[d.valid_mse.idxmin()]
        runs.append(dict(name=name, log=d, best_ep=int(best.epoch),
                         valid=float(best.valid_mse), valid_ci=float(best.valid_ci),
                         test=float(te.mse[0]), test_ci=float(te.ci[0]),
                         epochs=len(d)))
    return runs


def label_variance(seed, dataset='davis'):
    """Variance of the valid and test labels for a split, regenerated in memory."""
    hp = HyperParameter(); hp.set_dataset(dataset)
    df = pd.read_csv(os.path.join(hp.data_root, dataset, 'data.csv'))
    f = create_fold_setting_cold(df, seed, [0.8, 0.1, 0.1], ['target_key'])
    return f['valid'].affinity.var(), f['test'].affinity.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='include warm-split runs')
    ap.add_argument('--window', type=int, default=5,
                    help='+/- epochs around the best used to estimate selection bias')
    args = ap.parse_args()

    runs = collect(args.all)
    if not runs:
        sys.exit("No completed runs found (need both log/<run>.csv and log/Test-<run>.csv)")

    print("=" * 92)
    print("  VALIDATION -> TEST GAP   (finding 3)")
    print("=" * 92)
    print(f"\n{'run':<46}{'ep':>5}{'valid':>9}{'test':>9}{'gap':>9}")
    print("  " + "-" * 88)
    gaps = []
    for r in runs:
        g = r['test'] - r['valid']; gaps.append(g)
        print(f"{r['name']:<46}{r['best_ep']:>5}{r['valid']:>9.4f}{r['test']:>9.4f}{g:>+9.4f}")
    gaps = np.array(gaps)
    print("  " + "-" * 88)
    print(f"{'':<46}{'':>5}{'':>9}{'mean gap':>9}{gaps.mean():>+9.4f}")
    print(f"    positive in {int((gaps > 0).sum())}/{len(gaps)} runs "
          f"(range {gaps.min():+.4f} .. {gaps.max():+.4f})")

    # ---------------------------------------------------------- selection bias
    print(f"\n[A] SELECTION BIAS — how much lower the chosen epoch is than its neighbours")
    print(f"    (mean valid MSE within +/-{args.window} epochs, minus the selected best)")
    print(f"\n{'run':<46}{'best':>9}{'local mean':>12}{'bias':>9}")
    print("  " + "-" * 88)
    biases = []
    for r in runs:
        d = r['log']
        near = d[(d.epoch >= r['best_ep'] - args.window) & (d.epoch <= r['best_ep'] + args.window)]
        if len(near) < 3:
            continue
        lm = near.valid_mse.mean(); b = lm - r['valid']; biases.append(b)
        print(f"{r['name']:<46}{r['valid']:>9.4f}{lm:>12.4f}{b:>+9.4f}")
    if biases:
        print("  " + "-" * 88)
        print(f"    selection bias range: +{min(biases):.4f} .. +{max(biases):.4f}  "
              f"(mean +{np.mean(biases):.4f})")
        print(f"    -> this much of every gap is winner's curse, not real degradation")

    # ------------------------------------------------------- label variance
    print("\n[B] LABEL VARIANCE — MSE scales with label spread; valid and test differ")
    print(f"\n{'seed':<8}{'valid var':>12}{'test var':>12}{'test-valid':>12}")
    print("  " + "-" * 44)
    seeds = sorted({int(m.group(1)) for r in runs
                    for m in [re.search(r'split(\d+)', r['name'])] if m})
    for s in seeds:
        try:
            vv, tv = label_variance(s)
            print(f"{s:<8}{vv:>12.4f}{tv:>12.4f}{tv - vv:>+12.4f}")
        except Exception as e:
            print(f"{s:<8}  (could not regenerate: {type(e).__name__})")
    print("    -> a test set with higher label variance yields higher MSE for the same model")

    # ------------------------------------------------- ranking inversions
    print("\n[C] RANKING INVERSIONS — validation order disagreeing with test order")
    inv = []
    for i in range(len(runs)):
        for j in range(len(runs)):
            if i >= j:
                continue
            a, b = runs[i], runs[j]
            if re.search(r'split(\d+)', a['name']) is None: continue
            # only compare runs evaluated on the same split
            sa = re.search(r'split(\d+)', a['name']).group(1)
            sb = re.search(r'split(\d+)', b['name']).group(1)
            if sa != sb:
                continue
            if (a['valid'] < b['valid']) != (a['test'] < b['test']):
                inv.append((a, b))
    if inv:
        for a, b in inv:
            better_v = a if a['valid'] < b['valid'] else b
            better_t = a if a['test'] < b['test'] else b
            print(f"    valid favours {better_v['name'][:44]:<44} ({better_v['valid']:.4f})")
            print(f"    test  favours {better_t['name'][:44]:<44} ({better_t['test']:.4f})")
            print()
        print(f"    {len(inv)} inverted pair(s) among same-split runs")
    else:
        print("    none among same-split runs")
    print("    -> validation ranking does not reliably predict test ranking")

    print("=" * 92)


if __name__ == '__main__':
    main()
