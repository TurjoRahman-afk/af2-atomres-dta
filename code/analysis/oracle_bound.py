"""Reproduces README finding 2 — the remaining error is missing information, not
miscalibration.

A natural assumption when a model's absolute values are off is that the predictions are
merely mis-scaled and could be corrected after the fact. This script tests that directly by
computing an ORACLE bound: the best score achievable by ANY rank-preserving transform of the
model's output, fitted using the test labels themselves. That is cheating and unachievable
in practice, so it is a hard ceiling on what output-side work could ever deliver.

It reports:

  * where the error lives — the censored pKd = 5.0 floor vs real measurements, and by band
  * range compression — predicted spread vs true spread
  * clamping at the floor, which is free but nearly worthless
  * variance matching, which makes things worse
  * the oracle linear and oracle monotone (isotonic) ceilings
  * the HONEST linear calibration — fitted on validation only, i.e. the one you could
    actually deploy (this subsumes the former code/calib_check.py)
  * an honest split-half isotonic estimate of what calibration would really buy

Read-only. The split is regenerated in memory, so it never touches `datasets/` and works
regardless of which seed is on disk. NEEDS GPU — it will contend with a running training job.

    python code/analysis/oracle_bound.py                 # seed 41 champion
    python code/analysis/oracle_bound.py --seed 42
    python code/analysis/oracle_bound.py --tag _plddt    # a different run's checkpoint
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cold_split import create_fold_setting_cold                      # noqa: E402
from hyperparameter import HyperParameter                            # noqa: E402
from metrics import calculate_metrics                                # noqa: E402
from model_af2_atomres import MODEL as Model                # noqa: E402
from MyDataset import CustomDataSet, my_collate_fn, smile2graph, target2graph  # noqa: E402

import warnings; warnings.filterwarnings("ignore")
FLOOR = 5.0   # DAVIS reports Kd > 10 uM as pKd = 5.0 — a censored bound, not a measurement


def predict(seed, tag, dataset):
    hp = HyperParameter(); hp.set_dataset(dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = lambda d: pickle.load(open(d, 'rb'))
    drug_df = pd.read_csv(hp.drugs_dir); prot_df = pd.read_csv(hp.prots_dir)
    mol2vec = load(hp.mol2vec_dir); protvec = load(hp.protvec_dir); cmap = load(hp.contact_map)

    dc = {}
    for _, r in drug_df.iterrows():
        _, na, ei, ea = smile2graph(r['compound_iso_smiles'])
        dc[str(r['drug_key'])] = Data(x=na, edge_index=ei, edge_weight=ea)
    pc = {}
    for _, r in prot_df.iterrows():
        pid = str(r['target_key'])
        if pid not in cmap['contact_map']:
            continue
        _, tf, tei, ew = target2graph(cmap['contact_map'][pid], protvec["mat_dict"][pid])
        pc[pid] = Data(x=tf, edge_index=tei, edge_weight=ew)

    df = pd.read_csv(os.path.join(hp.data_root, dataset, 'data.csv'))
    fold = create_fold_setting_cold(df, seed, [0.8, 0.1, 0.1], ['target_key'])
    coll = lambda x: my_collate_fn(x, dev, hp, drug_df, prot_df, mol2vec, protvec,
                                   cmap, drug_graph_cache=dc, protein_graph_cache=pc)

    ckpt = f'./savemodel/{hp.dataset}-{hp.running_set}-split{seed}_new{tag}.pth'
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    m = nn.DataParallel(Model(hp, dev)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()

    out = {}
    for split in ('valid', 'test'):
        dl = DataLoader(CustomDataSet(fold[split], hp), batch_size=hp.Batch_size,
                        shuffle=False, drop_last=True, collate_fn=coll)
        P, Y = [], []
        with torch.no_grad():
            for mm, mmk, pm, pmk, dg, pg, aff in dl:
                o, _ = m(*[t.to(dev) for t in (mm, mmk, pm, pmk, dg, pg)])
                P += o.cpu().numpy().reshape(-1).tolist(); Y += aff.numpy().reshape(-1).tolist()
        out[split] = (np.array(P), np.array(Y))
    return out, os.path.basename(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--tag', default='_gnnprior')
    ap.add_argument('--dataset', default='davis')
    args = ap.parse_args()

    pred, ck = predict(args.seed, args.tag, args.dataset)
    Pv, Yv = pred['valid']
    P, Y = pred['test']
    mse, ci, rm = calculate_metrics(Y, P)
    print("=" * 84)
    print(f"  ORACLE BOUND ON OUTPUT-SIDE WORK  —  {ck}  (seed {args.seed})")
    print("=" * 84)
    print(f"\n  as-is:  MSE {mse:.4f}   CI {ci:.4f}   r2m {rm:.4f}   (n={len(Y)})")

    # ------------------------------------------------------- where the error lives
    floor = Y == FLOOR
    se = (Y - P) ** 2
    print(f"\n[1] WHERE THE ERROR LIVES")
    print(f"    {'region':<26}{'n':>6}{'share':>8}{'MSE':>9}{'bias':>9}{'% of err':>10}")
    for nm, m_ in [(f"censored floor (y={FLOOR})", floor), ("real measurement (y>5)", ~floor)]:
        if m_.sum():
            print(f"    {nm:<26}{m_.sum():>6}{100*m_.mean():>7.1f}%{se[m_].mean():>9.4f}"
                  f"{(P[m_]-Y[m_]).mean():>+9.3f}{100*se[m_].sum()/se.sum():>9.1f}%")
    print(f"\n    by affinity band:")
    for lo, hi in [(5.0, 5.001), (5.001, 6), (6, 7), (7, 8), (8, 12)]:
        b = (Y >= lo) & (Y < hi)
        if b.sum() < 5: continue
        lab = f"y = {FLOOR} (floor)" if hi == 5.001 else f"{lo:.0f} < y < {hi:.0f}"
        print(f"      {lab:<18} n={b.sum():>5}  MSE {se[b].mean():>7.4f}  "
              f"bias {(P[b]-Y[b]).mean():>+6.3f}")
    print(f"\n    true std {Y.std():.3f} vs predicted std {P.std():.3f} "
          f"-> range compressed to {100*P.std()/Y.std():.0f}% of truth")

    # ------------------------------------------------------------ transforms
    from sklearn.isotonic import IsotonicRegression
    print(f"\n[2] WHAT OUTPUT TRANSFORMS CAN DO")
    print(f"    {'transform':<44}{'MSE':>9}{'CI':>9}{'r2m':>9}")
    print(f"    {'as-is':<44}{mse:>9.4f}{ci:>9.4f}{rm:>9.4f}")

    below = P < FLOOR
    Pc = np.clip(P, FLOOR, None)
    m1, c1, r1 = calculate_metrics(Y, Pc)
    print(f"    {'clamp at the pKd floor':<44}{m1:>9.4f}{c1:>9.4f}{r1:>9.4f}")
    print(f"      ({100*below.mean():.1f}% of predictions below {FLOOR}, "
          f"by {(FLOOR-P[below]).mean():.3f} on average)" if below.sum() else "")

    Ps = (P - P.mean()) / P.std() * Y.std() + Y.mean()
    m2, c2, r2 = calculate_metrics(Y, Ps)
    print(f"    {'variance-matched (expand to true range)':<44}{m2:>9.4f}{c2:>9.4f}{r2:>9.4f}")

    A = np.vstack([P, np.ones_like(P)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    m3, c3, r3 = calculate_metrics(Y, a * P + b)
    print(f"    {'ORACLE linear (fitted ON TEST - cheating)':<44}{m3:>9.4f}{c3:>9.4f}{r3:>9.4f}")

    iso = IsotonicRegression(out_of_bounds='clip').fit(P, Y)
    m4, c4, r4 = calculate_metrics(Y, iso.predict(P))
    print(f"    {'ORACLE monotone (best possible - cheating)':<44}{m4:>9.4f}{c4:>9.4f}{r4:>9.4f}")

    av, bv = np.polyfit(Pv, Yv, 1)          # fitted on VALIDATION only — the deployable one
    m5, c5, r5 = calculate_metrics(Y, av * P + bv)
    print(f"    {'honest linear (fitted on VALIDATION)':<44}{m5:>9.4f}{c5:>9.4f}{r5:>9.4f}"
          f"   ({m5-mse:+.4f})")
    print(f"      y = {av:.4f}·pred {bv:+.4f}")

    rs = np.random.RandomState(0); idx = rs.permutation(len(P)); h = len(P) // 2
    fit, ev = idx[:h], idx[h:]
    iso2 = IsotonicRegression(out_of_bounds='clip').fit(P[fit], Y[fit])
    mh, _, _ = calculate_metrics(Y[ev], iso2.predict(P[ev]))
    m0, _, _ = calculate_metrics(Y[ev], P[ev])
    print(f"    {'honest monotone (fit half, apply to other)':<44}{mh:>9.4f}"
          f"{'':>9}{'':>9}   ({mh-m0:+.4f} vs {m0:.4f})")

    # -------------------------------------------------------------- verdict
    print(f"\n[3] VERDICT")
    print(f"    best achievable by ANY rank-preserving transform : {m4:.4f}")
    print(f"    fraction of error that survives it               : {100*m4/mse:.0f}%")
    print(f"    -> the model is not miscalibrated; it is missing information.")
    print(f"       No loss reshaping, clamping or calibration closes the gap.")
    print("=" * 84)


if __name__ == '__main__':
    main()
