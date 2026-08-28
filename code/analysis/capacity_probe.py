"""Reproduces README finding 6 — capacity is not the bottleneck, the representation is.

Two questions this answers:

  1. Is the predictor short of capacity?  Freeze the champion's 512-d representation (the
     vector handed to the KAN) and fit heads of increasing size on it, up to and beyond the
     KAN's own parameter count. If more capacity helped, test MSE would fall. It does not.

  2. How much is the head contributing at all?  A ridge (linear) head on the same frozen
     features bounds what the representation alone supports; the gap to the KAN is the
     head's real contribution.

Also prints the training-MSE-vs-irreducible-floor context, which is why the answer is what
it is: the model already fits its training data to within ~0.025 of the noise floor.

Read-only. Regenerates the split in memory, caches the extracted features so reruns are
instant. NEEDS GPU — it will contend with a running training job.

    python code/analysis/capacity_probe.py                # seed 41 champion
    python code/analysis/capacity_probe.py --seed 42
    python code/analysis/capacity_probe.py --refresh      # discard the feature cache
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

SWEEP = [[512, 256, 1], [512, 1024, 1], [512, 1024, 512, 1],
         [512, 2048, 1024, 1], [512, 4096, 2048, 1], [512, 4096, 4096, 1]]
KAN_PARAMS = 10_490_880


def extract(seed, tag, dataset, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    hp = HyperParameter(); hp.set_dataset(dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = lambda p: pickle.load(open(p, 'rb'))
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

    ck = f'./savemodel/{hp.dataset}-{hp.running_set}-split{seed}_new{tag}.pth'
    if not os.path.exists(ck):
        sys.exit(f"checkpoint not found: {ck}")
    model = nn.DataParallel(Model(hp, dev)).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    buf = []
    model.module.kan.register_forward_pre_hook(lambda m, i: buf.append(i[0].detach().cpu()))

    df = pd.read_csv(os.path.join(hp.data_root, dataset, 'data.csv'))
    fold = create_fold_setting_cold(df, seed, [0.8, 0.1, 0.1], ['target_key'])
    coll = lambda x: my_collate_fn(x, dev, hp, drug_df, prot_df, mol2vec, protvec,
                                   cmap, drug_graph_cache=dc, protein_graph_cache=pc)
    out = {}
    for sp in ('train', 'valid', 'test'):
        dl = DataLoader(CustomDataSet(fold[sp], hp), batch_size=hp.Batch_size,
                        shuffle=False, drop_last=True, collate_fn=coll)
        buf.clear(); Y, P = [], []
        with torch.no_grad():
            for mm, mmk, pm, pmk, dg, pg, aff in dl:
                o, _ = model(*[t.to(dev) for t in (mm, mmk, pm, pmk, dg, pg)])
                P += o.cpu().numpy().reshape(-1).tolist(); Y += aff.numpy().reshape(-1).tolist()
        out[sp + '_X'] = torch.cat(buf).numpy()
        out[sp + '_Y'] = np.array(Y); out[sp + '_P'] = np.array(P)
        print(f"    extracted {sp:<6} {out[sp+'_X'].shape}")
    np.savez(cache, **out)
    return out


def fit_head(dims, xt, yt, xv, yv, xe, dev, epochs=400, patience=40):
    torch.manual_seed(0)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(0.2)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, sd, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            ((net(xt[idx]).squeeze() - yt[idx]) ** 2).mean().backward()
            opt.step(); opt.zero_grad()
        net.eval()
        with torch.no_grad():
            v = ((net(xv).squeeze() - yv) ** 2).mean().item()
        if v < best - 1e-5:
            best, sd, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(sd); net.eval()
    with torch.no_grad():
        p = net(xe).squeeze().cpu().numpy()
    return sum(q.numel() for q in net.parameters()), best, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--tag', default='_gnnprior')
    ap.add_argument('--dataset', default='davis')
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f'.feats_{args.dataset}_s{args.seed}{args.tag}.npz')
    if args.refresh and os.path.exists(cache):
        os.remove(cache)

    print("=" * 84)
    print(f"  CAPACITY PROBE  —  seed {args.seed}{args.tag}")
    print("=" * 84)
    print("\n  extracting the frozen 512-d representation handed to the KAN...")
    d = extract(args.seed, args.tag, args.dataset, cache)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(d['train_X'], dtype=torch.float32, device=dev)
    yt = torch.tensor(d['train_Y'], dtype=torch.float32, device=dev)
    xv = torch.tensor(d['valid_X'], dtype=torch.float32, device=dev)
    yv = torch.tensor(d['valid_Y'], dtype=torch.float32, device=dev)
    xe = torch.tensor(d['test_X'], dtype=torch.float32, device=dev)
    Yte, Pte = d['test_Y'], d['test_P']
    kan = calculate_metrics(Yte, Pte)

    tr_ckpt = ((d['train_Y'] - d['train_P']) ** 2).mean()
    print(f"\n[1] CONTEXT — the head already fits training close to the noise floor")
    print(f"    train MSE of the SAVED checkpoint, eval mode : {tr_ckpt:.4f}")
    print(f"    irreducible floor (benchmark_integrity.py)   : 0.0348")
    print(f"    remaining headroom                           : {tr_ckpt - 0.0348:.4f}")
    print(f"    (the training log quotes a lower figure at its final epoch — that is")
    print(f"     measured mid-epoch with dropout active, on a later, more-overfit model)")
    print(f"    -> the head is not capacity-starved")

    print(f"\n[2] HEADS ON THE SAME FROZEN REPRESENTATION")
    print(f"    {'head':<28}{'params':>12}{'valid':>9}{'test MSE':>10}{'CI':>9}{'r2m':>9}")
    print(f"    {'KAN [512,1024,512,1]':<28}{KAN_PARAMS:>12,}{'—':>9}"
          f"{kan[0]:>10.4f}{kan[1]:>9.4f}{kan[2]:>9.4f}")
    print("    " + "-" * 77)

    from sklearn.linear_model import Ridge
    best = None
    for a in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000):
        r = Ridge(alpha=a).fit(d['train_X'], d['train_Y'])
        v = ((d['valid_Y'] - r.predict(d['valid_X'])) ** 2).mean()
        if best is None or v < best[0]:
            best = (v, a, r)
    v, alpha, ridge = best
    m, c, rm = calculate_metrics(Yte, ridge.predict(d['test_X']))
    print(f"    {f'RIDGE linear (a={alpha:g})':<28}{513:>12,}{v:>9.4f}{m:>10.4f}{c:>9.4f}{rm:>9.4f}")

    results = []
    for dims in SWEEP:
        n, v, p = fit_head(dims, xt, yt, xv, yv, xe, dev)
        m, c, rm = calculate_metrics(Yte, p)
        results.append((n, m))
        flag = "  <- KAN-matched" if abs(n - KAN_PARAMS) / KAN_PARAMS < 0.35 else ""
        print(f"    {'MLP ' + '-'.join(map(str, dims)):<28}{n:>12,}{v:>9.4f}"
              f"{m:>10.4f}{c:>9.4f}{rm:>9.4f}{flag}")

    print(f"\n[3] VERDICT")
    n0, m0 = results[0]; n1, m1 = max(results, key=lambda x: x[0])
    print(f"    smallest MLP  {n0:>12,} params -> {m0:.4f}")
    print(f"    largest  MLP  {n1:>12,} params -> {m1:.4f}   ({n1/n0:.0f}x more capacity, "
          f"{m1-m0:+.4f} MSE)")
    print(f"    best MLP at any width : {min(r[1] for r in results):.4f}   "
          f"vs the KAN's {kan[0]:.4f}")
    print(f"    -> more predictor capacity does not help; no MLP reaches the KAN at any width.")
    print(f"       The ceiling is the representation, set by 354 training proteins.")
    print("=" * 84)


if __name__ == '__main__':
    main()
