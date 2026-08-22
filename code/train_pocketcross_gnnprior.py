"""Training script for AF2-PocketCross-DTA — GNN-prior variant (model_pocketcross_gnnprior.py).
Simpler than train_pocketcross.py: no prot_struct / struct_cache needed since the pocket
score now comes from ProteinGraphNet's own per-residue embeddings, so this uses the plain
my_collate_fn (same one train.py uses). Plain MSE loss, matching the config that won
(0.3715) for the original pocket-prior version — isolates the ONE change: where the
pocket score comes from. Outputs tagged '_gnnprior' so nothing collides with other runs."""
import os, random, pickle, time, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_pocketcross_gnnprior import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn, smile2graph, target2graph
from torch_geometric.data import Data
from metrics import calculate_metrics
from cold_split import create_fold_setting_cold

import warnings; warnings.filterwarnings("ignore")


def assert_split_matches(hp, root, split_seed):
    """SPLIT_SEED only names output files — it does not select data. Fail loudly if the
    CSVs on disk were generated with a different cold_split.py SEED, instead of silently
    training on one split and labelling the results with another."""
    df = pd.read_csv(os.path.join(hp.data_root, hp.dataset, 'data.csv'))
    expect = set(create_fold_setting_cold(df, split_seed, [0.8, 0.1, 0.1], ['target_key'])['test'].target_key)
    actual = set(pd.read_csv(os.path.join(root, 'test.csv')).target_key)
    if expect != actual:
        raise SystemExit(
            f"Splits in {root} do NOT match SPLIT_SEED={split_seed} "
            f"({len(expect & actual)}/{len(expect)} test proteins agree). "
            f"Run: python code/cold_split.py --SEED {split_seed}")
    print(f"split check OK — data on disk is seed {split_seed} ({len(actual)} test proteins)")


def load_pickle(d):
    with open(d, 'rb+') as f:
        return pickle.load(f)


def build_graph_cache(drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map):
    print("Pre-building drug graph cache...")
    drug_cache = {}
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Drug graphs"):
        _, na, ei, ea = smile2graph(row['compound_iso_smiles'])
        drug_cache[str(row['drug_key'])] = Data(x=na, edge_index=ei, edge_weight=ea)
    print("Pre-building protein graph cache...")
    prot_cache = {}
    for _, row in tqdm(prot_df.iterrows(), total=len(prot_df), desc="Protein graphs"):
        pid = str(row['target_key'])
        if pid not in contact_map['contact_map']:
            continue
        _, tf, tei, ew = target2graph(contact_map['contact_map'][pid], protvec_dict["mat_dict"][pid])
        prot_cache[pid] = Data(x=tf, edge_index=tei, edge_weight=ew)
    print(f"Cache ready: {len(drug_cache)} drug, {len(prot_cache)} protein graphs")
    return drug_cache, prot_cache


def test(model, dataloader):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        mm, mmk, pm, pmk, dg, pg, aff = batch
        mm, mmk, pm, pmk, dg, pg = [t.to(device) for t in (mm, mmk, pm, pmk, dg, pg)]
        with torch.no_grad():
            out, _ = model(mm, mmk, pm, pmk, dg, pg)
        preds += out.cpu().numpy().reshape(-1).tolist()
        labels += aff.numpy().reshape(-1).tolist()
    return calculate_metrics(np.array(labels), np.array(preds))


if __name__ == "__main__":
    SEED = 0
    SPLIT_SEED = 43
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True

    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset-{hp.dataset}-{hp.running_set}  (AF2-PocketCross-DTA, GNN-prior variant)")

    drug_df = pd.read_csv(hp.drugs_dir); prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir); protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)

    root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    assert_split_matches(hp, root, SPLIT_SEED)
    train_set = CustomDataSet(pd.read_csv(os.path.join(root, 'train.csv')), hp)
    valid_set = CustomDataSet(pd.read_csv(os.path.join(root, 'valid.csv')), hp)
    test_set = CustomDataSet(pd.read_csv(os.path.join(root, 'test.csv')), hp)
    print("load dataset finished")

    dcache, pcache = build_graph_cache(drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map)
    collate = lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict,
                                      contact_map, drug_graph_cache=dcache, protein_graph_cache=pcache)
    tl = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, collate_fn=collate, pin_memory=True)
    vl = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, collate_fn=collate, pin_memory=True)
    testl = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, collate_fn=collate, pin_memory=True)

    os.makedirs('./savemodel', exist_ok=True); os.makedirs('./log', exist_ok=True)
    model = nn.DataParallel(Model(hp, device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))

    tag = '_gnnprior'
    best_model = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_new{tag}.pth'
    ckpt_path = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_new{tag}_checkpoint.pth'
    log_dir = f'./log/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_new{tag}.csv'

    best_valid_mse, patience, start_epoch = 10, 0, 1
    train_log, valid_log = [], []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state']); optimizer.load_state_dict(ck['optimizer_state'])
        start_epoch = ck['epoch'] + 1; best_valid_mse = ck['best_valid_mse']; patience = ck['patience']
        train_log, valid_log = ck['train_log'], ck['valid_log']
        print(f"Resumed from epoch {ck['epoch']} — best {best_valid_mse:.4f}, patience {patience}")
    else:
        print("No checkpoint — starting fresh")

    for epoch in range(start_epoch, hp.Epoch + 1):
        model.train(); pred, label = [], []
        t0 = time.time()
        for batch in tqdm(tl, desc=f"Epoch {epoch}/{hp.Epoch}", leave=False):
            mm, mmk, pm, pmk, dg, pg, aff = batch
            mm, mmk, pm, pmk, dg, pg, aff = [t.to(device) for t in (mm, mmk, pm, pmk, dg, pg, aff)]
            out, _ = model(mm, mmk, pm, pmk, dg, pg)
            loss = ((out.squeeze() - aff) ** 2).mean()
            pred += out.detach().cpu().numpy().reshape(-1).tolist()
            label += aff.detach().cpu().numpy().reshape(-1).tolist()
            loss.backward(); optimizer.step(); optimizer.zero_grad()

        tr = calculate_metrics(np.array(label), np.array(pred)); train_log.append(list(tr))
        print(f"Epoch {epoch} | Train MSE {tr[0]:.4f} | {time.time()-t0:.0f}s")
        vm, vc, vr = test(model, vl); valid_log.append([vm, vc, vr])
        print(f"Valid: mse {vm:.4f}, ci {vc:.4f}, rm2 {vr:.4f}")

        if vm < best_valid_mse:
            patience = 0; best_valid_mse = vm
            torch.save(model.state_dict(), best_model)
            print(f"Update best_mse at epoch {epoch}: {vm:.4f}")
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f"Early stop at epoch {epoch}. Best saved at {best_model}")
                break

        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'best_valid_mse': best_valid_mse, 'patience': patience,
                    'train_log': train_log, 'valid_log': valid_log}, ckpt_path)
        with open(log_dir, "w+", newline='') as f:
            wtr = csv.writer(f); wtr.writerow(["epoch", "train_mse", "train_ci", "train_r2m", "valid_mse", "valid_ci", "valid_r2m"])
            off = len(train_log) - len(valid_log)
            for i, r in enumerate(train_log, 1):
                vi = i - 1 - off
                v = valid_log[vi] if 0 <= vi < len(valid_log) else ['', '', '']
                wtr.writerow([i] + list(r) + list(v))

    print(f"Save log at {log_dir}")
    predModel = nn.DataParallel(Model(hp, device)).to(device)
    predModel.load_state_dict(torch.load(best_model))
    mse, ci, rm2 = test(predModel, testl)
    print(f"Test: mse {mse:.4f}, ci {ci:.4f}, rm2 {rm2:.4f}")
    pd.DataFrame({'mse': [mse], 'ci': [ci], 'rm2': [rm2]}).to_csv(
        f'./log/Test-{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_new{tag}.csv', index=False)
