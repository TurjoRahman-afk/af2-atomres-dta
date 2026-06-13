"""
Training script for LeanDTA (Path-B lean architecture).

Self-contained: reads the same data files as code/train.py, but writes its own
logs (lean_model/log/) and checkpoints (lean_model/savemodel/), so it never
collides with the existing pipeline or any running run.

Run:
    cd lean_model
    python train.py
"""

import os
import sys
import csv
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# reuse proven config / metrics / dataset wrapper from code/ (read-only)
_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.append(_CODE_DIR)
from hyperparameter import HyperParameter          # noqa: E402
from metrics import calculate_metrics              # noqa: E402
from MyDataset import CustomDataSet                 # noqa: E402

from model import LeanDTA
from dataset import build_caches, lean_collate_fn

import warnings
warnings.filterwarnings("ignore")

TAG = "davis-warm-lean"     # output filename tag
SPLIT_SEED = 42             # must match the split used to generate train/valid/test


def load_pickle(path):
    with open(path, "rb+") as f:
        return pickle.load(f)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            (dt, dm, dg, dfp, pr, pm, pg, y) = batch
            pred = model(dt.to(device), dm.to(device), dg.to(device), dfp.to(device),
                         pr.to(device), pm.to(device), pg.to(device))
            preds += pred.cpu().numpy().reshape(-1).tolist()
            labels += y.numpy().reshape(-1).tolist()
    return calculate_metrics(np.array(labels), np.array(preds))


def main():
    SEED = 0
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {hp.dataset}-{hp.running_set}")

    # data
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)

    root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    train_set = CustomDataSet(pd.read_csv(os.path.join(root, "train.csv")), hp)
    valid_set = CustomDataSet(pd.read_csv(os.path.join(root, "valid.csv")), hp)
    test_set = CustomDataSet(pd.read_csv(os.path.join(root, "test.csv")), hp)
    print("Loaded splits.")

    drug_graph_cache, drug_fp_cache, protein_graph_cache = build_caches(
        drug_df, prot_df, protvec_dict, contact_map
    )

    collate = lambda b: lean_collate_fn(
        b, hp, mol2vec_dict, protvec_dict,
        drug_graph_cache, drug_fp_cache, protein_graph_cache,
    )
    train_loader = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False,
                              drop_last=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False,
                             drop_last=True, collate_fn=collate, pin_memory=True)

    os.makedirs("./savemodel", exist_ok=True)
    os.makedirs("./log", exist_ok=True)

    model = LeanDTA(dim=128, n_heads=4, dropout=0.3, use_kan=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LeanDTA trainable parameters: {n_params/1e6:.2f}M")

    # lr 1e-4, weight_decay 1e-5 (1e-4 over-regularized in Run 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate,
                                 betas=(0.9, 0.999), weight_decay=1e-5)
    criterion = F.mse_loss

    best_path = f"./savemodel/{TAG}-split{SPLIT_SEED}.pth"
    ckpt_path = f"./savemodel/{TAG}-split{SPLIT_SEED}_checkpoint.pth"
    log_path = f"./log/{TAG}-split{SPLIT_SEED}.csv"

    train_log, valid_log = [], []
    best_valid_mse = 1e9
    patience = 0
    start_epoch = 1

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        best_valid_mse = ck["best_valid_mse"]
        patience = ck["patience"]
        train_log = ck["train_log"]; valid_log = ck["valid_log"]
        print(f"Resumed from epoch {ck['epoch']} (best MSE {best_valid_mse:.4f})")
    else:
        print("Starting fresh training.")

    epoch_times = []
    for epoch in range(start_epoch, hp.Epoch + 1):
        t0 = time.time()
        model.train()
        preds, labels = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{hp.Epoch}", unit="batch", leave=False)
        for batch in pbar:
            (dt, dm, dg, dfp, pr, pm, pg, y) = batch
            pred = model(dt.to(device), dm.to(device), dg.to(device), dfp.to(device),
                         pr.to(device), pm.to(device), pg.to(device))
            y = y.to(device)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            preds += pred.detach().cpu().numpy().reshape(-1).tolist()
            labels += y.detach().cpu().numpy().reshape(-1).tolist()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        tr_mse, tr_ci, tr_r2m = calculate_metrics(np.array(labels), np.array(preds))
        train_log.append([tr_mse, tr_ci, tr_r2m])

        va_mse, va_ci, va_r2m = evaluate(model, valid_loader, device)
        valid_log.append([va_mse, va_ci, va_r2m])

        epoch_times.append(time.time() - t0)
        eta = time.strftime("%H:%M:%S", time.gmtime(
            (sum(epoch_times) / len(epoch_times)) * (hp.Epoch - epoch)))
        print(f"Epoch {epoch}/{hp.Epoch} | train MSE {tr_mse:.4f} | "
              f"valid MSE {va_mse:.4f} CI {va_ci:.4f} r2m {va_r2m:.4f} | ETA {eta}")

        if va_mse < best_valid_mse:
            patience = 0
            best_valid_mse = va_mse
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (valid MSE {va_mse:.4f}) saved")
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f"Early stop at epoch {epoch}. Best valid MSE {best_valid_mse:.4f}")
                break

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_valid_mse": best_valid_mse, "patience": patience,
            "train_log": train_log, "valid_log": valid_log,
        }, ckpt_path)

        # 7-column log written every epoch (no destructive final overwrite)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_mse", "train_ci", "train_r2m",
                        "valid_mse", "valid_ci", "valid_r2m"])
            for i, (tr, va) in enumerate(zip(train_log, valid_log), 1):
                w.writerow([i] + list(tr) + list(va))

    # final test with best checkpoint
    model.load_state_dict(torch.load(best_path))
    te_mse, te_ci, te_r2m = evaluate(model, test_loader, device)
    print(f"\nTest | MSE {te_mse:.4f} | CI {te_ci:.4f} | r2m {te_r2m:.4f}")
    pd.DataFrame({"mse": [te_mse], "ci": [te_ci], "rm2": [te_r2m]}).to_csv(
        f"./log/Test-{TAG}-split{SPLIT_SEED}.csv", index=False)


if __name__ == "__main__":
    main()
