import os
import random
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn

import csv
import time
from tqdm import tqdm
from metrics import calculate_metrics

import warnings
warnings.filterwarnings("ignore")



def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)


def build_graph_cache(drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map):
    from MyDataset import smile2graph, target2graph
    from torch_geometric.data import Data

    print("Pre-building drug graph cache...")
    drug_graph_cache = {}
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Drug graphs"):
        drug_id = str(row['drug_key'])
        _, node_attr, edge_index, edge_attr = smile2graph(row['compound_iso_smiles'])
        drug_graph_cache[drug_id] = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)

    print("Pre-building protein graph cache...")
    protein_graph_cache = {}
    for _, row in tqdm(prot_df.iterrows(), total=len(prot_df), desc="Protein graphs"):
        prot_id = str(row['target_key'])
        if prot_id not in contact_map['contact_map']:
            continue
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id].copy()
        _, target_features, target_edge_index, target_edge_distance = target2graph(prot_contact_map, prot_mat)
        protein_graph_cache[prot_id] = Data(x=target_features, edge_index=target_edge_index, edge_weight=target_edge_distance)

    print(f"Cache ready: {len(drug_graph_cache)} drug graphs, {len(protein_graph_cache)} protein graphs")
    return drug_graph_cache, protein_graph_cache
    
def test(model, dataloader):
    model.eval()
    preds = []
    labels = []
    for batch_i, batch_data in enumerate(dataloader):
        mol_vec, prot_vec, mol_mat, mol_mat_mask,  prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

        mol_vec = mol_vec.to(device)
        prot_vec = prot_vec.to(device)
        mol_mat = mol_mat.to(device)
        mol_mat_mask = mol_mat_mask.to(device)
        prot_mat = prot_mat.to(device)
        prot_mat_mask = prot_mat_mask.to(device)
        drugh_graph = drugh_graph.to(device)
        protein_graph = protein_graph.to(device)
        affinity = affinity.to(device)


        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist()

    preds = np.array(preds)
    labels = np.array(labels)
    mse, ci, rm2 = calculate_metrics(labels, preds)
    return mse, ci, rm2

if __name__ == "__main__":
    SEED = 0          # weight init + batch shuffle seed (keep fixed across runs)
    SPLIT_SEED = 42   # data split seed — must match cold_split.py SEED
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    
    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Dataset-{hp.dataset}-{hp.running_set}") 
    print(f"Pretrain-{hp.mol2vec_dir}-{hp.protvec_dir}")
    save_metrics = {'mse':[], 'ci':[], 'rm2':[]}
    dataset_root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
    
    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)
    
    train_dir = os.path.join(dataset_root, f'train.csv')
    valid_dir = os.path.join(dataset_root, f'valid.csv')
    test_dir = os.path.join(dataset_root, f'test.csv')                 
    train_set = CustomDataSet(pd.read_csv(train_dir), hp)
    valid_set = CustomDataSet(pd.read_csv(valid_dir), hp)
    test_set = CustomDataSet(pd.read_csv(test_dir), hp)
    print("load dataset finished")

    drug_graph_cache, protein_graph_cache = build_graph_cache(drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map)

    collate = lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map,
                                      drug_graph_cache=drug_graph_cache, protein_graph_cache=protein_graph_cache)
    train_dataset_load = DataLoader(train_set, batch_size=hp.Batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)
    valid_dataset_load = DataLoader(valid_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=collate)
    test_dataset_load = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=collate)
    
    os.makedirs('./savemodel', exist_ok=True)
    os.makedirs('./log', exist_ok=True)

    model = nn.DataParallel(Model(hp, device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate, betas=(0.9, 0.999))
    criterion = F.mse_loss

    model_fromTrain = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}.pth'
    checkpoint_path = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_checkpoint.pth'

    train_log = []
    valid_log = []
    best_valid_mse = 10
    patience = 0
    start_epoch = 1

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found — resuming training from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_valid_mse = ckpt['best_valid_mse']
        patience = ckpt['patience']
        train_log = ckpt['train_log']
        valid_log = ckpt.get('valid_log', [])
        print(f"Resumed from epoch {ckpt['epoch']} — best MSE so far: {best_valid_mse:.4f}, patience: {patience}")
    else:
        print("No checkpoint found — starting fresh training")

    epoch_times = []

    for epoch in range(start_epoch, hp.Epoch + 1):
        epoch_start = time.time()

        # training
        model.train()
        pred = []
        label = []
        total_batches = len(train_dataset_load)
        pbar = tqdm(train_dataset_load, total=total_batches,
                    desc=f"Epoch {epoch}/{hp.Epoch}", unit="batch", leave=False)

        for batch_data in pbar:
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data

            mol_vec = mol_vec.to(device)
            prot_vec = prot_vec.to(device)
            mol_mat = mol_mat.to(device)
            mol_mat_mask = mol_mat_mask.to(device)
            prot_mat = prot_mat.to(device)
            prot_mat_mask = prot_mat_mask.to(device)
            drugh_graph = drugh_graph.to(device)
            protein_graph = protein_graph.to(device)
            affinity = affinity.to(device)

            predictions = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
            label = label + affinity.cpu().detach().numpy().reshape(-1).tolist()

            loss = criterion(predictions.squeeze(), affinity)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        pred = np.array(pred)
        label = np.array(label)
        mse_value, ci_value, rm2_value = calculate_metrics(label, pred)
        train_log.append([mse_value, ci_value, rm2_value])

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = hp.Epoch - epoch
        eta_seconds = avg_epoch_time * epochs_left
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        print(f"Epoch {epoch}/{hp.Epoch} | Train MSE: {mse_value:.4f} | Time: {epoch_time:.0f}s | ETA: {eta_str}")
            
        # valid
        mse, ci, rm2 = test(model, valid_dataset_load)
        valid_log.append([mse, ci, rm2])
        print(f'Valid at: mse: {mse}, ci: {ci}, rm2: {rm2}')

        # Early stop
        if mse < best_valid_mse :
            patience = 0
            best_valid_mse = mse
            # save model
            torch.save(model.state_dict(), model_fromTrain)
            print(f'Update best_mse, Valid at epoch: {epoch}: mse: {mse}, ci: {ci}, rm2: {rm2}')
        else:
            patience += 1
            if patience > hp.max_patience:
                print(f'Training stopped at epoch {epoch}, best model saved at {model_fromTrain}')
                break

        # Save checkpoint after every epoch so training can be resumed if interrupted
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_valid_mse': best_valid_mse,
            'patience': patience,
            'train_log': train_log,
            'valid_log': valid_log,
        }, checkpoint_path)

        # Write log CSV after every epoch so results are never lost on interruption
        log_dir = f"./log/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}.csv"
        with open(log_dir, "w+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse", "train_ci", "train_r2m", "valid_mse", "valid_ci", "valid_r2m"])
            valid_offset = len(train_log) - len(valid_log)
            for i, row in enumerate(train_log, 1):
                valid_idx = i - 1 - valid_offset
                v = valid_log[valid_idx] if 0 <= valid_idx < len(valid_log) else ['', '', '']
                writer.writerow([i] + row + list(v))

    log_dir = f"./log/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}.csv"
    with open(log_dir, "w+")as f:
        writer = csv.writer(f)
        writer.writerow(["mse",  "ci", "rm2"])
        for r in train_log:
            writer.writerow(r)
    print(f'Save log over at {log_dir}')

    # Test
    predModel = nn.DataParallel(Model(hp, device))
    predModel.load_state_dict(torch.load(model_fromTrain))
    predModel = predModel.to(device)    
    mse, ci, rm2 = test(predModel, test_dataset_load)
    print(f'Test at, mse: {mse}, ci: {ci}, rm2: {rm2}\n')
    save_metrics['mse'].append(mse)
    save_metrics['ci'].append(ci)
    save_metrics['rm2'].append(rm2)                          
        
    # save training log
    test_metrics = pd.DataFrame(save_metrics)    
    test_metrics.to_csv(f'./log/Test-{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}.csv', index=False)     
    print(f"Dataset-{hp.dataset}-{hp.running_set}")
