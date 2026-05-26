import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn
from metrics import calculate_metrics

import warnings
warnings.filterwarnings("ignore")

SPLIT_SEED = 43  # change this to test a different run
EPOCH = None     # set to an epoch number (e.g. 83) to test a specific snapshot, or None for current best

def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)

def test(model, dataloader):
    model.eval()
    preds, labels = [], []
    for batch_data in dataloader:
        mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drugh_graph, protein_graph, affinity = batch_data
        mol_vec = mol_vec.to(device)
        prot_vec = prot_vec.to(device)
        mol_mat = mol_mat.to(device)
        mol_mat_mask = mol_mat_mask.to(device)
        prot_mat = prot_mat.to(device)
        prot_mat_mask = prot_mat_mask.to(device)
        drugh_graph = drugh_graph.to(device)
        protein_graph = protein_graph.to(device)
        with torch.no_grad():
            pred = model(mol_vec, mol_mat, mol_mat_mask, prot_vec, prot_mat, prot_mat_mask, drugh_graph, protein_graph)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.numpy().reshape(-1).tolist()
    mse, ci, rm2 = calculate_metrics(np.array(labels), np.array(preds))
    return mse, ci, rm2

if __name__ == "__main__":
    hp = HyperParameter()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug_df = pd.read_csv(hp.drugs_dir)
    prot_df = pd.read_csv(hp.prots_dir)
    mol2vec_dict = load_pickle(hp.mol2vec_dir)
    protvec_dict = load_pickle(hp.protvec_dir)
    contact_map = load_pickle(hp.contact_map)

    test_dir = f'{hp.data_root}/{hp.dataset}/{hp.running_set}/test.csv'
    test_set = CustomDataSet(pd.read_csv(test_dir), hp)

    from MyDataset import smile2graph, target2graph
    from torch_geometric.data import Data
    from tqdm import tqdm

    drug_graph_cache = {}
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Drug graphs"):
        drug_id = str(row['drug_key'])
        _, node_attr, edge_index, edge_attr = smile2graph(row['compound_iso_smiles'])
        drug_graph_cache[drug_id] = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)

    protein_graph_cache = {}
    for _, row in tqdm(prot_df.iterrows(), total=len(prot_df), desc="Protein graphs"):
        prot_id = str(row['target_key'])
        if prot_id not in contact_map['contact_map']:
            continue
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id].copy()
        _, target_features, target_edge_index, target_edge_distance = target2graph(prot_contact_map, prot_mat)
        protein_graph_cache[prot_id] = Data(x=target_features, edge_index=target_edge_index, edge_weight=target_edge_distance)

    collate = lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map,
                                      drug_graph_cache=drug_graph_cache, protein_graph_cache=protein_graph_cache)
    test_loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=collate)

    if EPOCH is not None:
        model_path = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_epoch{EPOCH}.pth'
    else:
        model_path = f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}.pth'
    print(f"Loading model: {model_path}")
    predModel = nn.DataParallel(Model(hp, device))
    predModel.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    predModel = predModel.to(device)

    mse, ci, rm2 = test(predModel, test_loader)
    print(f"\nTest Results — split{SPLIT_SEED}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  CI:   {ci:.4f}")
    print(f"  r2m:  {rm2:.4f}")
