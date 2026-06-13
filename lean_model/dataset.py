"""
Dataset + collate for LeanDTA.

Reuses the PROVEN featurization from code/MyDataset.py (read-only import — nothing
in code/ is modified) and adds the two new inputs: Morgan fingerprints and the
per-residue mutation flag on the protein graph.
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

# make code/ importable (read-only reuse of the tested DRUG graph builder + padders)
_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.append(_CODE_DIR)
from MyDataset import smile2graph, matrix_pad_drug, matrix_pad_prot  # noqa: E402

from featurize import morgan_fingerprint, mutation_flag_vector


def build_protein_graph(distance_map, esm_mat, target_key):
    """Lean protein graph: ESM-C node features (+ global mutation flag) and edge_index only.

    Produces the SAME node features and edge topology as code/MyDataset.target2graph
    (ESM BOS/EOS strip, align to map size, forced self-loops + backbone edges,
    edges where distance > 0) but skips the RBF / edge_weight computation that
    LeanDTA never uses.
    """
    esm = esm_mat[1:-1, :]                                  # strip BOS/EOS
    size = min(esm.shape[0], distance_map.shape[0])
    esm = esm[:size, :]
    dmap = distance_map[:size, :size].copy()
    for i in range(size):
        if dmap[i, i] == 0.0:
            dmap[i, i] = 2.0                                # self-loop
        if i + 1 < size and dmap[i, i + 1] == 0.0:
            dmap[i, i + 1] = 3.8                            # backbone Cα-Cα
    rows, cols = np.where(dmap > 0.0)
    edge_index = torch.LongTensor(np.vstack([rows, cols]))  # [2, E]
    feats = torch.FloatTensor(esm)                          # [size, 1152]
    flag = torch.from_numpy(mutation_flag_vector(target_key, size))  # [size, 1] global flag
    x = torch.cat([feats, flag], dim=1)                     # [size, 1153]
    return Data(x=x, edge_index=edge_index)


def build_caches(drug_df, prot_df, protvec_dict, contact_map):
    """Pre-build everything keyed by id so the collate is fast.

    Returns:
      drug_graph_cache[did]   -> PyG Data(x=[N,88], edge_index)
      drug_fp_cache[did]      -> Tensor[2048]
      protein_graph_cache[pid]-> PyG Data(x=[S,1153] = ESM-C + mutation flag, edge_index)
    """
    drug_graph_cache, drug_fp_cache = {}, {}
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Drug feats"):
        did = str(row["drug_key"])
        smiles = row["compound_iso_smiles"]
        _, node_attr, edge_index, _ = smile2graph(smiles)
        drug_graph_cache[did] = Data(x=node_attr, edge_index=edge_index)  # GIN: no edge feats
        drug_fp_cache[did] = torch.from_numpy(morgan_fingerprint(smiles))

    protein_graph_cache = {}
    for _, row in tqdm(prot_df.iterrows(), total=len(prot_df), desc="Protein feats"):
        pid = str(row["target_key"])
        if pid not in contact_map["contact_map"]:
            continue
        cmap = contact_map["contact_map"][pid].copy()
        esm = protvec_dict["mat_dict"][pid]
        protein_graph_cache[pid] = build_protein_graph(cmap, esm, pid)

    print(f"Caches: {len(drug_graph_cache)} drugs, {len(protein_graph_cache)} proteins")
    return drug_graph_cache, drug_fp_cache, protein_graph_cache


def lean_collate_fn(batch_data, hp, mol2vec_dict, protvec_dict,
                    drug_graph_cache, drug_fp_cache, protein_graph_cache):
    B = len(batch_data)
    dmax, pmax = hp.drug_max_len, hp.prot_max_len
    ddim, pdim = hp.mol2vec_dim, hp.protvec_dim

    b_drug_tokens = torch.zeros((B, dmax, ddim), dtype=torch.float32)
    b_drug_mask = torch.zeros((B, dmax), dtype=torch.float32)
    b_prot_res = torch.zeros((B, pmax, pdim), dtype=torch.float32)
    b_prot_mask = torch.zeros((B, pmax), dtype=torch.float32)
    b_drug_fp = torch.zeros((B, 2048), dtype=torch.float32)
    b_label = torch.zeros(B, dtype=torch.float32)
    b_drug_graph, b_prot_graph = [], []

    for i, pair in enumerate(batch_data):
        did, pid, label = str(pair[0]), str(pair[2]), pair[4]
        drug_mat = mol2vec_dict["mat_dict"][did]      # tensor [L_d, 384]
        prot_mat = protvec_dict["mat_dict"][pid]      # numpy  [L_p, 1152]

        dpad, dmask = matrix_pad_drug(drug_mat, dmax)
        ppad, pmask = matrix_pad_prot(prot_mat, pmax)

        b_drug_tokens[i] = dpad
        b_drug_mask[i] = dmask
        b_prot_res[i] = ppad
        b_prot_mask[i] = pmask
        b_drug_fp[i] = drug_fp_cache[did]
        b_drug_graph.append(drug_graph_cache[did])
        b_prot_graph.append(protein_graph_cache[pid])
        b_label[i] = label

    return (
        b_drug_tokens, b_drug_mask, Batch.from_data_list(b_drug_graph), b_drug_fp,
        b_prot_res, b_prot_mask, Batch.from_data_list(b_prot_graph), b_label,
    )
