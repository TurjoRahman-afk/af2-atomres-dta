#This file handles data loading and preprocessing for drug-protein pairs 
import torch
import os.path as osp
from torch.utils.data import Dataset
from rdkit import Chem # for processing molecular structure 
import numpy as np
from torch_geometric.data import Data, Batch# for graph neural network 
import networkx as nx
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# standardize sequence lengths by padding or truncating
def matrix_pad_drug(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:         # if seq is shorter than max_len - pad with zeros and xreate a mask    
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = arr
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else: # otherwise, truncate and create a mask of ones
        new_arr = arr[:max_len]
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask

def matrix_pad_prot(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = torch.zeros((max_len, dim), dtype = torch.float32)
        vec_mask = torch.zeros((max_len), dtype = torch.float32)                            
        new_arr[:len] = torch.from_numpy(arr)
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = torch.from_numpy(arr[:max_len])
        vec_mask = torch.ones((max_len), dtype = torch.float32)  
        return new_arr, vec_mask

# convert protein into a graph structure
def target2graph(distance_map, protein_features_esm):
    target_edge_index = []
    target_edge_distance = []
    protein_features_esm = protein_features_esm[1:-1, :]
    target_size = protein_features_esm.shape[0]

    # AF2 contact map may cover fewer residues than ESM embedding — align to smaller
    map_size = distance_map.shape[0]
    target_size = min(target_size, map_size)
    protein_features_esm = protein_features_esm[:target_size, :]
    distance_map = distance_map[:target_size, :target_size].copy()

    # Force self-loops and consecutive backbone edges with representative distances
    for i in range(target_size):
        if distance_map[i, i] == 0.0:
            distance_map[i, i] = 2.0        # self-loop: min physical distance
        if i + 1 < target_size and distance_map[i, i + 1] == 0.0:
            distance_map[i, i + 1] = 3.8   # backbone Cα-Cα sequential distance

    index_row, index_col = np.where(distance_map > 0.0)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_distance.append(distance_map[i, j])

    target_feature = torch.FloatTensor(protein_features_esm)
    target_edge_index = torch.LongTensor(target_edge_index).transpose(1, 0)
    # v1: binary contact map — uniform edge weight 1.0 (not distance-scaled), to match the 0.19 setup
    edge_weight = torch.ones(len(target_edge_distance), dtype=torch.float32)

    return target_size, target_feature, target_edge_index, edge_weight

def target2struct(distance_map, protein_max, struct_dim=8, sigma=4.0, seq_exclude=2):
    """Per-residue structural features from the AF2 contact map, for the pocket prior.
    Aligned to ESM-C sequence positions with a +1 offset (position 0 = BOS token), so the
    features line up with the residues the interaction attention operates on.
    Returns a [protein_max, struct_dim] tensor.

    Uses the REAL Cα distances stored in the AF2 map (0 < d <= ~8A for contacts, 0 = no
    contact) rather than just a binary contact count, so the pocket prior sees actual
    packing geometry:
      [0] scaled total contact degree
      [1] tight-shell count   (<=5.5A)
      [2] mid-shell count     (5.5-6.5A)
      [3] loose-shell count   (6.5-8A)
      [4] distance-decay-weighted density  sum(exp(-d^2/2*sigma^2))  (closer = more weight)
      [5] mean contact distance (packing tightness)
      [6] z-scored total degree (relative to this protein)
      [7] real-residue flag

    Sequence-adjacent pairs (|i-j| <= seq_exclude) are excluded from every contact feature:
    the Cα-Cα backbone bond (~3.8A) puts every residue's immediate neighbors inside a
    naive 0-4A shell regardless of 3D fold, making that shell almost constant across all
    residues. Excluding them means every feature reflects true tertiary packing (the
    signal a pocket prior actually needs), not trivial chain connectivity.

    Shell boundaries (5.5 / 6.5 / 8A) are calibrated from the *actual* long-range Cα-Cα
    distance distribution (measured across a sample of Davis proteins after excluding
    sequence-local pairs): true non-bonded contacts almost never fall below ~4.5A, so a
    naive 0-4A bin is empty. 5.5/6.5 splits the populated 3.3-8A range into roughly even
    ~25/33/41% thirds instead.
    """
    dmap = np.asarray(distance_map, dtype=np.float32).copy()
    L = dmap.shape[0]
    idx = np.arange(L)
    seq_local = np.abs(idx[:, None] - idx[None, :]) <= seq_exclude
    dmap[seq_local] = 0.0   # drop bonded/near-sequence neighbors — keep only long-range (tertiary) contacts

    contact = dmap > 0
    deg = contact.sum(axis=1).astype(np.float32)
    close = ((dmap > 0) & (dmap <= 5.5)).sum(axis=1).astype(np.float32)
    mid = ((dmap > 5.5) & (dmap <= 6.5)).sum(axis=1).astype(np.float32)
    far = ((dmap > 6.5) & (dmap <= 8.0)).sum(axis=1).astype(np.float32)
    weighted = np.where(contact, np.exp(-(dmap ** 2) / (2 * sigma ** 2)), 0.0).sum(axis=1).astype(np.float32)
    dist_sum = np.where(contact, dmap, 0.0).sum(axis=1)
    mean_dist = np.divide(dist_sum, deg, out=np.zeros_like(dist_sum), where=deg > 0).astype(np.float32)

    mean_deg = float(deg.mean())
    std_deg = float(deg.std()) + 1e-6

    feats = np.zeros((protein_max, struct_dim), dtype=np.float32)
    end = min(L, protein_max - 1)
    if end > 0:
        feats[1:1 + end, 0] = deg[:end] / 20.0
        feats[1:1 + end, 1] = close[:end] / 10.0
        feats[1:1 + end, 2] = mid[:end] / 10.0
        feats[1:1 + end, 3] = far[:end] / 10.0
        feats[1:1 + end, 4] = weighted[:end] / 10.0
        feats[1:1 + end, 5] = mean_dist[:end] / 8.0
        feats[1:1 + end, 6] = (deg[:end] - mean_deg) / std_deg
        feats[1:1 + end, 7] = 1.0
    return torch.from_numpy(feats)


def pocketcross_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map,
                           drug_graph_cache=None, protein_graph_cache=None, struct_dim=8, struct_cache=None):
    """Like my_collate_fn but also returns per-residue AF2 structural features (prot_struct)."""
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)
    b_drug_mat = torch.zeros((batch_size, drug_max, hp.mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, hp.protvec_dim), dtype=torch.float32)
    b_prot_struct = torch.zeros((batch_size, protein_max, struct_dim), dtype=torch.float32)
    b_label = torch.zeros(batch_size, dtype=torch.float32)
    b_drug_graph, b_protein_graph = [], []

    for i, pair in enumerate(batch_data):
        drug_id, prot_id, label = pair[0], pair[2], pair[4]
        drug_id, prot_id = str(drug_id), str(prot_id)
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max)

        if drug_graph_cache is not None and drug_id in drug_graph_cache:
            drug_graph = drug_graph_cache[drug_id]
        else:
            _, node_attr, edge_index, edge_attr = smile2graph(
                drug_df.loc[drug_df['drug_key'] == pair[0], 'compound_iso_smiles'].iloc[0])
            drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)

        if protein_graph_cache is not None and prot_id in protein_graph_cache:
            protein_graph = protein_graph_cache[prot_id]
        else:
            _, tf, tei, ew = target2graph(prot_contact_map, prot_mat)
            protein_graph = Data(x=tf, edge_index=tei, edge_weight=ew)
        b_protein_graph.append(protein_graph)

        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
        if struct_cache is not None and prot_id in struct_cache:
            b_prot_struct[i] = struct_cache[prot_id]
        else:
            b_prot_struct[i] = target2struct(prot_contact_map, protein_max, struct_dim)
        b_label[i] = label

    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    return b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_prot_struct, b_drug_graph, b_protein_graph, b_label


def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                'Pt', 'Hg', 'Pb', 'X']]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['degree'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['ImplicitValence'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['num_h'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['hybridization'] == x) for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))


    return edge_index, edge_attr

"""
this particular function converts a SMILES string into a 
milecular graph representation for graph neural networks.
"""
def smile2graph(smile):
    #uses rdkit to convert SMILES into a molecular object
    mol = Chem.MolFromSmiles(smile)

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    mol_size = mol.GetNumAtoms()
    g = nx.DiGraph()
    
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                a_type=atom_i.GetSymbol(),
                a_num=atom_i.GetAtomicNum(),
                acceptor=0,
                donor=0,
                aromatic=atom_i.GetIsAromatic(),
                hybridization=atom_i.GetHybridization(),
                num_h=atom_i.GetTotalNumHs(),
                degree = atom_i.GetDegree(),
                # 5 more node features
                ExplicitValence=atom_i.GetExplicitValence(),
                FormalCharge=atom_i.GetFormalCharge(),
                ImplicitValence=atom_i.GetImplicitValence(),
                NumExplicitHs=atom_i.GetNumExplicitHs(),
                NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
            )
            
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor']

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                            b_type=e_ij.GetBondType(),
                      
                            IsConjugated=int(e_ij.GetIsConjugated()),
                            )
                
    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)         

    return mol_size, node_attr, edge_index, edge_attr

"""
This function is a custom batch collation function used by pytorch DataLoader.
It takes individual drug protein samples and combines them into batched tensors ready for the neural network."""
def my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map, isEsm=False, drug_graph_cache=None, protein_graph_cache=None):
    # get batch configuration
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # initialize empty batch tensors, creates tensors filled with zeros to store batch data
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    b_label = torch.zeros(batch_size, dtype=torch.float32)
    
    # for storing individual drug and protein graph 
    b_drug_graph = []    
    b_protein_graph = []
    
    # Process each sample in the batch
    for i, pair in enumerate(batch_data):   
        # extract sample data     
        drug_id, prot_id, label = pair[0], pair[2], pair[4]
        drug_smiles = drug_df.loc[drug_df['drug_key'] == drug_id, 'compound_iso_smiles'].iloc[0]
        prot_seq = prot_df.loc[prot_df['target_key'] == prot_id, 'target_sequence'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)        
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max) 

        # Drug graph — use cache if available, otherwise build
        if drug_graph_cache is not None and drug_id in drug_graph_cache:
            drug_graph = drug_graph_cache[drug_id]
        else:
            mol_size, node_attr, edge_index, edge_attr = smile2graph(drug_smiles)
            drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)

        # Protein graph — use cache if available, otherwise build
        if protein_graph_cache is not None and prot_id in protein_graph_cache:
            protein_graph = protein_graph_cache[prot_id]
        else:
            target_size, target_features, target_edge_index, edge_weight = target2graph(prot_contact_map, prot_mat)
            protein_graph = Data(x=target_features, edge_index=target_edge_index, edge_weight=edge_weight)
        b_protein_graph.append(protein_graph)
        
        
        # Store other values for the batch
        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
        b_label[i] = label
    
    # Batch graphs using PyG's built-in functionality
    # Batch.from_data_list() combines multiple graphs into a single large graph with batch indices 
    # this allows efficient parallel processing on GPU
    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    
    return b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_drug_graph, b_protein_graph, b_label


"""
the function is almost identical to my_collate_fn()
but specificalle designed for inference or prediction """
def pred_my_collate_fn(batch_data, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map, isEsm=False):
    batch_size = len(batch_data)
    drug_max = hp.drug_max_len
    protein_max = hp.prot_max_len
    mol2vec_dim = hp.mol2vec_dim
    protvec_dim = hp.protvec_dim
    
    # Mat for pretrain feat
    b_drug_mask = torch.zeros((batch_size, drug_max), dtype=torch.float32)
    b_prot_mask = torch.zeros((batch_size, protein_max), dtype=torch.float32)    
    b_drug_mat = torch.zeros((batch_size, drug_max, mol2vec_dim), dtype=torch.float32)
    b_prot_mat = torch.zeros((batch_size, protein_max, protvec_dim), dtype=torch.float32)
    
    b_drug_graph = []    
    b_protein_graph = []
    
    # Process each sample in the batch
    for i, pair in enumerate(batch_data):        
        drug_id, prot_id = pair[0], pair[2]
        drug_smiles = drug_df.loc[drug_df['drug_key'] == drug_id, 'compound_iso_smiles'].iloc[0]
        prot_seq = prot_df.loc[prot_df['target_key'] == prot_id, 'target_sequence'].iloc[0]        
        drug_id = str(drug_id)
        prot_id = str(prot_id)
        drug_mat = mol2vec_dict["mat_dict"][drug_id]
        prot_mat = protvec_dict["mat_dict"][prot_id]
        prot_contact_map = contact_map['contact_map'][prot_id]
        drug_mat_pad, drug_mask = matrix_pad_drug(drug_mat, drug_max)        
        prot_mat_pad, prot_mask = matrix_pad_prot(prot_mat, protein_max) 

        # Drug graph for PyTorch Geometric
        mol_size, node_attr, edge_index, edge_attr = smile2graph(drug_smiles)
        drug_graph = Data(x=node_attr, edge_index=edge_index, edge_weight=edge_attr)
        b_drug_graph.append(drug_graph)
        
        target_size, target_features, target_edge_index, edge_weight = target2graph(prot_contact_map, prot_mat)
        protein_graph = Data(x=target_features, edge_index=target_edge_index, edge_weight=edge_weight)
        b_protein_graph.append(protein_graph)
        
        
        # Store other values for the batch
        b_drug_mat[i] = drug_mat_pad
        b_drug_mask[i] = drug_mask
        b_prot_mat[i] = prot_mat_pad
        b_prot_mask[i] = prot_mask
    
    # Batch graphs using PyG's built-in functionality
    b_drug_graph = Batch.from_data_list(b_drug_graph)
    b_protein_graph = Batch.from_data_list(b_protein_graph)
    
    return b_drug_mat, b_drug_mask, b_prot_mat, b_prot_mask, b_drug_graph, b_protein_graph


class CustomDataSet(Dataset):
    def __init__(self, dataset, hp):    
        self.hp = hp
        self.dataset = dataset
        
    def __getitem__(self, index):
        return self.dataset.iloc[index,:]

    def __len__(self):
        return len(self.dataset)
