"""Smoke test: AF2-PocketCross-DTA (GNN-prior variant) on REAL data.
Checks: builds, forward+backward work, shapes correct, and — critically — that the
new per-residue GNN embedding (via to_dense_batch) actually aligns with real protein
lengths (not just that it doesn't crash)."""
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn
import model_pocketcross_gnnprior as M
from model_pocketcross_gnnprior import MODEL
import pickle

import warnings; warnings.filterwarnings("ignore")


def load_pickle(d):
    with open(d, 'rb+') as f:
        return pickle.load(f)


hp = HyperParameter()
dev = M.device

drug_df = pd.read_csv(hp.drugs_dir)
prot_df = pd.read_csv(hp.prots_dir)
mol2vec_dict = load_pickle(hp.mol2vec_dir)
protvec_dict = load_pickle(hp.protvec_dir)
contact_map = load_pickle(hp.contact_map)

root = f"{hp.data_root}/{hp.dataset}/{hp.running_set}"
df = pd.read_csv(f"{root}/test.csv").head(6)   # small real batch
dataset = CustomDataSet(df, hp)
collate = lambda x: my_collate_fn(x, dev, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map)
loader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=collate)

model = MODEL(hp, dev).to(dev)
n_params = sum(p.numel() for p in model.parameters())
model.train()

batch = next(iter(loader))
drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph, affinity = batch
drug_mat, drug_mask, prot_mat, prot_mask, affinity = [t.to(dev) for t in (drug_mat, drug_mask, prot_mat, prot_mask, affinity)]
drug_graph, protein_graph = drug_graph.to(dev), protein_graph.to(dev)

out, I = model(drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph)

print("\n================ SMOKE TEST: PocketCross (GNN-prior variant) ================")
print(f"params: {n_params/1e6:.2f} M")
print(f"OUTPUT  prediction {tuple(out.shape)}   (expected (6, 1))")
print(f"OUTPUT  interaction map I {tuple(I.shape)}")
print(f"any NaN in output? {torch.isnan(out).any().item()}")

# ---- the critical check: does the densified per-residue embedding actually align
# with real protein lengths, not just run without error? ----
pgn = model.protein_graph_model
pooled, residue_emb, residue_mask = pgn(protein_graph)
real_counts = residue_mask.sum(dim=1).tolist()
prot_seq_lens = df['target_sequence'].str.len().tolist()
print(f"\nAlignment check (per-sample):")
print(f"  residue_mask real-count (from GNN graph): {real_counts}")
print(f"  actual protein sequence length (raw csv): {prot_seq_lens}")
print(f"  position 0 (BOS slot) masked False for all samples? {(~residue_mask[:,0]).all().item()}")
print(f"  nonzero embedding only where mask is True? "
      f"{torch.allclose(residue_emb[~residue_mask], torch.zeros_like(residue_emb[~residue_mask]))}")

# ---- backward pass ----
w = torch.rand_like(affinity)
loss = ((out.squeeze() - affinity) ** 2).mean()
loss.backward()
g_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"\nBACKWARD loss={loss.item():.4f}  gradients_flow_to_all_params={g_ok}")
print(f"pocket MLP first-layer grad exists? {model.pocket.mlp[0].weight.grad is not None}")
print("================ DONE ================\n")
