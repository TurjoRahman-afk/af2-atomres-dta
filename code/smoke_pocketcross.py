"""Smoke test: build AF2-PocketCross-DTA, run one forward pass on dummy data, check shapes."""
import torch
from torch_geometric.data import Data, Batch
from hyperparameter import HyperParameter
import model_pocketcross as M
from model_pocketcross import MODEL

hp = HyperParameter()
dev = M.device
B, Nd, Lp, SD = 4, 220, 1200, 4      # batch, drug atoms, protein residues, struct feats
hp.struct_dim = SD

# --- dummy sequence inputs ---
drug_mat  = torch.randn(B, Nd, hp.mol2vec_dim, device=dev)      # ChemBERTa tokens
prot_mat  = torch.randn(B, Lp, hp.protvec_dim, device=dev)      # ESM-C residues
drug_mask = torch.zeros(B, Nd, device=dev);  drug_mask[:, :40]  = 1   # ~40 real atoms
prot_mask = torch.zeros(B, Lp, device=dev);  prot_mask[:, :300] = 1   # ~300 real residues
prot_struct = torch.randn(B, Lp, SD, device=dev)               # AF2 per-residue struct feats

# --- dummy graph batches (match the real collate format) ---
def drug_graph(n=40):
    ei = torch.randint(0, n, (2, n * 3))
    return Data(x=torch.randn(n, 88), edge_index=ei, edge_weight=torch.rand(ei.shape[1], 6))  # drug: [E,6]
def prot_graph(n=300):
    ei = torch.randint(0, n, (2, n * 4))
    return Data(x=torch.randn(n, 1152), edge_index=ei, edge_weight=torch.ones(ei.shape[1]))   # prot: [E]

dg = Batch.from_data_list([drug_graph() for _ in range(B)]).to(dev)
pg = Batch.from_data_list([prot_graph() for _ in range(B)]).to(dev)

# --- build + run ---
model = MODEL(hp, dev).to(dev)
n_params = sum(p.numel() for p in model.parameters())
model.train()
out, I = model(drug_mat, drug_mask, prot_mat, prot_mask, prot_struct, dg, pg)

print("\n================ SMOKE TEST: AF2-PocketCross-DTA ================")
print(f"params: {n_params/1e6:.2f} M")
print(f"INPUT   drug_mat {tuple(drug_mat.shape)}  prot_mat {tuple(prot_mat.shape)}  prot_struct {tuple(prot_struct.shape)}")
print(f"OUTPUT  prediction {tuple(out.shape)}   (expected ({B}, 1))")
print(f"OUTPUT  interaction_map I {tuple(I.shape)}   (expected ({B}, {Nd}, {Lp}))")
print(f"I rows sum to 1 over residues? {torch.allclose(I.sum(-1), torch.ones_like(I.sum(-1)), atol=1e-3)}")

# --- backward pass (make sure gradients flow / loss works) ---
y = torch.rand(B, device=dev) * 5 + 5           # dummy pKd in [5,10]
w = 1.0 + 0.5 * (y - 5.0)                        # weighted loss
loss = (w * (out.squeeze() - y) ** 2).mean()
loss.backward()
g_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"BACKWARD loss={loss.item():.4f}  gradients_flow={g_ok}")
print(f"pocket-bias beta (learnable) = {model.interaction.beta.item():.3f}, grad={model.interaction.beta.grad is not None}")
print("================ ALL CHECKS PASSED ================\n" if (out.shape==(B,1) and I.shape==(B,Nd,Lp) and g_ok) else "!!! CHECK FAILED !!!")
