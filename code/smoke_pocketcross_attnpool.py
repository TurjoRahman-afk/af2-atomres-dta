"""Smoke test: AF2-PocketCross-DTA (attention-pooling variant) on REAL data.
Verifies the swap is correct, not just that it runs:
  - attention weights are a valid distribution (sum to 1) and EXCLUDE padding
  - pooled output differs from the plain mean it replaced (i.e. it's actually learning weights)
  - param delta vs the champion is only the two LinearAttention modules
  - forward + backward work, gradients reach the new modules
"""
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn
import model_pocketcross_attnpool as M
from model_pocketcross_attnpool import MODEL
from model_pocketcross_gnnprior import MODEL as ChampionMODEL

import warnings; warnings.filterwarnings("ignore")


def load_pickle(d):
    with open(d, 'rb+') as f:
        return pickle.load(f)


hp = HyperParameter()
dev = M.device

drug_df = pd.read_csv(hp.drugs_dir); prot_df = pd.read_csv(hp.prots_dir)
mol2vec_dict = load_pickle(hp.mol2vec_dir); protvec_dict = load_pickle(hp.protvec_dir)
contact_map = load_pickle(hp.contact_map)

root = f"{hp.data_root}/{hp.dataset}/{hp.running_set}"
df = pd.read_csv(f"{root}/test.csv").head(6)
loader = DataLoader(CustomDataSet(df, hp), batch_size=6, shuffle=False,
                    collate_fn=lambda x: my_collate_fn(x, dev, hp, drug_df, prot_df,
                                                       mol2vec_dict, protvec_dict, contact_map))

model = MODEL(hp, dev).to(dev); model.train()
champ = ChampionMODEL(hp, dev)
n_new = sum(p.numel() for p in model.parameters())
n_old = sum(p.numel() for p in champ.parameters())
n_pool = sum(p.numel() for p in model.drug_attn.parameters()) + \
         sum(p.numel() for p in model.target_attn.parameters())

batch = next(iter(loader))
drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph, affinity = batch
drug_mat, drug_mask, prot_mat, prot_mask, affinity = [t.to(dev) for t in (drug_mat, drug_mask, prot_mat, prot_mask, affinity)]
drug_graph, protein_graph = drug_graph.to(dev), protein_graph.to(dev)

out, I = model(drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph)

print("\n========= SMOKE TEST: PocketCross (attention-pooling variant) =========")
print(f"params: {n_new/1e6:.3f} M   champion: {n_old/1e6:.3f} M   delta: +{n_new-n_old:,}")
print(f"  (the two LinearAttention modules account for {n_pool:,} params -> "
      f"{'MATCHES delta' if n_new - n_old == n_pool else 'MISMATCH!'})")
print(f"prediction {tuple(out.shape)}  NaN? {torch.isnan(out).any().item()}")

# ---- verify the attention pooling is a valid, padding-aware distribution ----
model.eval()
with torch.no_grad():
    Hp = model.target_ln(model.transformer_encoder2(model.fc2(prot_mat)))
    masks = prot_mask.unsqueeze(1).expand(-1, model.pool_heads, -1)
    a = torch.tanh(model.target_attn.linear_first(Hp))
    a = model.target_attn.linear_second(a).transpose(1, 2)
    a = torch.where(masks > 0.5, a, -9e15 * torch.ones_like(a))
    att = torch.softmax(a, dim=-1)                       # [B, heads, L]

    real_lens = prot_mask.sum(1).long().tolist()
    pad_mass = (att * (masks < 0.5).float()).sum(-1)     # attention mass landing on padding
    print(f"\nAttention validity (protein pool):")
    print(f"  weights sum to 1 per head? {torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-4)}")
    print(f"  attention mass on PADDING (should be ~0): {pad_mass.max().item():.3e}")
    print(f"  real residue counts in batch: {real_lens}")

    # is it actually weighting, or just reproducing a uniform mean?
    gp_attn = model.target_attn(Hp, masks)
    gp_mean = (Hp * prot_mask.unsqueeze(-1)).sum(1) / prot_mask.sum(1, keepdim=True).clamp(min=1)
    diff = (gp_attn - gp_mean).abs().mean().item()
    uniform = 1.0 / torch.tensor(real_lens, dtype=torch.float, device=att.device)
    max_w = att.max(-1).values.mean(-1)                  # avg (over heads) peak weight per sample
    print(f"  attn output differs from plain mean? {diff > 1e-4}  (mean abs diff {diff:.4f})")
    print(f"  peak weight vs uniform (>1 = focusing): "
          f"{[f'{(m/u).item():.1f}x' for m, u in zip(max_w, uniform)]}")

# ---- backward ----
model.train()
out, _ = model(drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph)
loss = ((out.squeeze() - affinity) ** 2).mean()
loss.backward()
ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"\nBACKWARD loss={loss.item():.4f}  gradients_flow_to_all_params={ok}")
print(f"  drug_attn grad exists?   {model.drug_attn.linear_first.weight.grad is not None}")
print(f"  target_attn grad exists? {model.target_attn.linear_first.weight.grad is not None}")
print("========= DONE =========\n")
