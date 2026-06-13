"""
LeanDTA — the Path-B lean architecture.

Assembles the building blocks from modules.py into the full model:

    Drug:    ChemBERTa tokens  + molecular graph + Morgan fingerprint
    Protein: ESM-C residues    + mutation-aware AF2 graph
    Interaction: bidirectional cross-attention + attention pooling
    Fusion:  low-rank bilinear product
    Head:    small MLP (or small KAN if use_kan=True)

Target trainable size: ~1.3M parameters (~10x smaller than the current model),
matched to a 68-drug / 442-protein dataset to stop memorizing.
"""

import torch
import torch.nn as nn

from modules import DrugEncoder, ProteinEncoder, CrossInteraction, BilinearFusion


class LeanDTA(nn.Module):
    def __init__(self, dim=128, n_heads=4, dropout=0.3,
                 chemberta_dim=384, esm_dim=1152, fp_dim=2048,
                 drug_node_dim=88, use_kan=False):
        super().__init__()

        self.drug_enc = DrugEncoder(
            chemberta_dim=chemberta_dim, fp_dim=fp_dim, node_dim=drug_node_dim,
            dim=dim, n_heads=n_heads, dropout=dropout,
        )
        # protein graph nodes = ESM-C (esm_dim) + 1-bit mutation flag
        self.prot_enc = ProteinEncoder(
            esm_dim=esm_dim, node_dim=esm_dim + 1,
            dim=dim, n_heads=n_heads, dropout=dropout,
        )

        self.cross = CrossInteraction(dim=dim, n_heads=n_heads, dropout=dropout)

        # combine the views on each side
        self.drug_combine = nn.Linear(dim * 3, dim)   # xd_attn + drug_graph + drug_fp
        self.prot_combine = nn.Linear(dim * 2, dim)   # xp_attn + prot_graph

        self.bilinear = BilinearFusion(dim, dim, out_dim=dim, rank=64, dropout=dropout)

        if use_kan:
            # keep the "KAN" brand option — but small, to avoid the old 8M-param head
            from kan import KAN
            self.head = KAN([dim, 64, 1])
        else:
            self.head = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
            )

    def forward(self, drug_tokens, drug_mask, drug_graph, drug_fp,
                prot_residues, prot_mask, prot_graph):
        """
        drug_tokens   [B, L_d, 384]   ChemBERTa per-token embeddings
        drug_mask     [B, L_d]        1 = real token, 0 = padding
        drug_graph    PyG Batch       .x [N_atoms, 88], .edge_index, .batch
        drug_fp       [B, 2048]       Morgan fingerprint (float)
        prot_residues [B, L_p, 1152]  ESM-C per-residue embeddings
        prot_mask     [B, L_p]        1 = real residue, 0 = padding
        prot_graph    PyG Batch       .x [N_res, 1153] (ESM-C + mutation flag), .edge_index, .batch
        """
        drug_pad = ~drug_mask.bool()    # True = padding
        prot_pad = ~prot_mask.bool()

        xd, drug_g, drug_f = self.drug_enc(drug_tokens, drug_pad, drug_graph, drug_fp)
        xp, prot_g = self.prot_enc(prot_residues, prot_pad, prot_graph)

        xd_attn, xp_attn = self.cross(xd, xp, drug_pad, prot_pad)

        drug_repr = self.drug_combine(torch.cat([xd_attn, drug_g, drug_f], dim=-1))
        prot_repr = self.prot_combine(torch.cat([xp_attn, prot_g], dim=-1))

        fused = self.bilinear(drug_repr, prot_repr)        # [B, dim]
        return self.head(fused).squeeze(-1)                # [B]


if __name__ == "__main__":
    # quick shape sanity check with random tensors + a tiny fake graph batch
    from torch_geometric.data import Data, Batch

    B, Ld, Lp = 4, 220, 1200
    model = LeanDTA()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.2f}M")

    drug_tokens = torch.randn(B, Ld, 384)
    drug_mask = torch.ones(B, Ld); drug_mask[:, 50:] = 0
    prot_res = torch.randn(B, Lp, 1152)
    prot_mask = torch.ones(B, Lp); prot_mask[:, 300:] = 0
    drug_fp = torch.randn(B, 2048)

    dg = Batch.from_data_list([
        Data(x=torch.randn(20, 88), edge_index=torch.randint(0, 20, (2, 40))) for _ in range(B)
    ])
    pg = Batch.from_data_list([
        Data(x=torch.randn(100, 1153), edge_index=torch.randint(0, 100, (2, 300))) for _ in range(B)
    ])

    out = model(drug_tokens, drug_mask, dg, drug_fp, prot_res, prot_mask, pg)
    print("Output shape:", out.shape)   # expect [4]
