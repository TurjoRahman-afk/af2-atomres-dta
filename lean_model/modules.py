"""
Building blocks for the lean DTA model (Path-B design).

Design rules baked into these modules:
  - Lean on FROZEN pretrained embeddings (ESM-C, ChemBERTa). The trainable
    layers on top are kept minimal (1 transformer layer, 3 GNN layers) so the
    model is too small to memorize a 68-drug / 442-protein dataset.
  - NO edge features in the GNN. Three independent ablations (KANPM, 3DProtDTA,
    our Runs 5-7) agree they don't help. Graph CONSTRUCTION + node features matter.
  - Attention pooling, NOT mean pooling, so the binding-relevant residues
    dominate instead of being averaged into noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class AttentionPool(nn.Module):
    """Learned-query attention pooling over a token/residue sequence.

    Replaces mean-pooling: a single learned query scores every position, so the
    few positions that matter (e.g. the binding pocket) dominate the pooled
    vector instead of being diluted across hundreds of irrelevant residues.
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.scale = dim ** 0.5

    def forward(self, x, key_padding_mask=None):
        # x: [B, L, D]   key_padding_mask: [B, L]  (True = padding position)
        scores = (x @ self.query) / self.scale            # [B, L]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (attn * x).sum(dim=1)                       # [B, D]


class GraphEncoder(nn.Module):
    """3-layer GIN encoder with add-pooling. Plain GIN, no edge features."""

    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
            d = hidden
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, batch)                  # [B, hidden]


class DrugEncoder(nn.Module):
    """Three complementary drug views: ChemBERTa sequence, molecular graph, Morgan FP."""

    def __init__(self, chemberta_dim=384, fp_dim=2048, node_dim=88,
                 dim=128, n_heads=4, dropout=0.2):
        super().__init__()
        # (a) ChemBERTa token sequence -> ONE light transformer layer
        self.tok_proj = nn.Linear(chemberta_dim, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.tok_encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.tok_ln = nn.LayerNorm(dim)
        # (b) molecular graph (RDKit atom/bond graph)
        self.graph = GraphEncoder(node_dim, dim, layers=3, dropout=dropout)
        # (c) Morgan fingerprint -> small MLP
        self.fp = nn.Sequential(
            nn.Linear(fp_dim, dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, tokens, pad_mask, graph, fp):
        # pad_mask: [B, L_d]  True = padding (ignored by attention)
        h = self.tok_encoder(self.tok_proj(tokens), src_key_padding_mask=pad_mask)
        xd = self.tok_ln(h)                                        # [B, L_d, dim]
        g = self.graph(graph.x, graph.edge_index, graph.batch)     # [B, dim]
        f = self.fp(fp)                                            # [B, dim]
        return xd, g, f


class ProteinEncoder(nn.Module):
    """ESM-C residue sequence + mutation-aware AlphaFold2 structure graph.

    The protein graph nodes carry the ESM-C embedding PLUS a 1-bit mutation flag,
    so the model can tell a mutant apart from its wildtype structure (the fix for
    the 72 Davis mutants that currently reuse the wildtype AF2 graph).
    """

    def __init__(self, esm_dim=1152, node_dim=1153, dim=128, n_heads=4, dropout=0.2):
        # node_dim = esm_dim + 1 (mutation flag appended to each residue node)
        super().__init__()
        self.res_proj = nn.Linear(esm_dim, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.res_encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.res_ln = nn.LayerNorm(dim)
        self.graph = GraphEncoder(node_dim, dim, layers=3, dropout=dropout)

    def forward(self, residues, pad_mask, graph):
        # pad_mask: [B, L_p]  True = padding (ignored by attention)
        h = self.res_encoder(self.res_proj(residues), src_key_padding_mask=pad_mask)
        xp = self.res_ln(h)                                          # [B, L_p, dim]
        g = self.graph(graph.x, graph.edge_index, graph.batch)       # [B, dim]
        return xp, g


class CrossInteraction(nn.Module):
    """Bidirectional cross-attention followed by attention pooling.

    Drug tokens attend to protein residues and vice versa; each side is then
    pooled with a learned-query AttentionPool (not mean).
    """

    def __init__(self, dim=128, n_heads=4, dropout=0.2):
        super().__init__()
        self.d2p = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.p2d = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.pool_d = AttentionPool(dim)
        self.pool_p = AttentionPool(dim)

    def forward(self, xd, xp, drug_pad, prot_pad):
        # drug queries protein
        xd_c, _ = self.d2p(query=xd, key=xp, value=xp, key_padding_mask=prot_pad)
        # protein queries drug
        xp_c, _ = self.p2d(query=xp, key=xd, value=xd, key_padding_mask=drug_pad)
        xd_attn = self.pool_d(xd_c, drug_pad)   # [B, dim]
        xp_attn = self.pool_p(xp_c, prot_pad)   # [B, dim]
        return xd_attn, xp_attn


class BilinearFusion(nn.Module):
    """Low-rank multiplicative drug x protein interaction (kept from your design)."""

    def __init__(self, drug_dim, prot_dim, out_dim=128, rank=64, dropout=0.2):
        super().__init__()
        self.dp = nn.Linear(drug_dim, rank)
        self.pp = nn.Linear(prot_dim, rank)
        self.out = nn.Linear(rank, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, drug, prot):
        d = torch.tanh(self.dp(drug))           # [B, rank]
        p = torch.tanh(self.pp(prot))           # [B, rank]
        return self.bn(self.drop(self.out(d * p)))
