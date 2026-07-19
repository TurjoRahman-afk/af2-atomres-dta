"""
AF2-PocketCross-DTA
===================
Structure-guided atom<->residue interaction model.

Backbone kept from KANPM (ChemBERTa/ESM-C transformers + graph GNNs + KAN).
NEW pieces (the contribution):
  - PocketPrior: per-residue "bindability" score from AF2 structural features
  - StructGuidedInteraction: drug-atom <-> protein-residue attention, biased by
    the pocket prior, producing an interpretable interaction map + a bilinear readout.

forward() returns (prediction, interaction_map) so the interaction map can be
inspected / visualised.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import HyperParameter
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch.nn import Linear
from kan import KAN

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- KEPT backbone
class GatedFusionLayer(nn.Module):
    def __init__(self, v_dim, q_dim, output_dim=128):
        super().__init__()
        self.v_transform = nn.Linear(v_dim, output_dim)
        self.q_transform = nn.Linear(q_dim, output_dim)
        self.gate_transform = nn.Linear(output_dim * 2, output_dim)
        self.act = nn.Tanh()

    def forward(self, v, q):
        v = self.act(self.v_transform(v))
        q = self.act(self.q_transform(q))
        gate = torch.sigmoid(self.gate_transform(torch.cat([v, q], dim=1)))
        return gate * v + (1 - gate) * q


class DrugGraphNet(nn.Module):
    def __init__(self, num_features_xd=88, dim=128, output_dim=128):
        super().__init__()
        self.conv0 = GCNConv(num_features_xd, dim)
        self.conv1 = GATConv(dim, dim); self.conv2 = GATConv(dim, dim)
        self.conv3 = GATConv(dim, dim); self.conv4 = GATConv(dim, dim); self.conv5 = GATConv(dim, dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(6)])
        self.fc = Linear(dim, output_dim)

    def forward(self, data):
        x, ei, ew, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)
        x = self.bns[0](F.relu(self.conv0(x, ei, ew.mean(dim=1))))
        for conv, bn in zip([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], self.bns[1:]):
            x = bn(F.relu(conv(x, ei)))
        x = global_add_pool(x, batch)
        return F.dropout(F.relu(self.fc(x)), p=0.2, training=self.training)   # [B, 128]


class ProteinGraphNet(nn.Module):
    def __init__(self, num_features_xd=1152, dim=256, output_dim=128):
        super().__init__()
        self.conv0 = GCNConv(num_features_xd, dim)
        self.conv1 = GATConv(dim, dim); self.conv2 = GATConv(dim, dim)
        self.conv3 = GATConv(dim, dim); self.conv4 = GATConv(dim, dim); self.conv5 = GATConv(dim, dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(6)])
        self.fc = Linear(dim, output_dim)

    def forward(self, data):
        x, ei, ew, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)
        x = self.bns[0](F.relu(self.conv0(x, ei, ew)))
        for conv, bn in zip([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], self.bns[1:]):
            x = bn(F.relu(conv(x, ei)))
        x = global_add_pool(x, batch)
        return F.dropout(F.relu(self.fc(x)), p=0.2, training=self.training)   # [B, 128]


# ----------------------------------------------------------------------------- NEW pieces
class PocketPrior(nn.Module):
    """Per-residue bindability score from AF2 structural features (e.g. contact degree)."""
    def __init__(self, struct_dim, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(struct_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, prot_struct):          # [B, Lp, struct_dim]
        return self.mlp(prot_struct).squeeze(-1)   # [B, Lp]  (pocket logits)


class StructGuidedInteraction(nn.Module):
    """Drug-atom <-> protein-residue attention, biased by the pocket prior."""
    def __init__(self, dim=128):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.tensor(1.0))   # pocket-bias strength (learned)
        self.drop = nn.Dropout(0.2)
        self.scale = dim ** 0.5

    def forward(self, Hd, Hp, drug_mask, prot_mask, pocket_logit):
        # Hd [B,Nd,D]  Hp [B,Lp,D]  masks [B,Nd]/[B,Lp] (1=real,0=pad)  pocket_logit [B,Lp]
        scores = torch.bmm(self.q(Hd), self.k(Hp).transpose(1, 2)) / self.scale   # [B,Nd,Lp]
        scores = scores + self.beta * pocket_logit.unsqueeze(1)                    # pocket bias on residues
        res_pad = (prot_mask < 0.5).unsqueeze(1)                                   # [B,1,Lp] True=pad
        scores = scores.masked_fill(res_pad, -9e15)
        I = self.drop(torch.softmax(scores, dim=-1))                              # [B,Nd,Lp] interaction map
        ctx = torch.bmm(I, Hp)                                                     # [B,Nd,D] residue-context per atom
        inter = Hd * ctx                                                          # [B,Nd,D] structure-weighted interaction
        am = drug_mask.unsqueeze(-1)                                              # [B,Nd,1]
        f_int = (inter * am).sum(dim=1) / am.sum(dim=1).clamp(min=1)              # [B,D] pooled over real atoms
        return f_int, I


class MODEL(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        d = 128
        self.fc3 = nn.Linear(hp.mol2vec_dim, d)     # drug 384 -> 128
        self.fc2 = nn.Linear(hp.protvec_dim, d)     # prot 1152 -> 128
        el = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(el, num_layers=3)
        el2 = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=1024)
        self.transformer_encoder2 = nn.TransformerEncoder(el2, num_layers=3)
        self.drug_ln = nn.LayerNorm(d)
        self.target_ln = nn.LayerNorm(d)

        self.drug_graph_model = DrugGraphNet()
        self.protein_graph_model = ProteinGraphNet()
        self.graph_fusion = GatedFusionLayer(d, d, d)

        self.struct_dim = getattr(hp, 'struct_dim', 4)
        self.pocket = PocketPrior(self.struct_dim)
        self.interaction = StructGuidedInteraction(d)

        # final = f_int(128) + drug_pool(128) + prot_pool(128) + graph(128) = 512
        self.kan = KAN([512, 1024, 512, 1])

    def forward(self, drug_mat, drug_mask, prot_mat, prot_mask, prot_struct, drug_graph, protein_graph):
        Hd = self.drug_ln(self.transformer_encoder(self.fc3(drug_mat)))     # [B,Nd,128]
        Hp = self.target_ln(self.transformer_encoder2(self.fc2(prot_mat)))  # [B,Lp,128]

        pocket_logit = self.pocket(prot_struct)                             # [B,Lp]
        f_int, I = self.interaction(Hd, Hp, drug_mask, prot_mask, pocket_logit)

        gd = (Hd * drug_mask.unsqueeze(-1)).sum(1) / drug_mask.sum(1, keepdim=True).clamp(min=1)   # [B,128]
        gp = (Hp * prot_mask.unsqueeze(-1)).sum(1) / prot_mask.sum(1, keepdim=True).clamp(min=1)   # [B,128]

        g = self.graph_fusion(self.drug_graph_model(drug_graph), self.protein_graph_model(protein_graph))  # [B,128]

        out = self.kan(torch.cat([f_int, gd, gp, g], dim=-1))              # [B,1]
        return out, I
