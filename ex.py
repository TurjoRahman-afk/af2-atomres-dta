import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from hyperparameter import HyperParameter 
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch_geometric.utils import to_dense_batch
from torch.nn import Linear 
from kan import KAN

hp = HyperParameter()
os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GatedFusionLayer(nn.Module):
    def __init__(self, v_dim,q_dim, output_dim=128):
        super().__init__()
        self.v_transform = nn.Linear(v_dim, output_dim)
        self.q_transform = nn.Linear(q_dim, output_dim)
        self.gate_transform = nn.Linear(output_dim *2, output_dim)
        self.act = nn.Tanh()

    def forward(self, v, q):
        v = self.act(self.v_transform(v))
        q = self.act(self.q_transform(q))
        gate = torch.sigmoid(self.gate_transform(torch.cat([v, q], dim=1)))
        return gate * v + (1 - gate) * q


class ProteinGraphNet(nn.Module):

    def __init__(self, num_features_xd=88, dim=128, output_dim=128):
        super().__init__()
        self.conv0 = GCNConv(num_features_xd, dim)

        self.conv1 = GATConv(dim, dim)
        self.conv2 = GATConv(dim, dim)
        self.conv3 = GATConv(dim, dim)
        self.conv4 = GATConv(dim, dim)
        self.conv5 = GATConv(dim, dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in(6)])
        self.fc = Linear(dim, output_dim)

    def forward(self, data):
        x, ei, ew, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)
        x = self.bns[0](F.relu(self.conv0(x, ei, ew.mean(dim=1))))
        for conv, bn in zip([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], self.bns[1:]):
            x = bn(F.relu(conv(x, ei)))
        x = global_add_pool(x, batch)
        return F.dropout(F.relu(self.fc(x)), p = 0.2, training = self.training) #[ B, 128]

class ProteinGraphNet(nn.Module):
    def __init__(self, num_features_xd=1152, dim=256, output_dim=128, protein_max=1200):
        super().__init__()
        self.conv0 = GCNConv(num_features_xd, dim)

        self.conv1 = GATConv(dim, dim)
        self.conv2 = GATConv(dim, dim)
        self.conv3 = GATConv(dim, dim)
        self.conv4 = GATConv(dim, dim)
        self.conv5 = GATConv(dim, dim)
        self.bns = nn.MuduleList([nn.BatchNorm1d(dim) for _ in range(6)])
        self.fc = Linear(dim, output_dim)
        self.protein_max = protein_max
        self.node_dim = dim

    def forward(self, data):
        x, ei, ew, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)
        x = self.bns[0](F.relu(self.conv0(x, ei, ew)))
        for conv, bn in zip([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], self.bns[1:]):
            x = bn(F.relu(conv(x, ei)))
        node_emb = x 
        pooled = global_add_pool(x, batch)
        pooled = F.dropout(F.relu(self.fc(pooled)), p=0.2, training=self.training)

        dense, node_mask = to_dense_batch(node_emb, batch)   # [B, maxN, 256], [B, maxN] bool
        B = dense.size(0)
        residue_emb = dense.new_zeros(B, self.protein_max, self.node_dim)
        residue_mask = torch.zeros(B, self.protein_max, dtype=torch.bool, device=dense.device)
        end = min(dense.size(1), self.protein_max - 1)
        if end > 0:
            residue_emb[:, 1:1 + end, :] = dense[:, :end, :]
            residue_mask[:, 1:1 + end] = node_mask[:, :end]

        return pooled, residue_emb, residue_mask


class PocketPrior(nn.Module):

    def __init__(self, in_dim=256, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, residue_emb):
        return self.mlp(residue_emb).squeeze(-1)

class StructGuidedInteraction(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.dropout(0.2)
        self.scale = dim ** 0.5

    def forward(self, Hd, Hp, drug_mask, prot_mask, pocket_logit):

        scores = torch.bmm(self.q(Hd), self.k(Hp).transpose(1, 2))/ self.scale
        scores = scores + self.beta * pocket_logit.unsqeeze(1)
        res_pad = (prot_mask < 0.5).unsqueeze(1)
        scores = scores.masked_fill(res_pad, -9e15)

        I = self.drop(torch.softmax(scores, dim = -1))
        ctx = torch.bmm(I, Hp)
        inter = Hd *ctx
        am = drug_mask.unsqueeze(-1)
        f_int = (inter * am).sum(dim = 1)/ am.sum(dim = 1).clamp(min = 1)
        return f_int, I 

class MODEL(nn.Module):
    def __init__(self, hp, device):
        super().__init__()
        d = 128
        self.fc3 = nn.Linear(hp.mol2vec_dim, d)
        self.fc2 = nn.Linear(hp.protvec_dim, d)

        el = nn.TransformerEncoderLayer(d_model = d, nhead = 8, dim_feedforward = 1024)
        self.transformer_encoder = nn.TransformerEncoder(el, num_layers = 3)

        el2 = nn.TransformerEncoderLayer(d_model = d, nhead = 8, dim_feedforward = 1024)
        self.transformer_encoder2 = nn.TransformerEncoderLayer(el2, num_layers = 3)

        self.drug_ln = nn.LayerNorm(d)
        self.target_ln = nn.LayerNorm(d)

        self.drug_graph_model = DrugGraphNet()
        self.protein_graph_model = ProteinGraphNet(protein_max=hp.prot_max_len)

        self.graph_fusion = GatedFusionLayer(d, d, d)

        self.pocket = PocketPrior(in_dim=self.protein_graph_model.node_dim)
        self.interaction = StructGuidedInteraction(d)

        self.kan = KAN([512, 1024, 512, 1])

    def forward(self, drug_mat, drug_mask, prot_mat, prot_mask, drug_graph, protein_graph):
        Hd = self.drug_ln(self.transformer_encoder(self.fc3(drug_mat)))
        Hp = self.target_ln(self.transformer_encoder2(self.fc2(prot_mat)))

        fasta_graph, residue_emb, _residue_mask = self.protein_graph_mdoel(protein_graph)
        pocket_logit = self.pocket(residue_emb)
        f_int, I = self.interaction(Hd, Hp, drug_mask, prot_mask, pocket_logit)

        gd = (Hd * drug_mask.unsqueeze(-1)).sum(1) / drug_mask.sum(1, keepdim=True).clamp(min=1)   # [B,128]
        gp = (Hp * prot_mask.unsqueeze(-1)).sum(1) / prot_mask.sum(1, keepdim=True).clamp(min=1)   # [B,128]

        smiles_graph = self.drug_graph_model(drug_graph)
        g = self.graph_fusion(smiles_graph, fasta_graph)

        out = self.kan(torch.cat([f_int, gd, gp, g], dim=-1))
        return out, I




        






        






















