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


class BilinearFusion(nn.Module):
    def __init__(self, drug_dim=256, prot_dim=256, output_dim=128, rank=64):
        super().__init__()
        # Low-rank factorization: avoids full drug_dim x prot_dim x output_dim tensor
        self.drug_proj = nn.Linear(drug_dim, rank)
        self.prot_proj = nn.Linear(prot_dim, rank)
        self.output_proj = nn.Linear(rank, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, drug, prot):
        d = torch.tanh(self.drug_proj(drug))    # [B, rank]
        p = torch.tanh(self.prot_proj(prot))    # [B, rank]
        interaction = d * p                      # element-wise [B, rank]
        out = self.output_proj(interaction)      # [B, output_dim]
        return self.bn(self.dropout(out))


class DrugGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=88,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(DrugGraphNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = Linear(dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight.mean(dim=1)))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return x          #smiles_graph[B, 128]


class ProteinGraphNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=1152,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(ProteinGraphNet, self).__init__()

        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.conv0 = GCNConv(num_features_xd, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.conv1 = GATConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.conv2 = GATConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.conv3 = GATConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.conv4 = GATConv(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.conv5 = GATConv(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = Linear(dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.batch.to(device)

        x = self.relu(self.conv0(x, edge_index, edge_weight))
        x = self.bn0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return x           #fasta_graph[B, 128]


class LinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, heads=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear_first = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        sentence_att = F.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = sentence_att.transpose(1, 2)
        minus_inf = -9e15 * torch.ones_like(sentence_att)
        e = torch.where(masks > 0.5, sentence_att, minus_inf)
        att = self.softmax(e)
        sentence_embed = att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.heads
        return avg_sentence_embed


class MODEL(nn.Module):

    def __init__(self, hp, device):
        super(MODEL, self).__init__()

        self.mol2vec_dim = hp.mol2vec_dim
        self.protvec_dim = hp.protvec_dim
        self.encoder_layers = 3
        self.encoder_heads = 8
        self.feedforward_dim = 1024
        self.dropout = 0.2

        self.drug_graph_model = DrugGraphNet(n_output=128)
        self.protein_graph_model = ProteinGraphNet(n_output=128)

        # Drug and Protein sequence encoders: separate transformer encoders for each modality
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=self.encoder_layers)

        # Cross-attention: drug tokens attend to protein, protein tokens attend to drug
        self.cross_attn_drug = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.2, batch_first=True)
        self.cross_attn_prot = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.2, batch_first=True)

        # Interaction branch (concatenated drug+protein sequence) — kept for additional context
        self.inter_attn_one = LinearAttention(128, 64, 8)

        self.drug_ln = nn.LayerNorm(128)
        self.target_ln = nn.LayerNorm(128)

        self.fc2 = nn.Linear(self.protvec_dim, 128)
        self.fc3 = nn.Linear(self.mol2vec_dim, 128)

        # Bilinear fusion: drug-side (seq+graph) x protein-side (seq+graph)
        self.bilinear = BilinearFusion(drug_dim=256, prot_dim=256, output_dim=256, rank=64)

        # Project cat_attn from 128 → 256 to match bilinear_out dimension
        self.cat_attn_proj = nn.Linear(128, 256)

        # KAN predictor: bilinear_out(256) + cat_attn_proj(256) = 512
        self.kan_predictor = KAN([512, 1024, 512, 1])

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def forward(self, drug, drug_mat, drug_mask, protein, prot_mat, prot_mask, drug_graph, protein_graph):

        smiles_graph = self.drug_graph_model(drug_graph)        # [B, 128]
        fasta_graph = self.protein_graph_model(protein_graph)   # [B, 128]

        # Drug sequence branch
        smiles_emb = self.transformer_encoder(self.fc3(drug_mat))   # [B, 220, 128]
        xd = self.drug_ln(smiles_emb)                               # [B, 220, 128]

        # Protein sequence branch
        fasta_emb = self.transformer_encoder2(self.fc2(prot_mat))   # [B, 1200, 128]
        xp = self.target_ln(fasta_emb)                              # [B, 1200, 128]

        # Cross-attention: drug attends to protein, protein attends to drug
        drug_pad_mask = ~drug_mask.bool()   # [B, 220]  True = padding position
        prot_pad_mask = ~prot_mask.bool()   # [B, 1200] True = padding position

        xd_cross, _ = self.cross_attn_drug(query=xd, key=xp, value=xp, key_padding_mask=prot_pad_mask)
        xp_cross, _ = self.cross_attn_prot(query=xp, key=xd, value=xd, key_padding_mask=drug_pad_mask)

        xd_attn = xd_cross.mean(dim=1)     # [B, 128]
        xp_attn = xp_cross.mean(dim=1)     # [B, 128]

        # Interaction branch: combined drug+protein sequence → attention pooling
        cat_f = torch.cat([xp, xd], dim=1)                         # [B, 1420, 128]
        # use each sample's real length (was hardcoded 128, which only masked the first sample in the batch)
        drug_lengths = drug_mask.sum(dim=1).long()                 # [B]
        prot_lengths = prot_mask.sum(dim=1).long()                 # [B]
        smiles_mask = self.generate_masks(xd, drug_lengths, 8)
        fasta_mask = self.generate_masks(xp, prot_lengths, 8)
        cat_mask = torch.cat([fasta_mask, smiles_mask], dim=-1)     # [B, 8, 1420]
        cat_attn = self.cat_attn_proj(self.inter_attn_one(cat_f, cat_mask))  # [B, 256]

        # Bilinear fusion: pair drug-side and protein-side features
        drug_side = torch.cat([xd_attn, smiles_graph], dim=-1)      # [B, 256]
        prot_side = torch.cat([xp_attn, fasta_graph], dim=-1)       # [B, 256]
        bilinear_out = self.bilinear(drug_side, prot_side)           # [B, 256]

        # Final prediction
        final = torch.cat([bilinear_out, cat_attn], dim=-1)         # [B, 512]
        out = self.kan_predictor(final)                              # [B, 1]
        return out
