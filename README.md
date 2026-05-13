# AF2-CrossKAN-DTA: Drug-Target Affinity Prediction with AlphaFold2 Structures, Cross-Attention, and Kolmogorov-Arnold Networks

---

## Overview

AF2-CrossKAN-DTA is a drug-target affinity (DTA) prediction model that predicts how strongly a drug molecule binds to a protein target. Strong binding means the drug is more likely to work — making DTA prediction a critical step in computational drug discovery.

This project builds on top of KANPM-DTA and introduces three architectural improvements:

1. **AlphaFold2 3D Contact Maps** — replaces sequence-predicted 2D contact maps with real 3D Cα distance matrices downloaded directly from the AlphaFold2 public database
2. **Cross-Attention** — drug and protein sequence tokens attend to each other instead of being encoded independently
3. **Bilinear Fusion** — replaces the soft-gated fusion with a low-rank bilinear product that explicitly models drug-protein feature interactions

---

## What's New Compared to KANPM-DTA

| Component | KANPM-DTA (Original) | AF2-CrossKAN-DTA (Ours) |
|-----------|---------------------|--------------------------|
| Protein contact map | ESM-2 predicted (2D, sequence-based) | AlphaFold2 real 3D Cα distance matrix |
| Sequence encoding | Linear Attention (independent) | Cross-Attention (drug ↔ protein) |
| Graph fusion | Gated Fusion (soft-gate, additive) | Bilinear Fusion (multiplicative, low-rank) |
| Predictor | KAN [512, 1024, 512, 1] | KAN [512, 1024, 512, 1] (same) |

---

## Model Architecture

```
Drug SMILES  ──► ChemBERTa ──► Drug Tokens [B, 220, 128]
Drug SMILES  ──► Graph Builder ──► Drug Graph (GNN) ──► [B, 128]
                                                              │
Protein Seq  ──► ESM-C ──► Protein Tokens [B, 1200, 128]    │
Protein Seq  ──► AlphaFold2 3D ──► Cα Contact Map            │
                        ──► Protein Graph (GNN) ──► [B, 128] │
                                                              │
          Cross-Attention (Drug queries Protein)              │
          Cross-Attention (Protein queries Drug)              │
                    │                                         │
          xd_attn [B, 128]              xp_attn [B, 128]     │
                    │                                         │
          drug_side = cat(xd_attn, drug_graph)  [B, 256]     │
          prot_side = cat(xp_attn, prot_graph)  [B, 256]     │
                    │                                         │
                    └─────────► BilinearFusion ◄──────────────┘
                                      │
                            bilinear_out [B, 256]
                                      │
                   cat_attn [B, 256] ─┘  (interaction attention)
                                      │
                         final = cat(bilinear_out, cat_attn)
                                   [B, 512]
                                      │
                            KAN [512 → 1024 → 512 → 1]
                                      │
                           Predicted Affinity Score
```

---

## The Three Changes Explained

### 1. AlphaFold2 3D Contact Maps

**Old approach (ESM-2):** Used ESM-2's contact prediction head to estimate which residues are spatially close. This is a learned prediction from sequence alone — it can be wrong.

**New approach (AlphaFold2):** For each protein in the dataset, we:
1. Query UniProt REST API to get the official accession ID from the gene name
2. Query AlphaFold2 EBI API to get the PDB file URL
3. Download the PDB file with full 3D atomic coordinates
4. Extract Cα (alpha-carbon) atom positions — one per residue
5. Compute all pairwise Euclidean distances in Angstroms
6. Threshold at 8Å → binary contact map (1 = in contact, 0 = not)

This gives a real, physics-grounded picture of which parts of the protein are physically close — exactly what the GNN needs to learn structural patterns.

**DAVIS dataset results:**
- 364 proteins → real AlphaFold2 3D structures
- 72 proteins → wildtype AlphaFold2 for mutant variants (e.g. EGFR(L858R) uses EGFR structure — valid because single point mutations don't change the overall fold)
- 3 proteins → removed (non-human organisms: Plasmodium falciparum, M. tuberculosis — no AF2 entry exists)

---

### 2. Cross-Attention (Drug ↔ Protein)

**Old approach (Linear Attention):** Drug and protein sequence tokens were each pooled into a single vector independently. No interaction between them at the sequence level.

**New approach (Cross-Attention):** Two `nn.MultiheadAttention` layers:
- Drug tokens **attend to** protein tokens → the drug learns which parts of the protein are relevant to it
- Protein tokens **attend to** drug tokens → the protein learns which residues are relevant to the drug

```python
xd_cross, _ = self.cross_attn_drug(query=xd, key=xp, value=xp, key_padding_mask=prot_pad_mask)
xp_cross, _ = self.cross_attn_prot(query=xp, key=xd, value=xd, key_padding_mask=drug_pad_mask)
xd_attn = xd_cross.mean(dim=1)   # [B, 128]
xp_attn = xp_cross.mean(dim=1)   # [B, 128]
```

This creates drug-aware protein representations and protein-aware drug representations before the graph fusion step.

---

### 3. Bilinear Fusion (Drug × Protein)

**Old approach (Gated Fusion):** A sigmoid gate controlled a weighted sum of drug graph and protein graph features. Additive — doesn't explicitly model how drug and protein features interact.

**New approach (Bilinear Fusion):** Low-rank bilinear product between drug-side and protein-side features:

```python
drug_side = cat(xd_attn, drug_graph)   # [B, 256]
prot_side = cat(xp_attn, prot_graph)   # [B, 256]

d = tanh(drug_proj(drug_side))         # [B, 64]
p = tanh(prot_proj(prot_side))         # [B, 64]
interaction = d * p                    # element-wise [B, 64]
out = output_proj(interaction)         # [B, 256]
```

The element-wise product in the shared rank-64 space forces the model to represent drug-protein feature interactions multiplicatively — not just add them together.

---

## Performance Results

### Original KANPM-DTA Results (Baseline)

| Dataset   | MSE Reduction | CI Increase | r2m Gain |
|-----------|--------------|-------------|----------|
| Davis     | 6.42%        | 0.45%       | 1.85%    |
| KIBA      | 4.86%        | 0.34%       | 0.90%    |
| Metz      | 4.44%        | 0.48%       | 0.84%    |
| BindingDB | 5.46%        | 0.80%       | 1.05%    |

#### BindingDB Full Comparison (Original KANPM-DTA)

| Model                        | MSE       | CI        | r2m       |
|------------------------------|-----------|-----------|-----------|
| DeepDTA (2018)               | 0.633     | 0.844     | 0.633     |
| AttentionDTA (2022)          | 0.745     | 0.542     | -         |
| GraphDTA-GIN (2021)          | 0.535     | 0.858     | -         |
| CoVAE (2021)                 | 0.512     | 0.847     | 0.412     |
| DeepCDA (2020)               | 0.848     | 0.722     | 0.531     |
| ELECTRA-DTA (2022)           | 0.650     | 0.837     | 0.670     |
| DoubleSG-DTA (2023)          | 0.533     | 0.862     | 0.726     |
| GDilatedDTA (2024)           | 0.483     | 0.868     | 0.730     |
| MF-DTA (2025)                | 0.569     | 0.865     | 0.737     |
| DeepDTAGen (2025)            | 0.458     | 0.876     | 0.760     |
| **KANPM-DTA Original**       | **0.433** | **0.883** | **0.768** |

---

### AF2-CrossKAN-DTA Results (To Be Filled After Training)

#### Davis (Warm Setting)

| Model                         | MSE | CI | r2m |
|-------------------------------|-----|----|-----|
| KANPM-DTA (Original)          | -   | -  | -   |
| **AF2-CrossKAN-DTA (Ours)**   | -   | -  | -   |

#### KIBA (Warm Setting)

| Model                         | MSE | CI | r2m |
|-------------------------------|-----|----|-----|
| KANPM-DTA (Original)          | -   | -  | -   |
| **AF2-CrossKAN-DTA (Ours)**   | -   | -  | -   |

---

## Ablation Study (To Be Filled After Training)

| Configuration | MSE | CI | r2m |
|---------------|-----|----|-----|
| Full AF2-CrossKAN-DTA | - | - | - |
| W/O Cross-Attention (use Linear Attention) | - | - | - |
| W/O Bilinear Fusion (use Gated Fusion) | - | - | - |
| W/O AlphaFold2 (use ESM-2 contact map) | - | - | - |
| W/O KAN (use MLP) | - | - | - |

---

## KAN Predictor

The final prediction head is a Kolmogorov-Arnold Network — each connection learns a B-spline function instead of a fixed weight.

```
Input:    [Batch, 512]
Layer 1:  512  → 1024
Layer 2:  1024 → 512
Layer 3:  512  → 1
Output:   [Batch, 1]
```

Each connection computes:
```
f(x) = w_base * SiLU(x)  +  Σ c_k * B_k(x)
```
where `B_k(x)` are B-spline basis functions and `c_k` are learned coefficients. This gives the model interpretable, learnable activation functions per connection rather than a fixed nonlinearity.

| Hyperparameter         | Value               |
|------------------------|---------------------|
| Layer Dimensions       | [512, 1024, 512, 1] |
| Spline Order           | 3 (cubic)           |
| Grid Size              | 5                   |
| Base Activation        | SiLU                |
| L1 Reg Weight          | 1.0                 |
| Entropy Reg Weight     | 1.0                 |

---

## Drug Graph Node Features

Each atom becomes a graph node with 88 features:

| Feature                              | Dimension |
|--------------------------------------|-----------|
| Atom Symbol (one-hot)                | 44        |
| Formal Charge                        | 1         |
| Explicit Valence                     | 1         |
| Number of Atomic Rings               | 1         |
| Hybridization Type                   | 3         |
| Donor / Acceptor                     | 2         |
| Degree (one-hot)                     | 11        |
| Radical Electrons                    | 1         |
| Aromatic                             | 1         |
| Explicit / Implicit Hydrogens        | 12        |
| Total H atoms on heavy atom          | 11        |
| **Total**                            | **88**    |

---

## Pretrained Models Used

| Model | Purpose | Output Dim |
|-------|---------|-----------|
| ChemBERTa (`DeepChem/ChemBERTa-77M-MTR`) | Drug SMILES → token embeddings | 384 |
| ESM-C (`esmc_600m`) | Protein sequence → residue embeddings | 1152 |
| AlphaFold2 (EBI database) | Protein → real 3D Cα contact map | L × L |

---

## Dataset Splits

Three split strategies to test generalization:

| Split | Training | Validation | Test |
|-------|----------|------------|------|
| Warm (drug+protein seen) | 23,882 | 2,985 | 2,985 |
| Unseen Protein | 23,868 | 2,992 | 2,992 |
| Unseen Drug | 23,706 | 3,073 | 3,073 |
| Unseen Pair | 18,954 | 308 | 308 |

3 non-human proteins (Plasmodium falciparum × 2, M. tuberculosis × 1) removed from all splits — no AlphaFold2 structure exists for these organisms.

---

## Setup

### Install Dependencies

```bash
pip install torch transformers rdkit biopython requests pandas numpy
pip install esm  # ESM-C
```

### Generate Embeddings

```bash
python pretrained/chemberta_pretraiend.py
python pretrained/esmC_pretraiend.py
```

### Generate AlphaFold2 Contact Maps

```bash
python pretrained/alphafold2_preprocess.py --dataset davis
python pretrained/alphafold2_preprocess.py --dataset kiba
python pretrained/alphafold2_preprocess.py --dataset metz
```

### Generate Dataset Splits

```bash
python code/cold_split.py --dataset davis
```

### Train

```bash
python code/train.py
```

---

## References

1. Liu et al. (2024) KAN: Kolmogorov-Arnold Networks — arXiv:2404.19756
2. Jumper et al. (2021) AlphaFold2 — Nature 596, 583–589
3. Ahmad et al. (2022) ChemBERTa-2 — arXiv:2209.01712
4. ESM Team (2024) ESM-C — Evolutionary Scale
5. Lin et al. (2023) ESM-2 — Science 379(6637)
6. Original KANPM-DTA — MD Youshuf Khan Rakib et al., Central South University
