# AF2-CrossKAN-DTA: Drug-Target Affinity Prediction with AlphaFold2 Structures, Cross-Attention, and Kolmogorov-Arnold Networks

---

## Overview

AF2-CrossKAN-DTA is a drug-target affinity (DTA) prediction model that predicts how strongly a drug molecule binds to a protein target. Strong binding means the drug is more likely to work — making DTA prediction a critical step in computational drug discovery.

**The three core contributions of this work:**

1. **AlphaFold2 3D Contact Maps** — uses real 3D Cα distance matrices from the AlphaFold2 public database instead of sequence-predicted contact maps, giving the protein graph branch genuine structural information
2. **Cross-Attention** — drug and protein sequence tokens attend to each other, creating drug-aware protein representations and protein-aware drug representations before fusion
3. **Bilinear Fusion + KAN** — a low-rank bilinear product explicitly models multiplicative drug-protein feature interactions, followed by a Kolmogorov-Arnold Network as the final predictor

---

## Model Architecture

```
Drug SMILES  ──► ChemBERTa ──► Drug Tokens [B, 220, 128]
Drug SMILES  ──► Graph Builder ──► Drug Graph (GNN) ──► [B, 128]

Protein Seq  ──► ESM-C ──► Protein Tokens [B, 1200, 128]
Protein Seq  ──► AlphaFold2 3D ──► Cα Contact Map
                       ──► Protein Graph (GNN) ──► [B, 128]

          Cross-Attention: Drug queries Protein → xd_attn [B, 128]
          Cross-Attention: Protein queries Drug  → xp_attn [B, 128]

          drug_side = cat(xd_attn, drug_graph)  [B, 256]
          prot_side = cat(xp_attn, prot_graph)  [B, 256]

                    BilinearFusion(drug_side, prot_side)
                              [B, 256]
                                 │
               cat_attn [B, 256] ┘  (interaction attention)
                                 │
                    final = cat(bilinear_out, cat_attn)
                                [B, 512]
                                 │
                       KAN [512 → 1024 → 512 → 1]
                                 │
                        Predicted Affinity Score
```

---

## Component Details

### AlphaFold2 3D Contact Maps

For each protein in the dataset we:
1. Query UniProt REST API to resolve the gene name to an accession ID
2. Query AlphaFold2 EBI API to get the PDB file URL
3. Download the PDB and extract Cα (alpha-carbon) atom 3D coordinates — one per residue
4. Compute all pairwise Euclidean distances in Angstroms
5. Threshold at 8Å → binary contact map (1 = in contact, 0 = not)

This gives the protein graph real 3D structural connectivity instead of predicted connectivity.

**DAVIS coverage:**
- 364 proteins → direct AlphaFold2 3D structure
- 72 proteins → wildtype AlphaFold2 for mutant variants (e.g. EGFR(L858R) uses EGFR — single point mutations do not change the overall fold)
- 3 proteins removed (non-human organisms with no AlphaFold2 entry)

---

### Cross-Attention

Two `nn.MultiheadAttention` layers replace independent sequence pooling:

```python
# Drug tokens attend to protein tokens
xd_cross, _ = self.cross_attn_drug(query=xd, key=xp, value=xp, key_padding_mask=prot_pad_mask)
# Protein tokens attend to drug tokens
xp_cross, _ = self.cross_attn_prot(query=xp, key=xd, value=xd, key_padding_mask=drug_pad_mask)

xd_attn = xd_cross.mean(dim=1)   # [B, 128]
xp_attn = xp_cross.mean(dim=1)   # [B, 128]
```

This ensures each drug representation is conditioned on the protein it is paired with, and vice versa.

---

### Bilinear Fusion

Low-rank bilinear product between drug-side and protein-side features:

```python
drug_side = cat(xd_attn, drug_graph)    # [B, 256]
prot_side = cat(xp_attn, prot_graph)    # [B, 256]

d = tanh(drug_proj(drug_side))          # [B, 64]  — rank-64 projection
p = tanh(prot_proj(prot_side))          # [B, 64]
interaction = d * p                     # element-wise product [B, 64]
out = output_proj(interaction)          # [B, 256]
```

The element-wise product in the shared low-rank space models multiplicative drug-protein feature interactions.

---

### KAN Predictor

The final predictor is a Kolmogorov-Arnold Network. Each connection learns a B-spline function instead of a fixed scalar weight:

```
f(x) = w_base * SiLU(x)  +  Σ c_k * B_k(x)
```

- `B_k(x)` are B-spline basis functions (smooth piecewise curves)
- `c_k` are learned coefficients — each connection has its own function

```
[B, 512] → [B, 1024] → [B, 512] → [B, 1]
```

| Hyperparameter     | Value               |
|--------------------|---------------------|
| Layer Dimensions   | [512, 1024, 512, 1] |
| Spline Order       | 3 (cubic)           |
| Grid Size          | 5                   |
| Base Activation    | SiLU                |
| L1 Reg Weight      | 1.0                 |
| Entropy Reg Weight | 1.0                 |

---

## Performance Results

### Prior Work (Comparison Baseline)

| Model                  | MSE       | CI        | r2m       |
|------------------------|-----------|-----------|-----------|
| DeepDTA (2018)         | 0.633     | 0.844     | 0.633     |
| GraphDTA-GIN (2021)    | 0.535     | 0.858     | -         |
| CoVAE (2021)           | 0.512     | 0.847     | 0.412     |
| ELECTRA-DTA (2022)     | 0.650     | 0.837     | 0.670     |
| DoubleSG-DTA (2023)    | 0.533     | 0.862     | 0.726     |
| GDilatedDTA (2024)     | 0.483     | 0.868     | 0.730     |
| MF-DTA (2025)          | 0.569     | 0.865     | 0.737     |
| DeepDTAGen (2025)      | 0.458     | 0.876     | 0.760     |

### AF2-CrossKAN-DTA Results (To Be Updated After Training)

#### Davis (Warm Setting)

| Model                       | MSE | CI | r2m |
|-----------------------------|-----|----|-----|
| **AF2-CrossKAN-DTA (Ours)** | -   | -  | -   |

#### KIBA (Warm Setting)

| Model                       | MSE | CI | r2m |
|-----------------------------|-----|----|-----|
| **AF2-CrossKAN-DTA (Ours)** | -   | -  | -   |

---

## Training Progress Log — DAVIS Warm

Training on RTX 5060 8GB. Best validation MSE updated each time model improves.
Metrics below are **training set** values. Best **validation MSE: 0.3969** (epoch 10).

| Epoch | Train MSE | CI | r2m |
|-------|-----------|----|-----|
| 1 | 0.8823 | 0.6701 | 0.0480 |
| 2 | 0.6365 | 0.7150 | 0.2131 |
| 3 | 0.6071 | 0.7266 | 0.2491 |
| 4 | 0.5855 | 0.7455 | 0.2764 |
| 5 | 0.5608 | 0.7544 | 0.3072 |
| 6 | 0.5462 | 0.7636 | 0.3257 |
| 7 | 0.5210 | 0.7639 | 0.3564 |
| 8 | 0.5034 | 0.7655 | 0.3783 |
| 9 | 0.4822 | 0.7777 | 0.4044 |
| 10 | 0.4740 | 0.7892 | 0.4142 |
| 20 | - | - | - |
| 30 | - | - | - |
| 40 | - | - | - |
| 50 | - | - | - |

---

## Ablation Study (To Be Updated After Training)

| Configuration | MSE | CI | r2m |
|---------------|-----|----|-----|
| Full AF2-CrossKAN-DTA | - | - | - |
| W/O Cross-Attention | - | - | - |
| W/O Bilinear Fusion | - | - | - |
| W/O AlphaFold2 (use predicted map) | - | - | - |
| W/O KAN (use MLP) | - | - | - |

---

## Drug Graph Node Features

Each atom becomes a node with 88 features:

| Feature                       | Dimension |
|-------------------------------|-----------|
| Atom Symbol (one-hot)         | 44        |
| Formal Charge                 | 1         |
| Explicit Valence              | 1         |
| Number of Atomic Rings        | 1         |
| Hybridization Type            | 3         |
| Donor / Acceptor              | 2         |
| Degree (one-hot)              | 11        |
| Radical Electrons             | 1         |
| Aromatic                      | 1         |
| Explicit / Implicit Hydrogens | 12        |
| Total H on heavy atom         | 11        |
| **Total**                     | **88**    |

---

## Pretrained Models

| Model | Purpose | Output |
|-------|---------|--------|
| ChemBERTa (`DeepChem/ChemBERTa-77M-MTR`) | Drug SMILES → token embeddings | 384-dim |
| ESM-C (`esmc_600m`) | Protein sequence → residue embeddings | 1152-dim |
| AlphaFold2 (EBI) | Protein → real 3D Cα contact map | L × L |

---

## Dataset Splits (DAVIS)

| Split | Train | Valid | Test |
|-------|-------|-------|------|
| Warm | 23,882 | 2,985 | 2,985 |
| Unseen Protein | 23,868 | 2,992 | 2,992 |
| Unseen Drug | 23,706 | 3,073 | 3,073 |
| Unseen Pair | 18,954 | 308 | 308 |

---

## Setup

### Install Dependencies

```bash
pip install torch transformers rdkit biopython requests pandas numpy
pip install esm
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
6. Shah et al. (2025) DeepDTAGen — Nature Communications 16(1):5021
7. Kang et al. (2025) MF-DTA — J. Biomedical Informatics
