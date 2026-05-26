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

### AF2-CrossKAN-DTA Results

To get stable and reproducible results we run 3 independent trainings on the DAVIS warm setting, each with a different data split seed (41, 42, 43 — the same protocol used in KANPM-DTA). The final reported MSE/CI/r2m are the mean and standard deviation across the 3 runs.

#### Davis (Warm Setting)

| Model | Split Seed | Test MSE | Test CI | Test r2m |
|-------|------------|----------|---------|----------|
| AF2-CrossKAN-DTA — Run 1 | 41 | 0.1954 | 0.8764 | 0.7190 |
| AF2-CrossKAN-DTA — Run 2 | 42 | 0.2223 | 0.8963 | 0.6653 |
| AF2-CrossKAN-DTA — Run 3 | 43 | - | - | - |
| **AF2-CrossKAN-DTA (Mean ± Std)** | — | - | - | - |

#### KIBA (Warm Setting)

| Model                       | MSE | CI | r2m |
|-----------------------------|-----|----|-----|
| **AF2-CrossKAN-DTA (Ours)** | -   | -  | -   |

---

## Training Progress Log — DAVIS Warm

Training on RTX 5060 8GB. Best validation MSE updated each time model improves.
Train MSE is the training set value. Valid MSE/CI/r2m are validation set values.

### Run 1 — Split Seed 41

> **Best Checkpoint — Epoch 100**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0795 |
> | Valid MSE | **0.1934** |
> | Valid CI | **0.8724** |
> | Valid r2m | **0.6914** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.4257 | 0.3904 | 0.8286 | 0.4375 |
| 20 | 0.3471 | 0.3197 | 0.8330 | 0.5349 |
| 30 | 0.2966 | 0.3084 | 0.8544 | 0.5465 |
| 40 | 0.2440 | 0.3275 | 0.8602 | 0.5156 |
| 50 | 0.2041 | 0.2522 | 0.8658 | 0.6273 |
| 60 | 0.1609 | 0.2300 | 0.8823 | 0.6301 |
| 70 | 0.1291 | 0.2299 | 0.8793 | 0.6186 |
| 80 | 0.1096 | 0.2295 | 0.8630 | 0.6339 |
| 90 | 0.0907 | 0.2177 | 0.8771 | 0.6093 |
| 100 | **0.0795** | **0.1934** | **0.8724** | **0.6914** |
| 110 | 0.0700 | 0.1992 | 0.8753 | 0.6587 |
| 120 | 0.0636 | 0.1997 | 0.8846 | 0.6626 |

### Run 2 — Split Seed 42

> **Best Checkpoint — Epoch 161**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0556 |
> | Valid MSE | **0.1869** |
> | Valid CI | **0.8970** |
> | Valid r2m | **0.7205** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.4251 | 0.4338 | 0.8176 | 0.4515 |
| 20 | 0.3450 | 0.3467 | 0.8317 | 0.5823 |
| 30 | 0.3014 | 0.2957 | 0.8541 | 0.6171 |
| 40 | 0.2463 | 0.2727 | 0.8612 | 0.6120 |
| 50 | 0.1993 | 0.2437 | 0.8691 | 0.6973 |
| 60 | 0.1613 | 0.2340 | 0.8817 | 0.6706 |
| 70 | 0.1350 | 0.2198 | 0.8893 | 0.6902 |
| 80 | 0.1132 | 0.2140 | 0.8849 | 0.7137 |
| 90 | 0.0924 | 0.2194 | 0.8868 | 0.6769 |
| 100 | 0.0794 | 0.2136 | 0.8849 | 0.7149 |
| 110 | 0.0687 | 0.2080 | 0.8847 | 0.6945 |
| 120 | 0.0639 | 0.2027 | 0.8973 | 0.6992 |
| 130 | 0.0672 | 0.1879 | 0.8935 | 0.7455 |
| 140 | 0.0633 | 0.1895 | 0.8918 | 0.7508 |
| 150 | 0.0552 | 0.1925 | 0.8941 | 0.7166 |
| 160 | 0.0476 | 0.1888 | 0.8920 | 0.7279 |
| **161** | **0.0556** | **0.1869** | **0.8970** | **0.7205** |

### Run 3 — Split Seed 43

> **Best Checkpoint — Epoch 83**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.1067 |
> | Valid MSE | **0.1921** |
> | Valid CI | **0.8871** |
> | Valid r2m | **0.7403** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.4253 | 0.3949 | 0.8272 | 0.5168 |
| 20 | 0.3329 | 0.3341 | 0.8451 | 0.5844 |
| 30 | 0.2936 | 0.3024 | 0.8532 | 0.6217 |
| 40 | 0.2455 | 0.2591 | 0.8737 | 0.6853 |
| 50 | 0.2070 | 0.2416 | 0.8798 | 0.6715 |
| 60 | 0.1653 | 0.2283 | 0.8820 | 0.6985 |
| 67 | 0.1481 | 0.2007 | 0.8786 | 0.7309 |
| 70 | 0.1407 | 0.2114 | 0.8893 | 0.7081 |
| 80 | 0.1117 | 0.2128 | 0.8791 | 0.7350 |
| **83** | **0.1067** | **0.1921** | **0.8871** | **0.7403** |

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
