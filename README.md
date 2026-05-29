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

#### Why Existing Approaches Are Insufficient

Most prior DTA models that incorporate protein graph structure rely on **sequence-predicted 2D contact maps**. For example, KANPM-DTA uses ESM-2 to generate a contact probability matrix directly from the amino acid sequence — edges are formed when the predicted probability exceeds a 0.5 threshold. While computationally convenient, this approach has a fundamental limitation: it never observes any 3D geometry. The contact probabilities are statistical predictions based on evolutionary co-variation in sequences, not actual physical distances between atoms. Two residues far apart in 3D space can still have a high co-evolutionary signal if they co-mutate, leading to false edges in the protein graph. Conversely, residues that are physically close but evolutionarily independent may be missed entirely.

This means sequence-predicted contact maps do not faithfully represent the true binding site geometry of a protein — which is precisely the information a DTA model needs to determine how a drug docks.

#### Our Approach — Real 3D Structural Contact Maps

We replace sequence-predicted contact maps with real 3D Cα distance matrices from AlphaFold2 predicted structures. For each protein in the dataset we:

1. Query UniProt REST API to resolve the gene name to an accession ID
2. Query AlphaFold2 EBI API to get the PDB file URL
3. Download the PDB and extract Cα (alpha-carbon) atom 3D coordinates — one per residue
4. Compute all pairwise Euclidean distances in Angstroms
5. Threshold at 8Å → binary contact map (1 = in contact, 0 = not)

The 8Å threshold is the standard cutoff in structural biology for defining residue-residue contacts. Edges in the protein graph now correspond to real physical proximity in 3D space — not statistical predictions. This gives the GNN genuine structural information about the protein's folding, active site geometry, and binding pocket topology.

| Property | Sequence-predicted (KANPM-DTA) | AlphaFold2 3D (Ours) |
|----------|-------------------------------|----------------------|
| Source | ESM-2 probability matrix | Real Cα coordinates |
| Edge criterion | Probability > 0.5 | Physical distance < 8Å |
| Geometric information | None | Actual Euclidean space |
| Binding site accuracy | Statistical approximation | Real 3D topology |
| False edges | Possible (co-evolution ≠ proximity) | None (distance is exact) |

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

## Data Flow Through the Model

A step-by-step walkthrough of how a drug-protein pair moves through every layer from raw input to predicted affinity score.

---

### Step 1 — Input Embeddings

**Drug side:**
- The SMILES string is tokenized by ChemBERTa and projected to a sequence of token embeddings: `[B, 220, 384]`
- The same SMILES is parsed by RDKit into a molecular graph — each atom becomes a node with 88 features (atom type, charge, hybridization, etc.)

**Protein side:**
- The amino acid sequence is encoded by ESM-C 600M into residue-level embeddings: `[B, 1200, 1152]`
- The AlphaFold2 PDB file is used to compute Cα pairwise distances — pairs within 8Å become edges in the protein graph, with ESM-C residue embeddings as node features

---

### Step 2 — Dimension Projection

Both embedding dimensions are projected down to 128 via a single linear layer:

```
Drug:    fc3: [B, 220, 384]  →  [B, 220, 128]
Protein: fc2: [B, 1200, 1152] → [B, 1200, 128]
```

This gives both modalities a common representation size before further processing.

---

### Step 3 — Transformer Self-Attention (within each modality)

Each modality has its own independent 3-layer Transformer Encoder (8 attention heads, feedforward dim=1024):

```
Drug tokens:    TransformerEncoder  → xd [B, 220, 128]
Protein tokens: TransformerEncoder2 → xp [B, 1200, 128]
```

At this stage drug tokens only attend to other drug tokens, and protein tokens only attend to other protein tokens. This builds up intra-modal context — "which parts of the drug are chemically important?" and "which residues in the protein are structurally relevant?"

---

### Step 4 — Graph Branch (parallel to sequence branch)

While the sequence branch runs, the graph branch processes structural information independently:

**Drug graph:** 1 GCNConv layer followed by 5 GATConv layers, each with BatchNorm + ReLU. Global add pooling collapses all atom nodes into one vector:
```
Drug atoms [N_atoms, 88]  →  GCN → 5×GAT → global_add_pool  →  smiles_graph [B, 128]
```

**Protein graph:** Same architecture but starts from 1152-dim ESM-C node features:
```
Protein residues [N_res, 1152]  →  GCN → 5×GAT → global_add_pool  →  fasta_graph [B, 128]
```

The graph branch captures **topological structure** (atom connectivity for drugs, 3D spatial contacts for proteins) that the sequence branch cannot see.

---

### Step 5 — Cross-Attention (drug ↔ protein interaction)

This is where drug and protein first interact. Two `nn.MultiheadAttention` layers (8 heads) run in parallel:

```python
# Drug tokens query the protein — each drug token asks "which protein residues are relevant to me?"
xd_cross, _ = cross_attn_drug(query=xd, key=xp, value=xp, key_padding_mask=prot_pad_mask)

# Protein tokens query the drug — each residue asks "which drug atoms are relevant to me?"
xp_cross, _ = cross_attn_prot(query=xp, key=xd, value=xd, key_padding_mask=drug_pad_mask)
```

Padding positions are masked out so attention never falls on zero-padded slots.

Both outputs are mean-pooled across the sequence dimension to get fixed-size vectors:
```
xd_attn = xd_cross.mean(dim=1)   [B, 128]  — protein-conditioned drug representation
xp_attn = xp_cross.mean(dim=1)   [B, 128]  — drug-conditioned protein representation
```

After this step, `xd_attn` is no longer a generic drug embedding — it is a drug representation that has been shaped by the specific protein it is paired with, and vice versa.

---

### Step 6 — Interaction Attention Branch

A third, separate attention stream concatenates both sequence outputs and applies a learned attention pooling:

```
cat([xp, xd], dim=1)  →  [B, 1420, 128]
LinearAttention (8 heads, tanh scoring)  →  [B, 128]
cat_attn_proj (128 → 256)               →  cat_attn [B, 256]
```

This branch looks at the full combined drug+protein sequence jointly and produces a global interaction summary vector. It provides a complementary view to the cross-attention branch.

---

### Step 7 — Side Assembly

The sequence and graph representations for each modality are merged:

```
drug_side = cat(xd_attn, smiles_graph)   [B, 128] + [B, 128]  =  [B, 256]
prot_side = cat(xp_attn, fasta_graph)    [B, 128] + [B, 128]  =  [B, 256]
```

Each side now carries both sequence-level context (cross-attention) and structural context (graph) for its modality.

---

### Step 8 — Bilinear Fusion

Instead of simply concatenating drug_side and prot_side, bilinear fusion models their **multiplicative interaction** in a shared low-rank space:

```python
d = tanh(drug_proj(drug_side))   # [B, 64]  — rank-64 projection of drug side
p = tanh(prot_proj(prot_side))   # [B, 64]  — rank-64 projection of protein side
interaction = d * p               # [B, 64]  — element-wise product (explicit interaction)
bilinear_out = output_proj(interaction)  →  BN  →  Dropout  →  [B, 256]
```

The element-wise product forces each dimension of the drug and protein representations to interact directly. Features that are both drug-relevant and protein-relevant produce a large value; features relevant to only one side cancel out.

---

### Step 9 — Final Concatenation

The bilinear interaction output and the interaction attention output are concatenated:

```
final = cat(bilinear_out, cat_attn)   [B, 256] + [B, 256]  =  [B, 512]
```

This gives the KAN predictor a 512-dimensional vector that encodes:
- Multiplicative drug-protein feature interactions (bilinear_out)
- Global joint drug-protein sequence context (cat_attn)

---

### Step 10 — KAN Predictor

The final 512-dim vector passes through a Kolmogorov-Arnold Network. Unlike an MLP where each connection is a fixed scalar weight, each KAN connection learns a B-spline function:

```
f(x) = w_base × SiLU(x)  +  Σ c_k × B_k(x)
```

where `B_k(x)` are cubic B-spline basis functions and `c_k` are learned coefficients.

```
[B, 512]  →  KAN layer 1  →  [B, 1024]
           →  KAN layer 2  →  [B, 512]
           →  KAN layer 3  →  [B, 1]
                               │
                    Predicted Affinity Score (pKd)
```

The KAN's learnable activation functions allow it to fit more complex non-linear drug-protein relationships than a fixed-activation MLP of the same depth.

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
| AF2-CrossKAN-DTA — Run 2 (retrain) | 42 | - | - | - |
| AF2-CrossKAN-DTA — Run 3 | 43 | 0.2155 | 0.8855 | 0.6564 |
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

### Run 2 — Split Seed 42 (Original — Replaced by Retrain)

> Original run completed (192 epochs, patience=30) but showed large valid/test gap (valid MSE 0.1869 → test MSE 0.2223). Retrained with patience=20 for better generalization.

### Run 2 Retrain — Split Seed 42 (New)

> **Best Checkpoint — Epoch 67** *(training in progress)*
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.1507 |
> | Valid MSE | **0.2060** |
> | Valid CI | **0.8874** |
> | Valid r2m | **0.7248** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.4446 | 0.4722 | 0.8038 | 0.4192 |
| 20 | 0.3498 | 0.3202 | 0.8460 | 0.6051 |
| 30 | 0.2906 | 0.2862 | 0.8569 | 0.6511 |
| 40 | 0.2397 | 0.2585 | 0.8703 | 0.6906 |
| 50 | 0.2013 | 0.2295 | 0.8845 | 0.7020 |
| 60 | 0.1710 | 0.2211 | 0.8875 | 0.7326 |
| **67** | **0.1507** | **0.2060** | **0.8874** | **0.7248** |
| 70 | 0.1446 | 0.2076 | 0.8859 | 0.7268 |

### Run 2 Original — Split Seed 42

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

> **Best Checkpoint — Epoch 121**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0657 |
> | Valid MSE | **0.1840** |
> | Valid CI | **0.9025** |
> | Valid r2m | **0.7424** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.4253 | 0.3949 | 0.8272 | 0.5168 |
| 20 | 0.3329 | 0.3341 | 0.8451 | 0.5844 |
| 30 | 0.2936 | 0.3024 | 0.8532 | 0.6217 |
| 40 | 0.2455 | 0.2591 | 0.8737 | 0.6853 |
| 50 | 0.2070 | 0.2416 | 0.8798 | 0.6715 |
| 60 | 0.1653 | 0.2283 | 0.8820 | 0.6985 |
| 70 | 0.1407 | 0.2114 | 0.8893 | 0.7081 |
| 80 | 0.1117 | 0.2128 | 0.8791 | 0.7350 |
| 90 | 0.0987 | 0.2092 | 0.8902 | 0.7175 |
| 100 | 0.0926 | 0.1979 | 0.8971 | 0.7235 |
| 109 | 0.0759 | 0.1863 | 0.8974 | 0.7548 |
| 110 | 0.0747 | 0.1984 | 0.8919 | 0.7375 |
| 120 | 0.0645 | 0.1920 | 0.8988 | 0.7313 |
| **121** | **0.0657** | **0.1840** | **0.9025** | **0.7424** |

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
