# Experimental History — AF2-AtomRes-DTA

> **This is the full historical record.** For the current model and results, see [README.md](README.md).
>
> Kept because the negative results are as informative as the positive ones — every failed
> variant here was a controlled, single-variable test with a natural early-stop and a real
> test number. Together they define what does and does not work for cold-start DTA on this
> dataset, and they exist so the same dead ends are not re-explored.
>
> **Note on code availability.** The working tree now carries only the champion
> (`code/model_af2_atomres.py` + `code/train_af2_atomres.py`) and its
> analysis tooling. The superseded implementations cited below — `model.py`/`train.py` (v4
> baseline) and `model_pocketcross.py`/`train_pocketcross.py` (hand-counted pocket prior,
> weighted loss, rich distance features) — were removed once their results were recorded here,
> the same way the attention-pooling variant was. **All are recoverable from git commit
> `a0937c8`:** `git checkout a0937c8 -- code/model.py code/train.py code/model_pocketcross.py code/train_pocketcross.py`

## Summary of every variant tried (DAVIS cold-protein, `unseen_prot`, seed 42)

| Variant | Test MSE | Test CI | Test r2m | Verdict |
|---------|----------|---------|----------|---------|
| Baseline (v4: cross-attention + bilinear fusion) | 0.4071 | 0.8382 | 0.4112 | superseded |
| PocketCross + weighted loss (a=0.5) | 0.3781 | 0.8450 | 0.4542 | ❌ worse than plain MSE |
| PocketCross, plain MSE, hand-counted pocket features | 0.3715 | 0.8453 | 0.4628 | superseded |
| PocketCross + rich distance pocket features (8-dim) | 0.4287 | 0.8180 | 0.4206 | ❌ worse than baseline |
| PocketCross + attention pooling (KANPM LinearAttention) | 0.3759 | 0.8420 | 0.4699 | ❌ worse |
| **PocketCross + GNN-derived residue prior** | **0.3519** | **0.8547** | **0.4928** | ✅ **adopted — current model** |

**Champion across 3 seeds (41/42/43): MSE 0.3618 ± 0.0109 · CI 0.8541 ± 0.0053 · r²ₘ 0.5360 ± 0.0392**

Run on seed 41 (compare against the seed-41 champion, **0.3601 / 0.8590 / 0.5460**, not the seed-42 row above):

| Variant | Test MSE | Test CI | Test r2m | Verdict |
|---------|----------|---------|----------|---------|
| PocketCross + AF2 pLDDT confidence as protein node feature | 0.3868 | 0.8299 | 0.5214 | ❌ worse on all three |

### Cross-cutting lessons

1. **Capacity changes fail in both directions.** Adding parameters (nonlinear ESM-C projection,
   attention pooling, richer hand-built features) and removing them (KAN shrink to [512,512,1]
   and [512,256,1], the "lean" rebuild) both underperformed. The model sits at a capacity
   sweet spot for ~442 proteins.
2. **The only changes that ever worked reused computation the model already performed.**
   The structure-guided interaction and the GNN-derived residue prior both re-routed existing
   internal representations rather than adding new inputs or capacity.
3. **Added input features consistently failed — now 0 for 3.** RBF edge features, the 8-dim
   distance-shell pocket features, and AF2 pLDDT confidence all hurt, despite each being
   better-motivated on paper than the thing it replaced. The pLDDT case is the strongest
   version of the test: it is genuinely *orthogonal* information (nothing else in the inputs
   can express "this contact is unreliable"), it cost only 256 parameters, and it still lost
   by +0.027 MSE. Treat "add a new input channel" as a red flag in this architecture.
4. **An ablation number is only valid inside the architecture it was measured on.** KANPM's
   ablation prices `W/O Linear Attention` at +0.054, but adding LinearAttention to this model
   made it worse — because this model already focuses attention via the interaction module,
   making the pooled attention redundant.
5. **Validation ranking does not predict test ranking — observed twice.** The attention-pooling
   run led on validation at matched epochs and still lost on test. The pLDDT run repeated it
   exactly: better validation than the champion (0.2550 vs 0.2578) and **worse** test
   (0.3868 vs 0.3601). With only 44 validation proteins, a validation edge under ~0.01 is noise.
6. **Output-side work is capped at 0.339 — it cannot reach KANPM.** A perfect rank-preserving
   rescale, fitted directly on the test labels (an oracle), reaches only 0.3389 versus KANPM's
   0.3140. **94% of the error survives any output transform**, so it is missing information,
   not miscalibration. No loss reshaping, clamping, or calibration closes the MSE gap.
7. **The training signal is exhausted.** Observed train MSE 0.0598 against an irreducible floor
   of 0.0348 — only 0.025 of headroom. This is why every regularizer failed: they reduce
   fitting, and there is almost nothing left to un-fit.
8. **The model genuinely learns; it does not memorize.** On 44 unseen proteins it beats a pure
   drug-identity lookup table by 48% (0.6950 → 0.3601). The cold-protein gap is a data-scale
   limit (354 training proteins), not a memorization artifact.

---

## Superseded architecture — v4 (cross-attention + bilinear fusion)

> ⚠️ **This section describes the OLD model, kept for the record.** Cross-attention and bilinear
> fusion were both **removed** when PocketCross replaced them. For the current architecture see
> [README.md](README.md).

---

## Overview

AF2-CrossKAN-DTA is a drug-target affinity (DTA) prediction model that predicts how strongly a drug molecule binds to a protein target. Strong binding means the drug is more likely to work — making DTA prediction a critical step in computational drug discovery.

**The three core contributions of this work:**

1. **AlphaFold2 3D Contact Maps** — builds binary protein contact maps from real 3D Cα coordinates in the AlphaFold2 public database (edge if Cα distance < 8Å) instead of sequence-predicted contact maps, giving the protein graph branch genuine structural topology
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

#### Our Approach — Binary Contact Maps from Real 3D Structure

We replace sequence-predicted contact maps with binary contact maps derived from real 3D Cα coordinates in AlphaFold2 predicted structures. For each protein in the dataset we:

1. Query UniProt REST API to resolve the gene name to an accession ID
2. Query AlphaFold2 EBI API to get the PDB file URL
3. Download the PDB and extract Cα (alpha-carbon) atom 3D coordinates — one per residue
4. Compute all pairwise Euclidean distances in Angstroms
5. Threshold at 8Å: residue pairs with Cα distance < 8Å become edges (binary contact, weight 1.0); pairs beyond 8Å are not connected

The 8Å threshold is the standard cutoff in structural biology for defining residue-residue contacts. The advantage over sequence-predicted maps is not edge *weighting* but edge **correctness**: every edge reflects true 3D proximity measured from the AlphaFold2 structure, rather than a co-evolutionary statistical guess. The graph topology therefore matches the protein's real binding-site geometry.

| Property | Sequence-predicted (KANPM-DTA) | AlphaFold2 3D (Ours) |
|----------|-------------------------------|----------------------|
| Source | ESM-2 probability matrix | Real Cα coordinates |
| Edge criterion | Probability > 0.5 | Real Cα distance < 8Å |
| Edge weight | **Contact probability** `W_ij = M_ij` (weighted) | **None — unweighted graph** (see note) |
| Geometric information | None | Edges set by true 3D distances |
| Binding site accuracy | Statistical approximation | Real 3D topology |
| False edges | Possible (co-evolution ≠ proximity) | None (distance is exact) |

> ⚠️ **Correction — this model uses no edge weights at all.** `target2graph` sets
> `edge_weight = torch.ones(...)`, and passing an all-ones weight to `GCNConv` is **bit-identical**
> to passing none (degree normalisation cancels any constant). The five `GATConv` layers never
> receive it — they are called `conv(x, ei)` and were built without `edge_dim`. So **0 of 6 graph
> layers use edge weights**; only `edge_index` (which residue pairs are connected) is used.
> Describing either model's edges as "binary (0/1)" was wrong: non-contacts are absent from
> `edge_index` rather than present with weight 0, and KANPM's edges carry continuous contact
> probabilities. The real contrast is **weighted (KANPM) vs unweighted (this model)**, which is a
> larger difference and is currently **confounded** with the contact-source difference in any
> head-to-head.

**DAVIS coverage:**
- 364 proteins → direct AlphaFold2 3D structure
- 72 proteins → wildtype AlphaFold2 for mutant variants (e.g. EGFR(L858R) uses EGFR — single point mutations do not change the overall fold)
- 6 proteins → backbone fallback (non-human organisms: 2× P. falciparum, 1× M. tuberculosis, 3 others with no AF2 entry) — all 442 proteins included in training

---

### Cross-Attention

Two `nn.MultiheadAttention` layers replace independent sequence pooling:

```python
# Drug tokens attend to protein tokens
xd_cross, _ = self.cross_attn_drug(query=xd, key=xp, value=xp, key_padding_mask=prot_pad_mask)
# Protein tokens attend to drug tokens
xp_cross, _ = self.cross_attn_prot(query=xp, key=xd, value=xd, key_padding_mask=drug_pad_mask)

xd_attn = self.drug_pool(xd_cross, smiles_mask)   # [B, 128]  — attention pooling
xp_attn = self.prot_pool(xp_cross, fasta_mask)    # [B, 128]  — attention pooling
```

This ensures each drug representation is conditioned on the protein it is paired with, and vice versa. The pooled outputs use KANPM's `LinearAttention` (learned weighted pool) rather than a plain mean.

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

Both embedding dimensions are projected down to 128 before further processing:

```
Drug:    fc3: [B, 220, 384]   →  [B, 220, 128]   (single linear)
Protein: fc2: [B, 1200, 1152] →  [B, 1200, 128]  (Linear 1152→512 → GELU → LayerNorm → Linear 512→128)
```

The protein side uses a 2-layer **nonlinear** projection (rather than a single flat linear) so more of ESM-C's 1152-dim signal is preserved through the 9× compression. This gives both modalities a common representation size.

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
Drug atoms [N_atoms, 88]  →  GCN(bond_scalar) → 5×GAT → global_add_pool  →  smiles_graph [B, 128]
```
The GCNConv layer is weighted by a scalar derived from the bond features (the 6-dim bond vector averaged to one value); the 5 GAT layers operate on the graph connectivity alone.

**Protein graph:** Same architecture but starts from 1152-dim ESM-C node features:
```
Protein residues [N_res, 1152]  →  GCN(binary contact) → 5×GAT → global_add_pool  →  fasta_graph [B, 128]
```
Edges are the binary AlphaFold2 contacts (Cα < 8Å, uniform weight 1.0); the GCNConv layer uses this binary adjacency and the 5 GAT layers operate on the connectivity. The graph topology reflects the protein's real 3D structure.

The graph branch captures **structure** (bond connectivity for drugs, real 3D residue contacts for proteins) that the sequence branch cannot see.

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

Both outputs are collapsed to fixed-size vectors via **attention pooling** (KANPM's `LinearAttention`, 8 heads) — a learned, masked weighted pool that focuses on binding-relevant residues instead of averaging all of them equally:
```
xd_attn = drug_pool(xd_cross, smiles_mask)   [B, 128]  — protein-conditioned drug representation
xp_attn = prot_pool(xp_cross, fasta_mask)    [B, 128]  — drug-conditioned protein representation
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

Davis dataset, warm/random split (log-scale pKd MSE — lower is better). Values taken from the respective papers; sorted best MSE first.

| Model                  | MSE       | CI        | r2m       |
|------------------------|-----------|-----------|-----------|
| 3DProtDTA (2023)       | 0.184     | 0.917     | 0.722     |
| HCAF-DTA (2025)        | 0.198     | 0.908     | 0.728     |
| GSAML-DTA (2022)       | 0.201     | 0.896     | 0.718     |
| DGraphDTA (2020)       | 0.202     | 0.904     | 0.700     |
| KANPM-DTA (2026)       | 0.204     | 0.898     | 0.715     |
| WGNN-DTA (2022)        | 0.208     | 0.900     | 0.691     |
| GraphDTA (2021)        | 0.229     | 0.893     | 0.685     |
| DeepDTA (2018)         | 0.261     | 0.878     | 0.630     |

> **Note:** these are Davis *warm/random-split* results as reported in each paper. Evaluation protocols differ (e.g. 3DProtDTA uses 5-fold CV), so treat this as the competitive landscape rather than a perfectly controlled head-to-head. KANPM-DTA is the direct predecessor; 3DProtDTA (also AlphaFold-structure-based) is the current best.

### AF2-CrossKAN-DTA Results

#### Davis (Warm Setting)

We run independent trainings on the DAVIS warm setting, each with a different data split seed (41, 42, 43, 32 — the same protocol used in KANPM-DTA). The reported MSE/CI/r2m are the mean and standard deviation across seeds. The model is **v4**: Adam lr 1e-4, no weight decay, flat LR, mean pooling, flat ESM-C projection `Linear(1152→128)`, full KAN [512, 1024, 512, 1], fixed interaction-attention masking. Output tag `_new`.

| Model | Split Seed | Test MSE | Test CI | Test r2m |
|-------|------------|----------|---------|----------|
| v4 — seed 41 | 41 | **0.1973** | 0.8843 | 0.7103 |
| v4 — seed 42 | 42 | **0.2340** | 0.8848 | 0.6707 |
| v4 — seed 43 | 43 | - | - | - |
| v4 — seed 32 | 32 | - | - | - |
| **v4 (Mean ± Std)** | — | - | - | - |

#### KIBA (Warm Setting)

| Model                       | MSE | CI | r2m |
|-----------------------------|-----|----|-----|
| **AF2-CrossKAN-DTA (Ours)** | -   | -  | -   |

---

## Training Progress Log — DAVIS Warm (v4, `_new`)

v4 config: Adam lr 1e-4, **no weight decay**, **flat LR**, mean pooling, flat ESM-C `fc2` `Linear(1152→128)`, full KAN [512,1024,512,1], fixed interaction masking, dropout 0.2. Output tag `_new`.

### Seed 41 (Complete)

> **Best Checkpoint — Epoch 83**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0769 |
> | Valid MSE | **0.2311** |
> | Valid CI | **0.8907** |
> | Valid r2m | **0.6866** |

> **Final Test Result**
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.1973** |
> | Test CI | **0.8843** |
> | Test r2m | **0.7103** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.3936 | 0.4141 | 0.8356 | 0.4612 |
| 20 | 0.3164 | 0.3350 | 0.8545 | 0.5645 |
| 30 | 0.2518 | 0.2883 | 0.8683 | 0.6484 |
| 40 | 0.2041 | 0.2633 | 0.8838 | 0.6289 |
| 50 | 0.1576 | 0.2625 | 0.8840 | 0.6199 |
| 60 | 0.1267 | 0.2610 | 0.8864 | 0.6258 |
| 70 | 0.0944 | 0.2355 | 0.8966 | 0.6533 |
| 80 | 0.0830 | 0.2395 | 0.8863 | 0.6790 |
| **83 (best)** | **0.0769** | **0.2311** | **0.8907** | **0.6866** |
| 90 | 0.0710 | 0.2410 | 0.8883 | 0.6390 |
| 100 | 0.0628 | 0.2415 | 0.8940 | 0.6563 |

### Seed 42 (Complete)

> **Best Checkpoint — Epoch 158**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0457 |
> | Valid MSE | **0.1987** |
> | Valid CI | **0.8831** |
> | Valid r2m | **0.7367** |

> **Final Test Result** (natural early-stop at ep178; tested on the ep158 best checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.2340** |
> | Test CI | **0.8848** |
> | Test r2m | **0.6707** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.3942 | 0.4220 | 0.7988 | 0.4393 |
| 20 | 0.3148 | 0.3370 | 0.8314 | 0.5673 |
| 30 | 0.2556 | 0.3007 | 0.8518 | 0.6077 |
| 40 | 0.2133 | 0.2787 | 0.8649 | 0.6102 |
| 50 | 0.1751 | 0.2563 | 0.8678 | 0.6322 |
| 60 | 0.1407 | 0.2436 | 0.8782 | 0.6389 |
| 70 | 0.1105 | 0.2339 | 0.8721 | 0.6875 |
| 80 | 0.0871 | 0.2249 | 0.8717 | 0.6751 |
| 90 | 0.0716 | 0.2285 | 0.8713 | 0.6574 |
| 100 | 0.0668 | 0.2138 | 0.8797 | 0.6876 |
| 110 | 0.0615 | 0.2228 | 0.8793 | 0.6798 |
| 120 | 0.0551 | 0.2089 | 0.8822 | 0.6734 |
| 130 | 0.0532 | 0.2117 | 0.8810 | 0.6911 |
| 140 | 0.0498 | 0.2057 | 0.8832 | 0.7184 |
| 150 | 0.0477 | 0.2054 | 0.8790 | 0.7456 |
| **158 (best)** | **0.0457** | **0.1987** | **0.8831** | **0.7367** |
| 160 | 0.0455 | 0.2055 | 0.8807 | 0.7163 |
| 170 | 0.0456 | 0.2094 | 0.8731 | 0.7405 |
| 178 (final) | 0.0446 | 0.2092 | 0.8768 | 0.7198 |

### Seed 43 (Pending)

_To be run._

### Seed 32 (Pending)

_To be run._

---

## Training Progress Log — DAVIS Cold-Protein (`unseen_prot`, v4, `_new`)

**Separate benchmark from warm split.** Here the 442 test proteins are **held out of training entirely** — the model must generalize to proteins it has never seen, so it cannot win by memorizing entity identities. This is the split where the AF2 contact map + ESM-C features are the *only* thing that can help. Same v4 model, `SPLIT_SEED = 42`.

> ⚠️ **Reset the yardstick:** absolute MSE is expected to be **~2× the warm number** (roughly 0.3–0.5, not 0.20) — that is normal for cold-protein and is *not* a regression. Judge this split by **CI (ranking ability)** and by comparison to *cold-protein* baselines, not by the warm 0.20.

### Seed 42 (Complete)

> **Best Checkpoint — Epoch 60**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.1277 |
> | Valid MSE | **0.3448** |
> | Valid CI | **0.8360** |
> | Valid r2m | **0.5310** |

> **Final Test Result** (natural early-stop at ep81; tested on the ep60 best checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.4071** |
> | Test CI | **0.8382** |
> | Test r2m | **0.4112** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.3940 | 0.5023 | 0.7987 | 0.4365 |
| 20 | 0.3007 | 0.3953 | 0.8329 | 0.5162 |
| 30 | 0.2497 | 0.3824 | 0.8302 | 0.5193 |
| 40 | 0.1997 | 0.4015 | 0.8380 | 0.4870 |
| 50 | 0.1587 | 0.4234 | 0.8333 | 0.4526 |
| **60 (best)** | **0.1277** | **0.3448** | **0.8360** | **0.5310** |
| 70 | 0.1094 | 0.3573 | 0.8340 | 0.5026 |
| 80 (final) | 0.0832 | 0.3699 | 0.8375 | 0.4988 |

_Result (Complete): cold-protein **test MSE 0.4071, CI 0.8382, r2m 0.4112** on 442 held-out proteins, evaluated on the ep60 best-valid checkpoint. As expected, absolute MSE is ~1.9× the warm number (0.211 → 0.407) — the cost of never seeing these proteins. The headline is the **CI: 0.838 on unseen proteins** vs 0.882 warm — ranking ability holds up far better than MSE, evidence the AF2/ESM-C features genuinely generalize rather than memorize. Note the low r2m (0.411): the pKd≈5 non-binder floor compresses the prediction range harder on cold proteins. No KANPM cold-protein baseline exists for a direct comparison; a same-split ablation (strip cross-attention/AF2 ≈ KANPM, run on `unseen_prot`) would be needed for a head-to-head._

---

## AF2-AtomRes-DTA — development log

> **Self-contained section for the new architecture.** Code: `code/model_pocketcross.py`, trained via `code/train_pocketcross.py`. If it outperforms the baseline, everything above this line can be removed and this becomes the main model.

## Motivation

Existing DTA models (KANPM, and the baseline above) **pool** the drug and protein each into a single vector, then combine them — discarding the actual binding event, which is specific **drug atoms** contacting specific **protein pocket residues**. AF2-AtomRes-DTA models that interaction **explicitly**, guided by 3D structure:

1. A **pocket prior** derived from the AF2 contact map scores each residue's bindability (burial/contact degree).
2. **Drug-atom ↔ protein-residue attention**, biased toward the structural pocket, produces an **interpretable interaction map** (which atom binds which residue).
3. A **weighted loss** upweights strong binders to fix the measured binder-lowballing (low r²ₘ).

## Architecture

```
 DRUG                                    PROTEIN                    AF2 3D structure
 SMILES → ChemBERTa → Transformer×3      seq → ESM-C → Transformer×3   per-residue
   → Hd [B,220,128] (per ATOM)             → Hp [B,1200,128] (per RES)  struct feats
        │        │                           │        │                    │
        │        │                           │        │              ┌───────────┐
        │        │                           │        │              │PocketPrior│ NEW
        │        │                           │        │              │  (MLP)    │
        │        └──────────┐     ┌──────────┘        │              └─────┬─────┘
        │                   ▼     ▼                    │              pocket score s
        │      ┌──────────────────────────────────────────────┐          │
        │      │ STRUCTURE-GUIDED ATOM↔RESIDUE ATTENTION       │◄─────────┘  NEW
        │      │ score = Qd·Kp/√d + β·s   →  I [B,220,1200]     │──► interpretable map
        │      │ f_int = Σ I·(Hd ⊙ Hp)   →  [B,128]            │
        │      └───────────────────────┬──────────────────────┘
        ▼ mean-pool        mean-pool ▼ │ f_int
     gd [B,128]            gp [B,128]  │
        │  DRUG graph ─┐  ┌─ PROT graph│      (KEPT: GNN branches
        │  GNN→[B,128] ▼  ▼ GNN→[B,128]│       + gated fusion)
        │          GatedFusion → g [B,128]
        └──────┬──────────┴──────┬──────┘
               ▼                  ▼
        concat[ f_int, gd, gp, g ] = [B,512] → KAN[512→1024→512→1] → affinity
        loss = weighted MSE (α=0.5, binders upweighted)
```

## What's Kept / New / Removed vs the baseline

| | Component |
|---|---|
| **Kept** | ChemBERTa & ESM-C encoders, transformers, drug+protein GNNs (GCN+5×GAT), gated graph fusion, KAN predictor |
| **New** | Pocket prior (AF2 contact-degree → MLP), structure-guided atom↔residue attention, interpretable interaction map, weighted loss |
| **Removed** | Cross-attention (drug↔prot), bilinear fusion, separate interaction-attention branch |

## Training Setup

| Setting | Value |
|---------|-------|
| Model file | `code/model_pocketcross.py` |
| Train script | `code/train_pocketcross.py` |
| Params | ~13.5 M |
| Structural features (`struct_dim`) | 4 (scaled/relative/z-scored contact degree + residue flag) |
| Pocket-bias strength β | learnable (init 1.0) |
| Loss | **weighted MSE**, `w = 1 + 0.5·(pKd − 5)` |
| Optimizer | Adam, lr 1e-4, no weight decay, flat LR |
| Batch size | 16 · Early stopping | patience 20 on valid MSE |
| Split | DAVIS cold-protein (`unseen_prot`), seed 42 |

---

## Results — DAVIS Cold-Protein (`unseen_prot`, seed 42)

| Model | Test MSE ↓ | Test CI ↑ | Test r²ₘ ↑ |
|-------|-----------|-----------|-----------|
| Baseline (old model, plain MSE) | 0.4071 | 0.8382 | 0.4112 |
| KANPM-DTA (target) | 0.314 | 0.857 | 0.556 |
| AF2-AtomRes-DTA (weighted loss) | 0.3781 | 0.8450 | 0.4542 |
| AF2-AtomRes-DTA (plain MSE, hand-counted pocket features) | 0.3715 | 0.8453 | 0.4628 |
| AF2-AtomRes-DTA (plain MSE, rich distance-features) — worse, not adopted | 0.4287 | 0.8180 | 0.4206 |
| AF2-AtomRes-DTA (plain MSE, attention pooling) — worse, not adopted | 0.3759 | 0.8420 | 0.4699 |
| **AF2-AtomRes-DTA (plain MSE, GNN-derived pocket prior) — best result** | **0.3519** | **0.8547** | **0.4928** |

### Seed 42 — Plain MSE ablation (Complete — natural early-stop ep82)

_Same model/split/seed as the weighted run above — **only the loss changed** (plain MSE instead of weighted) — isolating how much of the result comes from the architecture vs the loss. Trained to natural early-stop (best valid at ep62, patience 20 exhausted at ep82). Tested on the ep62 checkpoint._

> **Best Checkpoint — Epoch 62**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0458 |
> | Valid MSE | **0.3397** |
> | Valid CI | **0.8446** |
> | Valid r2m | **0.5585** |

> **Final Test Result** (natural early-stop ep82; tested on ep62 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3715** |
> | Test CI | **0.8453** |
> | Test r2m | **0.4628** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.2808 | 0.4001 | 0.8319 | 0.4918 |
| 27 | 0.1003 | 0.3520 | 0.8440 | 0.5384 |
| 44 | 0.0611 | 0.3448 | 0.8404 | 0.5557 |
| 48 | 0.0555 | 0.3439 | 0.8428 | 0.5390 |
| **62 (best)** | 0.0458 | **0.3397** | 0.8446 | 0.5585 |
| 70 | 0.0461 | 0.3496 | 0.8382 | 0.5294 |
| 82 (final) | 0.0436 | 0.3616 | 0.8325 | 0.5156 |

_**Result (Complete):** plain-MSE AF2-PocketCross test **MSE 0.3715 / CI 0.8453 / r²ₘ 0.4628** — **beats the weighted-loss run on all three metrics** (0.3781→0.3715 MSE, 0.8450→0.8453 CI, 0.4542→0.4628 r²ₘ). This contradicts the pre-registered prediction that weighted loss would help r²ₘ; at least at α=0.5 on this seed, plain MSE is simply better. **Attribution conclusion: the gain over the old baseline (0.4071→0.3715, −0.036) is driven by the architecture (structure-guided atom↔residue interaction), not the weighted loss.** This plain-MSE number is also the fair, KANPM-comparable headline result (field convention = plain MSE)._

### Seed 42 — Weighted loss (Complete — natural early-stop ep47)

_Trained to natural early-stop (best valid at ep27, patience 20 exhausted at ep47). Tested on the ep27 best-valid checkpoint._

> **Best Checkpoint — Epoch 27**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.1139 |
> | Valid MSE | **0.3460** |
> | Valid CI | **0.8370** |
> | Valid r2m | **0.5546** |

> **Final Test Result** (natural early-stop ep47; tested on ep27 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3781** |
> | Test CI | **0.8450** |
> | Test r2m | **0.4542** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.3052 | 0.4613 | 0.8272 | 0.4421 |
| 20 | 0.1625 | 0.3749 | 0.8338 | 0.5185 |
| **27 (best)** | 0.1139 | **0.3460** | 0.8370 | 0.5546 |
| 30 | 0.1035 | 0.3571 | 0.8360 | 0.5345 |
| 40 | 0.0755 | 0.3595 | 0.8318 | 0.5166 |
| 47 (final) | 0.0656 | 0.3882 | 0.8401 | 0.4776 |

_**Result (Complete):** AF2-AtomRes-DTA (weighted loss) test **MSE 0.3781 / CI 0.8450 / r²ₘ 0.4542** on cold-protein seed 42. **Beats the old baseline on all three metrics** (MSE 0.4071→0.3781, −0.029; CI 0.8382→0.8450; r²ₘ 0.4112→0.4542) — the structure-guided interaction + weighted loss both help. Still short of KANPM (0.314 / 0.857 / 0.556). Note: this bundles two changes (new architecture + weighted loss); a plain-MSE run of the same model is needed to attribute how much each contributes. Also the valid→test gap is smaller than the baseline's (0.346→0.378 = +0.032 vs baseline 0.345→0.407 = +0.062), a sign of better generalization._

### Seed 42 — Rich distance-based pocket features (Complete — natural early-stop ep49)

_Same model/split/seed/plain-MSE loss as the 0.3715 result above — **only the pocket-prior's input features changed**: 8 features derived from real AF2 Cα distances (tight/mid/loose shells at 5.5/6.5/8A, decay-weighted density, mean contact distance, z-scored degree, flag) instead of plain contact-degree, with sequence-adjacent pairs excluded so nothing reflects trivial backbone bonding. Trained to natural early-stop (best valid at ep29, patience 20 exhausted at ep49). Tested on the ep29 checkpoint._

> **Best Checkpoint — Epoch 29**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0968 |
> | Valid MSE | **0.3650** |
> | Valid CI | **0.8240** |
> | Valid r2m | **0.5447** |

> **Final Test Result** (natural early-stop ep49; tested on ep29 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.4287** |
> | Test CI | **0.8180** |
> | Test r2m | **0.4206** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 6 | 0.3591 | 0.4152 | 0.8230 | 0.5061 |
| 12 | 0.2393 | 0.3756 | 0.8286 | 0.5243 |
| 20 | 0.1554 | 0.4060 | 0.8222 | 0.5009 |
| **29 (best)** | 0.0968 | **0.3650** | 0.8240 | 0.5447 |
| 40 | 0.0717 | 0.3756 | 0.8333 | 0.4998 |
| 49 (final) | 0.0613 | 0.3873 | 0.8231 | 0.4863 |

_**Result (Complete) — negative finding:** test **MSE 0.4287 / CI 0.8180 / r²ₘ 0.4206** is WORSE than every other pocket-cross variant, and worse than even the pre-pocket-cross baseline (0.4071). Both valid AND test are worse than the winning plain-MSE run (best-valid 0.3650 vs that run's 0.3397; test 0.4287 vs that run's 0.3715) — this richer-feature version underperformed at every stage, not just at test time. The valid→test gap here (+0.064, 0.365→0.429) is also the largest seen across any run in this project. **Honest takeaway: enriching the pocket-prior from 4 simple degree-based features to 8 real-distance features did not help — it hurt, on both validation and test.** Plausibly the extra features gave the tiny pocket MLP more room to fit noise without adding real signal beyond what plain degree already captured. The simpler degree-only features (used in the 0.3715 best result) remain the better choice for this architecture. Reverting `RICH_STRUCT_FEATURES` to `False` (plain 4-feature pocket prior) is recommended going forward unless revisited with stronger regularization._

### Seed 42 — GNN-derived pocket prior (Complete — natural early-stop ep84) — 🏆 NEW BEST

_Same backbone/split/seed/plain-MSE loss as the 0.3715 result — **only the source of the pocket score changed**: instead of hand-counted contact-degree features (`target2struct`), the pocket score now comes from `ProteinGraphNet`'s own per-residue embeddings (already learned via real message-passing over the AF2 contact graph), densified via `to_dense_batch` instead of being discarded after pooling. No separate feature-engineering pipeline — `code/model_af2_atomres.py` / `code/train_af2_atomres.py`. Trained to natural early-stop (best valid at ep64, patience 20 exhausted at ep84). Tested on the ep64 checkpoint._

> **Best Checkpoint — Epoch 64**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0526 |
> | Valid MSE | **0.3185** |
> | Valid CI | **0.8504** |
> | Valid r2m | **0.5891** |

> **Final Test Result** (natural early-stop ep84; tested on ep64 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3519** |
> | Test CI | **0.8547** |
> | Test r2m | **0.4928** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 39 | 0.0785 | 0.3340 | 0.8486 | 0.5687 |
| 59 | 0.0550 | 0.3264 | 0.8398 | 0.5737 |
| **64 (best)** | 0.0526 | **0.3185** | 0.8504 | 0.5891 |
| 75 | 0.0480 | 0.3386 | 0.8466 | 0.5330 |
| 84 (final) | 0.0465 | 0.3390 | 0.8507 | 0.5391 |

_**Result (Complete) — new best:** test **MSE 0.3519 / CI 0.8547 / r²ₘ 0.4928** — **beats the previous champion (hand-counted pocket features) on all three metrics** (0.3715→0.3519 MSE, −0.0196; 0.8453→0.8547 CI; 0.4628→0.4928 r²ₘ). Reusing the protein GNN's own learned per-residue embeddings — instead of hand-counted contact-degree features — genuinely helps. The gap to KANPM narrows further: MSE gap 0.093 (baseline) → 0.058 (previous champion) → **0.038** (this result); CI gap is now only 0.002 (0.8547 vs 0.857) — essentially matched. r²ₘ gap remains the largest weak point (0.493 vs 0.556). This is now the standing best result for AF2-AtomRes-DTA on cold-protein seed 42._

### Seed 41 — GNN-derived pocket prior (Complete — natural early-stop ep49) — seed validation

_**Purpose: validation, not a new idea.** Identical model and config to the seed-42 champion — **only the data split changed** (`cold_split.py` SEED 41, `SPLIT_SEED = 41`). Every result in this project so far comes from seed 42 alone, and the champion's margin over the runner-up (0.0196) is **smaller than the 0.037 seed swing already observed on the warm split** (seed 41: 0.1973 vs seed 42: 0.2340). So a single-seed result cannot distinguish a real improvement from split luck. Verified this is a genuinely independent evaluation: **only 2 of 44 test proteins overlap with the seed-42 test set.**_

> **Best Checkpoint — Epoch 29**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0995 |
> | Valid MSE | **0.2578** |
> | Valid CI | **0.8719** |
> | Valid r2m | **0.5855** |

> **Final Test Result** (natural early-stop ep49; tested on ep29 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3601** |
> | Test CI | **0.8590** |
> | Test r2m | **0.5460** |

_**⚠️ Correction — the "seed 41 is an easier split" read was wrong.** Throughout this run, seed 41's **validation** was dramatically better than seed 42's (best 0.2578 vs 0.3185), and that was repeatedly interpreted as an easier split. **The test result contradicts it:** seed 41 tests at **0.3601**, slightly WORSE than seed 42's 0.3519. So a much better validation curve came with a slightly worse test score — **validation difficulty and test difficulty are independent**, and inferring split difficulty from the validation curve was unjustified. Seed 41's valid→test gap is +0.102 versus seed 42's +0.033._

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.2622 | 0.3253 | 0.8416 | 0.4955 |
| 12 | 0.2287 | 0.2878 | 0.8517 | 0.5665 |
| 14 | 0.1991 | 0.2809 | 0.8616 | 0.5778 |
| 20 | 0.1485 | 0.2852 | 0.8668 | 0.5427 |
| 28 | 0.1022 | 0.2694 | 0.8678 | 0.5548 |
| **29 (best so far)** | 0.0995 | **0.2578** | 0.8719 | 0.5855 |
| 32 | 0.0910 | 0.2655 | 0.8633 | 0.5814 |
| 40 | 0.0746 | 0.2611 | 0.8608 | 0.5776 |
| 49 (final) | 0.0598 | 0.3109 | 0.8479 | 0.4947 |

#### 📊 Champion across seeds — mean ± std (the point of the validation)

> ⚠️ **SUPERSEDED by the 3-seed table below.** The figures in this block are 2-seed and two of
> its conclusions did not survive seed 43: "cold-split test variance is small" was an artefact of
> having only two samples (MSE spread ±0.0058 → ±0.0109), and the CI gap of 0.0001 widened to
> 0.0029. Kept for the record; cite the 3-seed table instead.

| Metric | Seed 41 | Seed 42 | **Mean ± Std** | KANPM-DTA | Gap |
|--------|---------|---------|----------------|-----------|-----|
| Test MSE ↓ | 0.3601 | 0.3519 | **0.3560 ± 0.0058** | 0.314 ± 0.017 | +0.042 |
| Test CI ↑ | 0.8590 | 0.8547 | **0.8569 ± 0.0030** | 0.857 ± 0.011 | **+0.0001 — matched** |
| Test r²ₘ ↑ | 0.5460 | 0.4928 | **0.5194 ± 0.0376** | 0.556 ± 0.023 | +0.037 |

_**Three findings, one of which overturns an earlier assumption:**_

_**1. Cold-split test variance is small — much smaller than feared.** The two seeds differ by only **0.0082** MSE. The concern that drove this validation was the **0.0367** seed swing observed on the *warm* split — but that does not transfer to cold-protein. Consequently the champion's **0.0196** margin over the runner-up (0.3519 vs 0.3715) is **~2.4× the observed test spread**, so that improvement now looks real rather than seed luck. The concern was worth checking; the answer came back in the champion's favour. (Caveat: with n=2 the std is a weak estimate — seeds 43/32 would firm it up.)_

_**2. CI is statistically matched with KANPM.** 0.8569 ± 0.0030 vs their 0.857 ± 0.011 — a gap of 0.0001, far inside both error bars. On ranking ability for unseen proteins, this model is level with the target._

_**3. r²ₘ has the widest spread (± 0.038)** — nearly 5× the MSE spread, and it is the metric that swings most between seeds (0.546 vs 0.493). Any future r²ₘ "improvement" smaller than ~0.04 cannot be distinguished from seed noise, which retrospectively explains why the weighted-loss experiment (aimed at r²ₘ) produced an ambiguous signal._

#### Seed 43 — GNN-derived pocket prior (Complete — natural early-stop ep87)

_Third seed for the champion. Identical model and config; only the split changed
(`cold_split.py --SEED 43`, `SPLIT_SEED = 43`). Independent evaluation: only 8/44 test proteins
overlap seed 41's and 2/44 overlap seed 42's. First run to use the `assert_split_matches()`
startup guard, which verified the data on disk was genuinely seed 43 before training began._

> **Best Checkpoint — Epoch 67**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0571 |
> | Valid MSE | **0.3513** |

> **Final Test Result** (natural early-stop ep87, patience 20/20; tested on the ep67 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3735** |
> | Test CI | **0.8485** |
> | Test r2m | **0.5692** |

#### 📊 Champion across three seeds

| Metric | Seed 41 | Seed 42 | Seed 43 | **Mean ± Std** | 2-seed (superseded) | KANPM-DTA |
|--------|---------|---------|---------|----------------|---------------------|-----------|
| Test MSE ↓ | 0.3601 | 0.3519 | 0.3735 | **0.3618 ± 0.0109** | 0.3560 ± 0.0058 | 0.314 |
| Test CI ↑ | 0.8590 | 0.8547 | 0.8485 | **0.8541 ± 0.0053** | 0.8569 ± 0.0030 | 0.857 |
| Test r²ₘ ↑ | 0.5460 | 0.4928 | 0.5692 | **0.5360 ± 0.0392** | 0.5194 ± 0.0376 | 0.556 |

_**1. Every spread widened — n=2 had understated the variance.** MSE nearly doubled (±0.0058 →
±0.0109) and CI went ±0.0030 → ±0.0053. The earlier conclusion that "cold-split test variance is
small" was an artefact of two samples. **Any margin below ~0.011 MSE in this project cannot be
distinguished from split noise** — which still leaves the champion's 0.0196 win over the
runner-up standing (~1.8× the spread), but rules out anything smaller._

_**2. MSE standing falls from 4th to 6th of 9 — and the rank is meaningless.** At 0.3618 the model
sits 0.0008 behind MSGNN-DTA and 0.0022 ahead of FusionDTA, both an order of magnitude inside
±0.0109. Rows 4–7 of the comparison table are a statistical tie, not an ordering._

_**3. CI remains the durable result.** 0.8541 ± 0.0053 vs KANPM's 0.857 — a gap of 0.0029, against
0.0141 to the third-place model. The distance to first is ~5× smaller than the distance to third._

_**4. r²ₘ improved and is still the noisiest metric** (0.5194 → 0.5360, spread ±0.0392). The gap to
KANPM narrowed from 0.037 to 0.020 but stays well inside the spread, so no difference can be
claimed either way._

_**5. Validation mispredicted test for the fourth time.** Seed 43 had the worst validation MSE of
the three (0.3513 vs 0.2578 and 0.3185) and the worst validation r²ₘ throughout (0.4002 at ep26
vs ~0.54), yet returned the **best test r²ₘ of all three** (0.5692). See cross-cutting lesson 5
and `code/analysis/valid_test_gap.py`._

---

### Seed 42 — Attention pooling (Complete — natural early-stop ep46) — ❌ WORSE, not adopted

_Same model/split/seed/plain-MSE loss as the 0.3519 champion — **only the sequence-summary pooling changed**: the two summary vectors (`gd`, `gp`) are now produced by KANPM's learned `LinearAttention(128, 64, 8)` instead of a plain masked mean. Motivation: KANPM's ablation prices `W/O Linear Attention` at **+0.054 MSE** on unseen-protein — second only to the sequence branch and ~4× the graph branch (+0.014) — and KANPM has this component while the champion does not. Averaging 1200 residues equally dilutes the ~10–30 binding-site residues by ~40×. Cost: **+17,552 params (+0.13%)**, the cheapest change tried in this project; smoke test confirmed the param delta is exactly the two pooling modules. **Code and checkpoints were deleted after the negative result** — recoverable from git commit `3e0138b` if this is ever revisited._

> **Best Checkpoint — Epoch 25**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.1196 |
> | Valid MSE | **0.3256** |
> | Valid CI | **0.8496** |
> | Valid r2m | **0.5949** |

> **Final Test Result** (natural early-stop ep46; tested on ep25 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3759** |
> | Test CI | **0.8420** |
> | Test r2m | **0.4699** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.2812 | 0.3727 | 0.8409 | 0.5166 |
| 14 | 0.2166 | 0.3440 | 0.8442 | 0.5882 |
| **25 (best)** | 0.1196 | **0.3256** | 0.8496 | 0.5949 |
| 33 | 0.0906 | 0.3385 | 0.8445 | 0.5634 |
| 45 (final) | 0.0638 | 0.3522 | 0.8336 | 0.5347 |

_**Result (Complete) — negative finding:** test **MSE 0.3759 / CI 0.8420 / r²ₘ 0.4699** — **worse than the champion on all three metrics** (MSE +0.0240, CI −0.0127, r²ₘ −0.0230). Attention pooling is **not adopted**; the champion's plain-mean pooling stays._

_**Why the prediction failed.** The change was motivated by KANPM's ablation pricing `W/O Linear Attention` at +0.054 on unseen-protein. That evidence did not transfer, and in hindsight the reason is identifiable: **that ablation was measured on KANPM's architecture, which has no cross-attention** — there, `LinearAttention` is the *only* mechanism that can focus on relevant residues, so removing it is devastating. This model already has `StructGuidedInteraction`, a pocket-biased attention over residues. The summary vectors `gd`/`gp` are supplementary context, so a learned pool there is largely **redundant with focusing the model already does** — and the extra parameters cost more than the redundant focusing gained._

_Also notable: this run had a **larger valid→test gap** than the champion (+0.050 vs +0.034), i.e. it generalized worse despite an early validation lead — it reached 0.3256 by ep25 (champion needed ~ep59 for 0.3264) but then stalled, while the champion kept improving to 0.3185 by ep64. **Fast early validation progress was not predictive of final test quality.**_

---

### Seed 41 — AF2 pLDDT as a protein node feature (Complete — natural early-stop ep57) — ❌ WORSE, not adopted

_Same model/split/seed/plain-MSE loss as the seed-41 champion — **only the protein node features changed**: AlphaFold2's per-residue confidence (pLDDT, 0–100) is appended to each residue's ESM-C vector, scaled to [0,1], so `ProteinGraphNet` sees **1153** features instead of 1152. Code: `code/train_pocketcross_plddt.py`, `hp.use_plddt = True`. Cost: **+256 params (13,550,530 → 13,550,786)**._

_**Why a node feature and not an edge weight.** `edge_weight` reaches only `conv0` — the five GAT layers are constructed as `GATConv(dim, dim)` with no `edge_dim` and ignore edge information entirely. Node features propagate through all six message-passing layers. Adding `edge_dim` to the GATs would have added parameters to five layers, which is the pattern that has failed every previous time._

_**Motivation (measured, not assumed).** ~20% of residues in the DAVIS AF2 graphs sit at contact degree ≤ 4 — the signature of extended/disordered conformation — and their contacts are largely fictional. Nothing in the existing inputs can express "this contact is unreliable": the binary contact map cannot, and ESM-C cannot. Confirmed on real structures: **25.3% of all residues across 442 proteins are pLDDT < 50**, and low-confidence residues are **2.2× less connected** than high-confidence ones (mean contact degree 4.58 vs 10.22)._

> **Best Checkpoint — Epoch 37**
> | Metric | Value |
> |--------|-------|
> | Train MSE | 0.0833 |
> | Valid MSE | **0.2550** |
> | Valid CI | **0.8461** |
> | Valid r2m | **0.5869** |

> **Final Test Result** (natural early-stop ep57, patience 20 exhausted; tested on the ep37 checkpoint)
> | Metric | Value |
> |--------|-------|
> | Test MSE | **0.3868** |
> | Test CI | **0.8299** |
> | Test r2m | **0.5214** |

| Epoch | Train MSE | Valid MSE | Valid CI | Valid r2m |
|-------|-----------|-----------|----------|-----------|
| 10 | 0.2808 | 0.3149 | 0.8349 | 0.5257 |
| 20 | 0.1565 | 0.2611 | 0.8600 | 0.5787 |
| 30 | 0.1044 | 0.2818 | 0.8510 | 0.5293 |
| **37 (best)** | 0.0833 | **0.2550** | 0.8461 | 0.5869 |
| 45 | 0.0687 | 0.2637 | 0.8423 | 0.5748 |
| 50 | 0.0646 | 0.2721 | 0.8538 | 0.5457 |
| 57 (final) | 0.0634 | 0.2580 | 0.8527 | 0.5759 |

_**Result (Complete) — negative finding:** test **MSE 0.3868 / CI 0.8299 / r²ₘ 0.5214** — **worse than the seed-41 champion on all three metrics** (MSE +0.0267, CI −0.0291, r²ₘ −0.0246). The MSE penalty is **4.6× the observed cold-split seed spread (0.0058)**, so this is not seed noise. pLDDT is **not adopted**._

_**The validation trap fired again.** pLDDT had the *better* validation curve (best 0.2550 vs the champion's 0.2578) and the *worse* test score. Its valid→test gap is **+0.1318** versus the champion's +0.1023. This is the second time in this project that a validation lead inverted at test — see cross-cutting lesson 5._

_**Why the prediction failed.** The argument for pLDDT was that previous failed features (RBF edges, distance shells) were **redundant** — re-encoding distance information the graph topology already implied — whereas pLDDT is **orthogonal**, expressing something no other input can. That distinction did not save it. The more likely reading is cross-cutting lesson 2: this architecture only ever improved when a change **re-routed computation it already performed** (structure-guided interaction, GNN-derived residue prior). Every change that fed it a new input has lost, regardless of how orthogonal the information was._

_**Data quality was not the problem.** pLDDT coverage was **437/442 (98.9%)** real values, better than the contact maps' own 436/442. The 5 neutral-filled proteins (`ABL1p`, `IKK-epsilon`, `PFCDPK1(Pfalciparum)`, `PFPK5(Pfalciparum)`, `PFTAIRE2`) are genuine AlphaFold DB misses, not fetch failures. Verified: 0 length mismatches vs the contact maps, 0 NaN/Inf, range 16.8–98.9, and EGFR cross-checked against a direct download (mean 75.9, 22.8% < 50 — identical by both paths). Fetched with `pretrained/af2_plddt_only.py`, which is **read-only on the contact maps**, so the graphs stayed bit-identical to the champion's and pLDDT was genuinely the single changed variable._

---

## Diagnostics — what the remaining error actually is (2026-08)

> These are measurements on the trained seed-41 champion and on the DAVIS data itself, not new
> model variants. They exist because they **bound** what future work can achieve, and several of
> them close off directions that look attractive on paper.

### 1. The model learns; it does not memorize

Test-set comparison on the 44 held-out seed-41 proteins:

| Predictor | Test MSE | CI | r²ₘ |
|-----------|----------|----|-----|
| Global mean (knows nothing) | 0.8897 | 0.5000 | 0.0000 |
| **Drug-identity lookup table** (per-drug train mean, ignores the protein) | 0.6950 | 0.7489 | 0.2180 |
| **Champion** | **0.3601** | **0.8590** | **0.5460** |

The champion cuts error **48% below a pure drug lookup** on proteins it has never seen. The protein branch does real, transferable work — the cold-protein gap is a **data-scale limit (354 training proteins)**, not memorization.

### 2. The training signal is exhausted

Because 18.9% of training pairs are mutually contradictory (see §3), there is a hard floor on train MSE:

```
irreducible train floor (within-group variance)  : 0.0348
observed train MSE at ep49                       : 0.0598
remaining headroom                               : 0.0250
```

The model has extracted nearly everything extractable. **This explains why every regularizer failed** — weight decay, cosine LR, KAN shrink, the lean rebuild — they all reduce fitting, and there is almost nothing left to un-fit. It also means "train longer" is counterproductive: from ep29 (best) to ep49, train improved **40%** while validation got **21% worse**.

### 3. ⚠️ DAVIS itself is partly broken — 442 target keys, only 379 unique sequences

**The mutations were never applied to the sequences.** Mutant variants carry the *wildtype* sequence string; the mutation exists only in the target key's name. 81 keys collapse into 18 identical-sequence groups:

```
ABL1    15 keys   EGFR   12 keys   PIK3CA 10 keys   KIT   8 keys   FLT3  7 keys
RET      4 keys   MET     3 keys   BRAF    2 keys   ...
plus 7 DOMAIN-variant groups (RSK1/3/4, RPS6KA4/5, JAK1, TYK2 — KinDom.1 vs KinDom.2,
JH1-catalytic vs JH2-pseudokinase) and CDK4-cyclinD1 vs CDK4-cyclinD3 (differs only by a
partner protein that is not in the dataset at all)
```

**62 of 63 colliding key-pairs also share a byte-identical AF2 contact map** (the wildtype-for-mutant fallback). So the model receives **identical input** for e.g. `BRAF` and `BRAF(V600E)` and is asked to predict different affinities. Measured label disagreement between wildtype/mutant pairs: **mean MSE 0.3564** — the size of the entire test error.

Consequences, measured:

| | |
|---|---|
| Contradictory **training** pairs | **4,556 / 24,072 = 18.9%** |
| **Test** pairs with an identical (sequence, drug) in training | **340 / 2,992 = 11.4%** |
| Best possible MSE on those 340 (copy the train label) | 0.2259 |
| Their contribution to test MSE | **0.0257 of 0.3601 (≈7%) — provably unfixable by any architecture** |
| Colliding test proteins | `KIT(V559D-V654A)`, `ABL1(Q252H)p`, `RET`, `FLT3(D835H)`, `BRAF(V600E)` |

**This affects every paper on this benchmark, KANPM included.** It partly explains why the field plateaus around 0.31–0.36.

> **Repairing it is possible but changes the benchmark.** A mutation parser resolves the point/deletion
> mutations, but sequences are not canonically numbered — each protein needs its own offset (EGFR 0,
> PIK3CA +1, ABL1 +37) and some stored sequences are fragments. Only 5 families are well-constrained
> enough to trust (EGFR 17 muts, ABL1 13, PIK3CA 9, KIT 9, FLT3 5); BRAF/RET/MET/FGFR3/LRRK2 have 1–3
> mutations each and the offset search produces **spurious** matches (BRAF "resolves" at −553 when V600E
> should sit at position 600 exactly). Fixable share: ~60% of colliding test pairs.
> **Any fix makes results non-comparable to published numbers** — report it as a benchmark finding, or
> as a clearly-labelled secondary result, never as the headline.

### 4. Where the test error lives

| Region | n | share | MSE | mean pred vs true | **% of total error** |
|--------|---|-------|-----|-------------------|----------------------|
| Censored floor (y = 5.0) | 2,086 | 69.7% | 0.1212 | 5.119 vs 5.000 (**+0.119**) | **23.5%** |
| Real measurements (y > 5) | 906 | 30.3% | 0.9101 | 6.188 vs 6.600 (**−0.412**) | **76.5%** |

```
y = 5.0      MSE 0.121   bias +0.119
5 < y < 6    MSE 0.351   bias -0.012
6 < y < 7    MSE 0.817   bias -0.471
7 < y < 8    MSE 1.176   bias -0.601
8 < y < 12   MSE 2.355   bias -1.104   <- strong binders underpredicted by a full log unit
```

Three quarters of the error comes from 30% of the data, and underprediction grows monotonically with affinity. Predicted std is 0.809 vs true 0.943 — the model compresses its range to **86%** of truth.

### 5. Output-side transforms are capped at 0.339 — below KANPM is unreachable this way

| Transform | Test MSE | CI | r²ₘ |
|-----------|----------|----|-----|
| As-is | 0.3601 | 0.8590 | 0.5460 |
| Clamp at the pKd = 5.0 floor | 0.3593 (**−0.0008**) | 0.8585 | 0.5490 |
| Std-matched (expand to true range) | **0.3968** ❌ | 0.8590 | 0.4605 |
| *Oracle* linear (fitted **on test** — cheating) | 0.3525 | 0.8590 | 0.6034 |
| ***Oracle* monotone (best possible, cheating)** | **0.3389** | 0.8609 | 0.6186 |
| Honest monotone (fit on half, apply to other half) | 0.3564 (−0.0037) | — | — |

- **Clamping is worthless.** 30.9% of predictions fall below 5.0 but only by **0.022 on average** (worst 0.122) — the model already learned the floor.
- **Expanding the range hurts** (+0.037). The compression is correct shrinkage under uncertainty, not a defect.
- **94% of the error survives a perfect rank-preserving rescale.** It is missing information, not miscalibration.
- **But r²ₘ has real headroom** — the oracle reaches 0.6186 vs KANPM's 0.556, so a range-fidelity intervention (e.g. censored/Tobit likelihood for the 69.4% of labels that are `>` bounds, not measurements) is the one output-side idea still worth testing. It will not move MSE.

### 6. ⚠️ Correction — "3D not 2D" is not a defensible claim

The pipeline extracts real Cα coordinates and computes true distances, **but the distances are discarded**:

```python
# MyDataset.target2graph
target_edge_distance.append(distance_map[i, j])   # distances collected...
edge_weight = torch.ones(len(target_edge_distance), dtype=torch.float32)  # ...then overwritten
```

What the network receives is an **L×L binary adjacency matrix** — the same *type* of object as KANPM's ESM-2 contact map. No distances, coordinates, angles, or orientations reach any layer. (And `edge_weight` would only reach `conv0` anyway.)

AF2 is still doing most of the structural work — it supplies the **edge list**, and **52.4% of edges are impossible to derive from sequence order alone** (47.6% are \|i−j\| ≤ 2 sequential, 25.3% local fold, **27.1% long-range \|i−j\| > 10**). But the correct wording is **"structure-derived contact topology from AlphaFold2"**, contrasted with KANPM's **sequence-predicted, probability-weighted** topology — a claim about edge *provenance*, not dimensionality. Note KANPM's soft edges carry more information *per edge* than this model's binary ones; that difference is currently **confounded** with the topology-source difference in any head-to-head.

### 7. Literature position — the AF2 premise is contradicted, including by our own numbers

| Work | Finding |
|------|---------|
| **3DProtDTA** (2023, RSC Advances) | Residue-level protein graphs from **AlphaFold** structures on Davis/KIBA — the core idea here, published three years earlier |
| **CASTER-DTA** (2024, bioRxiv) | **Equivariant** GNNs over AlphaFold structures for DTA |
| **"When Does Structure Help?"** (2026, arXiv:2606.04228) | For binding affinity: ESM-2 Pearson r **0.449** vs AF2-single **0.307** vs AF2 pair-diagonal **0.278**. **Information bonus of AF2 over ESM-2 = −0.141 (negative.)** Structure helps other protein-property tasks, not affinity |

**Our own result agrees**: AF2-derived contacts give 0.356, KANPM's ESM-2-predicted contacts give 0.314. Swapping predicted contacts for real structure made it **worse** — independently reproducing the negative information bonus.

**Still uncontested:** nobody has run the controlled **contact-map-source swap** (AF2 edges vs ESM-2 edges, same architecture, same ESM-C nodes) on DAVIS **cold-protein**. That is a narrow but genuine gap — and the literature now predicts the outcome.

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

## Current Model & Training Setup

The current/official model is **v4** — binary AlphaFold2 contact maps, plain GAT graph layers, ChemBERTa + ESM-C sequence transformers, cross-attention, bilinear fusion, and a KAN predictor:

- **Fixed interaction-attention masking** — `generate_masks` uses each sample's real sequence length.
- **Mean pooling** — cross-attention outputs collapsed with plain `mean(dim=1)`.
- **Flat ESM-C projection** — `fc2` is a single `Linear(1152, 128)`.
- **Full KAN** — `[512, 1024, 512, 1]`.
- **KANPM-faithful optimizer** — plain Adam, lr 1e-4, **no weight decay**, **flat LR** (no scheduler), plain MSE.

| Setting | Value |
|---------|-------|
| KAN predictor | [512, 1024, 512, 1] (full) |
| ESM-C projection (`fc2`) | Linear(1152 → 128) (flat) |
| Pooling | mean over sequence |
| Optimizer | Adam, lr 1e-4, betas (0.9, 0.999), **no weight decay** |
| LR schedule | none (flat) |
| Dropout | 0.2 (graph nets, bilinear, cross-attn) |
| Batch size | 16 |
| Early stopping | patience 20 on validation MSE |
| Loss | MSE |

Results are reported as **mean ± std over seeds 41/42/43/32** (the spread is the consistency measure).

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
| Warm | 24,044 | 3,006 | 3,006 |
| Unseen Protein | 24,072 | 2,992 | 2,992 |
| Unseen Drug | 23,868 | 3,094 | 3,094 |
| Unseen Pair | 19,116 | 308 | 308 |

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
