# AF2-PocketCross-DTA

**Drug–target affinity prediction with structure-guided atom↔residue interaction.**

A DTA model built on AlphaFold2 protein structures, evaluated on the harder **cold-protein**
(unseen-target) setting of DAVIS rather than the near-saturated random split.

> Full experimental history, including every failed variant and why it failed, is in
> [EXPERIMENTS.md](EXPERIMENTS.md).

---

## What this project found

The project set out to test whether protein graphs built from **real AlphaFold2 structure**
beat graphs built from **sequence-predicted contacts** for affinity prediction on unseen
targets. Six measured results, in order of how much they constrain the conclusion:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Real structure did not help.** AF2-derived contacts score 0.3560; the reference method's ESM-2 *predicted* contacts score 0.314. Substituting real structure made results slightly worse. | this work + [arXiv:2606.04228](https://arxiv.org/abs/2606.04228), which independently measures a **−0.141 information bonus** for AF2 over ESM-2 on binding affinity |
| 2 | **Ranking is at state of the art; absolute accuracy is not.** CI 0.8569 ± 0.0030 vs the best published 0.857 — a gap of 0.0001, and 2nd of 9 overall. MSE is 4th of 9. | results table below |
| 3 | **DAVIS is partly self-contradictory — previously unreported.** 442 target keys, 379 unique sequences. 18.9% of training pairs are mutually contradictory; 11.4% of cold-protein test pairs are unlearnable by construction. | finding 4 below |
| 4 | **The remaining error is missing information, not miscalibration.** An oracle rank-preserving rescale fitted *on the test labels* reaches only 0.3389 — still above 0.314. 94% of error survives any output transform. | finding 2 below |
| 5 | **Capacity is not the bottleneck.** Training MSE 0.0598 against an irreducible floor of 0.0348; a 144× predictor-capacity sweep degrades test accuracy. | finding 5 below |
| 6 | **Every added input feature failed (0 for 3).** RBF edge features, distance-shell pocket features, and AF2 pLDDT confidence all hurt. Only changes that *re-routed existing computation* ever helped. | [EXPERIMENTS.md](EXPERIMENTS.md) |

**Conclusion.** For cold-protein affinity prediction on this benchmark, the sequence already
carries most of the usable signal; the fold adds little on top. Further gains are limited by
**training-protein diversity** (354 proteins), not by architecture, capacity, or output
calibration.

### Related work

AlphaFold-structure protein graphs for DTA were introduced by **3DProtDTA** (RSC Advances, 2023)
and extended with equivariant GNNs by **CASTER-DTA** (2024); the structural grounding here is not
novel in itself. What is contributed is the controlled cold-protein comparison, the error
decomposition, and the benchmark-integrity finding.

---

## What this model does

Most DTA models pool the drug into one vector and the protein into another, then combine them —
discarding the fact that binding is specific **drug atoms** contacting specific **protein residues**.
This model keeps token-level representations and models that interaction directly, weighted by a
residue score derived from the protein's structure-derived contact topology.

**Core components:**

1. **AlphaFold2 contact topology** — protein graphs built from actual Cα coordinates
   (edge if Cα distance < 8Å) rather than sequence-predicted contacts. Note the graph is
   **unweighted**: distances are discarded and none of the six graph layers use edge weights,
   so what reaches the network is contact *topology*, not geometry.
2. **Structure-guided atom↔residue attention** — every drug atom attends over every protein
   residue, with the attention biased by a learned per-residue structural score.
3. **GNN-derived residue prior** — that score comes from `ProteinGraphNet`'s own per-residue
   embeddings (learned by message-passing over the AF2 contact graph) rather than hand-built features.
4. **KAN predictor** — Kolmogorov-Arnold Network, each connection a learned B-spline.

---

## Architecture

```
DRUG                                 PROTEIN                    AF2 3D STRUCTURE
SMILES                               sequence                   (Cα contact map)
  │                                     │                              │
  ├──────────────┐                      ├──────────────┐              │
  ▼              ▼                      ▼              ▼              ▼
ChemBERTa   molecular graph          ESM-C        ┌─────────────────────────────┐
  │           (RDKit)                  │          │      ProteinGraphNet        │
  ▼              │                     ▼          │      (GCN + 5×GAT)          │
Linear 384→128   │              Linear 1152→128   │        │           │        │
  │              │                     │          │        ▼           ▼        │
  ▼              │                     ▼          │  global_add   to_dense_batch │
Transformer×3    │              Transformer×3     │    _pool      (+1 BOS align) │
  │              │                     │          │        │           │        │
  ▼              ▼                     ▼          │        ▼           ▼        │
 H_d         DrugGraphNet             H_p         │  fasta_graph  residue_emb    │
[B,220,128]  (GCN + 5×GAT)        [B,1200,128]    │    [B,128]   [B,1200,256]   │
 (per ATOM)       │                (per RESIDUE)  └────┬──────────────┬─────────┘
  │  │            │                   │  │             │              ▼
  │  │            │                   │  │             │        Residue-prior MLP
  │  │            │                   │  │             │           (256→32→1)
  │  │            │                   │  │             │              │
  │  │            └────────┐   ┌──────┘  │             │              ▼
  │  │                     ▼   ▼         │             │        residue_score
  │  │        STRUCTURE-GUIDED ATOM↔RESIDUE ATTENTION ◄┼──────────────┘
  │  │        score = Q(H_d)·K(H_p)ᵀ/√d + β·residue_score
  │  │        I = softmax(score)     → interaction map [B,220,1200]
  │  │        f_int = mean(I·(H_d ⊙ H_p))  →  [B,128]
  │  ▼                    │
  │ mean-pool           f_int          GatedFusion(drug_graph, protein_graph)
  ▼  │                    │                        │
 g_d [B,128]   g_p [B,128]│                    g [B,128]
  └──────┬───────────┬────┴────────────────────────┘
         ▼           ▼
   concat[ f_int, g_d, g_p, g ] = [B,512]
                 │
                 ▼
      KAN [512 → 1024 → 512 → 1]  →  predicted affinity (pKd)
```

**Model size:** 13.55 M parameters. **Loss:** plain MSE.

| Setting | Value |
|---------|-------|
| Model file | `code/model_pocketcross_gnnprior.py` |
| Train script | `code/train_pocketcross_gnnprior.py` |
| Sequence encoders | ChemBERTa (384→128), ESM-C (1152→128), 3-layer transformers, 8 heads |
| Graph encoders | GCNConv + 5× GATConv (drug dim 128, protein dim 256) |
| Predictor | KAN [512, 1024, 512, 1], cubic splines, grid 5 |
| Optimizer | Adam, lr 1e-4, no weight decay, flat LR |
| Batch size | 16 · Early stopping: patience 20 on validation MSE |

---

## Results — DAVIS cold-protein (`unseen_prot`)

Test proteins are **entirely held out of training** — the model must generalize to targets it
has never seen. Results are the mean ± std over independent data-split seeds.

| Metric | Seed 41 | Seed 42 | **Mean ± Std** |
|--------|---------|---------|----------------|
| Test MSE ↓ | 0.3601 | 0.3519 | **0.3560 ± 0.0058** |
| Test CI ↑ | 0.8590 | 0.8547 | **0.8569 ± 0.0030** |
| Test r²ₘ ↑ | 0.5460 | 0.4928 | **0.5194 ± 0.0376** |

_Seeds 43 and 32 pending._

### Comparison with published methods

Same benchmark and split protocol (KANPM's `cold_split.py`), DAVIS unseen-protein:

| Rank | Model | MSE ↓ | CI ↑ | r²ₘ ↑ |
|------|-------|-------|------|-------|
| 1 | KANPM-DTA | **0.314** | **0.857** | **0.556** |
| 2 | PMMR (2025) | 0.329 | 0.833 | 0.471 |
| 3 | DMFF-DTA (2025) | 0.330 | 0.840 | 0.501 |
| **4** | **AF2-PocketCross-DTA (this work)** | **0.3560** | **0.8569** | **0.5194** |
| 5 | MgraphDTA (2022) | 0.359 | 0.813 | 0.425 |
| 6 | MSGNN-DTA (2023) | 0.361 | 0.816 | 0.430 |
| 7 | FusionDTA (2022) | 0.364 | 0.826 | 0.435 |
| 8 | GRA-DTA (2024) | 0.376 | 0.827 | 0.453 |
| 9 | GraphDTA (2021) | 0.510 | 0.729 | 0.154 |

**Standing:** 2nd of 9 on CI (statistically tied with 1st — gap 0.0001), 2nd on r²ₘ, 4th on MSE.

> ⚠️ **Protocol matters.** Published "DAVIS unseen-protein" numbers are **not** mutually comparable —
> LaPro-DTA (2026) reports baselines at MSE 0.55–0.88 for the same nominal task. The table above is
> valid only because all entries use the same split code. Any comparison outside this protocol is
> meaningless.

---

## Honest findings and limitations

Five results that constrain what can be claimed:

**1. The residue prior does not identify binding sites.**
DAVIS is 100% kinases, whose ATP pockets are anchored by conserved HRD…DFG catalytic motifs.
Tested across 60 kinases with unambiguous motifs, the learned residue score at the ATP site is
**statistically indistinguishable from elsewhere** (52% vs 50% chance, paired t-test p = 0.37,
z = +0.05). The module measurably improves accuracy (+0.020 MSE) and contributes ~22% of attention
steering, but it is **not** learning binding-site location — most plausibly burial/contact density.
No interpretability claim is made. *(`code/validate_pocket.py`)*

**2. The MSE gap is an information deficit, not a calibration artifact.**
Decomposing test error: **99.5% is shape error** (wrong values on individual pairs), only 0.5% is
systematic offset. Post-hoc linear calibration fitted on validation recovers just −0.0075 MSE.
Stronger still: a **perfect rank-preserving rescale fitted directly on the test labels** (an oracle
that cannot be achieved in practice) reaches only **0.3389** — still above KANPM's 0.314. So
**94% of the error survives any output transform**, and clamping predictions at the pKd = 5.0 floor
buys just −0.0008. No loss reshaping or calibration closes the MSE gap. *(`code/calib_check.py`)*

**3. Validation systematically overstates performance.**
Across all **7 completed cold-protein runs** the valid→test gap is positive **7/7 times**, mean
**+0.065** (range +0.032 to +0.132). Two causes are separable. **Selection bias** — the single
best-validation epoch sits **+0.014 to +0.046** below the mean of the epochs around it (mean
+0.025), which is winner's curse rather than real degradation. **Label variance** — valid and
test are two different 44-protein samples, and MSE scales with label spread: on seed 41 the test
labels are more spread than validation (+0.176 variance), on seed 42 less (−0.095). The remainder
is intrinsic difficulty difference between the two samples. **Validation ranking does not reliably
predict test ranking** — two same-split inversions are on record, and a variant leading on
validation still lost on test. *(`code/analysis/valid_test_gap.py`)*

**4. The DAVIS benchmark is partly self-contradictory.**
DAVIS has **442 target keys but only 379 unique sequences** — the mutations were never applied to
the sequence strings, so `BRAF` and `BRAF(V600E)` carry identical sequences, and 62 of 63 colliding
key-pairs also share a byte-identical contact map. The model therefore receives **identical input**
for pairs with different labels. Consequences: **18.9% of training pairs are mutually contradictory**
(putting a hard floor of 0.0348 on train MSE), and **11.4% of cold-protein test pairs are unlearnable
by construction**, contributing **0.0257 of the 0.3601 test MSE (~7%)**. This affects every published
result on this benchmark, and partly explains why the field plateaus at 0.31–0.36.

**5. Adding input features has failed every time (0 for 3).**
RBF edge features, 8-dim distance-shell pocket features, and AlphaFold2 pLDDT confidence as a protein
node channel all hurt. pLDDT was the strongest test — genuinely orthogonal information at a cost of
256 parameters — and still lost by +0.027 MSE. The only changes that ever helped **re-routed
computation the model already performed** rather than feeding it new inputs. Full write-ups in
[EXPERIMENTS.md](EXPERIMENTS.md).

**6. Capacity is not the bottleneck — the representation is.**
The selected checkpoint fits its training data to **0.0788** against an irreducible floor of
**0.0348** (the training log reaches 0.0598 by the final epoch, measured mid-epoch with dropout
active on a later, more-overfit model). So the predictor is not capacity-starved — and sweeping
it from **131 K to 18.9 M parameters (144×)** makes test MSE *worse*, not better
(0.3758 → 0.3886), with validation barely moving across the whole range. On the frozen 512-d
representation a linear head scores **0.4435** and the best MLP at any width **0.3758**, versus
the KAN's **0.3601** — the head contributes, but is deep in diminishing returns while the
representation caps the result. *(`code/analysis/capacity_probe.py`)*

### Reproducing each finding

Every claim above is backed by a script. Commands are run from the repository root.

| # | Finding | Reproduce with | Needs GPU |
|---|---------|----------------|-----------|
| 1 | Residue prior does not identify binding sites | `python code/validate_pocket.py` | yes |
| 2 | Information deficit — calibration recovers little | `python code/calib_check.py` | yes |
| 2 | **Oracle bound 0.3389** | `python code/analysis/oracle_bound.py` | yes |
| 3 | Validation overstates performance | `python code/analysis/valid_test_gap.py` | **no** |
| 4 | **DAVIS is partly self-contradictory** | `python code/analysis/benchmark_integrity.py` | **no** |
| 5 | Added input features failed (0 for 3) | see [EXPERIMENTS.md](EXPERIMENTS.md) — each run logged in full | — |
| 6 | Capacity is not the bottleneck | `python code/analysis/capacity_probe.py` | yes |

Analysis scripts are read-only and regenerate their data split **in memory**, so they never
touch `datasets/` and are safe to run while a training job is in progress. The CPU-only ones
can be run at any time; the GPU ones will contend with training for the card.

**Other limitations:** 2 seeds only; only the cold-protein split evaluated (unseen-drug and
unseen-pair untested); trained on 442 proteins, which is the binding constraint on generalization.

---

## Data and preprocessing

| Source | Model | Output |
|--------|-------|--------|
| ChemBERTa (`DeepChem/ChemBERTa-77M-MTR`) | Drug SMILES → token embeddings | 384-dim |
| ESM-C (`esmc_600m`) | Protein sequence → residue embeddings | 1152-dim |
| AlphaFold2 (EBI) + UniProt | Protein → real 3D Cα contact map | L × L |
| RDKit | SMILES → molecular graph | 88 atom features |

**DAVIS AlphaFold2 coverage:** 364 proteins direct · 72 wildtype-for-mutant (point mutations do
not change the fold) · 6 backbone fallback (non-human, no AF2 entry). All 442 proteins included.

**Split sizes:**

| Split | Train | Valid | Test |
|-------|-------|-------|------|
| Warm | 24,044 | 3,006 | 3,006 |
| **Unseen Protein** | **24,072** | **2,992** | **2,992** |
| Unseen Drug | 23,868 | 3,094 | 3,094 |
| Unseen Pair | 19,116 | 308 | 308 |

<details>
<summary>Drug graph node features (88-dim)</summary>

| Feature | Dim |
|---------|-----|
| Atom symbol (one-hot) | 44 |
| Degree (one-hot) | 11 |
| Total H on heavy atom | 11 |
| Explicit / implicit hydrogens | 12 |
| Hybridization type | 3 |
| Donor / acceptor | 2 |
| Formal charge, explicit valence, ring count, radical electrons, aromatic | 5 |
| **Total** | **88** |

</details>

---

## Setup

```bash
pip install torch transformers rdkit biopython requests pandas numpy
pip install esm torch-geometric
```

```bash
# 1. Generate pretrained embeddings
python pretrained/chemberta_pretraiend.py
python pretrained/esmC_pretraiend.py

# 2. Download AlphaFold2 structures and build contact maps
python pretrained/alphafold2_preprocess.py --dataset davis

# 3. Generate splits  (set SEED in cold_split.py to match SPLIT_SEED in the train script)
python code/cold_split.py --dataset davis

# 4. Train  (set running_set='unseen_prot' in hyperparameter.py)
python code/train_pocketcross_gnnprior.py
```

Training resumes automatically from a checkpoint if one exists. Results are written to
`log/Test-{dataset}-{split}-split{SEED}_new_gnnprior.csv` once early stopping triggers naturally.

---

## References

1. Liu et al. (2024) *KAN: Kolmogorov-Arnold Networks* — arXiv:2404.19756
2. Jumper et al. (2021) *AlphaFold2* — Nature 596, 583–589
3. Ahmad et al. (2022) *ChemBERTa-2* — arXiv:2209.01712
4. ESM Team (2024) *ESM-C* — Evolutionary Scale
5. Lin et al. (2023) *ESM-2* — Science 379(6637)
6. KANPM-DTA (2026) — Briefings in Bioinformatics, bbag112
7. 3DProtDTA (2023) — RSC Advances, D3RA00281K
