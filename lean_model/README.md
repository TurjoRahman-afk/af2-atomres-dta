# LeanDTA — Path-B lean architecture (self-contained)

This folder is **completely separate** from `code/`. It does not import from or
modify the existing pipeline, so it cannot affect any running training (e.g. Run 7).

## Why this exists

Every run so far (v1, Runs 5–7) shows the same disease: train MSE → 0.06–0.08,
test MSE → 0.22–0.24. That gap is **memorizing, not learning**, caused by far too
much trainable capacity for a 68-drug / 442-protein dataset. Weight decay (Run 7)
over-corrected without helping. The fix is a **right-sized** model that leans on
frozen pretrained embeddings and fixes the specific leaks we found.

## What changed vs the current `code/model.py`

| | Current model | LeanDTA |
|--|--|--|
| Transformers | 2 × 3-layer (FFN 1024) | 2 × **1-layer** (FFN 512) |
| GNNs | 2 × 6-layer GAT, edge features | 2 × **3-layer GIN, no edge features** |
| Fusion paths | cross-attn + LinearAttention + bilinear | **cross-attn + bilinear only** |
| Pooling (seq) | **mean** | **attention pooling** |
| Drug views | graph + ChemBERTa | graph + ChemBERTa + **Morgan FP** |
| Protein graph | Cα distance / RBF | distance edges + **mutation flag** node feature |
| Head | KAN [512→1024→512→1] (~8M) | small MLP `[128→64→1]` (or small KAN) |
| Trainable params | ~12–15M | **~1.3M** |

Evidence behind each choice:
- **No edge features** — KANPM ablation, 3DProtDTA ablation (GINE no better than GIN),
  and our Runs 5–7 all agree they don't help.
- **Attention pooling** — both HCAF (max-pool) and 3DProtDTA (add-pool) beat mean.
- **Morgan FP** — 3DProtDTA ablation: adds non-overlapping info to the drug graph.
- **Mutation flag** — 3DProtDTA encodes it; fixes our 72 wildtype-for-mutant proteins.
- **Add-pool in GNN** — 3DProtDTA's best-pooling finding.

## Files

- `modules.py` — building blocks: `AttentionPool`, `GraphEncoder` (GIN),
  `DrugEncoder`, `ProteinEncoder`, `CrossInteraction`, `BilinearFusion`.
- `model.py` — `LeanDTA`, assembling the blocks. Run it directly for a shape /
  param-count sanity check: `python model.py`.
- `featurize.py` — the two new inputs: `morgan_fingerprint()` and the
  mutation-flag parser (`parse_mutation_positions`, `mutation_flag_vector`).
- `dataset.py` — `build_caches()` (drug graphs, Morgan FPs, mutation-aware
  protein graphs) and `lean_collate_fn()`. Reuses code/MyDataset's tested
  `smile2graph` / `target2graph` via a read-only import (nothing in code/ changes).
- `train.py` — full training loop. Reads the same data files as the existing
  pipeline; writes its own logs/checkpoints.

## Forward signature

```python
out = model(
    drug_tokens,    # [B, L_d, 384]  ChemBERTa per-token
    drug_mask,      # [B, L_d]       1 = real, 0 = pad
    drug_graph,     # PyG Batch: x [N_atoms, 88]
    drug_fp,        # [B, 2048]      Morgan fingerprint
    prot_residues,  # [B, L_p, 1152] ESM-C per-residue
    prot_mask,      # [B, L_p]       1 = real, 0 = pad
    prot_graph,     # PyG Batch: x [N_res, 1153] = ESM-C(1152) + mutation flag(1)
)   # -> [B] predicted affinity
```

## How to train

It's fully wired. The two new inputs (Morgan FP, mutation flag) are built
automatically inside `build_caches()`. Just run:

```bash
cd lean_model
python train.py
```

- Reads the same data as `code/train.py` (davis warm split42, the ChemBERTa /
  ESM-C / AF2 pkls). **No need to regenerate splits** (seed unchanged).
- Writes `lean_model/log/davis-warm-lean-split42.csv` (7-column per-epoch log)
  and `lean_model/savemodel/davis-warm-lean-split42.pth` (best checkpoint).
- Resumable: re-running picks up from the checkpoint.
- **Wait for Run 7 to finish first** — both want the GPU.

### Config (edit in `train.py` / inherited from `code/hyperparameter.py`)
- `dim=128`, `n_heads=4`, dropout `0.3`, `use_kan=False`
- Adam lr `1e-4`, **weight_decay `1e-5`** (1e-4 over-regularized in Run 7)
- batch 16, early-stop patience 20

## Quick check (no training)

```bash
cd lean_model
python model.py
# prints: Trainable parameters: 1.43M  and  Output shape: torch.Size([4])
```

## Verified
- `python model.py` → 1.43M params, output `[4]` ✓
- Morgan FP (2048-bit) and mutation parser (`ABL1(T315I)`→315→node 314; ignores
  gene names like `CSNK1G2` and tags like `phosphorylated`) ✓
- `dataset.py` + `train.py` import cleanly ✓

## Note
`build_caches` reuses `target2graph` from code/, which still returns the RBF
edge features — LeanDTA simply ignores them (it uses `x` + `edge_index` only,
per the "edge features don't help" finding). The mutation flag is appended to
each protein node, giving node dim 1153 = ESM-C 1152 + 1.
