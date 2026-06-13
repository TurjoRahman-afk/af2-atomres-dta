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

## What is NOT done yet (the integration step)

This is the **model only**. To train it you still need two data additions, which
touch the data pipeline and are intentionally left out of this folder:

1. **Morgan fingerprints** — compute a 2048-bit Morgan FP per drug (RDKit
   `AllChem.GetMorganFingerprintAsBitVect`) and feed `drug_fp`.
2. **Mutation flag** — append a 1-bit per-residue flag to the protein graph node
   features (1 at mutated positions, 0 elsewhere). Davis encodes mutants in the
   target name, e.g. `ABL1(T315I)`.

Once Run 7 finishes, the next step is wiring these into a copy of the data
loader + a training script. Ask and I'll build that part.

## Quick check

```bash
cd lean_model
python model.py
# prints: Trainable parameters: ~1.3M  and  Output shape: torch.Size([4])
```
