"""
New input features for LeanDTA (not present in the original pipeline):
  - Morgan fingerprints for drugs (2048-bit)
  - per-residue mutation flag for the protein graph (validated against sequence)
"""

import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


def morgan_fingerprint(smiles, n_bits=2048, radius=2):
    """2048-bit Morgan (ECFP4) fingerprint as a float32 vector."""
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    ConvertToNumpyArray(fp, arr)
    return arr


def parse_mutations(target_key):
    """Davis encodes point mutations in parentheses, e.g. 'ABL1(T315I)'.

    We only look INSIDE parentheses, so gene names ('CSNK1G2') and tags
    ('phosphorylated') are never mistaken for mutations.

    Returns a list of (wildtype_aa, position, mutant_aa) tuples, where position
    is 1-indexed. The wildtype letter lets us VALIDATE the position against the
    actual sequence (see mutation_flag_vector).
    """
    muts = []
    for paren in re.findall(r"\(([^)]*)\)", str(target_key)):
        for m in re.finditer(r"([A-Za-z])(\d+)([A-Za-z])", paren):
            muts.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return muts


def parse_mutation_positions(target_key):
    """Convenience: just the 1-indexed positions."""
    return [pos for _, pos, _ in parse_mutations(target_key)]


def mutation_flag_vector(target_key, num_nodes, sequence=None):
    """1-bit-per-residue flag aligned to protein graph nodes.

    Protein graph node i corresponds to residue (i+1) after the ESM BOS/EOS
    strip, so a mutation at 1-indexed position P maps to node index P-1.

    Robustness: if a `sequence` is given, we only set the flag when the residue
    at that index actually matches the wildtype letter from the mutation string
    (e.g. the 'T' of 'T315I'). If it doesn't match, the numbering is misaligned
    (truncated / renumbered structure), so we skip rather than flag the wrong
    residue. Without a sequence, we fall back to the raw P-1 mapping.

    Returns [num_nodes, 1] float32.
    """
    flag = np.zeros((num_nodes, 1), dtype=np.float32)
    for wt, pos, _mut in parse_mutations(target_key):
        idx = pos - 1
        if not (0 <= idx < num_nodes):
            continue                              # mutation outside the graph
        if sequence is not None and idx < len(sequence):
            if sequence[idx].upper() != wt:
                continue                          # numbering mismatch -> don't trust it
        flag[idx, 0] = 1.0
    return flag
