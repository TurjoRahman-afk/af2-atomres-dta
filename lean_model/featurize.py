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


def is_mutant(target_key):
    """True if the target is a mutant variant rather than the base/wildtype entry.

    Detects point mutations inside parentheses (e.g. 'ABL1(T315I)') and the common
    Davis non-point variants (del / ins / ITD / dup). Plain gene names ('CSNK1G2')
    and the phosphorylation tags are NOT treated as mutations.
    """
    if parse_mutations(target_key):
        return True
    for paren in re.findall(r"\(([^)]*)\)", str(target_key)):
        low = paren.lower()
        if any(k in low for k in ("del", "ins", "itd", "dup")):
            return True
    return False


def mutation_flag_vector(target_key, num_nodes):
    """Global mutation flag (3DProtDTA-style).

    Every node gets the SAME value: 1.0 if the protein is a mutant variant, else
    0.0. This avoids position-level mapping entirely — there is no per-residue
    index to get wrong — and is complementary to the ESM-C node features, which
    already differ at the mutated position because they come from the mutant
    sequence. Returns [num_nodes, 1] float32.
    """
    val = 1.0 if is_mutant(target_key) else 0.0
    return np.full((num_nodes, 1), val, dtype=np.float32)
