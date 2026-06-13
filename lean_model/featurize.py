"""
New input features for LeanDTA (not present in the original pipeline):
  - Morgan fingerprints for drugs (2048-bit)
  - per-residue mutation flag for the protein graph
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


def parse_mutation_positions(target_key):
    """Davis encodes point mutations in parentheses, e.g. 'ABL1(T315I)'.

    We only look INSIDE parentheses so gene names ('CSNK1G2') and tags
    ('phosphorylated') are never mistaken for mutations. Returns 1-indexed
    residue positions.
    """
    positions = []
    for paren in re.findall(r"\(([^)]*)\)", str(target_key)):
        for m in re.finditer(r"([A-Za-z])(\d+)([A-Za-z])", paren):
            positions.append(int(m.group(2)))
    return positions


def mutation_flag_vector(target_key, num_nodes):
    """1-bit-per-residue flag aligned to protein graph nodes.

    Protein graph node i corresponds to residue (i+1) after the ESM BOS/EOS
    strip, so a mutation at 1-indexed position P maps to node index P-1.
    Returns [num_nodes, 1] float32.
    """
    flag = np.zeros((num_nodes, 1), dtype=np.float32)
    for pos in parse_mutation_positions(target_key):
        idx = pos - 1
        if 0 <= idx < num_nodes:
            flag[idx, 0] = 1.0
    return flag
