"""
AlphaFold2 3D Structure Preprocessing Script
=============================================
Replaces ESM-2 predicted contact maps with real 3D Cα distance matrices
from AlphaFold2 PDB structures. Output pkl format is identical to the
existing contact_map pkl so MyDataset.py and train.py need no changes.

Usage:
    pip install biopython requests
    python pretrained/alphafold2_preprocess.py --dataset davis
    python pretrained/alphafold2_preprocess.py --dataset kiba
    python pretrained/alphafold2_preprocess.py --dataset metz

Output:
    pretrained/{dataset}/{dataset}_af2_contact_map.pkl
    Format: {'contact_map': {prot_id: np.ndarray([L, L])}}
    Values: 1.0 if Cα distance < 8Å, else 0.0  (compatible with target2graph threshold >= 0.5)
"""

import os
import io
import time
import pickle
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    warnings.simplefilter('ignore', PDBConstructionWarning)
except ImportError:
    raise ImportError("Run: pip install biopython")


ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
CONTACT_THRESHOLD_ANGSTROM = 8.0


def gene_name_to_uniprot(gene_name: str, organism: str = "Homo sapiens") -> Optional[str]:
    """Query UniProt REST API to get accession ID from gene name."""
    params = {
        "query": f"gene_exact:{gene_name} AND organism_name:{organism} AND reviewed:true",
        "format": "tsv",
        "fields": "accession",
        "size": 1
    }
    try:
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        if len(lines) >= 2:
            return lines[1].strip()
    except Exception as e:
        print(f"  UniProt lookup failed for {gene_name}: {e}")
    return None


def download_alphafold_pdb(uniprot_id: str) -> Optional[str]:
    """Download AlphaFold2 PDB file content as string."""
    url = ALPHAFOLD_URL.format(uniprot_id=uniprot_id)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"  AlphaFold2 PDB not found for {uniprot_id} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  Download failed for {uniprot_id}: {e}")
    return None


def extract_ca_coords(pdb_string: str) -> Optional[np.ndarray]:
    """Parse PDB string and extract Cα atom coordinates."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_string))
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())
        break  # only first model
    if len(ca_coords) == 0:
        return None
    return np.array(ca_coords)  # [L, 3]


def coords_to_contact_map(ca_coords: np.ndarray, threshold: float = CONTACT_THRESHOLD_ANGSTROM) -> np.ndarray:
    """
    Compute pairwise Cα distance matrix and threshold at given Angstroms.
    Returns binary matrix: 1.0 if distance < threshold, else 0.0.
    Compatible with target2graph() which thresholds at >= 0.5.
    """
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # [L, L, 3]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))         # [L, L]
    contact_map = (dist_matrix < threshold).astype(np.float32)
    return contact_map


def process_dataset(dataset: str, data_root: str = "./datasets", output_root: str = "./pretrained"):
    prot_csv = os.path.join(data_root, dataset, f"{dataset}_prots.csv")
    if not os.path.exists(prot_csv):
        raise FileNotFoundError(f"Protein CSV not found: {prot_csv}")

    prot_df = pd.read_csv(prot_csv)
    prot_ids = prot_df["target_key"].tolist()
    seq_len_map = dict(zip(prot_df["target_key"].astype(str), prot_df["target_sequence"].str.len()))
    print(f"Processing {len(prot_ids)} proteins for dataset: {dataset}")

    contact_map_dict = {}
    failed = []

    for i, prot_id in enumerate(prot_ids):
        print(f"[{i+1}/{len(prot_ids)}] {prot_id}")

        # Step 1: gene name → UniProt accession
        uniprot_id = gene_name_to_uniprot(prot_id)
        if uniprot_id is None:
            print(f"  Could not find UniProt ID for {prot_id}, skipping")
            failed.append(prot_id)
            continue

        print(f"  UniProt: {uniprot_id}")

        # Step 2: Download AlphaFold2 PDB
        pdb_string = download_alphafold_pdb(uniprot_id)
        if pdb_string is None:
            failed.append(prot_id)
            continue

        # Step 3: Extract Cα coordinates
        ca_coords = extract_ca_coords(pdb_string)
        if ca_coords is None or len(ca_coords) == 0:
            print(f"  No Cα atoms found for {prot_id}")
            failed.append(prot_id)
            continue

        print(f"  Extracted {len(ca_coords)} residues")

        # Step 4: Compute contact map
        contact_map = coords_to_contact_map(ca_coords)

        # Align length to sequence from CSV — prevents index errors in target2graph()
        expected_len = seq_len_map.get(str(prot_id))
        if expected_len and contact_map.shape[0] != expected_len:
            min_len = min(contact_map.shape[0], expected_len)
            contact_map = contact_map[:min_len, :min_len]
            print(f"  Length aligned: AF2={len(ca_coords)}, seq={expected_len} → cropped to {min_len}")

        contact_map_dict[str(prot_id)] = contact_map

        time.sleep(0.3)  # polite delay for API

    # Save pkl in same format as existing contact_map pkl
    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_af2_contact_map.pkl")

    with open(out_path, "wb") as f:
        pickle.dump({"contact_map": contact_map_dict}, f)

    print(f"\nSaved {len(contact_map_dict)} contact maps to: {out_path}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
        print("For failed proteins, fall back to ESM-2 contact maps or use ESMFold API.")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="davis",
                        choices=["davis", "kiba", "metz", "bindingDB"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--output_root", type=str, default="./pretrained")
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="Cα distance threshold in Angstroms (default: 8.0)")
    args = parser.parse_args()

    CONTACT_THRESHOLD_ANGSTROM = args.threshold
    out_path = process_dataset(args.dataset, args.data_root, args.output_root)
    print(f"\nDone. Update hyperparameter.py contact_map path to: {out_path}")
