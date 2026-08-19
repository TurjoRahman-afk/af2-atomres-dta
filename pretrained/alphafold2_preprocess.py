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
import re
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


ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
CONTACT_THRESHOLD_ANGSTROM = 8.0

# DAVIS uses informal aliases that don't match UniProt gene names
GENE_ALIASES = {
    "IKK-alpha":  "CHUK",
    "IKK-beta":   "IKBKB",
    "IKK-gamma":  "IKBKG",
    "p38-alpha":  "MAPK14",
    "p38-beta":   "MAPK11",
    "p38-gamma":  "MAPK12",
    "p38-delta":  "MAPK13",
    "ERK1":       "MAPK3",
    "ERK2":       "MAPK1",
    "JNK1":       "MAPK8",
    "JNK2":       "MAPK9",
    "JNK3":       "MAPK10",
    "S6K1":       "RPS6KB1",
    "S6K2":       "RPS6KB2",
    "MRCKA":      "CDC42BPA",
    "MRCKB":      "CDC42BPB",
    "PKAC-alpha": "PRKACA",
    "PKAC-beta":  "PRKACB",
    "MST1":       "STK4",
    "MST2":       "STK3",
    "HPK1":       "MAP4K1",
    "GCK":        "MAP4K2",
    "KHS1":       "MAP4K5",
    "HGK":        "MAP4K4",
    "MINK1":      "MINK1",
    "TNIK":       "TNIK",
}

# Organism name suffixes used in DAVIS protein IDs → full UniProt organism name
ORGANISM_SUFFIX_MAP = {
    "Pfalciparum":   "Plasmodium falciparum",
    "Mtuberculosis": "Mycobacterium tuberculosis",
    "Hsapiens":      "Homo sapiens",
}


def gene_name_to_uniprot(gene_name: str, organism: str = "Homo sapiens") -> Optional[str]:
    """Query UniProt REST API to get accession ID from gene name."""
    # Try reviewed (SwissProt) first, then unreviewed (TrEMBL) for non-model organisms
    for reviewed in ("true", "false"):
        params = {
            "query": f"gene_exact:{gene_name} AND organism_name:{organism} AND reviewed:{reviewed}",
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
            print(f"  UniProt lookup failed for {gene_name} ({organism}): {e}")
        if reviewed == "true":
            time.sleep(0.1)
    return None


def download_alphafold_pdb(uniprot_id: str) -> Optional[str]:
    """Query AlphaFold API to get the correct PDB URL, then download it."""
    api_url = ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id)
    try:
        api_resp = requests.get(api_url, timeout=15)
        if api_resp.status_code != 200 or not api_resp.json():
            print(f"  No AlphaFold2 entry for {uniprot_id}")
            return None
        pdb_url = api_resp.json()[0].get("pdbUrl")
        if not pdb_url:
            print(f"  No pdbUrl in AlphaFold2 response for {uniprot_id}")
            return None
        resp = requests.get(pdb_url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"  PDB download failed for {uniprot_id} (HTTP {resp.status_code})")
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
    """Compute pairwise Cα distance matrix. Store actual distances for contacts, 0.0 for non-contacts."""
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # [L, L, 3]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))         # [L, L] actual Angstrom distances
    contact_mask = dist_matrix < threshold
    dist_matrix[~contact_mask] = 0.0                        # zero out non-contacts
    return dist_matrix.astype(np.float32)                   # actual distances, 0 = no contact


def backbone_fallback_map(seq_len: int) -> np.ndarray:
    """Sequential backbone contact map for proteins where AlphaFold2 is unavailable.
    Connects each residue to itself and its immediate neighbors — always valid."""
    mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        mat[i, i] = 1.0
        if i > 0:
            mat[i, i - 1] = 1.0
        if i < seq_len - 1:
            mat[i, i + 1] = 1.0
    return mat


def extract_base_gene(prot_id: str) -> Optional[str]:
    """Extract base gene name from mutant/domain variants.
    e.g. EGFR(L858RT790M) -> EGFR, RSK1(KinDom.1-N-terminal) -> RSK1"""
    base = re.split(r'[\(\-]', prot_id)[0].strip()
    return base if base != prot_id else None


def _fetch_af2(gene_key: str, cache: dict) -> Optional[np.ndarray]:
    """Download AF2 for gene_key, cache result. Returns full contact map or None."""
    if gene_key in cache:
        return cache[gene_key]
    # Resolve alias → official gene name if needed
    lookup_name = GENE_ALIASES.get(gene_key, gene_key)
    uniprot_id = gene_name_to_uniprot(lookup_name)
    if not uniprot_id:
        cache[gene_key] = None
        return None
    pdb_string = download_alphafold_pdb(uniprot_id)
    if not pdb_string:
        time.sleep(0.3)
        cache[gene_key] = None
        return None
    ca_coords = extract_ca_coords(pdb_string)
    if ca_coords is None or len(ca_coords) == 0:
        cache[gene_key] = None
        return None
    cmap = coords_to_contact_map(ca_coords)
    cache[gene_key] = cmap
    time.sleep(0.2)
    return cmap


def parse_organism_suffix(prot_id: str) -> Optional[str]:
    """Extract organism from suffix like PFPK5(Pfalciparum) → 'Plasmodium falciparum'."""
    match = re.search(r'\(([^)]+)\)', prot_id)
    if match:
        suffix = match.group(1)
        return ORGANISM_SUFFIX_MAP.get(suffix)
    return None


def _fetch_af2_organism(gene_key: str, organism: str, cache: dict) -> Optional[np.ndarray]:
    """Download AF2 for gene_key in a specific organism."""
    cache_key = f"{gene_key}|{organism}"
    if cache_key in cache:
        return cache[cache_key]
    lookup_name = GENE_ALIASES.get(gene_key, gene_key)
    uniprot_id = gene_name_to_uniprot(lookup_name, organism)
    if not uniprot_id:
        cache[cache_key] = None
        return None
    pdb_string = download_alphafold_pdb(uniprot_id)
    if not pdb_string:
        time.sleep(0.3)
        cache[cache_key] = None
        return None
    ca_coords = extract_ca_coords(pdb_string)
    if ca_coords is None or len(ca_coords) == 0:
        cache[cache_key] = None
        return None
    cmap = coords_to_contact_map(ca_coords)
    cache[cache_key] = cmap
    time.sleep(0.2)
    return cmap


def try_get_contact_map(prot_id: str, seq_len: int, cache: dict) -> tuple:
    """Try AlphaFold2 download with alias and organism resolution. Returns (contact_map, source_label).
    Every protein gets a contact map — backbone fallback is the guaranteed last resort."""
    base = extract_base_gene(prot_id) or prot_id  # strips mutation/domain suffix
    organism_from_suffix = parse_organism_suffix(prot_id)

    # 1. Try exact protein name with human AF2 first
    cmap = _fetch_af2(prot_id, cache)
    if cmap is not None:
        min_len = min(cmap.shape[0], seq_len)
        return cmap[:min_len, :min_len], "af2"

    # 2. Try base gene (strip mutation/domain suffix) with human AF2
    if base != prot_id:
        cmap = _fetch_af2(base, cache)
        if cmap is not None:
            min_len = min(cmap.shape[0], seq_len)
            return cmap[:min_len, :min_len], f"af2-wildtype({base})"

    # 3. Try organism-specific AF2 if protein ID contains organism suffix
    if organism_from_suffix and organism_from_suffix != "Homo sapiens":
        cmap = _fetch_af2_organism(base, organism_from_suffix, cache)
        if cmap is not None:
            min_len = min(cmap.shape[0], seq_len)
            return cmap[:min_len, :min_len], f"af2-{organism_from_suffix}"
        # Also try with full prot_id as gene name
        cmap = _fetch_af2_organism(prot_id, organism_from_suffix, cache)
        if cmap is not None:
            min_len = min(cmap.shape[0], seq_len)
            return cmap[:min_len, :min_len], f"af2-{organism_from_suffix}"

    # 4. Guaranteed fallback — backbone sequential connectivity (all proteins included)
    return backbone_fallback_map(seq_len), "backbone-fallback"


def process_dataset(dataset: str, data_root: str = "./datasets", output_root: str = "./pretrained"):
    prot_csv = os.path.join(data_root, dataset, f"{dataset}_prots.csv")
    if not os.path.exists(prot_csv):
        raise FileNotFoundError(f"Protein CSV not found: {prot_csv}")

    prot_df = pd.read_csv(prot_csv)
    prot_ids = prot_df["target_key"].tolist()
    seq_len_map = dict(zip(prot_df["target_key"].astype(str), prot_df["target_sequence"].str.len()))
    print(f"Processing {len(prot_ids)} proteins for dataset: {dataset}")

    contact_map_dict = {}
    stats = {"af2": 0, "wildtype": 0, "fallback": 0}
    base_gene_cache = {}  # avoid re-downloading wildtype for multiple mutants

    for i, prot_id in enumerate(prot_ids):
        print(f"[{i+1}/{len(prot_ids)}] {prot_id}")
        seq_len = seq_len_map.get(str(prot_id), 1200)

        cmap, source = try_get_contact_map(str(prot_id), seq_len, base_gene_cache)
        contact_map_dict[str(prot_id)] = cmap
        print(f"  → {source} ({cmap.shape[0]} residues)")

        if source == "af2":
            stats["af2"] += 1
        elif "wildtype" in source:
            stats["wildtype"] += 1
        else:
            stats["fallback"] += 1

        time.sleep(0.2)

    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_af2_contact_map.pkl")

    with open(out_path, "wb") as f:
        pickle.dump({"contact_map": contact_map_dict}, f)

    print(f"\nSaved {len(contact_map_dict)}/{len(prot_ids)} contact maps to: {out_path}")
    print(f"  Real AF2 structures : {stats['af2']}")
    print(f"  Wildtype fallback   : {stats['wildtype']}")
    print(f"  Backbone fallback   : {stats['fallback']}")
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
    print(f"\nDone. Contact maps saved to: {out_path}")
