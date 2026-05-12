"""
Quick patch — fixes 3 remaining alias failures and removes 3 non-human proteins
from the existing davis_af2_contact_map.pkl without re-running all 442 proteins.
"""
import io
import pickle
import time
import requests
import numpy as np

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    warnings.simplefilter('ignore', PDBConstructionWarning)
except ImportError:
    raise ImportError("Run: pip install biopython")

ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
CONTACT_THRESHOLD = 8.0

# Proteins to REMOVE (non-human organisms — no AF2 exists)
TO_REMOVE = {"PFPK5(Pfalciparum)", "PFCDPK1(Pfalciparum)", "PKNB(Mtuberculosis)"}

# Proteins to FIX with correct gene name
TO_FIX = {
    "IKK-epsilon": "IKBKE",
    "PFTAIRE2":    "CDK15",
    "ABL1p":       "ABL1",
}

PKL_PATH = "./pretrained/davis/davis_af2_contact_map.pkl"


def gene_name_to_uniprot(gene_name):
    params = {
        "query": f"gene_exact:{gene_name} AND organism_name:Homo sapiens AND reviewed:true",
        "format": "tsv", "fields": "accession", "size": 1
    }
    resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
    lines = resp.text.strip().split("\n")
    return lines[1].strip() if len(lines) >= 2 else None


def download_af2_pdb(uniprot_id):
    resp = requests.get(ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id), timeout=15)
    if resp.status_code != 200 or not resp.json():
        return None
    pdb_url = resp.json()[0].get("pdbUrl")
    if not pdb_url:
        return None
    r = requests.get(pdb_url, timeout=30)
    return r.text if r.status_code == 200 else None


def pdb_to_contact_map(pdb_string):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", io.StringIO(pdb_string))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
        break
    if not coords:
        return None
    ca = np.array(coords)
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return (dist < CONTACT_THRESHOLD).astype(np.float32)


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    cmap_dict = data["contact_map"]

    print(f"Loaded {len(cmap_dict)} proteins from pkl.")

    # Step 1: Remove non-human proteins
    for prot_id in TO_REMOVE:
        if prot_id in cmap_dict:
            del cmap_dict[prot_id]
            print(f"REMOVED {prot_id}")

    # Step 2: Fix alias proteins
    for prot_id, gene_name in TO_FIX.items():
        print(f"Fixing {prot_id} → {gene_name}")
        uniprot_id = gene_name_to_uniprot(gene_name)
        if not uniprot_id:
            print(f"  Could not find UniProt ID for {gene_name}")
            continue
        print(f"  UniProt: {uniprot_id}")
        pdb = download_af2_pdb(uniprot_id)
        if not pdb:
            print(f"  No AF2 PDB found")
            continue
        cmap = pdb_to_contact_map(pdb)
        if cmap is None:
            print(f"  Could not extract Cα coords")
            continue
        # Crop to existing map size if needed
        old_len = cmap_dict[prot_id].shape[0] if prot_id in cmap_dict else cmap.shape[0]
        min_len = min(cmap.shape[0], old_len)
        cmap_dict[prot_id] = cmap[:min_len, :min_len]
        print(f"  → af2 ({cmap.shape[0]} residues, stored {min_len})")
        time.sleep(0.3)

    with open(PKL_PATH, "wb") as f:
        pickle.dump({"contact_map": cmap_dict}, f)

    print(f"\nDone. {len(cmap_dict)} proteins saved to {PKL_PATH}")
    print("Removed 3 non-human proteins. Fixed 3 alias proteins.")


if __name__ == "__main__":
    main()
