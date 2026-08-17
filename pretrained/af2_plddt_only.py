"""Fetch AlphaFold2 per-residue confidence (pLDDT) WITHOUT touching the contact maps.

Why this exists instead of re-running alphafold2_preprocess.py:

  1. The contact maps on disk are the ones the champion was trained on. Regenerating
     them risks changing them (AlphaFold DB updates, transient network failures
     downgrading a real structure to a backbone chain graph), which would make the
     pLDDT experiment a TWO-variable change instead of one. This script only ever
     reads {dataset}_af2_contact_map.pkl — it never writes it.

  2. alphafold2_preprocess.py has no retry logic: a single SSL hiccup permanently
     downgrades a protein. Here every network call retries with exponential backoff.

  3. Results are cached per protein, so the script is resumable and a re-run only
     retries the proteins that are still missing. Run it as many times as you like.

Usage:
    python pretrained/af2_plddt_only.py --dataset davis          # fetch / resume
    python pretrained/af2_plddt_only.py --dataset davis --report # just show status

Output: pretrained/{dataset}/{dataset}_af2_plddt.pkl   {'plddt': {prot_id: np.ndarray[L]}}
Cache:  pretrained/{dataset}/_plddt_fetch_cache.pkl    (safe to delete to start over)
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphafold2_preprocess import (  # noqa: E402
    ALPHAFOLD_API_URL, UNIPROT_SEARCH_URL, GENE_ALIASES,
    extract_base_gene, parse_organism_suffix, extract_ca_coords,
)

NEUTRAL_PLDDT = 50.0
MAX_ATTEMPTS = 4          # outer attempts, on top of urllib3's own retries
SAVE_EVERY = 10           # checkpoint the cache this often


def make_session() -> requests.Session:
    """Session that retries connection/SSL failures and 5xx with exponential backoff."""
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "af2-plddt-fetch/1.0"})
    return s


def uniprot_lookup(session, gene_name, organism="Homo sapiens"):
    """Gene name -> UniProt accession, retried. Returns accession or None."""
    for reviewed in ("true", "false"):
        params = {
            "query": f"gene_exact:{gene_name} AND organism_name:{organism} AND reviewed:{reviewed}",
            "format": "tsv", "fields": "accession", "size": 1,
        }
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = session.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
                r.raise_for_status()
                lines = r.text.strip().split("\n")
                if len(lines) >= 2 and lines[1].strip():
                    return lines[1].strip()
                break                      # valid empty answer — try the next 'reviewed'
            except Exception as e:
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"      uniprot {gene_name}: gave up after {MAX_ATTEMPTS} attempts ({type(e).__name__})")
                else:
                    time.sleep(2.0 * (attempt + 1))
    return None


def fetch_plddt(session, uniprot_id):
    """Download the AF2 model and return its per-residue pLDDT, retried."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            api = session.get(ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id), timeout=30)
            if api.status_code != 200 or not api.json():
                return None                # genuinely no AF2 entry — retrying won't help
            pdb_url = api.json()[0].get("pdbUrl")
            if not pdb_url:
                return None
            pdb = session.get(pdb_url, timeout=90)
            if pdb.status_code != 200:
                raise IOError(f"HTTP {pdb.status_code}")
            _, plddt = extract_ca_coords(pdb.text)
            return plddt
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                print(f"      af2 {uniprot_id}: gave up after {MAX_ATTEMPTS} attempts ({type(e).__name__})")
            else:
                time.sleep(2.0 * (attempt + 1))
    return None


def resolve(session, prot_id):
    """Mirror alphafold2_preprocess.try_get_contact_map's resolution order so the
    pLDDT comes from the same structure the contact map was built from."""
    base = extract_base_gene(prot_id) or prot_id
    organism = parse_organism_suffix(prot_id)

    for gene, org, label in (
        (prot_id, "Homo sapiens", "af2"),
        (base, "Homo sapiens", f"af2-wildtype({base})") if base != prot_id else (None, None, None),
        (base, organism, f"af2-{organism}") if organism and organism != "Homo sapiens" else (None, None, None),
        (prot_id, organism, f"af2-{organism}") if organism and organism != "Homo sapiens" else (None, None, None),
    ):
        if gene is None:
            continue
        uid = uniprot_lookup(session, GENE_ALIASES.get(gene, gene), org)
        if not uid:
            continue
        plddt = fetch_plddt(session, uid)
        if plddt is not None and len(plddt):
            return plddt, f"{label} [{uid}]"
        time.sleep(0.2)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davis")
    ap.add_argument("--output_root", default="./pretrained")
    ap.add_argument("--report", action="store_true", help="show status and exit")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_root, args.dataset)
    cmap_path = os.path.join(out_dir, f"{args.dataset}_af2_contact_map.pkl")
    cache_path = os.path.join(out_dir, "_plddt_fetch_cache.pkl")
    out_path = os.path.join(out_dir, f"{args.dataset}_af2_plddt.pkl")

    print(f"reading contact maps (READ-ONLY): {cmap_path}")
    with open(cmap_path, "rb") as f:
        contact_map = pickle.load(f)["contact_map"]
    prot_ids = list(contact_map)
    print(f"{len(prot_ids)} proteins\n")

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        done = sum(1 for v in cache.values() if v[0] is not None)
        print(f"resuming from cache: {done} already have real pLDDT, "
              f"{len(prot_ids) - done} to try\n")

    if args.report:
        report(cache, prot_ids)
        return

    session = make_session()
    todo = [p for p in prot_ids if cache.get(p, (None,))[0] is None]
    for i, pid in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {pid}")
        plddt, source = resolve(session, pid)
        if plddt is None:
            cache[pid] = (None, "failed")
            print("    -> no pLDDT (will retry on next run)")
        else:
            L = contact_map[pid].shape[0]
            p = plddt[:L]
            if len(p) < L:                       # AF2 model shorter than the contact map
                p = np.pad(p, (0, L - len(p)), constant_values=NEUTRAL_PLDDT)
            cache[pid] = (p.astype(np.float32), source)
            print(f"    -> {source}  L={L}  mean pLDDT {p.mean():.1f}  "
                  f"<50: {100*(p<50).mean():.0f}%")
        if i % SAVE_EVERY == 0:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        time.sleep(0.2)

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    # Build the final pkl: real pLDDT where we have it, neutral where we don't.
    plddt_dict = {}
    for pid in prot_ids:
        v = cache.get(pid, (None, "failed"))
        plddt_dict[pid] = v[0] if v[0] is not None else np.full(
            contact_map[pid].shape[0], NEUTRAL_PLDDT, dtype=np.float32)
    with open(out_path, "wb") as f:
        pickle.dump({"plddt": plddt_dict}, f)
    print(f"\nwrote {out_path}")
    report(cache, prot_ids)


def report(cache, prot_ids):
    real = [p for p in prot_ids if cache.get(p, (None,))[0] is not None]
    fail = [p for p in prot_ids if p not in real]
    print(f"\n=== pLDDT status ===")
    print(f"  real pLDDT : {len(real)}/{len(prot_ids)} ({100*len(real)/len(prot_ids):.1f}%)")
    print(f"  neutral 50 : {len(fail)}")
    if fail:
        print(f"  still missing: {', '.join(sorted(fail)[:12])}"
              f"{' ...' if len(fail) > 12 else ''}")
        print("  -> re-run this script to retry ONLY these")
    if real:
        allp = np.concatenate([cache[p][0] for p in real])
        print(f"  pooled mean pLDDT {allp.mean():.1f} | {100*(allp<50).mean():.1f}% of residues <50")


if __name__ == "__main__":
    main()
