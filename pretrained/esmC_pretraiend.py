import pickle
from tqdm import tqdm
import pandas as pd
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESMC.from_pretrained("esmc_600m").to("cuda").eval()

MAX_SEQ_LENGTH = 1200

def get_esmc_pretrain(model, df_dir, db_name):
    df = pd.read_csv(df_dir)

    emb_dict, emb_mat_dict, length_target = {}, {}, {}

    for prot_id, seq in tqdm(zip(df['target_key'], df['target_sequence']), total=len(df)):
        prot_id = str(prot_id)
        seq = str(seq)[:MAX_SEQ_LENGTH]
        protein = ESMProtein(sequence=seq)

        with torch.no_grad():
            logits_output = model.logits(
                model.encode(protein),
                LogitsConfig(sequence=True, return_embeddings=True)
            )

        reps = logits_output.embeddings[0].cpu().numpy()

        emb_mat_dict[prot_id] = reps
        emb_dict[prot_id] = reps.mean(axis=0)
        length_target[prot_id] = len(seq)

    with open(f'./pretrained/{db_name}/{db_name}_esmc_pretrain.pkl', 'wb') as f:
        pickle.dump({
            "dataset": db_name,
            "vec_dict": emb_dict,
            "mat_dict": emb_mat_dict,
            "length_dict": length_target
        }, f)

    print(f"Saved {db_name} ESM-C features.")


db_names = ['davis', 'kiba', 'metz']
df_dirs = [
    './datasets/davis/davis_prots.csv',
    './datasets/kiba/kiba_prots.csv',
    './datasets/metz/metz_prots.csv',
]

for db_name, df_dir in zip(db_names, df_dirs):
    try:
        print(f'Computing {db_name} protein embeddings with ESM-C.')
        get_esmc_pretrain(model, df_dir, db_name)
    except FileNotFoundError:
        print(f'Skipping {db_name} — CSV not found at {df_dir}')
