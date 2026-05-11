from tqdm import tqdm
import pickle
import torch
import pandas as pd
from transformers import AutoTokenizer, RobertaModel

def get_chem_pretrain(df_dir, db_name, max_smiles_length=220):
    df = pd.read_csv(df_dir)

    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()

    embeddings = {
        "dataset": db_name,
        "vec_dict": {},
        "mat_dict": {},
        "length_dict": {}
    }

    for drug_id, smile in tqdm(zip(df['drug_key'], df['compound_iso_smiles']),
                               total=len(df), desc="Processing SMILES"):
        drug_id = str(drug_id)
        smile = str(smile)[:max_smiles_length]

        with torch.no_grad():
            outputs = model(**tokenizer(smile, return_tensors='pt'))
            embeddings_out = outputs.last_hidden_state[0][1:-1]

        embeddings["mat_dict"][drug_id] = embeddings_out
        embeddings["vec_dict"][drug_id] = embeddings_out.mean(axis=0)
        embeddings["length_dict"][drug_id] = len(smile)

    output_path = f'./pretrained/{db_name}/{db_name}_chem_pretrained.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings for {len(embeddings['vec_dict'])} compounds to {output_path}")


db_names = ['davis', 'kiba', 'metz']
df_dirs = [
    './datasets/davis/davis_drugs.csv',
    './datasets/kiba/kiba_drugs.csv',
    './datasets/metz/metz_drugs.csv',
]

for db_name, df_dir in zip(db_names, df_dirs):
    try:
        print(f'Computing {db_name} drug embeddings with ChemBERTa.')
        get_chem_pretrain(df_dir, db_name)
    except FileNotFoundError:
        print(f'Skipping {db_name} — CSV not found at {df_dir}')
