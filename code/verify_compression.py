import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from model import MODEL as Model
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn
from metrics import calculate_metrics
from train import build_graph_cache
import warnings; warnings.filterwarnings("ignore")

def load_pickle(d):
    with open(d, 'rb+') as f: return pickle.load(f)

hp = HyperParameter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_SEED = 42

drug_df = pd.read_csv(hp.drugs_dir); prot_df = pd.read_csv(hp.prots_dir)
mol2vec_dict = load_pickle(hp.mol2vec_dir); protvec_dict = load_pickle(hp.protvec_dir)
contact_map = load_pickle(hp.contact_map)

root = os.path.join(hp.data_root, hp.dataset, hp.running_set)
test_set = CustomDataSet(pd.read_csv(os.path.join(root, 'test.csv')), hp)
dcache, pcache = build_graph_cache(drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map)
collate = lambda x: my_collate_fn(x, device, hp, drug_df, prot_df, mol2vec_dict, protvec_dict, contact_map,
                                  drug_graph_cache=dcache, protein_graph_cache=pcache)
loader = DataLoader(test_set, batch_size=hp.Batch_size, shuffle=False, drop_last=True, num_workers=0, collate_fn=collate)

model = nn.DataParallel(Model(hp, device))
model.load_state_dict(torch.load(f'./savemodel/{hp.dataset}-{hp.running_set}-split{SPLIT_SEED}_new.pth', map_location=device))
model = model.to(device).eval()

preds, labels = [], []
for batch in loader:
    mm, mmk, pm, pmk, dg, pg, aff = batch
    mm, mmk, pm, pmk, dg, pg = [t.to(device) for t in (mm, mmk, pm, pmk, dg, pg)]
    with torch.no_grad():
        p = model(mm, mmk, pm, pmk, dg, pg)
    preds += p.cpu().numpy().reshape(-1).tolist()
    labels += aff.cpu().numpy().reshape(-1).tolist()

preds, labels = np.array(preds), np.array(labels)
mse, ci, rm2 = calculate_metrics(labels, preds)

# linear fit pred ~ a*label + b  (slope a; compression => a well below 1)
a, b = np.polyfit(labels, preds, 1)
corr = np.corrcoef(labels, preds)[0, 1]

print("\n================ COMPRESSION CHECK (cold / unseen_prot, seed 42) ================")
print(f"n = {len(preds)}   (sanity: MSE {mse:.4f} / CI {ci:.4f} / rm2 {rm2:.4f}  -- should match README 0.407/0.838/0.411)")
print(f"\nLABEL  spread:  std={labels.std():.3f}   min={labels.min():.3f}  max={labels.max():.3f}  range={labels.max()-labels.min():.3f}")
print(f"PRED   spread:  std={preds.std():.3f}   min={preds.min():.3f}  max={preds.max():.3f}  range={preds.max()-preds.min():.3f}")
print(f"\n>>> pred_std / label_std = {preds.std()/labels.std():.3f}   (1.0 = no compression, <1 = compressed)")
print(f">>> regression slope pred~label = {a:.3f}   (1.0 ideal; <1 = compressed toward mean)")
print(f">>> correlation = {corr:.3f}")
print(f"\nHow far predictions reach into the binder tail:")
for thr in [6, 7, 8]:
    print(f"  labels>{thr}: {(labels>thr).mean()*100:4.1f}% of set | mean PRED there = {preds[labels>thr].mean():.3f} (true mean {labels[labels>thr].mean():.3f})")
print(f"  labels==floor(5.0): mean PRED there = {preds[labels<=5.01].mean():.3f} (true 5.0)")
# constant-predictor baselines for context
for c, name in [(labels.mean(),'mean'), (5.0,'floor')]:
    print(f"  [ref] constant-{name} predictor MSE = {((labels-c)**2).mean():.4f}")
print("================================================================================\n")
