"""Does the learned PocketPrior actually find real kinase ATP-binding sites?
Davis is 100% kinases. The ATP pocket is anchored by the canonical catalytic motifs
HRD (catalytic loop) and DFG (activation-loop start), which sit ~15-45 residues apart.
Test: is pocket_logit elevated in a window around those motifs vs elsewhere?"""
import re, pickle, torch, numpy as np, pandas as pd
from hyperparameter import HyperParameter
from MyDataset import target2graph
from torch_geometric.data import Data, Batch
import model_pocketcross_gnnprior as M
from model_pocketcross_gnnprior import MODEL
import warnings; warnings.filterwarnings("ignore")

def lp(d):
    with open(d,'rb+') as f: return pickle.load(f)
hp=HyperParameter(); dev=M.device
prot_df=pd.read_csv(hp.prots_dir); pv=lp(hp.protvec_dir); cm=lp(hp.contact_map)['contact_map']
model=MODEL(hp,dev).to(dev)
sd=torch.load('savemodel/davis-unseen_prot-split41_new_gnnprior.pth',map_location=dev,weights_only=False)
model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}); model.eval()

def find_catalytic(seq):
    """Canonical kinase: HRD ... DFG, DFG appearing 15-45 aa after HRD."""
    hits=[]
    for h in [m.start() for m in re.finditer('HRD', seq)]:
        for d in [m.start() for m in re.finditer('DFG', seq)]:
            if 15 <= d-h <= 45: hits.append((h,d))
    return hits

WIN=8   # +/- residues around each motif counted as "pocket region"
rows=[]; used=0
for _,r in prot_df.iterrows():
    pid=str(r['target_key']); seq=str(r['target_sequence'])
    if pid not in cm or pid not in pv['mat_dict']: continue
    hits=find_catalytic(seq)
    if len(hits)!=1: continue                      # unambiguous kinases only
    h,d=hits[0]
    _,tf,tei,ew=target2graph(cm[pid], pv['mat_dict'][pid])
    g=Batch.from_data_list([Data(x=tf,edge_index=tei,edge_weight=ew)]).to(dev)
    with torch.no_grad():
        _,remb,rmask=model.protein_graph_model(g)
        pl=model.pocket(remb)[0].cpu().numpy()     # [1200], +1 BOS offset
        mk=rmask[0].cpu().numpy()
    valid=np.where(mk)[0]
    if len(valid)<100: continue
    site=set()
    for pos in (h,d):
        for k in range(pos-WIN,pos+WIN+1):
            idx=k+1                                # +1 BOS offset
            if idx in valid: site.add(idx)
    site=np.array(sorted(site))
    if len(site)<10: continue
    other=np.setdiff1d(valid,site)
    z=(pl[site].mean()-pl[other].mean())/ (pl[other].std()+1e-9)
    rows.append((pid,len(valid),len(site),pl[site].mean(),pl[other].mean(),z))
    used+=1
    if used>=60: break

df=pd.DataFrame(rows,columns=['prot','n_res','n_site','site_mean','other_mean','z'])
print(f"proteins with unambiguous HRD...DFG catalytic motif: {len(df)}")
print(f"\nmean pocket_logit at ATP-site region : {df.site_mean.mean():+.4f}")
print(f"mean pocket_logit elsewhere          : {df.other_mean.mean():+.4f}")
print(f"mean difference (site - elsewhere)   : {(df.site_mean-df.other_mean).mean():+.4f}")
print(f"mean z-score (in units of protein-wise std): {df.z.mean():+.3f}")
print(f"proteins where site scored HIGHER    : {(df.site_mean>df.other_mean).sum()} / {len(df)}  "
      f"({(df.site_mean>df.other_mean).mean()*100:.0f}%)   [50% = chance]")
from scipy import stats
t,p=stats.ttest_rel(df.site_mean,df.other_mean)
print(f"paired t-test: t={t:.3f}, p={p:.2e}")
print(f"\nverdict: {'FINDS the ATP site' if p<0.05 and df.z.mean()>0.2 else 'NO reliable relation to the ATP site'}")
