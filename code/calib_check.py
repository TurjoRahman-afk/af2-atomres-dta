import sys, pickle, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from hyperparameter import HyperParameter
from MyDataset import CustomDataSet, my_collate_fn
import model_pocketcross_gnnprior as M
from model_pocketcross_gnnprior import MODEL
from metrics import calculate_metrics
import warnings; warnings.filterwarnings("ignore")
def lp(d):
    with open(d,'rb+') as f: return pickle.load(f)
hp=HyperParameter(); dev=M.device
drug_df=pd.read_csv(hp.drugs_dir); prot_df=pd.read_csv(hp.prots_dir)
m2=lp(hp.mol2vec_dir); pv=lp(hp.protvec_dir); cm=lp(hp.contact_map)
root=f"{hp.data_root}/{hp.dataset}/{hp.running_set}"
model=MODEL(hp,dev).to(dev)
sd=torch.load('savemodel/davis-unseen_prot-split41_new_gnnprior.pth',map_location=dev,weights_only=False)
model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}); model.eval()
def run(split):
    df=pd.read_csv(f"{root}/{split}.csv")
    ld=DataLoader(CustomDataSet(df,hp),batch_size=16,shuffle=False,drop_last=True,
        collate_fn=lambda x: my_collate_fn(x,dev,hp,drug_df,prot_df,m2,pv,cm))
    P,Y=[],[]
    for b in ld:
        dm,dmk,pm,pmk,dg,pg,aff=b
        dm,dmk,pm,pmk=[t.to(dev) for t in (dm,dmk,pm,pmk)]; dg,pg=dg.to(dev),pg.to(dev)
        with torch.no_grad(): o,_=model(dm,dmk,pm,pmk,dg,pg)
        P+=o.cpu().numpy().reshape(-1).tolist(); Y+=aff.numpy().reshape(-1).tolist()
    return np.array(P),np.array(Y)
pv_,yv=run('valid'); pt,yt=run('test')
np.savez('log/calib_preds_s41.npz',pv=pv_,yv=yv,pt=pt,yt=yt)
a,b=np.polyfit(pv_,yv,1)
print(f"CALIBRATION fit on VALID only:  y = {a:.4f}*pred + {b:+.4f}")
m0,c0,r0=calculate_metrics(yt,pt); m1,c1,r1=calculate_metrics(yt,a*pt+b)
print(f"\n{'':24s}{'MSE':>9s}{'CI':>9s}{'r2m':>9s}")
print(f"{'test  raw':24s}{m0:9.4f}{c0:9.4f}{r0:9.4f}")
print(f"{'test  recalibrated':24s}{m1:9.4f}{c1:9.4f}{r1:9.4f}")
print(f"{'CHANGE':24s}{m1-m0:+9.4f}{c1-c0:+9.4f}{r1-r0:+9.4f}")
print(f"\npred-vs-truth slope on test: raw {np.polyfit(yt,pt,1)[0]:.3f} -> recal {np.polyfit(yt,a*pt+b,1)[0]:.3f}  (1.0 = ideal)")
