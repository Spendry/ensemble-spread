"""
Does the collapse reflect a change in the IMPLIED ignorance, or a failure of the
ensemble to sample its own posterior?

For each width w:
  1. Draw M random-init networks. Their output covariance IS the width-w implicit
     prior (the empirical NNGP kernel). No training involved.
  2. Do exact GP regression with that kernel. This gives the CORRECT Bayesian
     posterior std in the hole under the width-w prior. Call it sigma_post(w).
  3. Compare to the measured ensemble spread sigma_ens(w) from the training runs.

READING:
  sigma_post flat  + sigma_ens collapsing -> ensemble fails to sample its own
                                             posterior. Collapse is an ESTIMATOR
                                             ARTIFACT. Supports Sam's claim.
  sigma_post collapsing too                -> the width-w prior really does get
                                             more confident. Collapse is real
                                             prior contraction, not an artifact.
"""
import numpy as np, torch, torch.nn as nn, json
torch.set_num_threads(1)

def data(n, rng, noise=0.05):
    half=n//2
    x=np.concatenate([rng.uniform(-3,-0.8,half), rng.uniform(0.8,3,n-half)])
    y=np.sin(2.5*x)+0.4*x+noise*rng.standard_normal(n)
    return x[:,None].astype(np.float32), y.astype(np.float32)

class Net(nn.Module):
    def __init__(s,w,depth=3):
        super().__init__(); L=[]; d=1
        for _ in range(depth): L+=[nn.Linear(d,w), nn.Tanh()]; d=w
        s.body=nn.Sequential(*L); s.head=nn.Linear(d,1)
    def forward(s,x): return s.head(s.body(x)).squeeze(-1)

rng=np.random.default_rng(0)
Xtr,Ytr=data(2000,rng); yscale=float(np.std(Ytr))
sel=np.random.default_rng(5).choice(len(Xtr),160,replace=False)
Xt=Xtr[sel]; Yt=Ytr[sel]
Xh=np.linspace(-0.6,0.6,40)[:,None].astype(np.float32)
Xi=np.linspace(1.0,2.8,40)[:,None].astype(np.float32)
ALL=torch.tensor(np.vstack([Xt,Xh,Xi])); nt,nh,ni=len(Xt),len(Xh),len(Xi)
NOISE=0.05**2
M=int(__import__('os').environ.get('MDRAWS','400'))

def gp_post(K, mu, Y):
    Ktt=K[:nt,:nt]+NOISE*np.eye(nt)
    L=np.linalg.cholesky(Ktt+1e-8*np.eye(nt))
    def block(a,b):
        Kab=K[a:b,:nt]; Kbb=K[a:b,a:b]
        V=np.linalg.solve(L,Kab.T)
        S=Kbb-V.T@V
        return np.sqrt(np.clip(np.diag(S),0,None))
    return block(nt,nt+nh), block(nt+nh,nt+nh+ni)

print("M =",M);print("width | prior std (hole) | GP POSTERIOR std hole | GP posterior std insup | measured ensemble hole")
ENS={8:0.7409,16:0.4549,32:0.2647,64:0.3487,128:0.1783,256:0.0955}
out=[]
for w in [8,16,32,64,128,256]:
    F=[]
    for s in range(M):
        torch.manual_seed(10000+s)
        m=Net(w)
        with torch.no_grad(): F.append(m(ALL).numpy())
    F=np.stack(F)
    mu=F.mean(0); K=np.cov(F,rowvar=False)
    sh,si=gp_post(K,mu,Yt)
    rec=dict(width=w, prior_hole=float(np.sqrt(np.diag(K)[nt:nt+nh]).mean()/yscale),
             post_hole=float(sh.mean()/yscale), post_insup=float(si.mean()/yscale),
             ens_hole=ENS[w])
    out.append(rec)
    print(f"{w:5d} | {rec['prior_hole']:.4f}          | {rec['post_hole']:.4f}                | "
          f"{rec['post_insup']:.4f}                 | {rec['ens_hole']:.4f}")

w=np.log2([r["width"] for r in out])
for key in ["prior_hole","post_hole","post_insup","ens_hole"]:
    v=np.log2([r[key] for r in out])
    print(f"log2 slope per doubling, {key:11s} = {np.polyfit(w,v,1)[0]:+.3f}")
json.dump(out,open("gp_conv_%d.json"%M,"w"),indent=1)

