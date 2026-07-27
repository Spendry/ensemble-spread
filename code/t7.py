"""
T7: where is the trained ensemble heading?

A wide trained ensemble converges to the NTK-limit predictive distribution. Its
covariance is NOT the NNGP posterior covariance:

  Sigma_NTK = K_hh + T_hX T_XX^-1 K_XX T_XX^-1 T_Xh
                   - (T_hX T_XX^-1 K_Xh + K_hX T_XX^-1 T_Xh)

  Sigma_NNGP = K_hh - K_hX (K_XX + s2 I)^-1 K_Xh      (the correct Bayesian answer)

Both computed from initialization only, no training. If Sigma_NTK stabilizes with
width, its value IS the asymptote of the ensemble spread at every larger width.

READING
  Sigma_NTK settles ABOVE Sigma_NNGP -> no crossing. Ensemble stays conservative
                                       at all widths. T7 kill condition fires.
  Sigma_NTK settles BELOW Sigma_NNGP -> crossing exists. Ensemble becomes
                                       overconfident past some width.
"""
import numpy as np, torch, torch.nn as nn, json, gc, time
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
sel=np.random.default_rng(5).choice(len(Xtr),100,replace=False)
Xt=Xtr[sel]; Yt=Ytr[sel]
Xh=np.linspace(-0.6,0.6,16)[:,None].astype(np.float32)
Xi=np.linspace(1.0,2.8,16)[:,None].astype(np.float32)
ALLnp=np.vstack([Xt,Xh,Xi]); ALL=torch.tensor(ALLnp)
nt,nh,ni=len(Xt),len(Xh),len(Xi); N=nt+nh+ni
S2=0.05**2

def nngp(w, M=800):
    F=[]
    for s in range(M):
        torch.manual_seed(50000+s); m=Net(w)
        with torch.no_grad(): F.append(m(ALL).numpy())
    F=np.stack(F); return np.cov(F,rowvar=False).astype(np.float64)

def ntk(w, seeds=3):
    acc=np.zeros((N,N))
    for sd in range(seeds):
        torch.manual_seed(90000+sd); m=Net(w)
        ps=list(m.parameters())
        G=np.empty((N,sum(p.numel() for p in ps)),dtype=np.float32)
        for i in range(N):
            m.zero_grad()
            g=torch.autograd.grad(m(ALL[i:i+1]).sum(), ps)
            G[i]=torch.cat([v.reshape(-1) for v in g]).numpy()
        acc += (G@G.T).astype(np.float64)
        del G; gc.collect()
    return acc/seeds

def stds(K, T):
    Ktt=K[:nt,:nt]+S2*np.eye(nt)
    Ttt=T[:nt,:nt]+1e-6*np.trace(T[:nt,:nt])/nt*np.eye(nt)
    Kinv=np.linalg.inv(Ktt); Tinv=np.linalg.inv(Ttt)
    out={}
    for name,a,b in [("hole",nt,nt+nh),("insup",nt+nh,N)]:
        Kbb=K[a:b,a:b]; KbX=K[a:b,:nt]; TbX=T[a:b,:nt]
        post = Kbb - KbX@Kinv@KbX.T
        A = TbX@Tinv
        ntk_cov = Kbb + A@K[:nt,:nt]@A.T - (A@KbX.T + KbX@A.T)
        out[name]=(float(np.sqrt(np.clip(np.diag(post),0,None)).mean()/yscale),
                   float(np.sqrt(np.clip(np.diag(ntk_cov),0,None)).mean()/yscale))
    return out

ENS={64:0.3487,128:0.1783,256:0.0955}
print(f"{'w':>6} {'NNGP post hole':>15} {'NTK-limit hole':>15} {'ratio':>7} "
      f"{'NNGP insup':>12} {'NTK insup':>11} {'measured ens':>13}")
res=[]
for w in [32,64,128,256,512,1024]:
    t0=time.time(); K=nngp(w); T=ntk(w); r=stds(K,T)
    rec=dict(width=w, post_hole=r["hole"][0], ntk_hole=r["hole"][1],
             post_insup=r["insup"][0], ntk_insup=r["insup"][1])
    res.append(rec)
    print(f"{w:6d} {rec['post_hole']:15.4f} {rec['ntk_hole']:15.4f} "
          f"{rec['ntk_hole']/rec['post_hole']:7.2f} {rec['post_insup']:12.4f} "
          f"{rec['ntk_insup']:11.4f} {ENS.get(w,float('nan')):13.4f}   [{time.time()-t0:.0f}s]",flush=True)
    del K,T; gc.collect()
json.dump(res,open("t7.json","w"),indent=1)
lw=np.log2([r["width"] for r in res])
for k in ["post_hole","ntk_hole"]:
    print(f"log2 slope {k:10s} = {np.polyfit(lw,np.log2([r[k] for r in res]),1)[0]:+.3f}")
print("last-3 slope ntk_hole =",
      f"{np.polyfit(lw[-3:],np.log2([r['ntk_hole'] for r in res[-3:]]),1)[0]:+.3f}")
