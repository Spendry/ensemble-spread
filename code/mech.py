"""
Mechanism probe for epistemic uncertainty collapse with width.

CANDIDATE: hole divergence is carried by KERNEL RANDOMNESS. A width-w net has a
random empirical kernel whose fluctuation falls as 1/sqrt(w). Seeds with different
kernels extrapolate differently where data does not constrain them. As w grows,
kernels concentrate, priors agree, extrapolations agree.

CONDITIONS (all measure across-seed std of prediction in the hole):
  RICH_full   different init + different batch order   (baseline)
  RICH_sameinit  SAME init, different batch order      -> optimizer path noise only
  RICH_sameorder different init, SAME batch order      -> init+kernel only
  LAZY_frozen    body frozen at init, train head only  -> pure random-feature kernel
                 regression; divergence here is kernel randomness with NO feature
                 learning at all.

ALSO: relative NTK drift ||Theta_end - Theta_init||_F / ||Theta_init||_F vs width.
If the mechanism is laziness, drift should fall with width and track the gap
between RICH and LAZY.
"""
import numpy as np, torch, torch.nn as nn, json, time, math
torch.set_num_threads(1)

def data(n, rng, noise=0.05):
    half = n//2
    x = np.concatenate([rng.uniform(-3,-0.8,half), rng.uniform(0.8,3,n-half)])
    y = np.sin(2.5*x) + 0.4*x + noise*rng.standard_normal(n)
    return x[:,None].astype(np.float32), y.astype(np.float32)

class Net(nn.Module):
    def __init__(s, w, depth=3):
        super().__init__(); L=[]; d=1
        for _ in range(depth): L += [nn.Linear(d,w), nn.Tanh()]; d=w
        s.body = nn.Sequential(*L); s.head = nn.Linear(d,1)
    def forward(s,x): return s.head(s.body(x)).squeeze(-1)

def ntk_gram(m, X):
    """Empirical NTK on probe points: Theta_ij = grad_theta f(x_i) . grad_theta f(x_j)."""
    G=[]
    for i in range(X.shape[0]):
        m.zero_grad()
        out = m(X[i:i+1])
        g = torch.autograd.grad(out.sum(), [p for p in m.parameters() if p.requires_grad],
                                retain_graph=False)
        G.append(torch.cat([v.reshape(-1) for v in g]))
    G = torch.stack(G)
    return (G @ G.T).detach().numpy()

def train(w, seed, Xtr, Ytr, frozen=False, order_seed=None, init_seed=None, steps=1200):
    torch.manual_seed(init_seed if init_seed is not None else seed)
    m = Net(w)
    if frozen:
        for p in m.body.parameters(): p.requires_grad=False
    params=[p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    X=torch.tensor(Xtr); Y=torch.tensor(Ytr)
    g = torch.Generator().manual_seed(order_seed if order_seed is not None else seed+7)
    for _ in range(steps):
        idx = torch.randint(0,len(X),(256,),generator=g)
        opt.zero_grad(); ((m(X[idx])-Y[idx])**2).mean().backward(); opt.step()
    with torch.no_grad(): tr=float(((m(X)-Y)**2).mean())
    return m, tr

rng=np.random.default_rng(0)
Xtr,Ytr = data(2000,rng)
yscale = float(np.std(Ytr))
hole  = torch.tensor(np.linspace(-0.6,0.6,200)[:,None].astype(np.float32))
insup = torch.tensor(np.linspace(1.0,2.8,200)[:,None].astype(np.float32))
probe = torch.tensor(np.linspace(-3,3,40)[:,None].astype(np.float32))

WIDTHS=[8,16,32,64,128,256]; NSEED=8
CONDS = {
 "RICH_full"     : dict(frozen=False, fix_init=False, fix_order=False),
 "RICH_sameinit" : dict(frozen=False, fix_init=True,  fix_order=False),
 "RICH_sameorder": dict(frozen=False, fix_init=False, fix_order=True),
 "LAZY_frozen"   : dict(frozen=True,  fix_init=False, fix_order=False),
}
out=[]; t0=time.time()
for cname,cfg in CONDS.items():
    for w in WIDTHS:
        ph,pi,trs,drifts=[],[],[],[]
        for s in range(NSEED):
            isd = 1234 if cfg["fix_init"] else s
            osd = 555  if cfg["fix_order"] else s+7
            if cname=="RICH_full":                       # measure NTK drift here only
                torch.manual_seed(isd); m0=Net(w); K0=ntk_gram(m0,probe)
            m,tr = train(w,s,Xtr,Ytr,frozen=cfg["frozen"],order_seed=osd,init_seed=isd)
            if cname=="RICH_full":
                K1=ntk_gram(m,probe)
                drifts.append(float(np.linalg.norm(K1-K0)/np.linalg.norm(K0)))
            m.eval()
            with torch.no_grad():
                ph.append(m(hole).numpy()); pi.append(m(insup).numpy())
            trs.append(tr)
        ph=np.stack(ph); pi=np.stack(pi)
        rec=dict(cond=cname,width=w,train_mse=float(np.mean(trs)),
                 hole=float(ph.std(0,ddof=1).mean()/yscale),
                 insup=float(pi.std(0,ddof=1).mean()/yscale),
                 ntk_drift=float(np.mean(drifts)) if drifts else None)
        out.append(rec)
        d = f" drift={rec['ntk_drift']:.3f}" if rec['ntk_drift'] is not None else ""
        print(f"{cname:15s} w={w:4d} trMSE={rec['train_mse']:.4f} "
              f"HOLE={rec['hole']:.4f} insup={rec['insup']:.4f}{d} [{time.time()-t0:.0f}s]",flush=True)
json.dump(out,open("mech.json","w"),indent=1)

# scaling fits
print("\nSCALING FITS  hole_div ~ a + b*w^(-alpha), and pure power law exponent")
from scipy.optimize import curve_fit
for cname in CONDS:
    rs=[r for r in out if r["cond"]==cname]
    w=np.array([r["width"] for r in rs],float); h=np.array([r["hole"] for r in rs])
    al,_=np.polyfit(np.log(w),np.log(h),1)
    try:
        p,_=curve_fit(lambda x,a,b,al: a+b*x**(-al), w,h,p0=[0.05,1.0,0.5],maxfev=20000)
        print(f"  {cname:15s} power-law alpha={-al:.3f} | floor a={p[0]:.4f} exponent={p[2]:.3f}")
    except Exception as e:
        print(f"  {cname:15s} power-law alpha={-al:.3f} | floor fit failed")
print("done",time.time()-t0)
