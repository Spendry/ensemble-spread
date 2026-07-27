"""
T3: Capacity scaling of representational divergence under controlled identifiability.

CONSTRUCTION
  Training inputs x in R^8, x ~ N(0,I) EXCEPT x2 := x1 (perfect collinearity).
  Target: y = 2*x1 + 0*x2 + 1.5*x3 - 1.0*x4 + 0.8*sin(2*x5) + 0.5*x6*x7 + eps

  On the training manifold, 2*x1 + 0*x2 == a*x1 + (2-a)*x2 for ALL a.
  => direction u_NI = (1,-1,0..)/sqrt2 is PROVABLY unconstrained by the data.
  => direction u_S  = (1, 1,0..)/sqrt2 is pinned at 2/sqrt2.
  => direction e_3                     is pinned at 1.5.

  This gives a referent with a known non-identifiable component and known
  identifiable components, established by construction rather than by measurement.

MEASUREMENT
  Train k seeds per width. At OFF-manifold probe points (x1,x2 drawn
  independently), take the Jacobian of each model and project onto the three
  directions. Across-seed dispersion of each projection = divergence.

PREDICTIONS
  PRH  : all divergences fall with width.
  CBR  : identifiable divergence falls; non-identifiable divergence flat or rising.
  K3   : fires if NI divergence falls at same normalized rate as ID.
"""
import numpy as np, torch, torch.nn as nn, json, time, math

torch.set_num_threads(1)
D = 8
SQ2 = math.sqrt(2.0)

def target(x):
    return (2.0*x[:,0] + 0.0*x[:,1] + 1.5*x[:,2] - 1.0*x[:,3]
            + 0.8*np.sin(2.0*x[:,4]) + 0.5*x[:,5]*x[:,6])

def make_train(n, rng, noise=0.05):
    x = rng.standard_normal((n, D))
    x[:,1] = x[:,0]                      # collinearity -> non-identifiability
    y = target(x) + noise*rng.standard_normal(n)
    return x.astype(np.float32), y.astype(np.float32)

def make_offmanifold(n, rng):
    """Probe points where x1 and x2 are independent: leaves the training manifold."""
    x = rng.standard_normal((n, D))
    return x.astype(np.float32)

class MLP(nn.Module):
    def __init__(self, w, depth=2):
        super().__init__()
        layers, d = [], D
        for _ in range(depth):
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)
    def feats(self, x):
        return self.body(x)

def train_one(w, seed, wd, Xtr, Ytr, steps=1200, depth=2):
    torch.manual_seed(seed)
    m = MLP(w, depth)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=wd)
    X = torch.tensor(Xtr); Y = torch.tensor(Ytr)
    n = X.shape[0]; bs = 512
    g = torch.Generator().manual_seed(seed + 99991)
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        loss = ((m(X[idx]) - Y[idx])**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        train_mse = float(((m(X) - Y)**2).mean())
    return m, train_mse

def jac_projections(m, Xprobe):
    """Mean Jacobian over probe points, projected onto the three directions."""
    X = torch.tensor(Xprobe, requires_grad=True)
    out = m(X)
    grad = torch.autograd.grad(out.sum(), X)[0].detach().numpy()
    g = grad.mean(axis=0)
    return dict(
        NI  = float((g[0] - g[1]) / SQ2),   # unconstrained by data
        SUM = float((g[0] + g[1]) / SQ2),   # pinned at 2/sqrt2 = 1.4142
        E3  = float(g[2]),                  # pinned at 1.5
        E8  = float(g[7]),                  # CONTROL: identifiable, pinned at 0
        norm= float(np.linalg.norm(g)),
    )

def linear_cka(A, B):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    num = np.linalg.norm(A.T @ B, 'fro')**2
    den = np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro')
    return float(num / den) if den > 0 else float('nan')

# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
Xtr, Ytr = make_train(3000, rng)
Xte, Yte = make_train(2000, np.random.default_rng(1))          # on-manifold holdout
Xoff     = make_offmanifold(1500, np.random.default_rng(2))    # off-manifold probes
Xoff_t   = torch.tensor(Xoff)

WIDTHS = [8, 16, 32, 64, 128, 256]
SEEDS  = list(range(16))
REGIMES = {"wd0": 0.0, "wd1e-3": 1e-3}

results = []
t0 = time.time()
for rname, wd in REGIMES.items():
    for w in WIDTHS:
        preds, feats, projs, mses = [], [], [], []
        for s in SEEDS:
            m, tr_mse = train_one(w, s, wd, Xtr, Ytr)
            m.eval()
            with torch.no_grad():
                te_mse = float(((m(torch.tensor(Xte)) - torch.tensor(Yte))**2).mean())
                p_off = m(Xoff_t).numpy()
                f_off = m.feats(Xoff_t).numpy()
            projs.append(jac_projections(m, Xoff))
            preds.append(p_off); feats.append(f_off); mses.append((tr_mse, te_mse))
        preds = np.stack(preds)
        ckas = [linear_cka(feats[i], feats[j])
                for i in range(len(SEEDS)) for j in range(i+1, len(SEEDS))]
        rec = dict(
            regime=rname, width=w,
            train_mse=float(np.mean([a for a,_ in mses])),
            test_mse_onmanifold=float(np.mean([b for _,b in mses])),
            NI_mean =float(np.mean([p["NI"]  for p in projs])),
            NI_std  =float(np.std ([p["NI"]  for p in projs], ddof=1)),
            SUM_mean=float(np.mean([p["SUM"] for p in projs])),
            SUM_std =float(np.std ([p["SUM"] for p in projs], ddof=1)),
            E3_mean =float(np.mean([p["E3"]  for p in projs])),
            E3_std  =float(np.std ([p["E3"]  for p in projs], ddof=1)),
            E8_mean =float(np.mean([p["E8"]  for p in projs])),
            E8_std  =float(np.std ([p["E8"]  for p in projs], ddof=1)),
            NI_all  =[p["NI"]  for p in projs],
            E8_all  =[p["E8"]  for p in projs],
            E3_all  =[p["E3"]  for p in projs],
            SUM_all =[p["SUM"] for p in projs],
            offman_pred_std=float(preds.std(axis=0, ddof=1).mean()),
            cka_mean=float(np.mean(ckas)),
        )
        results.append(rec)
        print(f"{rname:7s} w={w:4d} trMSE={rec['train_mse']:.4f} "
              f"teMSE={rec['test_mse_onmanifold']:.4f} | "
              f"NIstd={rec['NI_std']:.4f} E8std={rec['E8_std']:.4f} SUMstd={rec['SUM_std']:.4f} "
              f"E3 std={rec['E3_std']:.4f} | CKA={rec['cka_mean']:.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)

json.dump(results, open("t3_results_v2.json","w"), indent=1)
print("done", time.time()-t0)
