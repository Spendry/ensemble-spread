"""
C2: geometric non-identifiability (support hole).
C3: rotational non-identifiability (latent basis in a bottleneck autoencoder).
"""
import numpy as np, torch, torch.nn as nn, json, time, math
torch.set_num_threads(1)

def linear_cka(A,B):
    A=A-A.mean(0,keepdims=True); B=B-B.mean(0,keepdims=True)
    num=np.linalg.norm(A.T@B,'fro')**2
    den=np.linalg.norm(A.T@A,'fro')*np.linalg.norm(B.T@B,'fro')
    return float(num/den) if den>0 else float('nan')

# ===================== C2: SUPPORT HOLE =====================
# y = sin(2.5x) + 0.4x + noise, trained on [-3,-0.8] U [0.8,3].
# HOLE  [-0.6,0.6]  : provably unconstrained by data (no sample support)
# INSUP [1.0,2.8]   : constrained
def c2_data(n, rng):
    half=n//2
    x=np.concatenate([rng.uniform(-3,-0.8,half), rng.uniform(0.8,3,n-half)])
    y=np.sin(2.5*x)+0.4*x+0.05*rng.standard_normal(n)
    return x[:,None].astype(np.float32), y.astype(np.float32)

class MLP1(nn.Module):
    def __init__(s,w,din=1,depth=3):
        super().__init__(); L=[];d=din
        for _ in range(depth): L+=[nn.Linear(d,w),nn.Tanh()]; d=w
        s.body=nn.Sequential(*L); s.head=nn.Linear(d,1)
    def forward(s,x): return s.head(s.body(x)).squeeze(-1)

def run_c2():
    rng=np.random.default_rng(0)
    Xtr,Ytr=c2_data(2000,rng)
    hole=np.linspace(-0.6,0.6,300)[:,None].astype(np.float32)
    insup=np.linspace(1.0,2.8,300)[:,None].astype(np.float32)
    yscale=float(np.std(Ytr))
    out=[]
    for w in [8,16,32,64,128,256]:
        ph,pi=[],[]
        for s in range(12):
            torch.manual_seed(s); m=MLP1(w)
            opt=torch.optim.Adam(m.parameters(),lr=3e-3)
            X=torch.tensor(Xtr);Y=torch.tensor(Ytr)
            g=torch.Generator().manual_seed(s+7)
            for _ in range(1500):
                idx=torch.randint(0,len(X),(256,),generator=g)
                opt.zero_grad(); ((m(X[idx])-Y[idx])**2).mean().backward(); opt.step()
            m.eval()
            with torch.no_grad():
                ph.append(m(torch.tensor(hole)).numpy())
                pi.append(m(torch.tensor(insup)).numpy())
                tr=float(((m(X)-Y)**2).mean())
        ph=np.stack(ph); pi=np.stack(pi)
        rec=dict(width=w, train_mse=tr,
                 hole_div=float(ph.std(0,ddof=1).mean()/yscale),
                 insup_div=float(pi.std(0,ddof=1).mean()/yscale),
                 hole_seeds=ph.std(0,ddof=1).mean(), )
        out.append(rec)
        print(f"C2 w={w:4d} trMSE={tr:.4f} | HOLE div={rec['hole_div']:.4f} "
              f"IN-SUPPORT div={rec['insup_div']:.4f} | ratio={rec['hole_div']/rec['insup_div']:.1f}x",flush=True)
    return out

# ===================== C3: ROTATIONAL =====================
# x = A z + noise, z in R^k. Bottleneck AE with latent dim k.
# Reconstruction pins the k-dim SUBSPACE. Nothing pins the BASIS inside it.
# Rashomon set = O(k) orbit.
class AE(nn.Module):
    def __init__(s,d,k,w):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(d,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,k))
        s.dec=nn.Sequential(nn.Linear(k,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,d))
    def forward(s,x): return s.dec(s.enc(x))

def principal_angles_align(A,B):
    """1 - mean cos of principal angles between column spaces. 0 = same subspace."""
    Qa,_=np.linalg.qr(A-A.mean(0,keepdims=True))
    Qb,_=np.linalg.qr(B-B.mean(0,keepdims=True))
    sv=np.linalg.svd(Qa.T@Qb,compute_uv=False)
    return float(1-np.clip(sv,0,1).mean())

def basis_mismatch(A,B):
    """How far the optimal rotation is from identity. 0 = same basis, ~1.4 = unrelated."""
    A=(A-A.mean(0,keepdims=True)); B=(B-B.mean(0,keepdims=True))
    A=A/ (np.linalg.norm(A,axis=0,keepdims=True)+1e-9)
    B=B/ (np.linalg.norm(B,axis=0,keepdims=True)+1e-9)
    U,_,Vt=np.linalg.svd(A.T@B); R=U@Vt
    k=A.shape[1]
    return float(np.linalg.norm(R-np.eye(k),'fro')/math.sqrt(2*k))

def run_c3():
    d,k=16,4
    rng=np.random.default_rng(3)
    A=rng.standard_normal((d,k))/math.sqrt(k)
    Z=rng.standard_normal((4000,k))
    X=(Z@A.T+0.05*rng.standard_normal((4000,d))).astype(np.float32)
    Xp=(rng.standard_normal((800,k))@A.T).astype(np.float32)
    out=[]
    for w in [16,32,64,128,256]:
        lat=[];rec_mse=[]
        for s in range(12):
            torch.manual_seed(s); m=AE(d,k,w)
            opt=torch.optim.Adam(m.parameters(),lr=3e-3)
            T=torch.tensor(X); g=torch.Generator().manual_seed(s+11)
            for _ in range(1500):
                idx=torch.randint(0,len(T),(256,),generator=g)
                opt.zero_grad(); ((m(T[idx])-T[idx])**2).mean().backward(); opt.step()
            m.eval()
            with torch.no_grad():
                lat.append(m.enc(torch.tensor(Xp)).numpy())
                rec_mse.append(float(((m(T)-T)**2).mean()))
        n=len(lat); pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
        cka =np.mean([linear_cka(lat[i],lat[j]) for i,j in pairs])
        sub =np.mean([principal_angles_align(lat[i],lat[j]) for i,j in pairs])
        bas =np.mean([basis_mismatch(lat[i],lat[j]) for i,j in pairs])
        rec=dict(width=w,recon_mse=float(np.mean(rec_mse)),cka=float(cka),
                 subspace_misalign=float(sub),basis_mismatch=float(bas))
        out.append(rec)
        print(f"C3 w={w:4d} recMSE={rec['recon_mse']:.4f} | CKA={cka:.4f} "
              f"SUBSPACE misalign={sub:.4f} (identifiable) BASIS mismatch={bas:.4f} (non-identif.)",flush=True)
    return out

t0=time.time()
c2=run_c2(); print()
c3=run_c3()
json.dump(dict(c2=c2,c3=c3),open("t3_c2c3.json","w"),indent=1)
print("\ndone",time.time()-t0)
