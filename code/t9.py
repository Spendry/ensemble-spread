exec(open('t7.py').read().split('ENS={')[0])
import numpy as np, json
"""T9: does the in-support gap track observation noise (noise-fitting), or does it
come from the NTK kernel differing in SHAPE from the NNGP kernel?
Note: if Theta = c*K exactly, the NTK expression collapses to the NNGP posterior.
So any gap under matched ridge measures shape mismatch, not noise."""
res=[]
for w in [256,1024]:
    K=nngp(w); T=ntk(w); Ktt=K[:nt,:nt]; Ttt=T[:nt,:nt]
    sc=np.trace(Ttt)/np.trace(Ktt)
    # shape mismatch: correlation of the two kernels after unit-diagonal normalization
    dk=np.sqrt(np.diag(K)); dt=np.sqrt(np.diag(T))
    Kn=K/np.outer(dk,dk); Tn=T/np.outer(dt,dt)
    iu=np.triu_indices(N,1)
    print(f"\n=== w={w} ===  kernel shape corr(K_norm, Theta_norm) = "
          f"{np.corrcoef(Kn[iu],Tn[iu])[0,1]:.4f}   "
          f"mean|K_n - T_n| = {np.abs(Kn[iu]-Tn[iu]).mean():.4f}")
    print(f"{'noise s2':>10} {'hole post':>10} {'hole NTK':>9} {'ratio':>6} | "
          f"{'insup post':>11} {'insup NTK':>10} {'ratio':>6}")
    for mult in [0.1,0.3,1.0,3.0,10.0,30.0]:
        s2=S2*mult
        Ki=np.linalg.inv(Ktt+s2*np.eye(nt)); Ti=np.linalg.inv(Ttt+s2*sc*np.eye(nt))
        row={}
        for name,a,b in [("hole",nt,nt+nh),("insup",nt+nh,N)]:
            Kbb=K[a:b,a:b]; KbX=K[a:b,:nt]; A=T[a:b,:nt]@Ti
            po=np.sqrt(np.clip(np.diag(Kbb-KbX@Ki@KbX.T),0,None)).mean()/yscale
            nk=np.sqrt(np.clip(np.diag(Kbb+A@Ktt@A.T-(A@KbX.T+KbX@A.T)),0,None)).mean()/yscale
            row[name]=(po,nk)
        print(f"{s2:10.5f} {row['hole'][0]:10.4f} {row['hole'][1]:9.4f} "
              f"{row['hole'][1]/row['hole'][0]:6.2f} | {row['insup'][0]:11.4f} "
              f"{row['insup'][1]:10.4f} {row['insup'][1]/row['insup'][0]:6.2f}",flush=True)
        res.append(dict(width=w,s2=s2,hole_ratio=row['hole'][1]/row['hole'][0],
                        insup_ratio=row['insup'][1]/row['insup'][0]))
    del K,T; gc.collect()
json.dump(res,open("t9.json","w"),indent=1)
r=[x for x in res if x["width"]==1024]
lo=[x for x in r if abs(x["s2"]-S2*0.1)<1e-9][0]; hi=[x for x in r if abs(x["s2"]-S2*10)<1e-9][0]
print(f"\n100x noise change (w=1024): insup ratio {lo['insup_ratio']:.3f} -> {hi['insup_ratio']:.3f}"
      f"   hole ratio {lo['hole_ratio']:.3f} -> {hi['hole_ratio']:.3f}")
