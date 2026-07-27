exec(open('t7.py').read().split('ENS={')[0])
import numpy as np
for w in [256,1024]:
    K=nngp(w); T=ntk(w)
    Ktt=K[:nt,:nt]; Ttt=T[:nt,:nt]
    print(f"\n=== w={w} ===")
    print(f"  cond(K_XX)={np.linalg.cond(Ktt):.3e}   cond(T_XX)={np.linalg.cond(Ttt):.3e}")
    for name,a,b in [("hole",nt,nt+nh),("insup",nt+nh,N)]:
        Kbb=K[a:b,a:b]; KbX=K[a:b,:nt]; TbX=T[a:b,:nt]
        for label,ridge in [("noisy(s2)",S2),("ridgeless",1e-10*np.trace(Ktt)/nt)]:
            Ki=np.linalg.inv(Ktt+ridge*np.eye(nt))
            p=Kbb-KbX@Ki@KbX.T
            print(f"  NNGP {label:10s} {name:5s} std={np.sqrt(np.clip(np.diag(p),0,None)).mean()/yscale:.5f}")
        for label,ridge in [("noisy(s2)",S2*np.trace(Ttt)/np.trace(Ktt)),
                            ("ridgeless",1e-10*np.trace(Ttt)/nt)]:
            Ti=np.linalg.inv(Ttt+ridge*np.eye(nt)); A=TbX@Ti
            S=Kbb+A@Ktt@A.T-(A@KbX.T+KbX@A.T)
            print(f"  NTK  {label:10s} {name:5s} std={np.sqrt(np.clip(np.diag(S),0,None)).mean()/yscale:.5f}"
                  f"  (min eig {np.linalg.eigvalsh(S).min():+.2e})")
