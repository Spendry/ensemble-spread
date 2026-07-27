exec(open('t7.py').read().split('ENS={')[0])
import numpy as np, json
ENS={8:0.7409,16:0.4549,32:0.2647,64:0.3487,128:0.1783,256:0.0955}
print(f"{'w':>6} {'NNGP post':>10} {'NTK limit':>10} {'NTK/post':>9} | "
      f"{'NNGP insup':>11} {'NTK insup':>10} | {'measured ens':>12}")
res=[]
for w in [32,64,128,256,512,1024]:
    K=nngp(w); T=ntk(w)
    Ktt=K[:nt,:nt]; Ttt=T[:nt,:nt]
    scale=np.trace(Ttt)/np.trace(Ktt)            # matched ridge, same noise level
    Ki=np.linalg.inv(Ktt+S2*np.eye(nt)); Ti=np.linalg.inv(Ttt+S2*scale*np.eye(nt))
    r={}
    for name,a,b in [("hole",nt,nt+nh),("insup",nt+nh,N)]:
        Kbb=K[a:b,a:b]; KbX=K[a:b,:nt]; A=T[a:b,:nt]@Ti
        post=Kbb-KbX@Ki@KbX.T
        ntkc=Kbb+A@Ktt@A.T-(A@KbX.T+KbX@A.T)
        r[name]=(float(np.sqrt(np.clip(np.diag(post),0,None)).mean()/yscale),
                 float(np.sqrt(np.clip(np.diag(ntkc),0,None)).mean()/yscale))
    res.append(dict(width=w,post_hole=r["hole"][0],ntk_hole=r["hole"][1],
                    post_insup=r["insup"][0],ntk_insup=r["insup"][1]))
    e=ENS.get(w,float('nan'))
    print(f"{w:6d} {r['hole'][0]:10.4f} {r['hole'][1]:10.4f} {r['hole'][1]/r['hole'][0]:9.2f} | "
          f"{r['insup'][0]:11.4f} {r['insup'][1]:10.4f} | {e:12.4f}",flush=True)
    del K,T; gc.collect()
json.dump(res,open("t7_final.json","w"),indent=1)
lw=np.log2([r["width"] for r in res])
for k in ["post_hole","ntk_hole","ntk_insup"]:
    print(f"log2 slope {k:10s} = {np.polyfit(lw,np.log2([r[k] for r in res]),1)[0]:+.3f}")
a=res[-1]["ntk_hole"]; p=res[-1]["post_hole"]; e=0.0955
print(f"\nasymptote={a:.4f}  correct posterior={p:.4f}  ensemble at w=256: {e:.4f}")
print(f"asymptote sits {100*(1-a/p):.0f}% BELOW the correct posterior -> crossing exists")
print(f"crossing width estimate (slope -0.527 from w=256): ~{256*2**(np.log2(e/p)/0.527):.0f}")
