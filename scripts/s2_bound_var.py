"""
s2_bound_var.py  (escrito por el agente verificador, 9 nov; copiado del scratchpad)
Cota S2(rho) <= S2max(E, V): maxima S2 entre distribuciones diagonales en la base propia de H con media E y varianza V
(p_i = max(cuadratica en E_i, 0)).  Defasar en la base de H nunca baja S2, asi que el deficit S2_MRE(E) - S2max(E, V)
es la parte del deficit de S2 a E fija que OBLIGA la Var(H) del modelo.  Energias agrupadas en bins de 0.005.
Uso: python scripts/s2_bound_var.py N etiqueta:E:VarH:S2 [...]
"""
# max S2 over diagonal distributions p_i on the exact spectrum with fixed mean E and variance V.
# KKT: p_i = max(l0 + l1 x_i + l2 x_i^2, 0). Since S2(rho) <= S2(diag_H rho) <= S2max(E,V), this bounds the model's S2.
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mre_teff import spectrum, MRE
from scipy.optimize import brentq, fsolve, minimize
N=int(sys.argv[1]); Ev=np.sort(spectrum(N))
w=0.005; edges=np.arange(Ev[0]-w, Ev[-1]+2*w, w); n,_=np.histogram(Ev,edges); xb=0.5*(edges[1:]+edges[:-1]); m=n>0; n=n[m].astype(float); xb=xb[m]
mre=MRE(Ev)
def S2max(E0,V0):
    # dual: minimize 0.5*sum n*max(q,0)^2 - l.(1,E0,E0^2+V0) -> concave dual; use minimize on convex negative
    x=(xb-E0)/np.sqrt(V0)  # standardized
    tgt=np.array([1.0,0.0,1.0])
    def f(l):
        q=l[0]+l[1]*x+l[2]*x*x; qp=np.maximum(q,0)
        val=0.5*(n*qp*qp).sum()-l@tgt
        g=np.array([(n*qp).sum(),(n*qp*x).sum(),(n*qp*x*x).sum()])-tgt
        return val,g
    best=None
    for l0 in ([1e-6,-1e-6,0],[1e-7,0,0],[1e-6,-1e-7,1e-8]):
        r=minimize(f,np.array(l0,float),jac=True,method="BFGS",options=dict(gtol=1e-13,maxiter=20000))
        if best is None or r.fun<best.fun: best=r
    l=best.x; q=np.maximum(l[0]+l[1]*x+l[2]*x*x,0); P=n*q  # bin probabilities; per-state p=q
    resid=np.array([P.sum()-1,(P*x).sum(),(P*x*x).sum()-1])
    return -np.log((n*q*q).sum()/P.sum()**2), resid
T=3.0
runs=[a.split(":") for a in sys.argv[2:]]
for lab,E,V,S2 in runs:
    E,V,S2=float(E),float(V),float(S2)
    TE=brentq(lambda t: mre(t)[0]-E,0.05,50); SE=mre(TE)[1]
    # variance of MRE at TE
    f=lambda c: c-mre._sums(c)[1]/mre._sums(c)[0]-2*TE; c=brentq(f,mre.E[0]+1e-9,mre.E[-1]+70)
    p=np.maximum(c-mre.E,0);p/=p.sum(); Vm=(p*mre.E**2).sum()-((p*mre.E).sum())**2
    sm,res=S2max(E,Vm); s,res2=S2max(E,V)
    print(f"{lab}: E={E} V={V} VMRE(TE)={Vm:.2f} S2_MRE(E)={SE:.4f} check S2max(E,VMRE)={sm:.4f} | S2max(E,V_model)={s:.4f} -> deficit forced by V: {SE-s:+.4f} ; model S2={S2} total deficit {SE-S2:+.4f} ; resid {np.abs(res2).max():.1e}")
