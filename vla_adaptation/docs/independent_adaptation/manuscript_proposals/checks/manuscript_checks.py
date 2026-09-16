"""Numerical re-check of the existing manuscript claims that the Q3/Q4 drafts lean on
(iclr2027/theory_main.tex, theory_appendix.tex). All must print True."""
import numpy as np
rng = np.random.default_rng(1); ok = {}
# uniapp:perron: rho(B) formula and the three-inequality characterisation, 5000 random nonnegative 2x2
def perron(B): s,b,t,q = B.ravel(); return (s+q+np.sqrt((s-q)**2+4*b*t))/2
res = []
for _ in range(5000):
    B = rng.uniform(0, 1.3, (2,2)); r = max(abs(np.linalg.eigvals(B))); s,b,t,q = B.ravel()
    res.append(abs(perron(B)-r) < 1e-9 and ((r < 1) == (s < 1 and q < 1 and b*t < (1-s)*(1-q))))
ok["perron_formula_and_smallgain_iff"] = all(res)
# uniapp:weightedstorage: p = (I-B^T)^-1 1 > 0, B^T p = p - 1, lambda = max (B^T p)_i/p_i in [0,1), V_{k+1} <= lambda V_k + p^T d
res = []
for _ in range(2000):
    while True:
        B = rng.uniform(0, 1, (2,2))
        if perron(B) < 1: break
    p = np.linalg.solve(np.eye(2)-B.T, np.ones(2)); lam = max((B.T@p)/p)
    e = rng.uniform(0, 1, 2); d = rng.uniform(0, 1, 2); e1 = B@e + d
    res.append(np.all(p > 0) and np.allclose(B.T@p, p-1) and 0 <= lam < 1 and p@e1 <= lam*(p@e) + p@d + 1e-12)
ok["weighted_storage"] = all(res)
# q<1 is implied by sigma<1 and beta t < (1-sigma)(1-q) with beta t >= 0
ok["q_lt_1_redundant"] = all((lambda s,b,t,q: (not (s < 1 and b*t < (1-s)*(1-q))) or q < 1)(*rng.uniform(0,1.3,4)) for _ in range(5000))
# uniapp:jury: A_c = [[1,-1],[k,1-a-k]] Schur iff 0<a<2, 0<k<4-2a
res = []
for _ in range(5000):
    a, k = rng.uniform(-0.5, 2.5), rng.uniform(-0.5, 5)
    A = np.array([[1,-1],[k,1-a-k]]); schur = max(abs(np.linalg.eigvals(A))) < 1 - 1e-12
    res.append(schur == (0 < a < 2 and 0 < k < 4-2*a) or (abs(a) < 1e-3 or abs(k) < 1e-3 or abs(a-2) < 1e-3 or abs(k-(4-2*a)) < 1e-3))
ok["jury_region"] = all(res)
# uniapp:memoryformula, integratoroffset, legacydrift
def mem(lp, al, n, g=1.0, f=1.0):
    p, fh = 0.0, 0.0; out = []
    for _ in range(n): p = lp*p + g*(f-fh); fh = (1-al)*fh + al*f; out.append(p)
    return out
lp, al, n = 0.7, 0.08, 40; p = mem(lp, al, n)[-1]
ok["memory_formula"] = abs(p - (lp**n - (1-al)**n)/(lp-(1-al))) < 1e-12
p1 = mem(1.0, al, n)[-1]; ok["integrator_offset"] = abs(p1 - (1-(1-al)**n)/al) < 1e-12
def legacy(s, g, n, gam):
    p, fh = 0.0, 0.0
    for _ in range(n): p = p + g*(1-fh); fh = (1-gam)*fh + gam*s
    return p
s, gam = 0.6, 0.08; ok["legacy_drift"] = abs(legacy(s, 1.0, n, gam) - ((1-s)*n + s/gam*(1-(1-gam)**n))) < 1e-12
# invert-then-mask example: sqrt(101)
Mx = np.array([[1,1],[0,0.1]]); d = np.array([0,0.02]); c = np.linalg.solve(Mx, d)
ok["invert_then_mask_sqrt101"] = np.allclose(c, [-0.2, 0.2]) and abs(np.linalg.norm(d - np.array([1,0])*c[0])/np.linalg.norm(d) - np.sqrt(101)) < 1e-12
# product of two spectral-radius-.5 matrices: 2.25 + sqrt(5)
P = np.array([[.5,2],[0,.5]]) @ np.array([[.5,0],[2,.5]]); ok["product_radius"] = abs(max(abs(np.linalg.eigvals(P))) - (2.25+np.sqrt(5))) < 1e-12
# uniapp:dc scalar equilibrium: 1/(2 - 0.37) ~ 0.61
ok["dc_equilibrium_0.61"] = abs(1/(2-0.37) - 0.6135) < 1e-3
# innovation/NT coefficient rows (t,q,eta) from Prop. estimator + mismatch bound, checked on random constants
res = []
for _ in range(2000):
    al, l, m, eps, X, E, nu = rng.uniform(0,1,7); w = l*X + m*E + eps
    res.append(abs(((1-al)*E + al*w + nu) - ((al*l)*X + (1-al+al*m)*E + (al*eps+nu))) < 1e-12)
ok["innovation_row"] = all(res)
for k, v in ok.items(): print(f"{k:35s} {v}")
print("ALL PASS:", all(ok.values()))
