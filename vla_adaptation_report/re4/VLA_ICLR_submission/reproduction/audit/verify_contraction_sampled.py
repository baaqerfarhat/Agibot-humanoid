#!/usr/bin/env python3
"""Constructed checks of sampled lemmas; no VLA data or experiments."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(20260909)
checks = {}
def record(name, values, tol=5e-10):
    vals = np.asarray(values, dtype=float)
    maximum = float(np.max(vals))
    checks[name] = {'count': int(vals.size), 'maximum_signed_violation': maximum,
                    'tolerance': tol, 'passed': bool(maximum <= tol)}
    assert checks[name]['passed'], (name, maximum)
def sqrtm(P):
    eig, vec = np.linalg.eigh(P)
    return (vec * np.sqrt(eig)) @ vec.T

# Box projection, uncorrected noise influencing a common normalizer, update holds,
# and truth drift. The mask can select zero through all coordinates.
legacy, innovation = [], []
for i in range(12000):
    n = 6
    mask = rng.integers(0, 2, n)
    f, h = rng.uniform(-1, 1, (2, n))
    f_next = np.clip(f + rng.normal(0, .05, n), -1, 1)
    w = rng.normal(0, .7, n)
    z = f + w
    S = rng.normal(size=(n, n)) + 3*np.eye(n)
    P = np.diag(rng.integers(0, 2, n))
    gamma = rng.uniform(.001, 1)
    chi = int(rng.integers(0, 2))
    rho = rng.uniform(.05, 2)
    r = S @ z
    s = float(np.linalg.norm(P @ r) > .2)/(1 + np.linalg.norm(P @ r)**2/rho**2)
    alpha = gamma/(1 + np.linalg.norm(P @ S @ (z-h))**2/rho**2)
    E = np.linalg.norm(mask*(h-f)); eps = np.linalg.norm(mask*w)
    nu = np.linalg.norm(mask*(f_next-f))
    h_a = np.clip((1-chi*gamma)*h + chi*gamma*s*z, -1, 1)
    h_i = np.clip(h + chi*alpha*(z-h), -1, 1)
    legacy.append(np.linalg.norm(mask*(h_a-f_next)) -
                  ((1-chi*gamma)*E + chi*gamma*((1-s)*np.linalg.norm(mask*f)+s*eps)+nu))
    innovation.append(np.linalg.norm(mask*(h_i-f_next)) -
                      ((1-chi*alpha)*E + chi*alpha*eps+nu))
record('masked_legacy_recursion', legacy)
record('masked_actual_innovation_recursion', innovation)

# The separability assumption is substantive: coupled projection moves a selected
# coordinate even when its preprojection selected error is zero.
f = np.array([.5, 0]); v = np.array([.5, 2]); projected = v/np.linalg.norm(v)
projection_counterexample = {'preprojection_selected_error': 0.,
    'postprojection_selected_error': float(abs(projected[0]-f[0]))}
assert projection_counterexample['postprojection_selected_error'] > .25

# Interior derivative counterexample: independent central finite difference.
gamma, rho, z = .08, 1., .4
h = z-np.sqrt(3)*rho
def T(h): return h+gamma*(z-h)/(1+(z-h)**2/rho**2)
fd = (T(h+1e-5)-T(h-1e-5))/(2e-5)
derivative = 1+gamma/8
truth_factor = abs((T(h)-z)/(h-z))
record('innovation_derivative_formula', [abs(fd-derivative)-1e-9])
assert fd > 1 and abs(truth_factor-.98)<1e-12
rgrid=np.r_[np.linspace(0,100,100001),3.]
record('innovation_derivative_maximum', 1+gamma*(rgrid-1)/(1+rgrid)**2-derivative)

# Metric projection with genuinely nondiagonal Gamma. A rotated/parallelepiped
# constraint is represented as Gamma^(1/2) times a box, so its metric projection
# is available exactly. Compare arbitrary pairs, including active projections.
candidate_pairwise, candidate_truth, metric_conditions = [], [], []
for i in range(3500):
    n=4
    Q=rng.normal(size=(n,n)); Gamma=Q@Q.T+.4*np.eye(n)
    Ghalf=sqrtm(Gamma); Ginv=np.linalg.inv(Ghalf)
    Q=rng.normal(size=(n,n)); H=Q@Q.T+.1*np.eye(n)
    SI=Ghalf@H@Ghalf
    eig=np.linalg.eigvalsh(SI)
    step=rng.uniform(.001,1.999)/eig[-1]
    qobs=float(np.max(abs(1-step*eig)))
    b=rng.normal(size=n)
    def project(theta): return Ghalf@np.clip(Ginv@theta,-1,1)
    def update(theta): return project(theta+step*Gamma@(b-H@theta))
    th1,th2=Ghalf@rng.uniform(-1,1,n),Ghalf@rng.uniform(-1,1,n)
    candidate_pairwise.append(np.linalg.norm(Ginv@(update(th1)-update(th2))) -
                              qobs*np.linalg.norm(Ginv@(th1-th2)))
    truth=Ghalf@rng.uniform(-1,1,n)
    truth_next=Ghalf@np.clip(Ginv@truth+rng.normal(0,.05,n),-1,1)
    eps=b-H@truth
    candidate_truth.append(np.linalg.norm(Ginv@(update(th1)-truth_next)) -
        (qobs*np.linalg.norm(Ginv@(th1-truth))+step*np.linalg.norm(Ghalf@eps)+
         np.linalg.norm(Ginv@(truth_next-truth))))
record('candidate_metric_projection_pairwise',candidate_pairwise)
record('candidate_metric_projection_truth_error',candidate_truth)

# Actual masked schedule -> delayed snapshot -> time-varying metric tube.
# Analytic equality A_k^T P_{k+1} A_k = a_k^2 P_k permits occasional expansion.
snapshot, metric_violations, tube_violations, product_expansion = [], [], [], []
for family in ['legacy','innovation']:
    for run in range(40):
        n=3; horizon=100
        mask=np.array([1.,0.,1.]); gamma=.08
        hs=[rng.uniform(-.5,.5,n)]; fs=[rng.uniform(-.3,.3,n)]
        Bs=[np.linalg.norm(mask*(hs[0]-fs[0]))]; nus=[]
        x=rng.normal(0,.1,n); xr=rng.normal(0,.1,n)
        Q=rng.normal(size=(n,n)); P=Q@Q.T+np.eye(n)
        Ph=sqrtm(P); R=np.linalg.norm(Ph@(x-xr)); Rproduct=R
        for k in range(horizon):
            Q=rng.normal(size=(n,n)); Pnext=Q@Q.T+np.eye(n)
            Pnh=sqrtm(Pnext)
            U,_=np.linalg.qr(rng.normal(size=(n,n)))
            a=1.2 if k%2==0 else .5
            A=a*np.linalg.solve(Pnh,U@Ph)
            metric_violations.append(np.max(np.linalg.eigvalsh(A.T@Pnext@A-a*a*P)))
            tau=max(0,k-int(rng.integers(0,9)))
            inc=sum(np.linalg.norm(mask*(hs[j+1]-hs[j])) for j in range(tau,k))
            Bapp=min(Bs[k]+inc, Bs[tau]+sum(nus[tau:k]))
            Eapp=np.linalg.norm(mask*(hs[tau]-fs[k]))
            snapshot.append(Eapp-Bapp)
            zeta=rng.normal(0,.001,n)
            q=fs[k]-mask*hs[tau]+zeta
            Bin=rng.normal(size=(n,n))*.05
            L=np.linalg.norm(Pnh@Bin,2)
            disturbance=rng.normal(0,.001,n)
            eta=np.linalg.norm(Pnh@disturbance)
            g=L*(np.linalg.norm((1-mask)*fs[k])+Bapp+np.linalg.norm(zeta))+eta
            R=a*R+g
            x=A@x+Bin@q+disturbance; xr=A@xr
            tube_violations.append(np.linalg.norm(Pnh@(x-xr))-R)
            f_next=np.clip(fs[k]+rng.normal(0,.005,n),-.8,.8)
            nu=np.linalg.norm(mask*(f_next-fs[k])); nus.append(nu)
            w=rng.normal(0,.01,n); z=fs[k]+w
            chi=int(k%7 not in (2,3))
            if family=='legacy':
                s=float(np.linalg.norm(z)>.1)/(1+np.linalg.norm(z)**2/.5**2)
                hnext=np.clip((1-chi*gamma)*hs[k]+chi*gamma*s*z,-1,1)
                Bnext=(1-chi*gamma)*Bs[k]+chi*gamma*((1-s)*np.linalg.norm(mask*fs[k])+s*np.linalg.norm(mask*w))+nu
            else:
                alpha=gamma/(1+np.linalg.norm(z-hs[k])**2/.5**2)
                hnext=np.clip(hs[k]+chi*alpha*(z-hs[k]),-1,1)
                Bnext=(1-chi*alpha)*Bs[k]+chi*alpha*np.linalg.norm(mask*w)+nu
            hs.append(hnext); fs.append(f_next); Bs.append(Bnext)
            P,Ph=Pnext,Pnh
record('time_varying_discrete_metric_equality',metric_violations)
record('masked_snapshot_minimum_bound',snapshot)
record('scheduled_metric_tube',tube_violations)
# a_even=1.2, a_odd=.5 has rho=sqrt(.6) and C=1.2/rho for every interval.
rho=np.sqrt(.6); C=1.2/rho
for j in range(20):
 for k in range(j,50):
    product=np.prod([1.2 if i%2==0 else .5 for i in range(j,k)])
    product_expansion.append(product-C*rho**(k-j))
record('expanding_steps_windowed_product',product_expansion)

result={'seed':20260909,'scope':'Constructed algebra and numerical checks only; no VLA rollout validation.',
        'checks':checks,'coupled_projection_counterexample':projection_counterexample,
        'innovation_counterexample':{'gamma':gamma,'r':3.,'derivative_exact':derivative,
          'derivative_finite_difference':float(fd),'truth_error_multiplier':float(truth_factor)},
        'all_passed':all(c['passed'] for c in checks.values())}
# gamma was reused in the schedule but remains the documented .08.
output=Path(__file__).with_name('contraction_sampled_checks.json')
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'all_passed':result['all_passed'],'check_count':sum(c['count'] for c in checks.values()),
                  'output':str(output)},indent=2))
