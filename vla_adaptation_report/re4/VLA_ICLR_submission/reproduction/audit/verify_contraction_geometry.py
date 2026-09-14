#!/usr/bin/env python3
"""Constructed checks of the revised geodesic proof; no VLA data are used."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.linalg import expm

SEED = 20260909
rng = np.random.default_rng(SEED)
checks = []

def record(name, n, err, tol, description):
    checks.append(dict(name=name, cases=n, max_error=float(err), tolerance=tol,
                       passed=bool(err <= tol), description=description))

# A nonlinear, time-varying global coordinate change provides an exact metric.
# phi_x >= .85, so every pair has a unique geodesic and global norm bounds.
# F0 makes y=phi(x,t) satisfy ydot=-lambda*y.
def fields(x, t, rate):
    a = .15*np.sin(.4*t)
    adot = .06*np.cos(.4*t)
    phi = x + a*np.sin(x)
    g = 1+a*np.cos(x)
    gx = -a*np.sin(x)
    gt = adot*np.cos(x)
    phit = adot*np.sin(x)
    num = -rate*phi-phit
    F = num/g
    Fx = ((-rate*g-gt)*g-num*gx)/g**2
    return phi, g, gx, gt, F, Fx

n = 2000
metric_err = energy_err = cancel_err = 0.
for _ in range(n):
    x, xr = rng.uniform(-3, 3, 2)
    t = rng.uniform(0, 20)
    rate = rng.uniform(.1, 2)
    theta_error, disturbance = rng.normal(size=2)
    B = .7+.1*np.sin(x)
    phi, g, gx, gt, F, Fx = fields(x,t,rate)
    phir, gr, _, _, Fr, _ = fields(xr,t,rate)
    lhs = 2*g*gt+2*g*gx*F+2*g*g*Fx
    metric_err = max(metric_err, abs(lhs+2*rate*g*g))
    delta_phi = phi-phir
    p = g*delta_phi
    dx = F-B*theta_error+disturbance
    h = 1e-6
    def energy(sign):
        px=fields(x+sign*h*dx,t+sign*h,rate)[0]
        pr=fields(xr+sign*h*Fr,t+sign*h,rate)[0]
        return .5*(px-pr)**2
    derivative = (energy(1)-energy(-1))/(2*h)
    predicted = -rate*delta_phi**2-p*B*theta_error+p*disturbance
    energy_err = max(energy_err, abs(derivative-predicted))
    # Actual parameter storage derivative with exact composite law.
    gamma = rng.uniform(.2, 3)
    H = rng.uniform(.1, 2)
    kc = rng.uniform(.1, 2)
    eps, drift = rng.normal(size=2)
    estimate_dot=gamma*(B*p+kc*(-H*theta_error+eps))
    actual=predicted+theta_error/gamma*(estimate_dot-drift)
    target=-rate*delta_phi**2-kc*H*theta_error**2+p*disturbance+kc*theta_error*eps-theta_error/gamma*drift
    cancel_err=max(cancel_err,abs(actual-target))
record('time_varying_metric_inequality',n,metric_err,1e-12,
       'Exact differential inequality for M_c=(1+.15 sin(.4t) cos(x))^2.')
record('actual_endpoint_first_variation',n,energy_err,2e-7,
       'Central differences of geodesic half-energy along actual perturbed endpoint and nominal reference.')
record('composite_cross_term_cancellation',n,cancel_err,1e-12,
       'Parameter storage cancels actual endpoint coupling including metric time dependence.')

# Non-diagonal Gamma tests weighted projection; coordinate clipping would be wrong.
projection_violation=0.
projection_solver_error=0.
opt_failed=0
nproj=200
for _ in range(nproj):
    A=rng.normal(size=(3,3)); Gamma=A@A.T+.5*np.eye(3); P=np.linalg.inv(Gamma)
    h=rng.uniform(-1,1,3)
    active=rng.integers(0,3)
    h[active]=rng.choice([-1.,1.])
    true=rng.uniform(-1,1,3)
    velocity=rng.normal(size=3)
    lower=np.where(h<=-1,0.,-np.inf)
    upper=np.where(h>=1,0.,np.inf)
    objective=lambda v:.5*(v-velocity)@P@(v-velocity)
    jac=lambda v:P@(v-velocity)
    sol=minimize(objective,np.zeros(3),jac=jac,bounds=Bounds(lower,upper),
                 method='SLSQP',options={'ftol':1e-13,'maxiter':200})
    if not sol.success: opt_failed+=1
    normal=np.zeros(3); normal[active]=h[active]
    exact=velocity-Gamma@normal*max(normal@velocity,0.)/(normal@Gamma@normal)
    projection_solver_error=max(projection_solver_error,float(np.linalg.norm(sol.x-exact)))
    val=(h-true)@P@(exact-velocity)
    projection_violation=max(projection_violation,val)
record('weighted_tangent_projection',nproj,projection_violation,1e-12,
       'Non-diagonal parameter metric, random true points, active box faces; exact weighted half-space solution.')
record('weighted_projection_independent_solver',nproj,projection_solver_error,2e-6,
       'Exact weighted half-space projection cross-checked against independent constrained optimizer.')
if opt_failed:
    raise RuntimeError(f'{opt_failed} projection optimizations failed')

# The weighted test must retain cascades and match the 2-by-2 comparison condition.
weighted_err=0.
negative_semigroup_violation=0.
ncoupled=1000
for _ in range(ncoupled):
    a,c,b,d=np.exp(rng.uniform(-3,2,4))
    tau=d/b
    hcross=b*np.sqrt(tau)+d/np.sqrt(tau)
    weighted_err=max(weighted_err,abs(hcross*hcross-4*b*d))
    mat=np.array([[-a,b],[d,-c]])
    if a*c>b*d:
        corner=-np.linalg.solve(mat,np.array([.1,.2]))
        assert np.all(corner>=0)
    negative_semigroup_violation=max(negative_semigroup_violation,-float(expm(.1*mat).min()))
record('weighted_comparison_equivalence',ncoupled,weighted_err,2e-12,
       'min_tau h^2=4bd, including stable/unstable comparison cases.')
record('comparison_semigroup_positivity',ncoupled,negative_semigroup_violation,1e-14,
       'Metzler comparison propagates componentwise nonnegative forcing.')
report={'seed':SEED,'scope':'Constructed algebra/numerical checks; no VLA certificate or new empirical outcomes.',
        'checks':checks,'all_passed':all(x['passed'] for x in checks)}
out=Path(__file__).with_name('contraction_geometry_checks.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['all_passed']: raise SystemExit(1)
