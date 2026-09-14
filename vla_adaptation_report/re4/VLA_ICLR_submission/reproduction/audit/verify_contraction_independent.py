#!/usr/bin/env python3
"""Independent algebraic/constructed checks; not a formal proof or VLA validation."""
import json
from pathlib import Path
import numpy as np
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(9132026)
results = {}

# An exact nonlinear coordinate metric, independently chosen from the manuscript:
# phi(x)=sinh(x), M=cosh(x)^2, F0=-lambda*tanh(x).
# Bounds 1<=M<=cosh(2)^2 on [-2,2].
metric_error = 0.; variation_error = 0.
for _ in range(5000):
    xx, rr = rng.uniform(-2,2,size=2)
    ll=rng.uniform(.1,2); dd=rng.normal()
    mm=np.cosh(xx)**2; dmm=2*np.cosh(xx)*np.sinh(xx)
    field=-ll*np.tanh(xx); jac=-ll/np.cosh(xx)**2
    metric_error=max(metric_error,abs(dmm*field+2*mm*jac+2*ll*mm))
    ee=np.sinh(xx)-np.sinh(rr); px=np.cosh(xx)*ee
    actual=px*(field+dd)-ee*np.cosh(rr)*(-ll*np.tanh(rr))
    variation_error=max(variation_error,abs(actual+ll*ee**2-px*dd))
assert metric_error<1e-12 and variation_error<1e-12
results['nonconstant_metric_analytic_identities_numeric_check'] = {
    'cases':5000,
    'metric_identity_error':metric_error,
    'first_variation_identity_error':variation_error,
    'domain': '[-2,2], with endpoints and relevant perturbations contained',
    'metric': 'cosh(x)^2',
    'nominal_field': '-lambda*tanh(x)',
    'energy': '(sinh(x)-sinh(xr))^2/2',
}

# Interior normalizer counterexample, exact rational arithmetic.
gamma=Fraction(2,25); r=Fraction(3)
pair_multiplier=1+gamma*(r-1)/(1+r)**2
truth_multiplier=1-gamma/(1+r)
assert pair_multiplier == Fraction(101,100)
assert truth_multiplier == Fraction(49,50)
results['innovation_counterexample_exact'] = {'pair_multiplier':str(pair_multiplier),'truth_multiplier':str(truth_multiplier)}

# First variation/cancellation with a nonconstant state metric, nonidentity
# parameter metric, and active/inactive tangent-cone projection. The parameter
# set is an ellipsoid: theta=Gamma^(1/2)y, ||y||<=1.
max_joint_excess = -np.inf
max_pair_excess = -np.inf
n = 5000
for _ in range(n):
    A = rng.normal(size=(3,3))
    Gamma = A@A.T + .2*np.eye(3)
    evals, U = np.linalg.eigh(Gamma)
    Gh = (U*np.sqrt(evals))@U.T
    Hraw = rng.normal(size=(3,3))
    H = Hraw@Hraw.T + .1*np.eye(3)
    B = rng.normal(size=3)
    h = rng.normal(size=3); h /= np.linalg.norm(h)
    if rng.random() < .5:
        h *= rng.random()
    true = rng.normal(size=3); true *= .8*rng.random()/np.linalg.norm(true)
    z = h-true; tilde = Gh@z
    ex, er = rng.uniform(-1.5,1.5,size=2)
    ep = np.sinh(ex)-np.sinh(er)
    px = np.cosh(ex)*ep
    ll = rng.uniform(.1,2); kc = rng.uniform(.1,3)
    sig = B*px
    eps = rng.normal(size=3)*.05
    dr = rng.normal(size=3)*.02
    disturbance = rng.normal()*.03
    v = Gh@(sig + kc*(-H@tilde+eps))
    projected = v.copy()
    if np.linalg.norm(h) > 1-1e-12:
        projected -= h*max(0,float(h@v))
    exact = ep*(-ll*ep+np.cosh(ex)*(-B@tilde+disturbance)) + z@(projected-dr)
    upper = -ll*ep**2-kc*tilde@H@tilde+px*disturbance+kc*tilde@eps-z@dr
    max_joint_excess = max(max_joint_excess,exact-upper)
    # Two observers sharing the same exogenous H,b; projected on same ellipsoid.
    y1 = h
    y2 = rng.normal(size=3); y2 /= np.linalg.norm(y2)
    if rng.random()<.5: y2 *= rng.random()
    data = rng.normal(size=3)
    def field(y):
        vv = kc*Gh@(data-H@Gh@y)
        if np.linalg.norm(y)>1-1e-12: vv -= y*max(0,float(y@vv))
        return vv
    dy=y1-y2
    exact_pair = dy@(field(y1)-field(y2))
    upper_pair = -kc*(Gh@dy)@H@(Gh@dy)
    max_pair_excess=max(max_pair_excess,exact_pair-upper_pair)
assert max_joint_excess < 1e-10 and max_pair_excess < 1e-10
results['nonconstant_metric_projected_composite']={'cases':n,'max_positive_excess':max(0,float(max_joint_excess))}
results['projected_common_data_observers']={'cases':n,'max_positive_excess':max(0,float(max_pair_excess))}

# Independent sampled map whose geodesic distance is known exactly.
# F(x,q)=asinh(a*sinh(x)+q), d(x,y)=|sinh(x)-sinh(y)|.
# J0^2 M(F0)=a^2 M(x), and input gain is exactly one.
metric_error=0.; distance_excess=-np.inf; input_error=0.
for _ in range(n):
    a=rng.uniform(.05,1.1); xx, yy=rng.uniform(-1,1,size=2); q=rng.uniform(-.2,.2)
    ff0=np.arcsinh(a*np.sinh(xx)); ffy=np.arcsinh(a*np.sinh(yy)); ff=np.arcsinh(a*np.sinh(xx)+q)
    jac=a*np.cosh(xx)/np.cosh(ff0)
    metric_error=max(metric_error,abs(jac**2*np.cosh(ff0)**2-a*a*np.cosh(xx)**2))
    input_error=max(input_error,abs(abs(np.sinh(ff)-np.sinh(ff0))-abs(q)))
    distance_excess=max(distance_excess,abs(np.sinh(ff)-np.sinh(ffy))-a*abs(np.sinh(xx)-np.sinh(yy))-abs(q))
assert metric_error<1e-12 and input_error<1e-12 and distance_excess<1e-12
results['sampled_metric_geodesic_input']={'cases':n,'metric_identity_error':metric_error,'input_identity_error':input_error,'max_positive_tube_excess':max(0,float(distance_excess))}

# Why mask nonexpansiveness needs separable projection: projection onto x1=x2
# can move the selected coordinate when the input difference has zero there.
z1=np.array([0.,1.]); z2=np.zeros(2); D=np.diag([1.,0.])
proj=lambda z: np.repeat(np.mean(z),2)
assert np.linalg.norm(D@(z1-z2)) == 0
assert np.linalg.norm(D@(proj(z1)-proj(z2))) == .5
results['coupled_projection_mask_counterexample']={'selected_input_difference':0.,'selected_output_difference':.5}

results['scope']='Exact rational counterexample and constructed numerical checks of analytical identities only; not a formal proof of all hypotheses and not verification of any VLA experiment.'
(ROOT/'contraction_independent_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
