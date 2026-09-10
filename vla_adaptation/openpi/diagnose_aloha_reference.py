"""Exploratory ALOHA reference diagnostics; never issues qualification.

Compares a declared affine/position-history grid using an inner split of the
original fitting episodes. Original validation episodes are then reported as
already-examined diagnostics, not reused as fresh qualification of a new model.
No simulator, policy inference, success-based selection, or current artifact edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log',type=Path,default=ROOT/'results/composite_followup/calibration/aloha_healthy_seed2600.json')
parser.add_argument('--reference',type=Path,default=ROOT/'results/composite_followup/calibration/aloha_reference_bimanual.json')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
LOG=args.log
d=json.loads(LOG.read_text())
if len(d) != 10:
    raise ValueError('This declared diagnostic split requires the original ten episodes')
if args.out.exists():
    raise ValueError('Refusing to overwrite an existing diagnostic')
states=np.r_[0:6,7:13]
episodes=[]
for ep in d:
    q=np.asarray(ep['q'])[:,states]; before=np.asarray(ep['q_before'])[:,states]
    u=np.asarray(ep['u'])[:,states]
    previous=np.vstack([before[0], before[:-1]])
    old_u=np.vstack([before[0], u[:-1]])
    episodes.append((before,previous,u,old_u,q))

def fit(ids, order, command_lag, structure):
    groups=[np.arange(12)] if structure=='full' else [np.arange(6),np.arange(6,12)]
    pieces=[]
    for group in groups:
        xx=[]; yy=[]
        for i in ids:
            p,p2,u,u2,y=episodes[i]
            blocks=[p[:,group]] + ([p2[:,group]] if order==2 else []) + [u[:,group]] + ([u2[:,group]] if command_lag else [])
            xx.append(np.concatenate(blocks,axis=1)[1:]); yy.append(y[1:,group])
        X=np.concatenate(xx); Y=np.concatenate(yy)
        mx=X.mean(0); my=Y.mean(0); Xc=X-mx; Yc=Y-my
        coef=np.linalg.solve(Xc.T@Xc+1e-6*np.eye(X.shape[1]), Xc.T@Yc).T
        intercept=my-coef@mx
        width=len(group); A=np.zeros((order*width,order*width))
        A[:width,:order*width]=coef[:,:order*width]
        if order==2: A[width:,:width]=np.eye(width)
        pieces.append(dict(group=group, coef=coef, intercept=intercept,
                           spectral_radius=float(max(abs(np.linalg.eigvals(A)))),
                           design_condition=float(np.linalg.cond(Xc))))
    return dict(order=order,command_lag=command_lag,structure=structure,pieces=pieces)

def errors(model,ids):
    singles=[]; rollouts=[]; per_episode=[]
    for i in ids:
        p,p2,u,u2,y=episodes[i]; one=np.zeros_like(y); rollout=np.zeros_like(y)
        r=p[0].copy(); r2=p[0].copy()
        for t in range(len(y)):
            next_r=np.zeros(12)
            for part in model['pieces']:
                g=part['group']
                extra_u=[u2[t,g]] if model['command_lag'] else []
                xin=np.concatenate([p[t,g]]+([p2[t,g]] if model['order']==2 else [])+[u[t,g]]+extra_u)
                xref=np.concatenate([r[g]]+([r2[g]] if model['order']==2 else [])+[u[t,g]]+extra_u)
                one[t,g]=part['coef']@xin+part['intercept']
                next_r[g]=part['coef']@xref+part['intercept']
            r2,r=r,next_r; rollout[t]=r
        s=y-one; a=y-rollout
        singles.append(s); rollouts.append(a)
        per_episode.append(dict(episode=i,one=np.sqrt(np.mean(s*s,0)),rollout=np.sqrt(np.mean(a*a,0))))
    s=np.concatenate(singles); a=np.concatenate(rollouts)
    return dict(one=np.sqrt(np.mean(s*s,0)),rollout=np.sqrt(np.mean(a*a,0)),
                max_position_norm=float(max(np.linalg.norm(a,axis=1))))

specs=[(1,False,'full'),(1,False,'arms'),(2,False,'full'),(2,False,'arms'),(2,True,'full'),(2,True,'arms')]
report=dict(schema_version=1,
            description='Exploratory revised-model diagnostics; old validation is not fresh qualification',
            source_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (LOG,args.reference,Path(__file__))},
            state_indices=states.tolist(),command_indices=states.tolist(),
            settings=dict(ridge=1e-6,dt=.02,inner_fit_episodes=list(range(4)),
                          inner_check_episodes=[4,5],refit_episodes=list(range(6)),
                          old_validation_diagnostic_episodes=list(range(6,10)),
                          fit_drops_first_transition=True,
                          initial_reference_history='repeat initial position; initial previous command also holds position'),
            limits=['No candidate receives a qualification flag. The .005-rad per-state gate is unchanged.',
                    'All model choices are exploratory after the original 12D failure was observed.',
                    'Inner check stays inside the original training pool; original validation has been examined and is not fresh.',
                    'A revised model requires new independently collected qualification trajectories.',
                    'No contact/force/qvel measurements are in this input log. Derivatives and gripper commands are proxies.',
                    'Autonomous references use only raw command and their own position history after reset.',
                    'The local linear spectral radius does not certify physical closed-loop stability.'],
            candidates=[])
for order,lag,structure in specs:
    name=f'order{order}_ulag{int(lag)}_{structure}'
    m=fit(range(4),order,lag,structure)
    inner=errors(m,[4,5])
    full=fit(range(6),order,lag,structure)
    report['candidates'].append(dict(name=name,
        training_only_development_episodes=[4,5],training_fit_episodes=list(range(4)),
        inner_check=inner,refit_train=errors(full,range(6)),old_validation_diagnostic=errors(full,range(6,10)),
        spectral_radii=[p['spectral_radius'] for p in full['pieces']],
        design_conditions=[p['design_condition'] for p in full['pieces']]))

base=json.loads(args.reference.read_text())['model']
A=np.asarray(base['A']); B=np.asarray(base['B']); c=np.asarray(base['offset'])
rr=[]; lag1=[]; vel=[]; cmdvel=[]; accel=[]; gripclose=[]; gripmotion=[]
for ep,(p,p2,u,u2,y) in zip(d,episodes):
    res=y-(p@A.T+u@B.T+c)
    rr.append(res[2:]); lag1.append(res[1:-1])
    vel.append((p-p2)[2:]/.02); cmdvel.append((u-u2)[2:]/.02)
    accel.append(((y-p)-(p-p2))[2:]/(.02**2))
    gu=np.asarray(ep['u'])[:,13]; gq=np.asarray(ep['q'])[:,13]
    gripclose.append((np.diff(gu,prepend=gu[0]) < -.001)[2:])
    gripmotion.append(np.abs(np.diff(gq,prepend=gq[0]))[2:])
R=np.concatenate(rr); lag=np.concatenate(lag1); V=np.concatenate(vel); U=np.concatenate(cmdvel); Acc=np.concatenate(accel)
def corr(x,y):
    return np.asarray([np.corrcoef(x[:,j],y[:,j])[0,1] for j in range(12)])
report['residual_diagnostics']=dict(lag1_correlation=corr(R,lag),position_velocity_correlation=corr(R,V),
    command_velocity_correlation=corr(R,U),achieved_acceleration_correlation=corr(R,Acc),
    command_velocity_rms=np.sqrt(np.mean(U*U,0)),position_velocity_rms=np.sqrt(np.mean(V*V,0)))
speed=np.linalg.norm(U[:,6:],axis=1); res=np.linalg.norm(R[:,6:],axis=1)
lo,hi=np.quantile(speed,[.25,.75]); close=np.concatenate(gripclose)
report['residual_diagnostics'].update(right_error_low_command_speed=float(np.sqrt(np.mean(res[speed<=lo]**2))),
    right_error_high_command_speed=float(np.sqrt(np.mean(res[speed>=hi]**2))),
    right_error_when_right_gripper_closing=float(np.sqrt(np.mean(res[close]**2))),
    right_error_when_not_closing=float(np.sqrt(np.mean(res[~close]**2))),
    right_gripper_closing_fraction=float(close.mean()),
    caveat='No contact/force/qvel telemetry here. Gripper commands and finite differences are diagnostic proxies, not causal evidence.')
def enc(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('x') as output:
    json.dump(report,output,default=enc,indent=2,allow_nan=False)
    output.write('\n')
for row in report['candidates']:
    print(row['name'], 'inner max one/roll',max(row['inner_check']['one']),max(row['inner_check']['rollout']),
          'old val max one/roll',max(row['old_validation_diagnostic']['one']),max(row['old_validation_diagnostic']['rollout']),
          'rho',row['spectral_radii'])
print(json.dumps(report['residual_diagnostics'],default=enc,indent=2))
