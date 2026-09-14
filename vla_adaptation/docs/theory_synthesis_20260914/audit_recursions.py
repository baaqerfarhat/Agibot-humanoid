"""Post hoc arithmetic audit, 2026-09-14. No simulator, no edits to archived scorers.
Observed same-run w is an algebraic decomposition, not an independently bounded uncertainty.
"""
import argparse, hashlib, json, pathlib, subprocess
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=pathlib.Path)
args=parser.parse_args()
OUT={"status":"post hoc consistency audit; not a predictive certificate", "source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"runs":{}}
for run in ['T1_headline','T4_innov']:
    episodes={}
    with (ROOT/'results/re4_theory/telemetry'/run/'telemetry.jsonl').open() as stream:
        header=json.loads(next(stream))
        for line in stream:
            d=json.loads(line)
            if d.get('type')=='step' and d.get('phase')=='rollout' and d.get('arm')=='adaptive':
                episodes.setdefault(d['episode'],[]).append(d)
    a,cfg=header['args'],header['config']
    assert not a.get('gate_stats') and a.get('profile')=='step' and not a.get('onset')
    assert a['deadzone_mode']=='zero' and not a.get('freeze_after')
    Mi=np.array(cfg['M_inv']);M=np.array(cfg['M']);bias=np.zeros(6) if cfg['bias'] is None else np.array(cfg['bias'])
    corr=np.flatnonzero(cfg['mask']);gamma=a['gamma'];clip=a['clip']
    failures=0; max_error=0.;vector_error=0.; bounds=[];errnorm=[];w_norm=[];meanw=[];steps=0;deads=0;clips=0;corrclips=0;last_ratios=[];slist=[];unmatched_model_eps=[];eps_old=[];force_bias=[];force_model=[]
    for e,ep in sorted(episodes.items()):
        ep=sorted(ep,key=lambda d:d['t']); B=float(np.linalg.norm(np.array(ep[0]['f_hat_before'])[corr]-np.array(ep[0]['f_true'])[corr]))
        epws=[];last_ratio=None
        for d in ep:
            r,F,fb,fa=[np.array(d[k],float) for k in ['r','f_true','f_hat_before','f_hat']]
            assert np.all(np.abs(F)<=clip) and np.array_equal(F, np.array(ep[0]['f_true']))
            z=Mi@r-bias; w=z-F; effective_s=0. if d['deadzone_fired'] else float(d['attenuation'])
            if a['law']=='legacy':
                pre=(1-gamma)*fb+gamma*effective_s*z
                epre=(1-gamma)*(fb-F)+gamma*effective_s*w-gamma*(1-effective_s)*F
                B=(1-gamma)*B+gamma*effective_s*np.linalg.norm(w[corr])+gamma*(1-effective_s)*np.linalg.norm(F[corr])
                force_bias.append(gamma*(1-effective_s)*np.linalg.norm(F[corr]))
            else:
                pre=fb+gamma*effective_s*(z-fb)
                epre=(1-gamma*effective_s)*(fb-F)+gamma*effective_s*w
                B=(1-gamma*effective_s)*B+gamma*effective_s*np.linalg.norm(w[corr])
                force_bias.append(0.)
            expected=np.clip(pre,-clip,clip)
            max_error=max(max_error,float(np.max(np.abs(expected-fa))))
            vector_error=max(vector_error,float(np.max(np.abs(pre-F-epre))))
            E=np.linalg.norm((fa-F)[corr]);viol=E>B+1e-12
            failures+=int(viol);bounds.append(float(B));errnorm.append(float(E));w_norm.append(float(np.linalg.norm(w[corr])))
            steps+=1;deads+=bool(d['deadzone_fired']);clips+=bool(np.any(np.abs(pre)>clip));corrclips+=bool(np.any(np.abs(pre[corr])>clip))
            slist.append(effective_s);epws.append(w)
            unmatched_model_eps.append(np.linalg.norm((r-M@F)[corr]));eps_old.append(np.linalg.norm((r-M@(F-fb))[corr]))
            force_model.append(gamma*effective_s*np.linalg.norm(w[corr]));last_ratio=B/max(E,1e-15)
        last_ratios.append(last_ratio);meanw.append(np.mean(epws[-50:],axis=0)[corr].tolist())
    OUT['runs'][run]={
        'telemetry_sha256':hashlib.sha256((ROOT/'results/re4_theory/telemetry'/run/'telemetry.jsonl').read_bytes()).hexdigest(),
        'law':a['law'],'episodes':len(episodes),'steps':steps,'corrected_channels':corr.tolist(),
        'update_map_max_absolute_error_all_channels':max_error,'error_recursion_max_algebra_error_before_projection':vector_error,
        'deadzone_steps':deads,'projection_steps_any_channel':clips,'projection_steps_corrected_channels':corrclips,
        'corrected_triangle_bound_coverage':1-failures/steps,'corrected_triangle_bound_violations':failures,
        'median_bound_over_actual_error':float(np.median(np.array(bounds)/np.maximum(errnorm,1e-15))),
        'median_final_bound_over_final_actual_error':float(np.median(last_ratios)),
        'median_actual_error_norm_corrected':float(np.median(errnorm)),
        'median_observed_w_norm_corrected':float(np.median(w_norm)),
        'median_last50_mean_w_corrected':np.median(meanw,axis=0).tolist(),
        'median_effective_attenuation':float(np.median(slist)),
        'median_model_error_forcing_term':float(np.median(force_model)),
        'median_legacy_attenuation_bias_forcing_term':float(np.median(force_bias)),
        'median_total_fault_model_eps_norm':float(np.median(unmatched_model_eps)),
        'median_archived_4_2_eps_norm':float(np.median(eps_old)),
    }
assert all(v['update_map_max_absolute_error_all_channels']<1e-12 and v['corrected_triangle_bound_violations']==0 for v in OUT['runs'].values())
text=json.dumps(OUT,indent=2)+'\n'
if args.out:
    args.out.write_text(text)
print(text)
