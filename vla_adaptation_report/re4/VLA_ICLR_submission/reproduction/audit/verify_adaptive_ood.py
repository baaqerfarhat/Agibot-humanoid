"""Reproduce constructed adaptive-control checks and Fig. fig_theory_ood.

Run: python audit/verify_adaptive_ood.py
Requires Python 3, NumPy, SciPy and Matplotlib. No robot/VLA data are used.
The randomized tests check algebra, not a proof or experimental certification.
The scalar ODE is a deliberately constructed plant with an ideal current
regression. It does not model acquisition of informative real-robot data.
"""
from pathlib import Path
import json

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260909


def unit(rng, n):
    x = rng.normal(size=n)
    return x / np.linalg.norm(x)


def randomized_checks(trials=5000):
    rng = np.random.default_rng(SEED)
    excess = dict(unweighted=-np.inf, weighted=-np.inf, projection=-np.inf,
                  jump=-np.inf, radial=-np.inf)
    spd = dict(unweighted=0, weighted=0)
    for _ in range(trials):
        p, n = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        U, _ = np.linalg.qr(rng.normal(size=(p, p)))
        Gamma = U @ np.diag(rng.uniform(0.12, 3.0, size=p)) @ U.T
        Ginv = np.linalg.inv(Gamma)
        gm, gM = np.linalg.eigvalsh(Gamma)[[0, -1]]
        R = rng.normal(size=(p, p))
        H = R @ R.T + rng.uniform(0.1, 1.0) * np.eye(p)
        mu = np.linalg.eigvalsh(H)[0]
        S = rng.normal(size=(p, n))
        cs = float(rng.uniform(0.1, 2.0))
        S *= cs / np.linalg.norm(S, 2)
        e, t = rng.normal(size=n), rng.normal(size=p)
        X, Z = np.linalg.norm(e), np.sqrt(t @ Ginv @ t)
        lam, kc, cd = rng.uniform(0.4, 2.0), rng.uniform(0.3, 3.0), rng.uniform(0.4, 1.8)
        kappa = float(rng.choice([0.0, 1.0, rng.uniform(0, 1)]))
        d0, b0, nu = rng.uniform(0, 0.08, size=3)
        ellx, elltheta, mx, mtheta = rng.uniform(0, 0.12, size=4)
        d = unit(rng, n) * (d0 + ellx * X + elltheta * Z)
        b = unit(rng, p) * (b0 + mx * X + mtheta * Z)
        v = unit(rng, p) * nu
        sigma = S @ e
        edot = -lam * e - S.T @ t + cd * d
        raw = Gamma @ (kappa * sigma + kc * (-H @ t + b))
        # Estimate at origin on the face n^T estimate <= 0; theta*=-t is feasible.
        normal = unit(rng, p)
        if normal @ t < 0:
            normal = -normal
        projected = raw - Gamma @ normal * max(0.0, normal @ raw) / (normal @ Gamma @ normal)
        p_excess = float(t @ Ginv @ (projected - raw))
        excess['projection'] = max(excess['projection'], p_excess)
        tdot = projected - v
        a = lam - cd * ellx
        c = kc * mu * gm - kc * np.sqrt(gM) * mtheta
        qz = kc * np.sqrt(gM) * b0 + nu / np.sqrt(gm)
        for label, tau in [('unweighted', 1.0), ('weighted', float(np.exp(rng.uniform(-3, 3))))]:
            h = (abs(kappa-tau)*cs*np.sqrt(gM) + tau*cd*elltheta + kc*np.sqrt(gM)*mx)/np.sqrt(tau)
            K = np.array([[a, -h / 2], [-h / 2, c]])
            q = np.array([np.sqrt(tau) * cd * d0, qz])
            z = np.array([np.sqrt(tau) * X, Z])
            actual = float(tau * e @ edot + t @ Ginv @ tdot)
            upper = float(-z @ K @ z + q @ z)
            excess[label] = max(excess[label], actual - upper)
            beta = float(np.linalg.eigvalsh(K)[0])
            if beta > 0:
                spd[label] += 1
                radial_upper = -beta * (z @ z) + np.linalg.norm(q) * np.linalg.norm(z)
                excess['radial'] = max(excess['radial'], actual - radial_upper)
        jump = rng.normal(size=p)
        before = np.sqrt(X*X + t @ Ginv @ t)
        after = np.sqrt(X*X + (t-jump) @ Ginv @ (t-jump))
        excess['jump'] = max(excess['jump'], float(after-before-np.sqrt(jump @ Ginv @ jump)))
    assert max(excess.values()) < 1e-10, excess

    # Information-only update is a stable triangular cascade here even when
    # the unweighted certificate fails. This guards against necessity claims.
    lam, information_gain, cs, tau = 0.2, 0.1, 1.0, 0.01
    K1 = np.array([[lam, -cs/2], [-cs/2, information_gain]])
    Ktau = np.array([[lam, -np.sqrt(tau)*cs/2], [-np.sqrt(tau)*cs/2, information_gain]])
    exact_dynamics = np.array([[-lam, -cs], [0.0, -information_gain]])
    unweighted_min = float(np.linalg.eigvalsh(K1)[0])
    weighted_min = float(np.linalg.eigvalsh(Ktau)[0])
    assert unweighted_min < 0 < weighted_min
    assert np.max(np.real(np.linalg.eigvals(exact_dynamics))) < 0
    return {
        'trials': trials, 'seed': SEED,
        'maximum_inequality_excess': {k: float(v) for k, v in excess.items()},
        'positive_definite_samples': spd,
        'prediction_only_certificate_counterexample': {
            'lambda': lam, 'kc_mu': information_gain, 'tau': tau,
            'unweighted_min_eigenvalue': unweighted_min,
            'weighted_min_eigenvalue': weighted_min,
            'actual_system_eigenvalues': np.linalg.eigvals(exact_dynamics).tolist(),
            'meaning': 'Failure of the unweighted sufficient condition does not imply instability.'
        }
    }


def stale_memory_checks(trials=500):
    rng = np.random.default_rng(SEED + 1)
    identity_error, bound_excess = 0.0, -np.inf
    for _ in range(trials):
        p, obs, count, now = 3, 2, 20, 10.0
        v = rng.normal(size=p) * .03
        jumps = [(3.0, rng.normal(size=p)), (7.0, rng.normal(size=p))]
        theta = lambda time: v*time + sum((jump for jt, jump in jumps if jt <= time), np.zeros(p))
        H, h, decomposition = np.zeros((p,p)), np.zeros(p), np.zeros(p)
        upper = 0.0
        for ti in rng.uniform(0, now, size=count):
            Y = rng.normal(size=(obs,p))
            weight = rng.uniform(0, 1)
            noise = unit(rng, obs) * rng.uniform(0, .02)
            target = Y @ theta(ti) + noise
            H += weight * Y.T @ Y
            h += weight * Y.T @ target
            decomposition += weight * Y.T @ (noise + Y @ (theta(ti)-theta(now)))
            variation = np.linalg.norm(v)*(now-ti) + sum(np.linalg.norm(jump) for jt,jump in jumps if ti < jt <= now)
            upper += weight*np.linalg.norm(Y,2)*(np.linalg.norm(noise)+np.linalg.norm(Y,2)*variation)
        bias = h - H @ theta(now)
        identity_error = max(identity_error, float(np.linalg.norm(bias-decomposition)))
        bound_excess = max(bound_excess, float(np.linalg.norm(bias)-upper))
    assert identity_error < 1e-10 and bound_excess < 1e-10
    # The data are informative but from the wrong constant regime.
    H, old, current = 2.0, 0.4, 1.1
    h = H * old
    bias = h - H * current
    return {
        'trials': trials, 'maximum_decomposition_error': identity_error,
        'maximum_bias_bound_excess': bound_excess,
        'exact_scalar_old_regime': {
            'H': H, 'old_theta': old, 'current_theta': current,
            'b_H': bias, 'prediction_only_equilibrium': h/H,
            'equilibrium_parameter_error': h/H-current,
            'meaning': 'Positive information alone does not remove stale-regime bias.'
        }
    }


def constructed_example():
    lam, kc, mu = 1.0, 2.0, 0.65
    dbar, bbar, nubar = 0.02, 0.006, 0.03
    Q = float(np.hypot(dbar, kc*bbar+nubar))
    # Segment starts explicitly contain the post-jump theta*. Drift is bounded.
    segments = [(0.0, 5.0, 0.7, 0.0), (5.0, 10.0, 1.15, -0.03),
                (10.0, 16.0, 0.4, 0.0)]
    curves, summary = {}, {}
    for kappa in (0.0, 1.0):
        K = np.array([[lam, -(1-kappa)/2], [-(1-kappa)/2, kc*mu]])
        beta = float(np.linalg.eigvalsh(K)[0])
        state, theta_previous, bound_start = np.array([0.25, 0.0]), None, None
        fields = {k: [] for k in ['time', 'e', 'parameter_error', 'theta', 'theta_hat', 'W', 'bound']}
        jump_excess, energy_excess, derivative_excess = -np.inf, -np.inf, -np.inf
        for start, end, theta_start, drift in segments:
            previous_W = None if theta_previous is None else np.hypot(state[0],state[1]-theta_previous)
            if bound_start is None:
                bound_start = float(np.hypot(state[0], state[1]-theta_start))
            else:
                jump = theta_start-theta_previous
                bound_start += abs(jump)
                post_W = np.hypot(state[0],state[1]-theta_start)
                jump_excess = max(jump_excess, float(post_W-previous_W-abs(jump)))
            def rhs(time, y):
                e, estimate = y
                theta_star = theta_start + drift*(time-start)
                error = estimate-theta_star
                d = dbar*np.sin(2.3*time)
                bH = bbar*np.sin(1.7*time + 0.4)
                # Synthetic measurement generator sees the plant parameter;
                # the update itself receives only its observed statistic h.
                h_observed = mu*theta_star + bH
                return [-lam*e-error+d, kappa*e+kc*(h_observed-mu*estimate)]
            times = np.linspace(start, end, int((end-start)*200)+1)
            sol = solve_ivp(rhs, (start,end), state, method='DOP853', t_eval=times,
                            rtol=1e-11, atol=1e-13, max_step=0.025)
            assert sol.success
            theta = theta_start + drift*(times-start)
            error = sol.y[1]-theta
            W = np.hypot(sol.y[0],error)
            envelope = np.exp(-beta*(times-start))*bound_start + Q/beta*(1-np.exp(-beta*(times-start)))
            energy_excess = max(energy_excess,float(np.max(W-envelope)))
            for i, time in enumerate(times):
                edot, hatdot = rhs(time,sol.y[:,i])
                dV = sol.y[0,i]*edot + error[i]*(hatdot-drift)
                derivative_excess=max(derivative_excess,float(dV+beta*W[i]**2-Q*W[i]))
            values=[times,sol.y[0],error,theta,sol.y[1],W,envelope]
            for name,value in zip(fields,values):
                fields[name].extend(value.tolist())
            state, theta_previous, bound_start = sol.y[:,-1], float(theta[-1]), float(envelope[-1])
        assert energy_excess < 1e-8 and derivative_excess < 1e-8 and jump_excess < 1e-8
        curves[kappa]={name:np.array(value) for name,value in fields.items()}
        # Segment integration avoids counting duplicated jump instants as time.
        integration = np.trapezoid
        summary[str(int(kappa))] = {
            'beta_unweighted': beta, 'Q': Q, 'radius_floor_bound': Q/beta,
            'maximum_W_minus_bound':energy_excess,
            'maximum_dV_minus_radial_bound':derivative_excess,
            'maximum_jump_inequality_excess':jump_excess,
            'integrated_squared_tracking_error':float(integration(curves[kappa]['e']**2,curves[kappa]['time'])),
            'integrated_squared_parameter_error':float(integration(curves[kappa]['parameter_error']**2,curves[kappa]['time'])),
            'peak_absolute_tracking_error':float(np.max(np.abs(curves[kappa]['e']))),
        }
    return curves, {
        'status':'constructed analytical illustration; not robot or VLA validation',
        'plant':'e_dot=-lambda e-(theta_hat-theta_star)+d',
        'update':'theta_hat_dot=kappa e+kc*(-H*(theta_hat-theta_star)+b_H)',
        'regression':'Ideal current regression h(t)=H theta_star(t)+b_H(t), H=0.65 throughout; no retained-data acquisition is modeled.',
        'lambda':lam, 'Gamma':1.0, 'kc':kc, 'H':mu,
        'initial_e':0.25, 'initial_theta_hat':0.0,
        'disturbance':'0.02 sin(2.3 t)', 'regression_bias':'0.006 sin(1.7 t+0.4)',
        'theta_segments':[{'start':a,'end':b,'theta_start':c,'drift':d} for a,b,c,d in segments],
        'integrator':{'method':'DOP853','rtol':1e-11,'atol':1e-13,'max_step':0.025},
        'comparison':summary,
        'interpretation':'The composite branch improves this unweighted sufficient rate. Actual parameter and tracking transients can trade off; no general speed dominance is claimed.'
    }


def make_figure(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axs = plt.subplots(1,3,figsize=(7.15,2.3),sharex=True)
    colors={0.0:'#286995',1.0:'#BA572D'}
    names={0.0:r'Prediction only ($\kappa=0$)',1.0:r'Composite ($\kappa=1$)'}
    for kappa,curve in curves.items():
        color=colors[kappa]
        axs[0].plot(curve['time'],curve['e'],color=color,lw=1.3,label=names[kappa])
        axs[1].plot(curve['time'],curve['parameter_error'],color=color,lw=1.3)
        axs[2].plot(curve['time'],curve['W'],color=color,lw=1.3)
        axs[2].plot(curve['time'],curve['bound'],color=color,lw=1.0,ls='--')
    for ax in axs:
        for onset in (5,10):
            ax.axvline(onset,color='#9B9B9B',ls=':',lw=.7,zorder=0)
        ax.set_xlim(0,16)
        ax.set_xlabel('Time (arbitrary units)')
        ax.set_xticks([0,5,10,15])
        ax.grid(alpha=.16,lw=.5)
    axs[0].set_title('(a) Tracking error')
    axs[0].set_ylabel(r'$e$')
    axs[1].set_title('(b) Parameter error')
    axs[1].set_ylabel(r'$\widetilde\theta$')
    axs[2].set_title('(c) Radius and bounds')
    axs[2].set_ylabel(r'$W=\sqrt{e^2+\widetilde\theta^2}$')
    axs[0].axhline(0,color='black',lw=.5,alpha=.3)
    axs[1].axhline(0,color='black',lw=.5,alpha=.3)
    axs[2].text(.98,.94,'Solid: simulated radius\nDashed: analytic envelope',transform=axs[2].transAxes,ha='right',va='top',fontsize=6.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
    fig.suptitle('Constructed control example: identical data and gains',y=1.02,fontsize=9)
    fig.subplots_adjust(left=.065,right=.993,bottom=.22,top=.72,wspace=.36)
    target=ROOT/'paper'/'fig_theory_ood.pdf'
    fig.savefig(target,bbox_inches='tight',pad_inches=.025)
    fig.savefig(ROOT/'audit'/'fig_theory_ood_preview.png',dpi=180,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    return str(target.relative_to(ROOT))


def main():
    checks = {'scope':'Reproducible algebra checks and a newly constructed scalar plant; no original experimental outcomes are recomputed.'}
    checks['randomized_theorem_checks']=randomized_checks()
    checks['stale_memory_checks']=stale_memory_checks()
    curves,checks['constructed_example']=constructed_example()
    checks['figure']=make_figure(curves)
    checks['result']='passed'
    (ROOT/'audit'/'adaptive_ood_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    np.savez_compressed(ROOT/'audit'/'constructed_ood_trajectories.npz',
                        **{f'kappa_{int(k)}_{field}':value for k,curve in curves.items() for field,value in curve.items()})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
