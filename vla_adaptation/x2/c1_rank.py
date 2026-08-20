"""PLAN_X2_DEPLOYMENT Phase C1 (P1): rank of the action->task map G on X2, in
BOX-PICKUP configurations.

    G = J_task * Mc^-1 * S^T * Kp * D                             (PLAN_CROSS_EMBODIMENT eq. 1)
    Mc^-1 = M^-1 - M^-1 Jc^T (Jc M^-1 Jc^T)^+ Jc M^-1             (contact-consistent)

G depends only on configuration, so the robot is posed kinematically at frames of the
reference box-pickup clip -- no policy rollout needed. Prop. 1's premise is rank(G) = p.
Everything stays in PhysX generalized coordinates (6 root DoF first, then 31 joints);
Kp and D are policy-ordered and mapped in via dof_ids, the permutation that
make_mass_matrix_fn warns is silently scrambling if skipped.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("adaptation").resolve()))
import paths, dataclasses

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("c1_rank.json")
ckpt = paths.resolve_ckpt()
paths.enter_holosoma()
from holosoma.utils.eval_utils import (CheckpointConfig, init_eval_logging,
                                       load_saved_experiment_config)
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
init_eval_logging()

saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(ckpt)))
mt = saved_cfg.command.setup_terms["motion_command"]
mc = mt.params["motion_config"]
if isinstance(mc, dict):
    mc = dict(mc); mc["motion_file"] = str(paths.MOTION); mc["motion_dir"] = ""
    mc["use_adaptive_timesteps_sampler"] = False; mc["start_at_timestep_zero_prob"] = 1.0
    mt.params["motion_config"] = mc
else:
    mt.params["motion_config"] = dataclasses.replace(
        mc, motion_file=str(paths.MOTION), motion_dir="",
        use_adaptive_timesteps_sampler=False, start_at_timestep_zero_prob=1.0)
saved_cfg.termination.terms.pop("bad_tracking", None)

eval_cfg = saved_cfg.get_eval_config()
object.__setattr__(eval_cfg.training, "headless", True)
object.__setattr__(eval_cfg.training, "num_envs", 1)
object.__setattr__(eval_cfg.training, "max_eval_steps", 10)
env, device, app = setup_simulation_environment(eval_cfg)

sim = env.simulator
robot = sim._robot
view = robot.root_physx_view
bodies = list(robot.data.body_names)
dof_ids = np.asarray(sim.dof_ids, dtype=int)
N_DOF, N_ROOT = 31, 6

HANDS = [bodies.index("left_sphere_hand_link"), bodies.index("right_sphere_hand_link")]
FEET  = [bodies.index("left_ankle_roll_link"),  bodies.index("right_ankle_roll_link")]
print(f"hands {HANDS}  feet {FEET}", flush=True)

# Kp and action scale: policy-ordered -> physx rows
rows = N_ROOT + dof_ids
kp = np.asarray(env.p_gains.detach().cpu().numpy(), float).reshape(-1)
dsc = np.asarray(env.action_scales.detach().cpu().numpy(), float).reshape(-1)
print(f"Kp range [{kp.min():.1f}, {kp.max():.1f}]   action_scale range [{dsc.min():.3f}, {dsc.max():.3f}]")
St = np.zeros((N_ROOT + N_DOF, N_DOF))       # generalized-force selection
St[rows, np.arange(N_DOF)] = 1.0
B = St @ np.diag(kp * dsc)                   # (37, 31): action -> generalized force

ref = np.load(paths.MOTION)
qkey = next(k for k in ref.files if "dof_pos" in k or "joint_pos" in k)
Q = np.asarray(ref[qkey], float)
print(f"motion '{qkey}' {Q.shape}")

def snapshot():
    M = view.get_generalized_mass_matrices()[0].detach().cpu().numpy().astype(float)
    J = view.get_jacobians()[0].detach().cpu().numpy().astype(float)
    return M, J

def gmat(M, J, task_bodies, contact_bodies):
    Minv = np.linalg.pinv(M)
    if contact_bodies:
        Jc = np.vstack([J[b] for b in contact_bodies])
        Mc = Minv - Minv @ Jc.T @ np.linalg.pinv(Jc @ Minv @ Jc.T) @ Jc @ Minv
    else:
        Mc = Minv
    Jt = np.vstack([J[b] for b in task_bodies])
    return Jt @ Mc @ B

rng = np.random.default_rng(0)
frames = np.linspace(0, len(Q) - 1, 12).astype(int)
recs, Mtrace = [], []
for f in frames:
    q = Q[f][:N_DOF]
    jp = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
    jp[0, dof_ids if len(dof_ids) == N_DOF else slice(None)] = \
        __import__("torch").as_tensor(q, dtype=jp.dtype, device=jp.device)
    robot.write_joint_state_to_sim(jp, jv * 0.0)
    sim.step() if hasattr(sim, "step") else None
    M, J = snapshot()
    Mtrace.append(float(np.trace(M)))
    for label, tb in (("right_hand", [HANDS[1]]), ("both_hands", HANDS)):
        G = gmat(M, J, tb, FEET)
        s = np.linalg.svd(G, compute_uv=False)
        p = G.shape[0]
        recs.append({"frame": int(f), "task": label, "p": p,
                     "rank": int(np.linalg.matrix_rank(G, tol=s.max() * 1e-6)),
                     "sv_max": float(s.max()), "sv_min": float(s.min()),
                     "cond": float(s.max() / max(s.min(), 1e-300))})

# SELF-CHECK: if the kinematic writes did not take, M is identical at every frame and
# every number above is one configuration reported twelve times.
spread = (max(Mtrace) - min(Mtrace)) / max(abs(np.mean(Mtrace)), 1e-12)
print(f"\nSELF-CHECK trace(M) relative spread across frames: {spread:.3e} "
      f"({'POSES VARY - ok' if spread > 1e-6 else 'IDENTICAL - writes did NOT take'})")

for lab in ("right_hand", "both_hands"):
    rs = [r for r in recs if r["task"] == lab]
    p = rs[0]["p"]; full = sum(1 for r in rs if r["rank"] == p)
    print(f"\n{lab}: p={p}  full rank in {full}/{len(rs)} frames")
    print(f"  sv_min  min {min(r['sv_min'] for r in rs):.4g}  med {np.median([r['sv_min'] for r in rs]):.4g}")
    print(f"  cond    med {np.median([r['cond'] for r in rs]):.4g}  max {max(r['cond'] for r in rs):.4g}")

OUT.write_text(json.dumps({"checkpoint": str(ckpt), "motion": str(paths.MOTION),
                           "frames": frames.tolist(), "trace_M_spread": spread,
                           "kp_range": [kp.min(), kp.max()], "records": recs}, indent=2))
print(f"\nwrote {OUT}")
if app: close_simulation_app(app)
