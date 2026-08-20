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
sys.path.insert(0, "/home/mtaheri/ws_AgibotX2/Agibot-humanoid/adaptation")
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
import torch
sim = env.simulator
robot = sim._robot
view = robot.root_physx_view
sim_ctx = sim.sim
DT = float(getattr(env, "sim_dt", 0.005))
bodies = list(robot.data.body_names); dof_ids = np.asarray(sim.dof_ids, dtype=int)
N_DOF, N_ROOT = 31, 6
HAND = bodies.index("right_sphere_hand_link")
FEET = [bodies.index("left_ankle_roll_link"), bodies.index("right_ankle_roll_link")]
rows = N_ROOT + dof_ids
St = np.zeros((N_ROOT + N_DOF, N_DOF)); St[rows, np.arange(N_DOF)] = 1.0
ref = np.load(paths.MOTION); Q = np.asarray(ref["joint_pos"], float)
q0 = Q[len(Q)//2][:N_DOF]

jp0 = robot.data.joint_pos.clone(); jv0 = robot.data.joint_vel.clone()
root0 = robot.data.root_state_w.clone()
jp0[0, :] = torch.as_tensor(q0, dtype=jp0.dtype, device=jp0.device)

def set_state():
    robot.write_root_state_to_sim(root0.clone())
    robot.write_joint_state_to_sim(jp0.clone(), jv0.clone() * 0.0)

def read_JM():
    M = view.get_generalized_mass_matrices()[0].detach().cpu().numpy().astype(float)
    J = view.get_jacobians()[0].detach().cpu().numpy().astype(float)
    return M, J

def hand_vel():
    return robot.data.body_state_w[0, HAND, 7:13].detach().cpu().numpy().astype(float)

def apply_and_step(tau):
    set_state()
    robot.set_joint_effort_target(torch.as_tensor(tau, dtype=jp0.dtype, device=jp0.device).view(1, -1))
    robot.write_data_to_sim()
    sim_ctx.step(render=False)
    robot.update(DT)
    return hand_vel()

print("\n" + "="*70)
print("GATE 0 — does J vary with the kinematic write? (C1 only checked M)")
set_state(); _, Ja = read_JM()
jp1 = jp0.clone(); jp1[0, :] = torch.as_tensor(Q[10][:N_DOF], dtype=jp0.dtype, device=jp0.device)
robot.write_joint_state_to_sim(jp1, jv0*0.0); _, Jb = read_JM()
dJ = np.abs(Ja[HAND] - Jb[HAND]).max()
print(f"  max|dJ_hand| between two poses: {dJ:.4e}   {'PASS' if dJ > 1e-6 else 'FAIL - J IS STALE'}")
set_state()

print("\nGATE 1 — replay determinism (same state + same torque, twice)")
tau0 = np.zeros(N_DOF)
v_a = apply_and_step(tau0); v_b = apply_and_step(tau0)
rep = np.abs(v_a - v_b).max()
print(f"  max|v_a - v_b| = {rep:.4e}   {'PASS' if rep < 1e-9 else 'FAIL - NOT DETERMINISTIC'}")

set_state(); M, J = read_JM()
Minv = np.linalg.pinv(M)
Jc = np.vstack([J[b] for b in FEET])
Mc = Minv - Minv @ Jc.T @ np.linalg.pinv(Jc @ Minv @ Jc.T) @ Jc @ Minv
Gtau = J[HAND] @ Mc @ St                                   # (6, 31) task accel per joint torque

print("\nGATE 2 — epsilon plateau (cosine of FD vs G, one fixed direction)")
rng = np.random.default_rng(0)
d = rng.standard_normal(N_DOF); d /= np.linalg.norm(d)
pred_dir = Gtau @ d
v0 = apply_and_step(tau0)
for eps in (1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0, 100.0):
    fd = (apply_and_step(eps * d) - v0) / DT / eps
    c = float(fd @ pred_dir / (np.linalg.norm(fd) * np.linalg.norm(pred_dir) + 1e-300))
    print(f"  eps={eps:>7.3g}  cos={c:+.4f}   |fd|={np.linalg.norm(fd):.4g}  |pred|={np.linalg.norm(pred_dir):.4g}")

print("\nDIAGNOSTIC — are the feet actually in contact, and does the contact model matter?")
set_state()
fz = robot.data.body_state_w[0, :, 2].detach().cpu().numpy()
print(f"  foot heights z: left {fz[FEET[0]]:.4f}  right {fz[FEET[1]]:.4f}   (ground z=0)")
print(f"  hand height z:  {fz[HAND]:.4f}   root z: {float(root0[0,2]):.4f}")
variants = {"both_feet(12 constraints)": FEET, "left_foot_only": [FEET[0]],
            "no_contact(free-floating)": []}
rng2 = np.random.default_rng(1)
dirs = [rng2.standard_normal(N_DOF) for _ in range(16)]
dirs = [d/np.linalg.norm(d) for d in dirs]
fds = []
for dk in dirs:
    fds.append((apply_and_step(1.0 * dk) - v0) / DT)
for name, cbs in variants.items():
    if cbs:
        Jc2 = np.vstack([J[b] for b in cbs])
        Mc2 = Minv - Minv @ Jc2.T @ np.linalg.pinv(Jc2 @ Minv @ Jc2.T) @ Jc2 @ Minv
    else:
        Mc2 = Minv
    G2 = J[HAND] @ Mc2 @ St
    cs2 = [float(f @ (G2@dk) / (np.linalg.norm(f)*np.linalg.norm(G2@dk)+1e-300))
           for f, dk in zip(fds, dirs)]
    rt = [float(np.linalg.norm(f)/(np.linalg.norm(G2@dk)+1e-300)) for f, dk in zip(fds, dirs)]
    print(f"  {name:<28} cos mean {np.mean(cs2):+.4f}  median {np.median(cs2):+.4f}"
          f"   |fd|/|pred| med {np.median(rt):.3f}")

print("\nGATE 3 — 32-direction holdout at the plateau eps")
EPS = 1.0
cs, ratios = [], []
for k in range(32):
    dk = rng.standard_normal(N_DOF); dk /= np.linalg.norm(dk)
    fd = (apply_and_step(EPS * dk) - v0) / DT / EPS
    pr = Gtau @ dk
    cs.append(float(fd @ pr / (np.linalg.norm(fd)*np.linalg.norm(pr) + 1e-300)))
    ratios.append(float(np.linalg.norm(fd) / (np.linalg.norm(pr) + 1e-300)))
cs = np.array(cs); ratios = np.array(ratios)
print(f"  cosine: mean {cs.mean():+.4f}  min {cs.min():+.4f}  median {np.median(cs):+.4f}")
print(f"  |fd|/|pred|: median {np.median(ratios):.4f}  IQR [{np.percentile(ratios,25):.3f}, {np.percentile(ratios,75):.3f}]")
print(f"  VERDICT: {'PASS' if cs.mean() > 0.9 else 'FAIL'} (walker gate was cos +0.954)")

import json
Path(sys.argv[1] if len(sys.argv)>1 else "c2.json").write_text(json.dumps(
    {"dt": DT, "dJ_max": dJ, "replay_max_diff": rep, "eps_used": EPS,
     "cos_mean": cs.mean(), "cos_min": cs.min(), "cos_median": float(np.median(cs)),
     "ratio_median": float(np.median(ratios)), "cosines": cs.tolist()}, indent=2))
if app: close_simulation_app(app)
