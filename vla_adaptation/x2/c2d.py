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
sim = env.simulator; robot = sim._robot; view = robot.root_physx_view
bodies = list(robot.data.body_names); dof_ids = np.asarray(sim.dof_ids, dtype=int)
N_DOF, N_ROOT = 31, 6; DT = float(getattr(env, "sim_dt", 0.005))
HAND = bodies.index("right_sphere_hand_link")
FEET = [bodies.index("left_ankle_roll_link"), bodies.index("right_ankle_roll_link")]

env.reset_all()
for _ in range(40): sim.sim.step(render=False); robot.update(DT)
jp = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
root = robot.data.root_state_w.clone()
def set_state():
    robot.write_root_state_to_sim(root.clone()); robot.write_joint_state_to_sim(jp.clone(), jv.clone())
def probe(tau):
    set_state()
    robot.set_joint_effort_target(torch.as_tensor(tau, dtype=jp.dtype, device=jp.device).view(1,-1))
    robot.write_data_to_sim(); sim.sim.step(render=False); robot.update(DT)
    return robot.data.body_state_w[0, HAND, 7:13].detach().cpu().numpy().astype(float)

set_state()
Mraw = view.get_generalized_mass_matrices()[0].detach().cpu().numpy().astype(float)
J    = view.get_jacobians()[0].detach().cpu().numpy().astype(float)

EPS = 50.0                       # well above the 1.9e-4 replay noise floor
v0 = probe(np.zeros(N_DOF))
rng = np.random.default_rng(11)
dirs = [d/np.linalg.norm(d) for d in (rng.standard_normal(N_DOF) for _ in range(16))]
FD = [ (probe(EPS*d) - v0)/DT/EPS for d in dirs ]
print(f"FD magnitudes: median |fd| = {np.median([np.linalg.norm(f) for f in FD]):.4g}")

def evaluate(label, Minv, rows):
    St = np.zeros((N_ROOT+N_DOF, N_DOF)); St[rows, np.arange(N_DOF)] = 1.0
    Jc = np.vstack([J[b] for b in FEET])
    Mc = Minv - Minv@Jc.T@np.linalg.pinv(Jc@Minv@Jc.T)@Jc@Minv
    for cn, Mx in (("free", Minv), ("contact", Mc)):
        G = J[HAND] @ Mx @ St
        cs = [float(f@(G@d)/(np.linalg.norm(f)*np.linalg.norm(G@d)+1e-300)) for f, d in zip(FD, dirs)]
        rt = [float(np.linalg.norm(f)/(np.linalg.norm(G@d)+1e-300)) for f, d in zip(FD, dirs)]
        print(f"  {label:<34} {cn:<8} cos {np.mean(cs):+.4f}  |fd|/|pred| {np.median(rt):.4f}")

rows_rootfirst  = N_ROOT + dof_ids
rows_jointfirst = dof_ids
print("\n=== which PhysX convention reproduces the sim? ===")
evaluate("pinv(M), root-first",  np.linalg.pinv(Mraw), rows_rootfirst)
evaluate("M as inverse, root-first", Mraw,             rows_rootfirst)
evaluate("pinv(M), joint-first", np.linalg.pinv(Mraw), rows_jointfirst)
evaluate("M as inverse, joint-first", Mraw,            rows_jointfirst)
print("\n(identity ordering, no dof_ids permutation)")
evaluate("pinv(M), root-first, no perm", np.linalg.pinv(Mraw), N_ROOT + np.arange(N_DOF))
evaluate("M as inverse, root-first, no perm", Mraw,            N_ROOT + np.arange(N_DOF))
if app: close_simulation_app(app)
