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
St = np.zeros((N_ROOT+N_DOF, N_DOF)); St[N_ROOT+dof_ids, np.arange(N_DOF)] = 1.0

env.reset_all()
for _ in range(40): sim.sim.step(render=False); robot.update(DT)   # settle properly
z = robot.data.body_state_w[0,:,2].detach().cpu().numpy()
print(f"settled: foot z L {z[FEET[0]]:.4f} R {z[FEET[1]]:.4f}  root z {float(robot.data.root_state_w[0,2]):.4f}")

print("\ncurrent actuator gains as seen by the sim:")
ks = robot.data.joint_stiffness[0].detach().cpu().numpy()
kd = robot.data.joint_damping[0].detach().cpu().numpy()
lim = robot.data.joint_effort_limits[0].detach().cpu().numpy() if hasattr(robot.data,'joint_effort_limits') else None
print(f"  stiffness [{ks.min():.1f}, {ks.max():.1f}]   damping [{kd.min():.2f}, {kd.max():.2f}]"
      + (f"   effort_limit [{lim.min():.1f}, {lim.max():.1f}]" if lim is not None else ""))

jp = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
root = robot.data.root_state_w.clone()
def set_state():
    robot.write_root_state_to_sim(root.clone())
    robot.write_joint_state_to_sim(jp.clone(), jv.clone())
def probe(tau):
    set_state()
    robot.set_joint_effort_target(torch.as_tensor(tau, dtype=jp.dtype, device=jp.device).view(1,-1))
    robot.write_data_to_sim(); sim.sim.step(render=False); robot.update(DT)
    return robot.data.body_state_w[0, HAND, 7:13].detach().cpu().numpy().astype(float)

def run(label):
    set_state()
    M = view.get_generalized_mass_matrices()[0].detach().cpu().numpy().astype(float)
    J = view.get_jacobians()[0].detach().cpu().numpy().astype(float)
    Minv = np.linalg.pinv(M)
    Jc = np.vstack([J[b] for b in FEET])
    Mc = Minv - Minv@Jc.T@np.linalg.pinv(Jc@Minv@Jc.T)@Jc@Minv
    G = J[HAND] @ Mc @ St
    v0 = probe(np.zeros(N_DOF))
    rng = np.random.default_rng(3); cs, rt = [], []
    for _ in range(16):
        d = rng.standard_normal(N_DOF); d /= np.linalg.norm(d)
        fd = (probe(1.0*d) - v0)/DT
        pr = G @ d
        cs.append(float(fd@pr/(np.linalg.norm(fd)*np.linalg.norm(pr)+1e-300)))
        rt.append(float(np.linalg.norm(fd)/(np.linalg.norm(pr)+1e-300)))
    print(f"  {label:<34} cos mean {np.mean(cs):+.4f} median {np.median(cs):+.4f}"
          f"   |fd|/|pred| med {np.median(rt):.4f}")
    return np.mean(cs), np.median(rt)

print("\n=== FD vs G, PD ACTIVE (as C2 ran it) ===")
run("PD active")
print("\n=== FD vs G, PD ZEROED (only my effort acts) ===")
zero = torch.zeros_like(robot.data.joint_stiffness)
robot.write_joint_stiffness_to_sim(zero); robot.write_joint_damping_to_sim(zero)
robot.update(DT)
ks2 = robot.data.joint_stiffness[0].detach().cpu().numpy()
print(f"  stiffness now [{ks2.min():.1f}, {ks2.max():.1f}]")
run("PD zeroed")
if app: close_simulation_app(app)
