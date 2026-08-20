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
N_DOF, N_ROOT = 31, 6
HAND = bodies.index("right_sphere_hand_link")
FEET = [bodies.index("left_ankle_roll_link"), bodies.index("right_ankle_roll_link")]
sc = sim.sim
print("=== what is one sim.sim.step() actually worth? ===")
for m in ("get_physics_dt","get_rendering_dt"):
    try: print(f"  {m}() = {getattr(sc, m)()}")
    except Exception as e: print(f"  {m}: {type(e).__name__}")
print(f"  env.dt={getattr(env,'dt',None)}  env.sim_dt={getattr(env,'sim_dt',None)}"
      f"  decimation={getattr(env,'decimation',None)}")
cfg = getattr(sc, 'cfg', None)
for a in ("dt","substeps","render_interval"):
    if cfg is not None and hasattr(cfg, a): print(f"  sim.cfg.{a} = {getattr(cfg,a)}")

env.reset_all()
for _ in range(40): sc.step(render=False); robot.update(0.005)
DT_PHYS = sc.get_physics_dt()
jp = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone(); root = robot.data.root_state_w.clone()
def set_state():
    robot.write_root_state_to_sim(root.clone()); robot.write_joint_state_to_sim(jp.clone(), jv.clone())
def probe(tau, nstep=1):
    set_state()
    t = torch.as_tensor(tau, dtype=jp.dtype, device=jp.device).view(1,-1)
    for _ in range(nstep):
        robot.set_joint_effort_target(t); robot.write_data_to_sim(); sc.step(render=False); robot.update(DT_PHYS)
    return robot.data.body_state_w[0, HAND, 7:13].detach().cpu().numpy().astype(float)

set_state()
Mraw = view.get_generalized_mass_matrices()[0].detach().cpu().numpy().astype(float)
J = view.get_jacobians()[0].detach().cpu().numpy().astype(float)
Minv = np.linalg.pinv(Mraw)
St = np.zeros((N_ROOT+N_DOF, N_DOF)); St[N_ROOT+dof_ids, np.arange(N_DOF)] = 1.0
G = J[HAND] @ Minv @ St
EPS = 50.0
rng = np.random.default_rng(11)
dirs = [d/np.linalg.norm(d) for d in (rng.standard_normal(N_DOF) for _ in range(16))]
print(f"\nusing DT_PHYS = {DT_PHYS}")
for nstep in (1, 2, 4):
    v0 = probe(np.zeros(N_DOF), nstep)
    FD = [(probe(EPS*d, nstep) - v0)/(DT_PHYS*nstep)/EPS for d in dirs]
    cs = [float(f@(G@d)/(np.linalg.norm(f)*np.linalg.norm(G@d)+1e-300)) for f,d in zip(FD,dirs)]
    rt = [float(np.linalg.norm(f)/(np.linalg.norm(G@d)+1e-300)) for f,d in zip(FD,dirs)]
    print(f"  nstep={nstep}  cos {np.mean(cs):+.4f}   |fd|/|pred| {np.median(rt):.4f}")
    if nstep == 1:
        lin = [float(f[:3]@(G@d)[:3]/(np.linalg.norm(f[:3])*np.linalg.norm((G@d)[:3])+1e-300)) for f,d in zip(FD,dirs)]
        ang = [float(f[3:]@(G@d)[3:]/(np.linalg.norm(f[3:])*np.linalg.norm((G@d)[3:])+1e-300)) for f,d in zip(FD,dirs)]
        print(f"    split: linear-only cos {np.mean(lin):+.4f}   angular-only cos {np.mean(ang):+.4f}")
if app: close_simulation_app(app)
