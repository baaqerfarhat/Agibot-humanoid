"""PLAN_X2_DEPLOYMENT Phase A — the X2 manipulation harness a VLA can be driven through.

Supplies the three things GR00T needs from an X2 scene, plus the scoring the plan's gates
are written against:

    RGB            egocentric camera on head_pitch_link, rendered at the VLA's input size
    proprioception 31-DoF joint position/velocity from the articulation
    language       the task string (fixed for box pickup)
    success        adaptation.adapt_experiments_isaac.box_metrics -- lifted / carried /
                   placed, not merely "stayed upright"

**One process, many episodes.** Camera-enabled startup initialises the RTX renderer and
costs MINUTES against ~20 s for a physics-only boot, so the harness is a long-lived object:
build it once, call `run_episode()` repeatedly. Relaunching per episode makes startup
dominate any multi-episode screen.

Four boot gotchas, each of which fails without naming its cause (see the module notes in
`docs/PLAN_X2_DEPLOYMENT.md`):
  1. OMNI_KIT_ACCEPT_EULA=YES, or Isaac blocks on an interactive prompt.
  2. The saved v31 config embeds an ABSOLUTE motion path from another machine.
  3. paths.enter_holosoma() must precede every holosoma import.
  4. Cameras need `logger.video.enabled = True` BEFORE the app is created -- that is what
     passes --enable_cameras, without which replicator's OmniGraph never exists and
     annotator.attach() dies with "Invalid object in Py_Graph".

CAMERA STATUS (20 Aug): NOT WORKING, and the failure is upstream of this file.
The prim attaches, the render product is created, and the app launches with
`enable_cameras=True` (verified in args_cli) -- but the annotator returns shape (0,)
forever. So does holosoma's OWN video-recorder annotator in the same process, which is
the tell: this is not a bug in how we attach. Drive mechanisms tried, all empty:
sim.sim.step(render=True), sim.sim.render(), video_recorder.capture_frame(),
SimulationApp.update(). rep.orchestrator.step() -- replicator's documented driver --
blocks indefinitely instead.

Next things to try, in order: (a) run holosoma's own eval CLI with video recording on and
see whether IT ever writes a non-empty mp4 -- if not, this is a pre-existing environment
problem rather than anything the harness introduced; (b) use IsaacLab's Camera /
TiledCamera sensor instead of raw replicator, which owns the graph lifecycle properly.

Usage:
    h = X2Harness(with_camera=True)
    h.setup()
    for seed in range(20):
        r = h.run_episode(seed)
        print(seed, r["success"], r["frames"].shape if r["frames"] is not None else None)
    h.close()
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ADAPTATION = Path(__file__).resolve().parents[2] / "adaptation"
# ace_adapt (ExportedPolicy) lives in the vendored package dir, and the adaptation
# scripts put BOTH on sys.path -- see the header of adapt_experiments_isaac.py.
sys.path.insert(0, str(ADAPTATION / "ACC_ADAPTATION_PACKAGE"))
sys.path.insert(0, str(ADAPTATION))

TASK_PROMPT = "pick up the box and carry it"
CAM_RES = (224, 224)          # GR00T / pi0 native input size
HEAD_LINK = "head_pitch_link"


class X2Harness:
    def __init__(self, with_camera: bool = True, res=CAM_RES, steps: int = 500,
                 env_id: int = 0):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        self.with_camera = with_camera
        self.res = res
        self.steps = steps
        self.env_id = env_id
        self._ann = None
        self._rep = None

    # ------------------------------------------------------------------ setup
    def setup(self):
        import paths
        self.paths = paths
        ckpt = paths.resolve_ckpt()
        paths.enter_holosoma()

        import dataclasses
        from holosoma.utils.eval_utils import (CheckpointConfig, init_eval_logging,
                                               load_checkpoint,
                                               load_saved_experiment_config)
        from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
        from holosoma.utils.helpers import get_class
        from holosoma.utils.sim_utils import setup_simulation_environment
        from holosoma.utils.config_utils import CONFIG_NAME
        init_eval_logging()

        ckpt_cfg = CheckpointConfig(checkpoint=str(ckpt))
        saved_cfg, saved_wandb = load_saved_experiment_config(ckpt_cfg)

        # (2) the embedded motion path points at the original author's machine
        mt = saved_cfg.command.setup_terms["motion_command"]
        mc = mt.params["motion_config"]
        if isinstance(mc, dict):
            mc = dict(mc)
            mc.update(motion_file=str(paths.MOTION), motion_dir="",
                      use_adaptive_timesteps_sampler=False,
                      start_at_timestep_zero_prob=1.0, freeze_at_timestep_zero_prob=0.0)
            mt.params["motion_config"] = mc
        else:
            mt.params["motion_config"] = dataclasses.replace(
                mc, motion_file=str(paths.MOTION), motion_dir="",
                use_adaptive_timesteps_sampler=False, start_at_timestep_zero_prob=1.0,
                freeze_at_timestep_zero_prob=0.0)
        saved_cfg.termination.terms.pop("bad_tracking", None)

        eval_cfg = saved_cfg.get_eval_config()
        object.__setattr__(eval_cfg.training, "headless", True)
        object.__setattr__(eval_cfg.training, "num_envs", 1)
        object.__setattr__(eval_cfg.training, "max_eval_steps", self.steps)
        if self.with_camera:
            # (4) THIS is what passes --enable_cameras at launch.
            object.__setattr__(eval_cfg.logger.video, "enabled", True)

        env, device, app = setup_simulation_environment(eval_cfg)
        self.env, self.device, self.app = env, device, app

        log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(),
                                     task_name="eval")
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_cfg.save_config(str(log_dir / CONFIG_NAME))
        object.__setattr__(eval_cfg.algo.config, "eval_callbacks", {})

        checkpoint = load_checkpoint(ckpt_cfg.checkpoint, str(log_dir))
        algo = get_class(eval_cfg.algo._target_)(
            device=device, env=env, config=eval_cfg.algo.config,
            log_dir=str(log_dir), multi_gpu_cfg=None)
        algo.setup()
        algo.attach_checkpoint_metadata(saved_cfg, saved_wandb)
        algo.load(str(checkpoint))
        algo._create_eval_callbacks()
        algo._pre_evaluate_policy()
        self.algo = algo
        self.task = algo._unwrap_env()
        self.ctrl_dt = float(self.task.dt)

        from ace_adapt import ExportedPolicy
        self.pol = ExportedPolicy(str(paths.POLICY_NPZ))
        self.ref_pos = self.pol.ref_pos
        self.task.reset_all()

        if self.with_camera:
            self._attach_camera()
        return self

    def _attach_camera(self):
        """Egocentric camera on the head link. Must run AFTER the app exists."""
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except ImportError:
            from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        head = f"/World/envs/env_{self.env_id}/Robot/{HEAD_LINK}"
        cam_path = head + "/EgoCam"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(18.0)
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.0, 0.05))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))   # look down +x

        rp = rep.create.render_product(cam_path, self.res)
        ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu",
                                                  do_array_copy=True)
        ann.attach([rp])
        self._rep, self._ann = rep, ann
        print(f"  [cam] attached {self.res} at {cam_path}", flush=True)

    # -------------------------------------------------------------- capture
    def grab_rgb(self):
        """One RGB frame, or None while the render graph is still warming up."""
        if self._ann is None:
            return None
        # NOTE: do NOT call rep.orchestrator.step() here. It can block indefinitely when
        # the render pipeline is driven by the sim loop rather than by replicator -- that
        # call, not GPU contention, is what hung three probe runs for an hour each. The
        # annotator is populated by sim.sim.step(render=True); it simply returns an empty
        # buffer for the first few frames while the graph warms up.
        # The rollout steps PHYSICS only; nothing drives the renderer, so the annotator
        # stays empty unless we ask for a render pass here.
        try:
            self.task.simulator.sim.render()
        except Exception as e:
            if not getattr(self, "_render_warned", False):
                print("  [cam] render() failed:", type(e).__name__, str(e)[:60])
                self._render_warned = True
        a = np.asarray(self._ann.get_data())
        if a.size == 0:
            return None
        return a[:, :, :3].astype(np.uint8) if a.ndim == 3 and a.shape[2] >= 3 else a

    # -------------------------------------------------------------- episodes
    def run_episode(self, seed: int, capture_every: int = 5, max_steps: int | None = None):
        """One scored episode. Returns the rollout record plus box metrics and frames."""
        import eval_adapt_isaac as base
        from adapt_experiments_isaac import box_metrics
        from holosoma.utils.common import seeding

        seeding(seed, torch_deterministic=False)
        frames, steps_at = [], []

        # _rollout on this branch has no on_step hook (that lives on lift-feasibility),
        # but obs_hook fires once per control step and can pass the obs through
        # untouched, which is all a frame grab needs.
        step_n = [0]

        def obs_hook(obs):
            i = step_n[0]; step_n[0] += 1
            if self.with_camera and i % capture_every == 0:
                f = self.grab_rgb()
                if f is not None:
                    frames.append(f); steps_at.append(i)
            return obs

        r = base._rollout(self.algo, self.task, None,
                          max_steps or self.steps, self.ref_pos, self.ctrl_dt,
                          obs_hook=obs_hook)
        r.update(box_metrics(r["records"], self.ctrl_dt))
        r["frames"] = np.stack(frames) if frames else None
        r["frame_steps"] = steps_at
        r["prompt"] = TASK_PROMPT
        r["seed"] = seed
        return r

    def observation(self, rgb=None):
        """The observation dict a VLA client consumes. State is 31-DoF joint pos/vel."""
        robot = self.task.simulator._robot
        return {
            "image": rgb if rgb is not None else self.grab_rgb(),
            "state": np.concatenate([
                robot.data.joint_pos[0].detach().cpu().numpy(),
                robot.data.joint_vel[0].detach().cpu().numpy()]).astype(np.float32),
            "prompt": TASK_PROMPT,
        }

    def close(self):
        try:
            from holosoma.utils.sim_utils import close_simulation_app
            if getattr(self, "app", None):
                close_simulation_app(self.app)
        except Exception:
            pass


def main():
    """Phase A gate: the trained policy runs through the harness and it reports success."""
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--out", default="phaseA_gate.json")
    a = ap.parse_args()

    out_path = Path(a.out).resolve()
    h = X2Harness(with_camera=not a.no_camera, steps=a.steps).setup()
    rows = []
    for s in range(a.episodes):
        r = h.run_episode(s)
        fr = r["frames"]
        print(f"  seed {s}: survival {r['survival']:4d}  lifted {r['lifted']}  "
              f"carry {r['carry_s']:.2f}s  placed {r['placed']}  success {r['success']}  "
              f"frames {None if fr is None else fr.shape}", flush=True)
        rows.append({k: (float(r[k]) if isinstance(r[k], (int, float, np.floating)) else r[k])
                     for k in ("seed", "survival", "lifted", "carry_s", "max_box_z",
                               "final_dist", "placed", "success")}
                    | {"n_frames": 0 if fr is None else int(fr.shape[0])})
    n_ok = sum(1 for r in rows if r["success"])
    n_fr = sum(r["n_frames"] for r in rows)
    print(f"\nPHASE A GATE: success {n_ok}/{len(rows)}   frames captured {n_fr}")
    print("  gate requires: a non-VLA policy completes the task AND the harness scores it")
    print(f"  -> {'PASS' if n_ok > 0 and (a.no_camera or n_fr > 0) else 'FAIL'}")
    out_path.write_text(json.dumps({"episodes": rows, "success": n_ok,
                                    "n": len(rows), "frames": n_fr}, indent=2))
    print(f"wrote {out_path}")
    h.close()


if __name__ == "__main__":
    main()
