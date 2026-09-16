#!/usr/bin/env python3
"""Freeze the FrozenYet Adaptive configuration bundle (EXPERIMENT_PLAN.md section 2, item 6): the actual serialised
predictor W (6 x (K+2): K+1 FIR taps per channel and an affine offset), sensitivity M, OUT scaling, correction support,
observer constants, normalisation support, and the hashes of every source they were derived from.

Resolution of the historical configurations (written explicitly, as the plan requires):
  * W is the deployed reference calibration: the ridge FIR (K = 6, ridge 1e-2) fitted on the shipped healthy log
    results/phase05/error_signal_so3.json (three episodes, initial state 45), exactly as the headline runner and the
    E1 v2 continuation driver fit it (adaptive_law.fit_plant with default arguments). It is NOT the E2 thirty-episode
    partition predictor U (r_y tap sum 0.1458) nor the response-constrained C: those were evaluated in E2 and Q6 and
    are development information here; the headline task results and the E1 continuations used this fit.
  * M is the probed sensitivity results/phase05/openloop_so3.json (the diagonal matrix both arms use).
  * Constants: gamma .08, deadzone .008, normalisation scale .15, all-channel normalisation, rotation support {3,4,5}.
  * Correction caps are scenario parameters (.05 reference / .025 cap_half / .05 healthy), deliberately different from
    the historical .15 projection box; they are not part of the bundle.
"""
import argparse, hashlib, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE))
import adaptive_law as AL


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant-log", default=str(HERE.parent / "results/phase05/error_signal_so3.json"))
    ap.add_argument("--openloop", default=str(HERE.parent / "results/phase05/openloop_so3.json"))
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    W = AL.fit_plant(a.plant_log); M = np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"], float)
    assert W.shape == (6, AL.K_FIR + 2) and M.shape == (6, 6)
    bundle = dict(name="fya_recovery_deadline_v1 configuration bundle", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  K=AL.K_FIR, W=W.tolist(), W_tap_sums=W[:, :AL.K_FIR + 1].sum(1).tolist(), W_offset=W[:, -1].tolist(), M=M.tolist(), M_diag=np.diag(M).tolist(),
                  OUT=AL.OUT.tolist(), corr_dims=[3, 4, 5], gamma=0.08, dead=0.008, norm_r=0.15, norm_channels="all",
                  fir_convention="K = 6 means seven command samples (current and six previous), newest first; W[:, -1] is the affine offset",
                  sources=dict(plant_log=a.plant_log, plant_log_sha256=sha(a.plant_log), openloop=a.openloop, openloop_sha256=sha(a.openloop),
                               adaptive_law_sha256=sha(HERE / "adaptive_law.py"), fit="adaptive_law.fit_plant(plant_log) default ridge 1e-2, per-channel"),
                  resolution="historical reference calibration (headline runner, E1 v2); not the E2 partition predictor U or the constrained C")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(bundle, indent=1))
    print("wrote", a.out, "sha256", sha(a.out)); print("W tap sums", np.round(bundle["W_tap_sums"], 4).tolist(), "M diag", np.round(bundle["M_diag"], 4).tolist())


if __name__ == "__main__":
    main()
