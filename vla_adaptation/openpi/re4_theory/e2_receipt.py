#!/usr/bin/env python3
"""Freeze the E2 core artifacts (iclr2027/EXPERIMENTS_ASAP.md, section 2): both fitted predictor
arrays (U unconstrained, C with the rotation-y tap sum pinned to the qualified finite-horizon value),
their hashes, the fitting episode ids, the probe sources, M and its hash, OUT, masks, observer
constants, code revision, and the ordered manifests. Verifies numerically that the five
unconstrained channels are identical between U and C."""
import hashlib, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(HERE / "openpi"))
import adaptive_law as AL
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
log = HERE / "results/iclr_unified_v1/sources/e2_fit_qual_spatial_init40_43.json"; fit = list(range(30))
openloop = HERE / "results/phase05/openloop_so3.json"; M = np.array(json.loads(openloop.read_text())["M"])
U = AL.fit_plant(str(log), episodes=fit); C = AL.fit_plant(str(log), episodes=fit, dc={4: 0.254})
d = json.loads(log.read_text()); keys = d["records"][0]["episode_keys"]
same = [bool(np.allclose(U[i], C[i], atol=0, rtol=0)) for i in range(6)]
probes = sorted(str(p.relative_to(HERE)) for p in (HERE / "results/iclr_unified_v1/E2_probe").glob("*.json"))
manifests = {s: dict(path=f"results/iclr_unified_v1/manifests/{s}_E2_core.json", sha256=sha(HERE / f"results/iclr_unified_v1/manifests/{s}_E2_core.json"),
                     ordered_keys=[(m["task"], m["init"], m["sampler_seed"]) for m in json.loads((HERE / f"results/iclr_unified_v1/manifests/{s}_E2_core.json").read_text())["scenarios"]])
             for s in ("libero_spatial", "libero_10")}
rec = dict(created="2026-09-15", prereg="prereg_records/PREREG_E2_CORE.md", code_revision=subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           runner_sha256=sha(HERE / "openpi/adaptive_law.py"), server_sha256=sha(HERE / "openpi/ace_server.py"),
           predictor=dict(form="per-axis FIR, K=6, ridge lambda=0.01 (X'X + lam I), intercept free; taps + intercept per channel", K=AL.K_FIR, ridge=0.01,
                          fit_log=str(log.relative_to(HERE)), fit_log_sha256=sha(log), fit_episode_indices=fit, fit_episode_keys=[keys[i] for i in fit],
                          U=U.tolist(), U_sha256=hashlib.sha256(U.tobytes()).hexdigest(), C=C.tolist(), C_sha256=hashlib.sha256(C.tobytes()).hexdigest(),
                          U_tap_sums=U[:, :7].sum(1).tolist(), C_tap_sums=C[:, :7].sum(1).tolist(), constraint=dict(channel=4, tap_sum=0.254, label="matched finite-horizon (40-step) response, PREREG_E2_PROBE_QUALIFICATION.md"),
                          unconstrained_channels_identical=same, probe_sources=probes, probe_sha256={p: sha(HERE / p) for p in probes}),
           sensitivity=dict(path=str(openloop.relative_to(HERE)), sha256=sha(openloop), M=M.tolist(), M_diag=np.diag(M).tolist(), used_by="both arms, unchanged"),
           OUT=AL.OUT.tolist(), correction_mask=[3, 4, 5], observer=dict(law="innov", norm_channels="corrected", gamma=0.08, dead=0.008, norm_r=0.15, clip=0.15, deadzone_mode="zero", bias="none"),
           fault=dict(faulted="+0.05 on all six action channels", healthy=0.0, oracle="--static-corr -0.05 x6 masked to {3,4,5} (matched-authority, translation remains)"),
           sampler=dict(schedule="fold_in(fold_in(key(sampler_seed), episode_ordinal), call_index), control re-issued at every episode start; the handshake's probe calls consume the first keys identically in every arm",
                        note="the key depends on the episode ordinal: arms must run the manifest in the same order; preserved in episodes.csv and telemetry"),
           manifests=manifests, arms=["healthy_off", "healthy_U", "healthy_C", "faulted_off", "faulted_U", "faulted_C", "faulted_oracle"])
out = HERE / "results/iclr_unified_v1/E2_core/RECEIPT.json"; out.write_text(json.dumps(rec, indent=1))
print("unconstrained channels identical:", same, "| U/C r_y tap sums", round(rec["predictor"]["U_tap_sums"][4], 4), round(rec["predictor"]["C_tap_sums"][4], 4), "| wrote", out.relative_to(HERE))
