# Answers to your list

Everything is in `ACC_ADAPTATION_PACKAGE.zip` (35 MB, self-contained).

## One correction first

**I do not have a working run.** My frozen policy FALLS at ~2 s in every run. That is the whole
reason for this exchange — yours reportedly completes the pickup, mine doesn't. So there is no
"successful `env_dump.npz`" from my side (your item 4). `reference_env_dump.npz` in the package is
my **failing** run, for you to diff against yours.

Likewise `box_present=False` is not a placeholder — my scene genuinely has no box.

## Must-have (to run `run_dump.py`)

1. **Full package** — in the zip: `ace_adapt.py`, `run_mujoco_demo.py`, `run_dump.py`,
   `evaluate.py`, `viewer_demo.py`, `assets/`. Only deps are `numpy`, `mujoco`, `scipy`.
2. **Policy file** — `assets/x2_box_policy_v31.npz`
   sha256 `17D21A8301E32ACB9A91741EA73B449DEF4C4D126FDF7AC3EA5021750A34DB91`
   Its `meta_json.run_path` = `20260730_215012-x2_box_v31_flatfoot-locomotion/model_202500.pt`.
   From `baaqerfarhat/Agibot-humanoid`, `box_pickup/policy/`.
3. **Robot XML** — `assets/robot_full_flat_ground_excl.xml`
   sha256 `5727FE486FA7076136AF3C09EF34F8A6D82FCA8F85B17702D553F55B31DEF6DF`
   From `YujinAnn/agibot_control_functions`, branch `policy_rosBridge_sim`, `robots/x2/`.
   **No box in it.** Caveat: that repo's own `x2.yaml` shows this model is the harness for the
   20-DoF *walking* policy (it soft-locks waist/head/wrists). It may be the wrong model for the
   31-DoF box task — one of the things I want your answer on.
4. **My dump** — `reference_env_dump.npz` (seed 600, 61 steps to the fall). Failing, not working.

## Must-have (to run the adaptation)

5. **Scripts** — `evaluate.py` (paired frozen vs adapted + matched null), `ace_adapt.py` (the
   method itself). ACE offline attribution is not in the package; results are in `RESULTS.md` §5.
6. **Commands**
   ```
   python run_mujoco_demo.py --seeds 32                    # frozen baseline
   python run_mujoco_demo.py --adapt --seeds 32            # adapted
   python evaluate.py --seeds 32 --seed0 600               # paired + sign tests
   python evaluate.py --seeds 32 --seed0 600 --null        # + matched random control
   python run_dump.py --out mine.npz --seed 600 --steps 200
   ```
7. **Config** — defaults in `ace_adapt.AdaptConfig`: layer 2, Γ=3e-4, leak 1e-2, gx_level 1,
   error mask legs+waist, engage_step 0. Loop: control 50 Hz, physics 500 Hz (decimation 10),
   gain scale 1.2, leg target filter 0.8 (legs only), rate limit 0.15 rad/step. Full contract in
   `INTEGRATION.md`.
8. **Expected outputs** — `RESULTS.md`. Summary: adaptation cuts leg tracking error ~37%
   (14.7°→9.2°) and delays the fall ~14% (1.94→2.22 s). **It never prevents the fall** (0/32
   either way). No saved adapted weights or video; regenerate with the commands above.

## Nice-to-have

9. **Env** — Python 3.11, `mujoco` 3.11.0, numpy, scipy. No conda needed. My runs were on
   Windows native and WSL Ubuntu-22.04; both agree.
10. **MuJoCo or Isaac** — everything above is MuJoCo. The policy was trained in Isaac (holosoma).
    I also ran it in holosoma's MuJoCo model: survival 2.15 s vs 1.97 s, still falls.
11. **Box** — not in my scene at all. **Is it in yours, as a physical body?**
12. **ACE results** — `RESULTS.md` §5: 24 seeds × 40 draws/layer. Selects layer 2
    (ACE −0.125 ± 0.081); layer 3 harmful (+0.206 ± 0.098).

## What I need back

`run_dump.py` output from your **working** run, plus:
- which policy file (path + size)
- which reference motion (path + number of frames) — mine is the 734-frame one baked into the
  `.npz`; the repo also ships a *different* 434-frame clip
- which robot model file
- is the box physically in the scene
