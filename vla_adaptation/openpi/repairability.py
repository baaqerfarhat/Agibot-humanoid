"""Ground truth: can each layer actually IMPLEMENT the repair? -- one cheap test per site.

Every selection criterion so far has been judged against a single known-repairable site
(`action_out_proj/bias`, verified 100%). Choosing an estimator by how well it ranks one known
answer is overfitting, so this measures the answer for EVERY site and produces the vector the
criteria should be scored against.

Method, per site:
  1. Probe with K seeded draws; record each draw's mean action-change vector v_k in R^7.
  2. Least-squares the coefficients a that make sum_k a_k v_k point along u*, the normalised
     action shift that cancels the fault, then scale to the magnitude the repair needs.
  3. Apply that directed edit (server-side `combo`) and run episodes with the fault on.

Repairability(l) = success rate achieved. A layer that cannot produce a constant offset --
because its effect depends on the input -- will fail here no matter how "important" it is,
which is exactly the distinction ACE cannot draw.
"""
from __future__ import annotations

import argparse, json, pathlib, time
import numpy as np
import main as lm
from openpi_client import image_tools
from paired_probe import Probe
from ace_screen_v2 import SITES
from reachability_score import repair_direction, element, repair_magnitude
import ace_screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--matched-c", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--obs", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--scales", default="0.5,1.0,1.5")
    a = ap.parse_args()
    matched = json.loads(a.matched_c.read_text())
    u = repair_direction()          # unit direction
    u_mag = repair_magnitude()      # the shift actually required
    scales = [float(x) for x in a.scales.split(",")]

    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=True))
    els = []
    for tid in range(4):
        env, desc, inits = pr.env_for(tid)
        env.reset(); obs = env.set_init_state(inits[10])
        for _ in range(12):
            obs, *_ = env.step(lm.LIBERO_DUMMY_ACTION)
        plan = []
        for step in range(a.obs // 4 * 15):
            if step % 15 == 0:
                els.append(element(obs, desc))
            if not plan:
                plan = list(pr.client.infer(els[-1])["actions"][: a.replan_steps])
            obs, _, done, _ = env.step(np.asarray(plan.pop(0), float).tolist())
            if done: break
    base_a = [np.asarray(pr.client.infer(e)["actions"], float)[:, :7].mean(0) for e in els]

    eps = [((i % 10), 10 + i // 10) for i in range(a.episodes)]
    ace_screen.SEVERITY = 0.05
    state = {"blocks": []}
    if a.out.exists():
        state = json.loads(a.out.read_text())
    done = {(b["site"], b["scale"]) for b in state["blocks"]}

    print(f"{len(els)} obs, {a.probes} probes/site, {a.episodes} episodes/test\n")
    for s in SITES:
        if s not in matched:
            continue
        c = matched[s]["c"]
        V = []
        for k in range(a.probes):
            pr.control(dict(site=s, seed=3000 + k, c_rel=c, pin_rng=True, dims=7))
            dA = np.array([np.asarray(pr.client.infer(e)["actions"], float)[:, :7].mean(0) - b
                           for e, b in zip(els, base_a)])
            V.append(dA.mean(0))
        V = np.array(V)                                  # (K,7) achievable mean effects
        coef, *_ = np.linalg.lstsq(V.T, u * u_mag, rcond=None)  # fit the REQUIRED shift
        pred = V.T @ coef
        cosine = float(pred @ u / (np.linalg.norm(pred) + 1e-12))
        print(f"{s:<34} directed-fit cos={cosine:+.3f}  |pred|={np.linalg.norm(pred):.4f}")
        for sc in scales:
            if (s, sc) in done:
                continue
            combo = [[3000 + k, float(sc * coef[k])] for k in range(a.probes)]
            pr.control(dict(site=s, combo=combo, c_rel=c, dims=7, pin_rng=False))
            t0 = time.time()
            succ = sum(pr.rollout(t, i)[0] for t, i in eps)
            rec = dict(site=s, scale=sc, cosine=cosine, successes=int(succ), n=len(eps),
                       success_rate=succ / len(eps), wall_s=round(time.time() - t0, 1))
            state["blocks"].append(rec)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(state, indent=1))
            print(f"    scale {sc:>4}  {succ}/{len(eps)} = {100*rec['success_rate']:>5.1f}%"
                  f"  ({rec['wall_s']}s)")
    print("=== REPAIRABILITY DONE")


if __name__ == "__main__":
    main()
