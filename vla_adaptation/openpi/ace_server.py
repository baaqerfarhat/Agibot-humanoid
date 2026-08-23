"""Policy server for the ACE screen: serves pi0.5-LIBERO with one layer perturbed.

**Why this file exists instead of a two-line mutation of the model.** openpi serves through
`nnx_utils.module_jit`, which does `graphdef, state = nnx.split(module)` ONCE at wrap time
and closes over that `state` -- its own docstring says the module state is "frozen to
whatever it was when module_jit was called". So the obvious approach, mutating
`model.action_out_proj.kernel.value` in place between episodes, is a SILENT NO-OP through
the serving path: inference keeps using the captured state, every draw returns the baseline,
and an ACE screen built on it would spend 400 episodes measuring its own sampler noise.

Here the state is a jit ARGUMENT instead of a capture, so swapping it costs no
recompilation (same shapes and dtypes -> same signature -> cache hit) and the perturbation
is unambiguously what inference uses.

Control channel is a JSON file, because the client lives in a different venv (LIBERO needs
python 3.8). The client writes {site, draw, seed}; this server applies the draw and writes
an ACK carrying the RELATIVE displacement it actually applied. The client refuses to run
episodes until that ack reads c within tolerance -- the perturbation is verified live on
every draw rather than assumed.

    python ace_server.py --control /tmp/ace_control.json --ack /tmp/ace_ack.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.policies import policy_config as _pc
from openpi.training import config as _config
import openpi.shared.download as download
from openpi.serving import websocket_policy_server as _server

C_REL = 0.02   # prereg §3: ||Delta||_F / ||W||_F, identical across sites


@dataclasses.dataclass(frozen=True)
class Site:
    name: str
    path: str          # slash-joined path in the nnx state tree
    index: int | None  # leading-axis index for stacked layers, else None


SITES = [
    Site("action_out_proj/bias", "action_out_proj/bias/.value", None),
    Site("action_out_proj/kernel", "action_out_proj/kernel/.value", None),
    Site("action_in_proj/kernel", "action_in_proj/kernel/.value", None),
    Site("time_mlp_out/kernel", "time_mlp_out/kernel/.value", None),
    Site("expert/mlp_1/linear/L0", "PaliGemma/llm/layers/mlp_1/linear/.value", 0),
    Site("expert/mlp_1/linear/L8", "PaliGemma/llm/layers/mlp_1/linear/.value", 8),
    Site("expert/mlp_1/linear/L17", "PaliGemma/llm/layers/mlp_1/linear/.value", 17),
    Site("llm/mlp/linear/L0", "PaliGemma/llm/layers/mlp/linear/.value", 0),
    Site("llm/mlp/linear/L17", "PaliGemma/llm/layers/mlp/linear/.value", 17),
    Site("img/MlpBlock_0/Dense_1/kernel/B26",
         "PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel/.value", 26),
]
BY_NAME = {s.name: s for s in SITES}


def path_str(path) -> str:
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


def site_stats(state, site: Site) -> dict:
    """||W||_F and numel for the SITE (the slice, for stacked layers)."""
    for path, v in jax.tree_util.tree_flatten_with_path(state)[0]:
        if path_str(path) == site.path:
            w = v[site.index] if site.index is not None else v
            w32 = jnp.asarray(w, jnp.float32)
            return dict(fro=float(jnp.linalg.norm(w32)), numel=int(np.prod(w32.shape)),
                        shape=list(w32.shape))
    raise KeyError(f"{site.path} not found in the state tree")


def rel_tol(numel: int) -> float:
    """How far the REALISED ||Delta||/||W|| may sit from c before a draw is called dead.

    rho is set so the ratio equals c in EXPECTATION; the realised norm of a Gaussian draw
    is chi-distributed, with relative sd 1/sqrt(2*numel). That is 12.5% at the 32-element
    bias site, so a fixed +-10% band rejects a third of its legitimate draws -- which is
    exactly what aborted the first run. The band therefore scales as 6 sigma, floored at
    25% so huge layers still get a meaningful check. This guards against a perturbation
    that never landed (rel ~ 0); it is not a constraint on the estimator, which stays
    N(0, rho^2 I) as pre-registered.
    """
    return float(max(0.25, 6.0 / np.sqrt(2.0 * numel)))


def rho_for(stats: dict, c: float = C_REL) -> float:
    """prereg §3: rho = c * ||W||_F / sqrt(numel), so ||Delta||_F/||W||_F == c."""
    return c * stats["fro"] / np.sqrt(stats["numel"])


BIAS_PATH = "action_out_proj/bias/.value"


def bias_edited_state(base_state, add):
    """base_state with a fixed vector ADDED to action_out_proj/bias.

    Separate from the ACE draw: this is a deliberate, computed edit (the analytic repair),
    not a random perturbation. Returns (state, applied_l2).
    """
    add = np.asarray(add, np.float32)
    applied = {}

    def f(path, v):
        if path_str(path) != BIAS_PATH:
            return v
        w32 = jnp.asarray(v, jnp.float32)
        pad = jnp.zeros_like(w32).at[: add.shape[0]].set(jnp.asarray(add))
        applied["l2"] = float(jnp.linalg.norm(pad))
        return jnp.asarray(w32 + pad, v.dtype)

    st = jax.tree_util.tree_map_with_path(f, base_state)
    if "l2" not in applied:
        raise KeyError("action_out_proj/bias not found")
    return st, applied


def perturbed_state(base_state, site: Site, rho: float, seed: int):
    """base_state with N(0, rho^2) added at exactly one site. Returns (state, applied_rel)."""
    key = jax.random.key(seed)
    applied = {}

    def f(path, v):
        if path_str(path) != site.path:
            return v
        w32 = jnp.asarray(v, jnp.float32)
        target = w32[site.index] if site.index is not None else w32
        delta = rho * jax.random.normal(key, target.shape, jnp.float32)
        applied["rel"] = float(jnp.linalg.norm(delta) / jnp.linalg.norm(target))
        applied["dnorm"] = float(jnp.linalg.norm(delta))
        out = w32.at[site.index].add(delta) if site.index is not None else w32 + delta
        return jnp.asarray(out, v.dtype)

    new_state = jax.tree_util.tree_map_with_path(f, base_state)
    if "rel" not in applied:
        raise KeyError(f"site {site.name} never matched")
    return new_state, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=pathlib.Path, required=True)
    ap.add_argument("--ack", type=pathlib.Path, required=True)
    ap.add_argument("--norms-out", type=pathlib.Path, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = _config.get_config(a.config)
    policy = _pc.create_trained_policy(cfg, download.maybe_download(a.checkpoint))
    model = policy._model
    graphdef, base_state = nnx.split(model)

    # ||W||_F per site, recorded BEFORE any rollout (prereg §3)
    norms = {s.name: site_stats(base_state, s) for s in SITES}
    for n, st in norms.items():
        st["rho"] = rho_for(st)
    if a.norms_out:
        a.norms_out.parent.mkdir(parents=True, exist_ok=True)
        a.norms_out.write_text(json.dumps(norms, indent=1))
    logging.info("site norms recorded: %s", {k: round(v["fro"], 2) for k, v in norms.items()})

    sample_kwargs = dict(policy._sample_kwargs)

    @jax.jit
    def sample(state, rng, obs):
        return nnx.merge(graphdef, state).sample_actions(rng, obs, **sample_kwargs)

    holder = {"state": base_state, "stamp": None, "pin_rng": False}

    def _sample_actions(rng, obs, **kw):
        _apply_control_if_changed()
        # pin_rng freezes the flow sampler's noise so a weight change can be isolated from
        # sampler stochasticity. Diagnostic only -- the screen itself scores real rollouts.
        if holder["pin_rng"]:
            rng = jax.random.key(0)
        return sample(holder["state"], rng, obs)

    def _apply_control_if_changed():
        if not a.control.exists():
            return
        stamp = a.control.stat().st_mtime_ns
        if stamp == holder["stamp"]:
            return
        req = json.loads(a.control.read_text())
        holder["stamp"] = stamp
        holder["pin_rng"] = bool(req.get("pin_rng", False))
        if req.get("bias_add") is not None:
            st, ap = bias_edited_state(base_state, req["bias_add"])
            holder["state"] = st
            want = float(np.linalg.norm(np.asarray(req["bias_add"], np.float32)))
            # ok means "the edit that landed is the edit that was asked for" -- a
            # deliberately ZERO edit (the k=0 control) is a legitimate request, so this
            # compares against the request rather than demanding a non-zero norm.
            ack = dict(site="bias_add", draw=req.get("draw"), applied_l2=ap["l2"],
                       requested_l2=want, n=len(req["bias_add"]),
                       ok=bool(abs(ap["l2"] - want) <= 1e-4 + 1e-3 * want),
                       pin_rng=holder["pin_rng"], applied_rel=0.0)
        elif req.get("site") is None:
            holder["state"] = base_state
            ack = dict(site=None, draw=req.get("draw"), applied_rel=0.0, ok=True,
                       pin_rng=holder["pin_rng"])
        else:
            site = BY_NAME[req["site"]]
            st = norms[site.name]
            # c_rel lets a probe sweep the perturbation SCALE; the screen leaves it at the
            # pre-registered C_REL and is unaffected.
            c = float(req.get("c_rel", C_REL))
            rho = rho_for(st, c) if c != C_REL else st["rho"]
            state, applied = perturbed_state(base_state, site, rho, int(req["seed"]))
            holder["state"] = state
            ack = dict(site=site.name, draw=req.get("draw"), seed=req["seed"], c_rel=c,
                       rho=rho, fro=st["fro"], numel=st["numel"],
                       applied_rel=applied["rel"], delta_fro=applied["dnorm"],
                       pin_rng=holder["pin_rng"],
                       rel_tol=rel_tol(st["numel"]),
                       ok=bool(abs(applied["rel"] - c) <= rel_tol(st["numel"]) * c))
        ack["stamp"] = stamp
        a.ack.write_text(json.dumps(ack, indent=1))
        logging.info("control applied: %s", ack)

    policy._sample_actions = _sample_actions
    _apply_control_if_changed()

    logging.info("ACE server ready on :%d (state is a jit ARGUMENT, not a capture)", a.port)
    _server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
