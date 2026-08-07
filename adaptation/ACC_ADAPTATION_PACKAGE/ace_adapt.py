"""Causality-guided single-layer online adaptation (ACC 2026) — environment-agnostic core.

Method
------
Adapt ONE layer of a frozen policy online, driven by the tracking error:

    delta_L      = g(x)^T P e                          error signal, in ACTION space
    delta_l      = Psi_l(a_l) . W_{l+1}^T delta_{l+1}   backprop through activation Jacobians
    Wdot_{l*}    = Gamma . delta_{l*} z_{l*-1}^T - gamma . W_{l*}

`-gamma W` is the leakage term that keeps the weights bounded (sigma-modification). No
counterfactual model, no probing, no gradient of a learned objective — one backward pass per
control step through the layers above the adapted one.

This module has NO simulator dependency. Wire it to any environment by supplying, each control
step, the policy observation and the joint tracking error. See INTEGRATION.md.

Reference: M. Taheri, S.-J. Chung, F. Y. Hadaegh, "Closing the Loop Inside Neural Networks:
Causality-Guided Layer Adaptation for Fault Recovery Control", ACC 2026.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# Policy: the exported MLP (weights + observation normaliser + reference motion)
# --------------------------------------------------------------------------------------
class ExportedPolicy:
    """A policy exported as .npz: W0..Wn / b0..bn, `mean`, `std`, `meta_json`.

    Activation is ELU on hidden layers, linear output. The forward pass caches
    pre-activations and activations because the adaptation needs both.
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(str(d["meta_json"]))
        n = int(d["n_layers"])
        self.W0 = [d[f"W{i}"].astype(np.float64) for i in range(n)]   # frozen reference
        self.b = [d[f"b{i}"].astype(np.float64) for i in range(n)]
        self.mean = d["mean"].astype(np.float64)
        self.std = d["std"].astype(np.float64)
        # Reference motion carried in the export (optional for adaptation itself).
        self.ref_pos = d["ref_joint_pos"].astype(np.float64) if "ref_joint_pos" in d else None
        self.ref_vel = d["ref_joint_vel"].astype(np.float64) if "ref_joint_vel" in d else None
        self.ref_quat = d["ref_quat_xyzw"].astype(np.float64) if "ref_quat_xyzw" in d else None

    @property
    def n_layers(self) -> int:
        return len(self.W0)

    def layer_shapes(self):
        return [w.shape for w in self.W0]

    def forward(self, obs, W=None):
        """Return (action, pre_activations, activations). `activations[0]` is normalised obs."""
        W = self.W0 if W is None else W
        x = (obs - self.mean) / self.std
        a, z = [], [x]
        for i in range(len(W)):
            ai = W[i] @ x + self.b[i]
            a.append(ai)
            x = ai if i == len(W) - 1 else np.where(ai > 0, ai, np.expm1(np.minimum(ai, 0.0)))
            z.append(x)
        return a[-1], a, z


def _elu_jacobian(a):
    return np.where(a > 0, 1.0, np.exp(np.minimum(a, 0.0)))


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class AdaptConfig:
    """Defaults are the configuration confirmed on held-out seeds; see RESULTS.md.

    layer        index of the adapted weight matrix (0-based). 2 for a 4-layer policy.
    gain         Gamma. THE sensitive knob — see RESULTS.md for the stability boundary.
    leak         gamma, the sigma-modification term keeping weights bounded.
    p_gain       P, scalar Lyapunov weight.
    gx_level     input-mapping fidelity (see `delta_L` below):
                   0 = action_scale only
                   1 = action_scale * Kp   <-- confirmed best
                   2 = + inverse mass matrix (needs `mass_matrix_fn`)
    error_joints substring list selecting which joint errors are regulated.
                 None = all joints. ("hip","knee","ankle","waist") is the confirmed default.
    engage_step  control step at which adaptation starts. 0 = from the first step (best).
    """
    layer: int = 2
    gain: float = 3e-4
    leak: float = 1e-2
    p_gain: float = 1.0
    gx_level: int = 1
    error_joints: tuple | None = ("hip", "knee", "ankle", "waist")
    engage_step: int = 0
    max_weight_drift: float = 5.0     # safety: abort adaptation if ||W - W0||_F exceeds this


# --------------------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------------------
class LayerAdapter:
    """Single-layer Lyapunov adaptation around a frozen exported policy.

    Usage per control step:

        action = adapter.act(obs)              # forward pass with the adapted weights
        ...apply action to the plant, step the sim/robot...
        adapter.update(joint_error)            # one backward pass + weight update

    `joint_error` is q - q_ref for the actuated joints, in radians, same ordering as the
    policy's joint list.
    """

    def __init__(self, policy: ExportedPolicy, cfg: AdaptConfig | None = None,
                 action_scale=None, kp=None, joint_names=None, mass_matrix_fn=None):
        self.pol = policy
        self.cfg = cfg or AdaptConfig()
        meta = policy.meta
        self.joint_names = list(joint_names or meta.get("joint_names", []))
        self.action_scale = np.asarray(
            action_scale if action_scale is not None else meta["action_scale"], dtype=float)
        self.kp = np.asarray(kp if kp is not None else meta["joint_stiffness"], dtype=float)
        self.mass_matrix_fn = mass_matrix_fn

        if self.cfg.error_joints is None:
            self.err_mask = np.ones(len(self.action_scale))
        else:
            self.err_mask = np.array(
                [1.0 if any(k in n for k in self.cfg.error_joints) else 0.0
                 for n in self.joint_names])
            if self.err_mask.sum() == 0:
                raise ValueError("error_joints matched no joint names")

        self.reset()

    # -- lifecycle ---------------------------------------------------------------------
    def reset(self):
        """Restore the frozen weights. Call at the start of every episode."""
        self.W = [w.copy() for w in self.pol.W0]
        self.step = 0
        self.diverged = False
        self._cache = None

    @property
    def weight_drift(self) -> float:
        return float(np.linalg.norm(self.W[self.cfg.layer] - self.pol.W0[self.cfg.layer]))

    # -- forward -----------------------------------------------------------------------
    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        action, a, z = self.pol.forward(obs, self.W)
        self._cache = (a, z)
        return action

    # -- error signal ------------------------------------------------------------------
    def delta_L(self, joint_error):
        """delta_L = g(x)^T P e, the action-space direction that reduces tracking error.

        The policy emits a joint TARGET driven by a PD servo, so the action->torque map is
        exact and diagonal: d(tau)/d(a) = Kp diag(action_scale). Composing with the plant
        dynamics gives g(x) = M^-1 S^T Kp diag(s), hence

            delta_L = diag(s) Kp S M^-1 P e

        gx_level selects how much of that chain is used. Level 1 (drop M^-1) is the confirmed
        best: M^-1 is correct physics but weights by inverse inertia, which points the
        correction at the lightest joints (wrists/head) whose tracking errors do not matter
        for the task. Fidelity has to be about the quantity being regulated.
        """
        e = np.asarray(joint_error, dtype=float) * self.err_mask
        s, kp, P = self.action_scale, self.kp, self.cfg.p_gain

        if self.cfg.gx_level == 0:
            return -P * (s * e)
        if self.cfg.gx_level == 1:
            return -P * (s * kp * e)
        if self.cfg.gx_level == 2:
            if self.mass_matrix_fn is None:
                raise ValueError("gx_level=2 requires mass_matrix_fn")
            M = np.asarray(self.mass_matrix_fn(), dtype=float)   # actuated block, (n, n)
            try:
                acc = np.linalg.solve(M, P * e)
            except np.linalg.LinAlgError:
                return -P * (s * kp * e)
            return -(s * kp * acc)
        raise ValueError(f"unknown gx_level {self.cfg.gx_level}")

    # -- update ------------------------------------------------------------------------
    def update(self, joint_error, dt):
        """One adaptation step. `dt` is the CONTROL period (seconds), not the physics step."""
        self.step += 1
        if self.diverged or self.step <= self.cfg.engage_step or self._cache is None:
            return
        a, z = self._cache
        L, layer = self.pol.n_layers, self.cfg.layer

        d = self.delta_L(joint_error)
        for l in range(L - 1, layer, -1):                 # backprop to the adapted layer
            d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)

        Wdot = self.cfg.gain * np.outer(d, z[layer]) - self.cfg.leak * self.W[layer]
        self.W[layer] = self.W[layer] + dt * Wdot

        # Divergence guard. WITHOUT THIS a run whose weights blow up to NaN silently scores as
        # a PERFECT run, because NaN fails every failure comparison (`height < thresh` and
        # `err > limit` are both False). Never omit it.
        if not np.isfinite(self.W[layer]).all() or self.weight_drift > self.cfg.max_weight_drift:
            self.diverged = True
            self.W[layer] = self.pol.W0[layer].copy()


__all__ = ["ExportedPolicy", "AdaptConfig", "LayerAdapter"]
