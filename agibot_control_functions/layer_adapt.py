#!/usr/bin/env python3
"""Online single-layer policy adaptation, hardware build. numpy only, no ROS.

Transcribed from `adaptation/ACC_ADAPTATION_PACKAGE/ace_adapt.py` (M. Taheri,
S.-J. Chung, F. Y. Hadaegh, "Closing the Loop Inside Neural Networks:
Causality-Guided Layer Adaptation for Fault Recovery Control", ACC 2026) with two
deliberate changes for running on a real robot:

  1. The leak is `-gamma*(W - W0)` instead of `-gamma*W`. The shipped form decays the
     adapted layer toward ZERO, which erodes 13.7% of the layer's weight norm over one
     734-step episode even under PERFECT tracking. Around a trained controller the leak
     should pull back to the trained weights, not to the origin.
  2. Safety latching. Any drift/NaN trip restores W0 and disables adaptation for the
     rest of the run instead of allowing it to re-engage.

Kept verbatim: the update law itself, the input map `g(x)`, and the divergence guard.

The update is one backward pass through the layers above the adapted one, so it is
cheap: measured 1.9 ms forward + 3.3 ms update on the dev box, ~26% of a 20 ms tick.
Re-measure on the robot CPU before trusting it (the deploy script does this for you).

This module is deliberately free of ROS and of the simulator so it can be unit-checked
offline against the reference implementation; see `--self-check` in
`deploy_x2_box_adapt.py`.
"""

from __future__ import annotations

import numpy as np

# Error-mask presets, keyed by the joint-name substrings they select.
#   "waist"      -- the ONLY preset that helped in Isaac. Adapting the leg joints
#                   fights the balance controller: on a floating base the legs are what
#                   hold the robot up, so driving their tracking error to zero spends
#                   stance authority the policy needs. Under a knee fault this preset
#                   extended survival 53% (p = 0.0011, 6 seeds); with the legs included
#                   the robot was down in 2.2 s having never lifted the box.
#   "legs_waist" -- the paper's default. Catastrophic on our policy, healthy or faulted.
#   "sagittal_legs" -- knees, hip pitch, ankle pitch. Derived from the v33 deployment by
#                   comparing hardware against the SAME policy and clip in Isaac
#                   (`analyze_deploy_logs.py --sim`): leg tracking error is the only thing
#                   that degrades, 8.7 -> 16.8 deg, with knees +15-18 deg and ankle pitch
#                   +14 deg, while the waist transfers unchanged (1.9 -> 2.2 deg, so
#                   "waist" has nearly nothing to correct here) and the arms are slightly
#                   BETTER on hardware. These six joints also obey their commands (servo
#                   error 4-9 deg) and are almost never clipped by a joint limit, unlike
#                   the roll joints -- adapting a joint pinned at its stop cannot reach
#                   the plant. UNVALIDATED: needs an Isaac gain sweep before hardware use.
#   "rise"       -- waist plus the sagittal legs, for the stand-up. Its case is the
#                   2026-08-14 static-pose test (E2), which held the robot still and
#                   compared the torque each joint settled at against the same pose
#                   held in MuJoCo. Split left/right, the roll axes came out as almost
#                   pure squeeze -- hip_roll net load 1.0 Nm against 11.5 Nm of the two
#                   legs pressing into each other, ankle_roll 1.0 against 5.3 -- while
#                   the sagittal axes carried real net load, hip_pitch +10.1 and knee
#                   -9.9 Nm. That distinction decides what can be adapted. With both
#                   feet planted the roll axes close a kinematic loop through the
#                   floor, so a roll command cannot move the joint, only squeeze
#                   harder; an integral term pointed there winds up without ever
#                   reducing its error. It is the concrete reason "legs_waist" is
#                   catastrophic. The sagittal axes are free to move and carry a real
#                   load, and the waist droops 0.163 rad at the carry pose.
MASK_PRESETS = {
    "waist": ("waist",),
    "legs_waist": ("hip", "knee", "ankle", "waist"),
    "sagittal_legs": ("knee", "hip_pitch", "ankle_pitch"),
    "rise": ("waist", "knee", "hip_pitch", "ankle_pitch"),
    "waist_arms": ("waist", "shoulder", "elbow"),
    "all": None,
}


def _elu_jacobian(a: np.ndarray) -> np.ndarray:
    return np.where(a > 0.0, 1.0, np.exp(np.minimum(a, 0.0)))


class OnlineLayerAdapter:
    """Lyapunov adaptation of one weight matrix around a frozen policy.

    Per control step, in this order:

        adapter.update(joint_error, dt)   # closes the loop on the PREVIOUS action
        action = adapter.act(obs)         # forward pass with the current weights

    `joint_error` is `q_measured - q_reference` in radians, policy joint order, and must
    be the error produced by the action that `act()` last returned.
    """

    def __init__(self, W0, b, mean, std, joint_names, action_scale, kp,
                 layer: int = 2, gain: float = 3e-4, leak: float = 1e-2,
                 p_gain: float = 1.0, mask=("waist",), max_drift: float = 1.0,
                 engage_step: int = 0):
        self.W0 = [np.asarray(w, dtype=np.float64) for w in W0]
        self.b = [np.asarray(x, dtype=np.float64) for x in b]
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.joint_names = list(joint_names)
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        self.kp = np.asarray(kp, dtype=np.float64)

        self.layer = int(layer)
        self.gain = float(gain)
        self.leak = float(leak)
        self.p_gain = float(p_gain)
        self.max_drift = float(max_drift)
        self.engage_step = int(engage_step)

        if not 0 <= self.layer < len(self.W0) - 1:
            raise ValueError(
                f"layer {self.layer} must be a hidden layer in 0..{len(self.W0) - 2}; "
                "the output layer is unusable (diverges at any useful gain)"
            )

        if mask is None:
            self.err_mask = np.ones(len(self.action_scale))
        else:
            self.err_mask = np.array(
                [1.0 if any(k in n for k in mask) else 0.0 for n in self.joint_names],
                dtype=np.float64,
            )
            if self.err_mask.sum() == 0:
                raise ValueError(f"mask {mask} matched none of {self.joint_names}")
        self.masked_joints = [n for n, m in zip(self.joint_names, self.err_mask) if m > 0]

        self.disabled = False
        self.disabled_reason = ""
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        """Restore the trained weights. Adaptation does NOT persist across takes."""
        self.W = [w.copy() for w in self.W0]
        self.step = 0
        self.peak_drift = 0.0
        self._cache = None

    def freeze(self, reason: str = "frozen by operator") -> None:
        """Stop adapting and revert to the trained weights, permanently for this run."""
        if not self.disabled:
            self.disabled = True
            self.disabled_reason = reason
            self.W[self.layer] = self.W0[self.layer].copy()

    @property
    def weight_drift(self) -> float:
        return float(np.linalg.norm(self.W[self.layer] - self.W0[self.layer]))

    @property
    def drift_fraction(self) -> float:
        """Drift as a fraction of the trained layer's norm -- the interpretable number."""
        return self.weight_drift / float(np.linalg.norm(self.W0[self.layer]))

    # ---------------------------------------------------------------- forward
    def _forward(self, obs, W):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        a, z = [], [x]
        for i in range(len(W)):
            ai = W[i] @ x + self.b[i]
            a.append(ai)
            x = ai if i == len(W) - 1 else np.where(ai > 0, ai, np.expm1(np.minimum(ai, 0.0)))
            z.append(x)
        return a[-1], a, z

    def act(self, obs) -> np.ndarray:
        """Action from the CURRENT (possibly adapted) weights. Caches for update()."""
        action, a, z = self._forward(obs, self.W)
        self._cache = (a, z)
        return action

    def act_frozen(self, obs) -> np.ndarray:
        """Action the untouched trained policy would have produced. No caching."""
        return self._forward(obs, self.W0)[0]

    # ---------------------------------------------------------------- update
    def delta_L(self, joint_error) -> np.ndarray:
        """`delta_L = g(x)^T P e`, the action-space direction that reduces tracking error.

        The policy emits a joint TARGET tracked by a PD servo, so the action->torque map
        is exact and diagonal: d(tau)/d(a) = Kp diag(action_scale). This is `gx_level=1`
        from the paper, which drops `M^-1`; adding the inverse inertia aims the
        correction at the lightest joints (wrists, head) whose tracking errors do not
        matter, and measured WORSE in both his testbed and ours (divergence on 6/6 seeds
        with the true PhysX inertia).

        Note `action_scale * Kp` is invariant to a PD-gain retune, because
        `action_scale = cfg_scale * effort / Kp`. The v31 -> v33 waist change
        (kp 20 -> 60, scale 0.6 -> 0.2) leaves this signal identical, which is why the
        v31-tuned gain is a defensible starting point on v33.
        """
        e = np.asarray(joint_error, dtype=np.float64) * self.err_mask
        return -self.p_gain * (self.action_scale * self.kp * e)

    def update(self, joint_error, dt: float) -> None:
        """One adaptation step. `dt` is the CONTROL period, not the physics step."""
        self.step += 1
        if self.disabled or self.gain == 0.0:
            return
        if self.step <= self.engage_step or self._cache is None:
            return

        a, z = self._cache
        L = len(self.W)

        d = self.delta_L(joint_error)
        for l in range(L - 1, self.layer, -1):
            d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)

        Wdot = (self.gain * np.outer(d, z[self.layer])
                - self.leak * (self.W[self.layer] - self.W0[self.layer]))
        self.W[self.layer] = self.W[self.layer] + dt * Wdot

        # Divergence guard. WITHOUT THIS a run whose weights go NaN scores as a PERFECT
        # run, because NaN fails every failure comparison. Never remove it.
        if not np.isfinite(self.W[self.layer]).all():
            self.freeze("weights went non-finite")
            return
        # Record before any revert, so the post-run report shows the value that tripped.
        drift = self.weight_drift
        self.peak_drift = max(self.peak_drift, drift)
        if drift > self.max_drift:
            self.freeze(f"drift {drift:.3f} exceeded max_drift {self.max_drift:.3f}")
