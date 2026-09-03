"""The ACC-2026 adaptation law as published, plus the theory-preserving knobs.

Eq. (4) of Taheri, Chung & Hadaegh:

    Wdot_{l*} = Gamma . delta_{l*} z_{l*-1}^T  -  gamma W_{l*}

Differences from `ACC_ADAPTATION_PACKAGE/ace_adapt.py`, and why each matters:

* **P is a positive diagonal, not a scalar, and it replaces the error mask.**
  The mask zeroes rows of the regulated error, which makes P singular and breaks
  `Vdot <= -e^T Q e` on the masked directions -- the best-performing Isaac config
  sits outside Theorem 1. With a diagonal PD nominal (K = diag(k_j)) the theorem
  needs only `Q = 1/2 (PK + K^T P) = diag(p_j k_j) > 0`, i.e. ANY p_j > 0. So a
  small-but-nonzero leg weight buys the same de-emphasis with the proof intact.

* **Gamma is a diagonal matrix, per the paper, not a scalar.** Optionally scaled
  per row of the adapted layer by the downstream row norm of W_{l*+1}, so rows
  with little influence on the output are not updated as hard as rows with much.
  Any positive diagonal is admissible.

* **Leak centre is selectable.** `zero` is eq. (4) as published (sigma-modification,
  the bound carries ||W*||). `w0` recentres on the frozen weights, which is the
  variant the repo adopted; its bound carries ||W0 - W*|| instead.

* **Spectral guard.** Assumption 2 gets its boundedness from spectral normalisation
  during training. This policy was not spectrally normalised, so `||W|| < 1` cannot
  be imposed after the fact without destroying it. The faithful analogue is to
  forbid adaptation from RAISING the layer's Lipschitz constant above the frozen
  policy's: project back to sigma_max(W0) whenever it is exceeded.
"""
from __future__ import annotations

import numpy as np
from ace_adapt import LayerAdapter, _elu_jacobian


class PaperAdapter(LayerAdapter):
    """Eq. (4) with diagonal Gamma, diagonal P, selectable leak centre, spectral guard.

    Subclass and override the class attributes; the experiment harness constructs
    adapters as `cls(pol, cfg, joint_names=..., mass_matrix_fn=...)`, so per-variant
    settings have to live on the class.
    """

    LEAK_CENTRE = "zero"      # "zero" = eq. (4) as published; "w0" = recentred
    P_LEG = 1.0               # diagonal P weight on hip/knee/ankle
    P_WAIST = 1.0             # ... on the waist joints
    P_OTHER = 1.0             # ... on arms/head
    GAMMA_ROW_NORM = False    # scale Gamma rows by downstream influence
    SPECTRAL_GUARD = False    # forbid raising sigma_max above the frozen value

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        p = np.full(len(self.action_scale), self.P_OTHER, dtype=float)
        for i, n in enumerate(self.joint_names):
            if any(k in n for k in ("hip", "knee", "ankle")):
                p[i] = self.P_LEG
            elif "waist" in n:
                p[i] = self.P_WAIST
        assert (p > 0).all(), "P must be positive definite -- see Theorem 1"
        self.P_diag = p
        self.err_mask = np.ones_like(p)   # neutralise the inherited binary mask

        layer = self.cfg.layer
        n_rows = self.pol.W0[layer].shape[0]
        if self.GAMMA_ROW_NORM:
            infl = np.linalg.norm(self.pol.W0[layer + 1], axis=0)
            infl = infl / infl.mean()
            self.Gamma = self.cfg.gain / np.maximum(infl, 1e-3)
        else:
            self.Gamma = np.full(n_rows, self.cfg.gain, dtype=float)

        self.sigma0 = float(np.linalg.norm(self.pol.W0[layer], 2))
        self._u = None

    def delta_L(self, joint_error):
        """delta_L = g(x)^T P e with P = diag(P_diag), g = diag(action_scale * Kp)."""
        e = np.asarray(joint_error, dtype=float)
        s, kp = self.action_scale, self.kp
        if self.cfg.gx_level == 0:
            return -(s * self.P_diag * e)
        if self.cfg.gx_level == 1:
            return -(s * kp * self.P_diag * e)
        if self.cfg.gx_level == 2:
            if self.mass_matrix_fn is None:
                raise ValueError("gx_level=2 requires mass_matrix_fn")
            M = np.asarray(self.mass_matrix_fn(), dtype=float)
            try:
                acc = np.linalg.solve(M, self.P_diag * e)
            except np.linalg.LinAlgError:
                return -(s * kp * self.P_diag * e)
            return -(s * kp * acc)
        raise ValueError(f"unknown gx_level {self.cfg.gx_level}")

    def _sigma_max(self, W):
        """Power iteration, warm-started -- avoids an SVD in the 20 ms control tick."""
        if self._u is None:
            self._u = np.random.default_rng(0).standard_normal(W.shape[1])
            self._u /= np.linalg.norm(self._u)
        u = self._u
        for _ in range(2):
            v = W @ u
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                return 0.0
            v /= nv
            u = W.T @ v
            nu = np.linalg.norm(u)
            if nu < 1e-12:
                return 0.0
            u /= nu
        self._u = u
        return float(np.linalg.norm(W @ u))

    def update(self, joint_error, dt):
        self.step += 1
        if self.diverged or self.step <= self.cfg.engage_step or self._cache is None:
            return
        a, z = self._cache
        L, layer = self.pol.n_layers, self.cfg.layer

        d = self.delta_L(joint_error)
        for l in range(L - 1, layer, -1):
            d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)

        centre = 0.0 if self.LEAK_CENTRE == "zero" else self.pol.W0[layer]
        Wdot = (self.Gamma[:, None] * np.outer(d, z[layer])
                - self.cfg.leak * (self.W[layer] - centre))
        self.W[layer] = self.W[layer] + dt * Wdot

        if self.SPECTRAL_GUARD:
            s = self._sigma_max(self.W[layer])
            if s > self.sigma0 > 0:
                self.W[layer] *= self.sigma0 / s

        if not np.isfinite(self.W[layer]).all() or self.weight_drift > self.cfg.max_weight_drift:
            self.diverged = True
            self.W[layer] = self.pol.W0[layer].copy()


# ---- variants -------------------------------------------------------------------
class PaperExact(PaperAdapter):
    """Eq. (4) verbatim: P = I, no mask, leak toward zero. Never run in Isaac before."""


class PaperW0Leak(PaperAdapter):
    """Eq. (4) with the leak recentred on the frozen weights."""

    LEAK_CENTRE = "w0"


class WeightedP01(PaperAdapter):
    """Legs/arms de-emphasised 100x but P still positive definite. Theorem 1 intact."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.01
    P_OTHER = 0.01


class WeightedP10(PaperAdapter):
    """Legs/arms de-emphasised 10x."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.1
    P_OTHER = 0.1


class WeightedP01Spec(WeightedP01):
    """Weighted P plus the spectral guard (Assumption 2's analogue)."""

    SPECTRAL_GUARD = True


class WeightedP01RowNorm(WeightedP01):
    """Weighted P plus per-row Gamma scaled by downstream influence."""

    GAMMA_ROW_NORM = True


class PaperW0LeakSpec(PaperW0Leak):
    """Unmasked law + spectral guard. Offline the unmasked law raises sigma_max to
    1.023x the frozen value while the weighted-P variants lower it, so this is where
    Assumption 2's analogue actually binds -- and where it can be tested as a fix for
    the divergences the unmasked law shows at this gain."""

    SPECTRAL_GUARD = True


class WeightedP003(PaperAdapter):
    """P_leg = 0.003 -- between the 0.01 knee and the singular boundary."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.003
    P_OTHER = 0.003


class WeightedP001(PaperAdapter):
    """P_leg = 0.001 -- closest admissible point to the binary mask."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.001
    P_OTHER = 0.001


class WeightedP003Spec(WeightedP003):
    """P_leg = 0.003 with the spectral guard; the guard halved variance on the
    unmasked arm, and variance is the weighted-P arms' only weakness."""

    SPECTRAL_GUARD = True


# ---- gating: the "at least as good as frozen" floor the law currently lacks ---------
#
# The envelope study says adaptation helps exactly when the FROZEN policy is already
# failing, and costs ~200 steps of survival when it is not. Neither fault severity nor
# fault site predicts which side you are on -- right_knee:0.3 leaves frozen at 356 steps
# (adaptation helps) while right_hip_pitch:0.3 leaves it at 734 (adaptation hurts).
#
# So gate on the thing that actually predicts it: is the tracking error worse than this
# policy's own healthy profile at this point in the motion? A FIXED threshold cannot do
# this -- the healthy profile swings 7.4 deg to 17.2 deg across the motion -- so the gate
# compares against the per-frame nominal and latches on once the excess is sustained.
_NOMINAL_PROFILE = None


def _nominal_profile():
    global _NOMINAL_PROFILE
    if _NOMINAL_PROFILE is None:
        import pathlib
        p = pathlib.Path(__file__).resolve().parent / "isaac_runs" / "isaac_frozen_npz_seed600.npz"
        _NOMINAL_PROFILE = np.load(p, allow_pickle=True)["leg_err_deg"].astype(float)
    return _NOMINAL_PROFILE


class GatedAdapter(WeightedP01Spec):
    """Weighted-P + spectral guard, engaged only once tracking degrades past nominal.

    The motion frame advances one per control step from frame 0 in this eval config,
    so `self.step` indexes the nominal profile directly.
    """

    ENGAGE_MARGIN_DEG = 4.0
    ENGAGE_HOLD = 25          # consecutive steps above nominal+margin before latching

    def reset(self):
        super().reset()
        self._above = 0
        self._engaged = False
        self._engage_step = -1
        self._legs = np.array([i for i, n in enumerate(self.joint_names)
                               if any(k in n for k in ("hip", "knee", "ankle"))])

    def update(self, joint_error, dt):
        if not self._engaged:
            e = np.asarray(joint_error, dtype=float)
            leg_err = float(np.degrees(np.abs(e[self._legs])).mean())
            prof = _nominal_profile()
            nominal = float(prof[min(self.step, len(prof) - 1)])
            self._above = self._above + 1 if leg_err > nominal + self.ENGAGE_MARGIN_DEG else 0
            if self._above >= self.ENGAGE_HOLD:
                self._engaged = True
                self._engage_step = self.step
            else:
                self.step += 1        # PaperAdapter.update would have done this
                return
        super().update(joint_error, dt)


class GatedAdapterTight(GatedAdapter):
    """Same gate, more eager: smaller margin and shorter hold."""

    ENGAGE_MARGIN_DEG = 2.5
    ENGAGE_HOLD = 15


# ---- ACE: the paper's offline layer attribution, in THIS environment ----------------
#
#   ACE_l = E[||edot|| | do(W_l + Delta_l)] - E[||edot|| | W_l],   Delta_l ~ N(0, rho_l^2 I)
#
# and the selected layer is argmin ACE. This has never been run in Isaac: layer 2 was
# chosen in the MuJoCo testbed on the HEALTHY policy, at 1.5 sem from zero and (per
# RESULTS.md) post-hoc relative to the empirical sweep. The paper prescribes evaluating
# ACE on the OOD scenarios you deploy under -- i.e. per fault. Since adaptation helps at
# right_knee:0.3 and hurts at left_knee:0.3, the question is whether ACE selects a
# DIFFERENT layer for the two.
#
# Delta is drawn with ||Delta_l||_F = ACE_REL * ||W0_l||_F so the intervention is a
# matched RELATIVE displacement across layers of very different size (W0 is 512x164,
# W3 is 31x128). Each seed draws its own direction, so seeds are the Monte Carlo draws;
# pairing each against frozen_npz on the same seed gives a paired ACE estimate, which is
# tighter than the unpaired difference of expectations in eq. (2).
class _AceProbe(PaperAdapter):
    ACE_LAYER = 2
    ACE_REL = 0.02

    def reset(self):
        super().reset()
        L = self.ACE_LAYER
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        d = rng.standard_normal(self.pol.W0[L].shape)
        d *= (self.ACE_REL * np.linalg.norm(self.pol.W0[L])) / np.linalg.norm(d)
        self.W[L] = self.pol.W0[L] + d

    def update(self, joint_error, dt):
        self.step += 1      # pure do-operator intervention: no adaptation


class AceL0(_AceProbe):
    ACE_LAYER = 0


class AceL1(_AceProbe):
    ACE_LAYER = 1


class AceL2(_AceProbe):
    ACE_LAYER = 2


class AceL3(_AceProbe):
    ACE_LAYER = 3


# ---- compliant-grasp variants -------------------------------------------------------
#
# A deformable box means the commanded squeeze is never achieved: the palms travel in,
# the object yields, and joint error PERSISTS in the arms. The knee-fault tuning weights
# arms at P_OTHER = 0.01, so the tuned adapter is blind to exactly that error. These
# variants put the weight back on the arms while keeping the legs de-emphasised, which
# is what the envelope study showed the legs need.
class WeightedPArm(PaperAdapter):
    """Arms and waist regulated, legs de-emphasised. For grasp-side errors."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.01
    P_WAIST = 1.0
    P_OTHER = 1.0
    SPECTRAL_GUARD = True


class WeightedPArmOnly(PaperAdapter):
    """Only the arms regulated -- isolates the grasp from the balance chain."""

    LEAK_CENTRE = "w0"
    P_LEG = 0.01
    P_WAIST = 0.01
    P_OTHER = 1.0
    SPECTRAL_GUARD = True


# ---- alternative error signals -------------------------------------------------------
#
# Everything measured so far says the LAW is fine and the regulated error is the problem:
# joint-position error is not the control objective on a floating base (and on the arms it
# is actively anti-task). The paper's e is the controlled-output error, so the question is
# what that output should be here. These variants keep the law, P, Gamma, leak and guard
# identical and change only e.
class _ContextAdapter(WeightedP01Spec):
    """Base for adapters that need more than joint position error."""

    def reset(self):
        super().reset()
        self._ctx = {}

    def set_context(self, **kw):
        self._ctx = kw

    def _ref_vel(self):
        rv = self.pol.ref_vel
        return rv[min(self.step, len(rv) - 1)] if rv is not None else 0.0


class VelocityErrorAdapter(_ContextAdapter):
    """e = qdot - qdot_ref. Regulates rate rather than position."""

    def delta_L(self, joint_error):
        v = self._ctx.get("dof_vel")
        if v is None:
            return super().delta_L(joint_error)
        return super().delta_L(np.asarray(v, float) - self._ref_vel())


class SlidingErrorAdapter(_ContextAdapter):
    """e = s = (qdot - qdot_ref) + LAMBDA (q - q_ref), the Slotine sliding variable.

    Standard in the adaptive-control tradition the paper builds on: s = 0 is an
    exponentially stable manifold in the tracking error, so regulating s regulates
    position AND its rate, and adds damping the pure-position error lacks.
    """

    LAMBDA = 5.0

    def delta_L(self, joint_error):
        v = self._ctx.get("dof_vel")
        e = np.asarray(joint_error, float)
        if v is None:
            return super().delta_L(e)
        s = (np.asarray(v, float) - self._ref_vel()) + self.LAMBDA * e
        return super().delta_L(s)


class SlidingErrorAdapterLowGain(SlidingErrorAdapter):
    """Same, at 1/3 the gain: s is larger in magnitude than e alone."""


class BaseAttitudeAdapter(_ContextAdapter):
    """e defined on the BASE attitude, mapped to the leg joints by a Jacobian transpose.

    This is the repo's open question #2 in its cheapest form: on a floating base what
    falls over is the trunk, not the joints, so regulate trunk roll/pitch directly.
    The joint map is a hand-built J^T -- roll error onto the roll-axis leg joints,
    pitch error onto the pitch-axis chain -- not the true centroidal map, so treat a
    positive result as motivation for doing this properly rather than as the final word.
    """

    ROLL_GAIN = 1.0
    PITCH_GAIN = 1.0

    def reset(self):
        super().reset()
        n = len(self.joint_names)
        self._jt_roll = np.zeros(n)
        self._jt_pitch = np.zeros(n)
        for i, nm in enumerate(self.joint_names):
            sgn = 1.0 if nm.startswith("left") else -1.0
            if "hip_roll" in nm:
                self._jt_roll[i] = sgn
            elif "ankle_roll" in nm:
                self._jt_roll[i] = -sgn
            elif "hip_pitch" in nm:
                self._jt_pitch[i] = 1.0
            elif "knee" in nm:
                self._jt_pitch[i] = -1.0
            elif "ankle_pitch" in nm:
                self._jt_pitch[i] = 1.0

    def delta_L(self, joint_error):
        q = self._ctx.get("root_quat_xyzw")
        if q is None or self.pol.ref_quat is None:
            return super().delta_L(joint_error)
        qr = self.pol.ref_quat[min(self.step, len(self.pol.ref_quat) - 1)]
        R = _quat_mat(q).T @ _quat_mat(qr)          # reference relative to measured base
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        e = self.ROLL_GAIN * roll * self._jt_roll + self.PITCH_GAIN * pitch * self._jt_pitch
        return super().delta_L(-e)                   # sign: drive the base back to reference


def _quat_mat(q):
    x, y, z, w = np.asarray(q, float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class BaseAttitudeAdapterHi(BaseAttitudeAdapter):
    """Base-attitude error needs ~50x the gain of joint error: the signal is a few
    hundredths of a radian and the J^T map touches only 12 of 31 joints."""


class SlidingLambda2(SlidingErrorAdapter):
    """s = edot + 2 e -- less position weight, more damping."""

    LAMBDA = 2.0


class SlidingLambda10(SlidingErrorAdapter):
    """s = edot + 10 e -- closer to pure position error."""

    LAMBDA = 10.0


# ---- saturation-relief adaptation ----------------------------------------------------
#
# Measured on the first rise (frames 110-150), hardware, |a| = 4 is the effort limit:
#     hip_pitch   |a| 4.3-4.9, peaks 9.2, saturated 52-60% of the window
#     waist_pitch |a| 1.2, knee 1.1-2.1, ankle_pitch 0.8-1.1  -- never at the limit
#
# An error-driven law pushes hardest where the error is largest, which is the SATURATED
# hips -- where extra command yields exactly zero extra torque. The spare capacity in the
# waist and knees is never recruited because those joints track their reference well and
# so generate almost no error signal. Error-nulling and load-redistribution are different
# objectives; this class does the second.
#
# The waist cannot lift the pelvis (empirical dz/dq = -0.005 m/rad; the waist is ABOVE the
# root). What it can do is pitch the torso upright, pulling the CoM back over the feet and
# reducing the hip moment -- i.e. it UNLOADS the hip rather than adding lift. The knee and
# ankle do add lift (dz/dq = -0.108 and -0.057).
class SaturationReliefAdapter(WeightedP01Spec):
    """Drive the update from unmet actuator demand, not tracking error.

    delta_L is built from how far the hip command exceeds the effort limit, and is
    applied to the joints that can relieve it. Still eq. (4) for the weight update --
    only the controlled output has changed.
    """

    LIMIT = 4.0            # |a| at the effort limit, by construction of action_scale
    RELIEF = {"waist_pitch": -1.0,      # toward upright: CoM back, hip moment down
              "knee": -0.6,             # extend: raises the pelvis
              "ankle_pitch": -0.3}      # extend: raises the pelvis, smaller authority
    SOURCE = ("hip_pitch",)             # the joints whose saturation we are relieving
    # The excess is in ACTION units (up to ~5) where a normal tracking error is ~0.1 rad,
    # so delta_L would be ~50x its usual size and blow through the drift bound. This
    # converts the excess into an equivalent-radian scale.
    RELIEF_SCALE = 0.02

    def reset(self):
        super().reset()
        n = len(self.joint_names)
        self._src = np.array([1.0 if any(k in nm for k in self.SOURCE) else 0.0
                              for nm in self.joint_names])
        self._relief = np.zeros(n)
        for i, nm in enumerate(self.joint_names):
            for key, w in self.RELIEF.items():
                if key in nm:
                    self._relief[i] = w
        self._last_action = np.zeros(n)
        self.sat_ticks = 0

    def act(self, obs):
        a = super().act(obs)
        self._last_action = np.asarray(a, float)
        return a

    def delta_L(self, joint_error):
        a = self._last_action
        excess = np.maximum(np.abs(a) - self.LIMIT, 0.0) * self._src
        total = float(excess.sum())
        if total <= 0.0:
            return np.zeros_like(a)
        self.sat_ticks += 1
        # scale into the same units delta_L normally carries (action_scale * Kp * e)
        return -(self.action_scale * self.kp) * (self._relief * total * self.RELIEF_SCALE)


class SaturationReliefStrong(SaturationReliefAdapter):
    """Same, with the waist weighted harder relative to the leg extensors."""

    RELIEF = {"waist_pitch": -1.0, "knee": -0.25, "ankle_pitch": -0.1}


# ---- anticipatory (lead) error -------------------------------------------------------
#
# The law as written is purely reactive: delta_L uses the error NOW, so a correction only
# begins once the robot has already fallen behind. The rise is ~40 control steps, so by
# the time the error is informative the outcome is largely decided.
#
# The reference is known ahead of time, so anticipation is free: regulate against where
# the reference WILL be, e = q(t) - q_ref(t+H). That is lead compensation, and it needs
# no forward model -- expanding it gives
#     e_lead = [q(t) - q_ref(t)] + [q_ref(t) - q_ref(t+H)]
#            = joint_error      + a known reference increment
# so it costs one table lookup per step. H is in control steps (50 Hz).
class LookaheadAdapter(WeightedP01Spec):
    HORIZON = 15                      # steps of lead (15 = 0.3 s)

    def delta_L(self, joint_error):
        rp = self.pol.ref_pos
        if rp is None:
            return super().delta_L(joint_error)
        n = len(rp)
        now = rp[min(self.step, n - 1)]
        ahead = rp[min(self.step + self.HORIZON, n - 1)]
        return super().delta_L(np.asarray(joint_error, float) + (now - ahead))


class LookaheadShort(LookaheadAdapter):
    HORIZON = 5                       # 0.1 s


class LookaheadLong(LookaheadAdapter):
    HORIZON = 30                      # 0.6 s -- most of the rise


# ---------------------------------------------------------------------------
# Stability error instead of tracking error
# ---------------------------------------------------------------------------
class StabilityAdapter(WeightedP01Spec):
    """Adapt on the CoM-over-foot violation, not on tracking error.

    The reference this policy tracks is infeasible in single support: it parks the
    CoM up to 300 mm lateral of the stance foot's roll axis, which needs 127 N-m of
    ankle_roll restoring moment from a foot that can transmit at most 21.1 N-m (the
    ground reaction acts inside a 50 mm sole half width). Measured by
    box_pickup/check_ankle_roll_feasible.py on the shipped clip.

    So driving adaptation on `q - q_ref`, which is what every other adapter here
    does, pushes the robot INTO the pose that cannot be held. On hardware the run
    that survived tracked WORST through those windows; the ones that followed the
    reference fell. The error therefore has to be a stability quantity.

    f(q) = |CoM_y - stance_ankle_y| - bound,  clipped at 0

    with bound = tau_limit / (m g), the lateral offset a saturated ankle_roll can
    still hold -- derived from the mass and the actuator limit, not chosen. The
    joint-space error handed to the law is the direction of steepest increase of f,
    scaled by the violation:

        e = f * grad_q(f) / ||grad_q(f)||

    grad is taken numerically over all 31 joints (0.46 ms per FK, ~11 s over a
    741-step rollout) rather than over a hand-picked lateral subset, so no joint is
    privileged by assumption.

    f = 0 gives e = 0, so the adapter is inert whenever the CoM is inside what the
    ankle can hold, and only acts on the violation. It engages in single support
    only: in double support the moment is shared between two feet and the bound
    does not apply as stated.
    """

    # P = 1, not WeightedP01's 0.01. This error is nonzero only on the single-support
    # frames that violate the bound (~4% of a rollout), where the tracking-error
    # adapters see a nonzero error every step. With P=0.01 and gain 1e-4 the weight
    # drift came out at 2.4e-05 against the 0.021 those adapters produce -- inert.
    P_LEG = 1.0
    P_OTHER = 1.0
    P_WAIST = 1.0

    CONTACT = 0.020        # sole this far above the floor still carries load
    TAU_ROLL = 24.0        # ankle_roll effort limit, N-m
    G = 9.81
    EPS = 1e-4             # finite-difference step, rad

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ctx = None
        self._robot = None
        self._sole = None
        self._mass_total = None
        self._engaged = 0
        self._steps = 0

    def _lazy_setup(self):
        if self._robot is not None:
            return
        import sys as _sys
        from pathlib import Path as _P
        bp = _P(__file__).resolve().parents[1] / "box_pickup"
        if str(bp) not in _sys.path:
            _sys.path.insert(0, str(bp))
        from rebuild_reference_motion import FIXED_FRAMES, URDF, Robot, load_masses
        self._robot = Robot()
        self._feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self._sole = {b: [np.asarray(FIXED_FRAMES[k][1]) for k in FIXED_FRAMES
                          if "sphere" in k and FIXED_FRAMES[k][0] == b]
                      for b in self._feet}
        self._mass_total = sum(m for m, _ in load_masses(URDF).values())

    def set_context(self, **kw):
        self._ctx = kw

    def _sole_h_and_com(self, q, jn, rp, rq):
        out = self._robot.fk(q, jn, rp, rq)
        h = []
        for b in self._feet:
            p, R = out[b]
            h.append(min((p + R @ s)[2] for s in self._sole[b]))
        com = self._robot.com(out)[0]
        return np.array(h), com, out

    def _violation(self, q, jn, rp, rq, floor):
        h, com, out = self._sole_h_and_com(q, jn, rp, rq)
        down = (h - floor) < self.CONTACT
        if down.sum() != 1:                     # only single support is bounded
            return 0.0, None
        i = int(np.argmax(down))
        stance_y = out[self._feet[i]][0][1]
        bound = self.TAU_ROLL / (self._mass_total * self.G)
        return max(abs(com[1] - stance_y) - bound, 0.0), i

    def delta_L(self, joint_error):
        self._steps += 1
        if self._ctx is None:
            return np.zeros_like(np.asarray(joint_error, dtype=float))
        self._lazy_setup()
        q = np.asarray(self._ctx["dof_pos"], dtype=float)
        jn = self._ctx["dof_names"]
        rp = np.asarray(self._ctx["root_pos"], dtype=float)
        rq = np.asarray(self._ctx["root_quat_wxyz"], dtype=float)
        floor = float(self._ctx.get("floor", 0.0))

        f0, stance = self._violation(q, jn, rp, rq, floor)
        if f0 <= 0.0:
            return np.zeros_like(q)             # inside what the ankle can hold
        self._engaged += 1

        grad = np.zeros_like(q)
        for j in range(len(q)):
            qp = q.copy(); qp[j] += self.EPS
            fj, _ = self._violation(qp, jn, rp, rq, floor)
            grad[j] = (fj - f0) / self.EPS
        n2 = float(grad @ grad)
        if n2 < 1e-12:
            return np.zeros_like(q)
        # Gauss-Newton step, not a normalised direction. f is in metres and grad in
        # m/rad, so f*grad/||grad|| would hand the law an error in METRES while it
        # expects joint space (radians, like q - q_ref). f*grad/||grad||^2 has units
        # m / (m/rad) = rad and is the joint displacement that linearly zeroes the
        # violation: f(q + dq) ~ f + grad.dq = 0 for dq = -f*grad/||grad||^2.
        e = f0 * grad / n2
        s = self._spectral_scale() if hasattr(self, "_spectral_scale") else 1.0
        return -(s * self.P_diag * e)

    @property
    def engaged_fraction(self):
        return self._engaged / max(self._steps, 1)


class PreemptiveStanceAdapter(StabilityAdapter):
    """Act before the robot commits to a single-support phase it cannot hold.

    StabilityAdapter failed, and the measurement says why: its error f = |CoM_y -
    stance_y| - bound is only nonzero ONCE the robot is already in single support and
    already outside the bound. By then the foot is unloaded and the choice is made;
    pushing on the last layer cannot retract it. Measured over a gain sweep with real
    authority (drift 2.8e-04 / 2.0e-03 / 1.6e-02, the last comparable to the 0.021 the
    tracking adapters produce), single-support frames went 25 -> 44 / 36 / 48 against
    the frozen control: monotonically WORSE with gain.

    That is the same defect Baaqer identified in two candidate trigger signals: they
    are measured during single support, but what separates the runs is how much single
    support happens at all, so they read the consequence of the choice rather than the
    choice.

    Two changes follow.

    1. GATE ON THE REFERENCE, NOT THE ROBOT. The clip is known ahead of time, so the
       frames whose single support is infeasible can be computed offline once, exactly
       as box_pickup/check_ankle_roll_feasible.py does: in single support the whole
       weight passes through one foot, the ankle_roll moment is m*g times the lateral
       CoM-to-roll-axis distance, and tau_limit buys a fixed offset. The adapter
       engages when any frame in [t, t+LOOKAHEAD] is one of those -- BEFORE the robot
       is in the pose.

    2. DRIVE THE SWING FOOT DOWN, NOT THE CoM BACK. Keeping both soles loaded is what
       prevents the commitment; pulling the CoM back only fights it afterwards. The
       error is the Gauss-Newton step that lowers the higher sole:

           g(q) = max(sole_height) - CONTACT,  clipped at 0
           e    = g * grad_q(g) / ||grad_q(g)||^2

    Inert whenever the upcoming reference is feasible, so it costs nothing on a clip
    that does not ask for the impossible -- which is the property that makes it a
    runtime fix rather than a retrain.
    """

    LOOKAHEAD = 25          # frames (0.5 s at 50 Hz) of warning before commitment

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._bad = None    # frames whose reference single support is infeasible

    def _infeasible_frames(self, ref_clip):
        """Frames where the reference asks more ankle_roll moment than tau allows."""
        self._lazy_setup()
        d = np.load(ref_clip, allow_pickle=True)
        bn = [str(x) for x in d["body_names"]]
        bp, bq = np.asarray(d["body_pos_w"]), np.asarray(d["body_quat_w"])
        from urdf_fk import quat_wxyz_to_mat
        n = len(bp)
        h = np.zeros((n, 2)); ay = np.zeros((n, 2))
        for i, b in enumerate(self._feet):
            j = bn.index(b)
            for f in range(n):
                Rm = quat_wxyz_to_mat(bq[f, j])
                h[f, i] = min((bp[f, j] + Rm @ s)[2] for s in self._sole[b])
                ay[f, i] = bp[f, j][1]
        # CoM of the reference pose, from the same body set
        from rebuild_reference_motion import load_masses, URDF
        mass = load_masses(URDF)
        com_y = np.zeros(n); tot = 0.0
        for link, (m, c) in mass.items():
            if link not in bn:
                continue
            j = bn.index(link); tot += m
            for f in range(n):
                Rm = quat_wxyz_to_mat(bq[f, j])
                com_y[f] += m * (bp[f, j] + Rm @ np.asarray(c))[1]
        com_y /= max(tot, 1e-9)
        bound = self.TAU_ROLL / (self._mass_total * self.G)
        down = h < self.CONTACT
        bad = np.zeros(n, dtype=bool)
        single = down.sum(1) == 1
        idx = np.where(single)[0]
        for f in idx:
            k = int(np.argmax(down[f]))
            if abs(com_y[f] - ay[f, k]) > bound:
                bad[f] = True
        return bad

    def set_context(self, **kw):
        super().set_context(**kw)
        if self._bad is None:
            clip = kw.get("ref_clip")
            if clip:
                try:
                    self._bad = self._infeasible_frames(clip)
                    print(f"[adapter] {int(self._bad.sum())} of {len(self._bad)} reference "
                          f"frames are infeasible single support", flush=True)
                except Exception as exc:                      # never fail the rollout
                    print(f"[adapter] could not precompute infeasible frames: {exc}", flush=True)
                    self._bad = np.zeros(1, dtype=bool)

    def _swing_excess(self, q, jn, rp, rq):
        h, _, _ = self._sole_h_and_com(q, jn, rp, rq)
        return max(float(h.max()) - self.CONTACT, 0.0)

    def delta_L(self, joint_error):
        self._steps += 1
        if self._ctx is None or self._bad is None:
            return np.zeros(len(np.asarray(joint_error)))
        f = int(self._ctx.get("frame", 0))
        hi = min(f + self.LOOKAHEAD, len(self._bad))
        if f >= len(self._bad) or not self._bad[f:hi].any():
            return np.zeros(len(np.asarray(joint_error)))   # upcoming clip is feasible

        self._lazy_setup()
        q = np.asarray(self._ctx["dof_pos"], dtype=float)
        jn = self._ctx["dof_names"]
        rp = np.asarray(self._ctx["root_pos"], dtype=float)
        rq = np.asarray(self._ctx["root_quat_wxyz"], dtype=float)
        g0 = self._swing_excess(q, jn, rp, rq)
        if g0 <= 0.0:
            return np.zeros_like(q)      # both soles already down: nothing to prevent
        self._engaged += 1
        grad = np.zeros_like(q)
        for j in range(len(q)):
            qp = q.copy(); qp[j] += self.EPS
            grad[j] = (self._swing_excess(qp, jn, rp, rq) - g0) / self.EPS
        n2 = float(grad @ grad)
        if n2 < 1e-12:
            return np.zeros_like(q)
        e = g0 * grad / n2
        s = self._spectral_scale() if hasattr(self, "_spectral_scale") else 1.0
        return -(s * self.P_diag * e)


class ComSlidingAdapter(StabilityAdapter):
    """Adaptive law on the CoM sliding variable, with a residual estimate that has units.

    Both earlier attempts drove on a POSITION error and both failed. Reactive
    (|CoM_y - stance_y| - bound) is nonzero only once the robot is already outside
    what the ankle can hold: single-support frames went 25 -> 48 with gain, worse
    monotonically. Preemptive (gate on the reference, drive the swing foot down) was
    indistinguishable from the control over 6 seeds: 43.5 +/- 10.0 against 37.2 +/-
    9.3, p = 0.28.

    Two things change here.

    SLIDING VARIABLE, so the law acts before the boundary is crossed:

        e = CoM_y - stance_y                (lateral offset, m)
        s = e_dot + LAMBDA * e              (Slotine's sliding variable)

    s is large while the CoM is still travelling outward, before |e| exceeds the
    bound, which is the lead time a position threshold cannot give. LAMBDA sets how
    far ahead: the surface s = 0 is reached ~1/LAMBDA seconds before e would be.

    RESIDUAL ESTIMATE WITH A MAGNITUDE, rather than a bare gradient step:

        theta_hat_dot = GAMMA_TH * s - SIGMA * theta_hat        (sigma-modification)
        F_cmd         = -(KD_S * s + theta_hat)                 (lateral force, N/kg)

    theta_hat estimates the unmodelled lateral acceleration driving the CoM out, in
    m/s^2. It is the quantity to watch: for this robot a residual of order the
    gravitational term g * (offset / height) ~ 9.81 * 0.1 / 0.6 ~ 1.6 m/s^2 is
    physically plausible, so an estimate that runs to tens or hundreds is diverging
    and one that stays at 1e-3 is inert. Both are visible in `theta_trace`.

    The scalar task-space demand maps to joint space through the CoM Jacobian, which
    is the physically correct mapping and keeps the magnitude meaningful:

        e_joint = -F_cmd * J / ||J||^2,   J = d(CoM_y - stance_y)/dq

    sigma-modification bounds theta_hat without needing persistent excitation, which
    matters because the CoM is only interesting for part of the clip.
    """

    LAMBDA = 3.0        # 1/s. sliding surface lead; ~0.33 s of anticipation
    GAMMA_TH = 2.0      # adaptation rate for the residual estimate
    SIGMA = 0.5         # leak on theta_hat (sigma-modification)
    KD_S = 1.0          # proportional term on the sliding variable

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.theta_hat = 0.0
        self._prev_e = None
        self.theta_trace = []
        self.s_trace = []
        self.ejoint_trace = []

    def reset(self):
        super().reset()
        self.theta_hat = 0.0
        self._prev_e = None
        self.theta_trace = []
        self.s_trace = []
        self.ejoint_trace = []

    def _lateral_offset(self, q, jn, rp, rq):
        """CoM_y - stance_y. Uses the loaded foot; in double support, the nearer one."""
        h, com, out = self._sole_h_and_com(q, jn, rp, rq)
        down = h < self.CONTACT
        ys = [out[b][0][1] for b in self._feet]
        if down.sum() == 1:
            ref_y = ys[int(np.argmax(down))]
        elif down.sum() == 2:
            ref_y = min(ys, key=lambda y: abs(com[1] - y))
        else:
            return None
        return float(com[1] - ref_y)

    def delta_L(self, joint_error):
        self._steps += 1
        n = len(np.asarray(joint_error))
        if self._ctx is None:
            return np.zeros(n)
        self._lazy_setup()
        q = np.asarray(self._ctx["dof_pos"], dtype=float)
        jn = self._ctx["dof_names"]
        rp = np.asarray(self._ctx["root_pos"], dtype=float)
        rq = np.asarray(self._ctx["root_quat_wxyz"], dtype=float)
        dt = float(self._ctx.get("dt", 0.02))

        e = self._lateral_offset(q, jn, rp, rq)
        if e is None:                     # airborne: nothing to push against
            self._prev_e = None
            self.theta_trace.append(self.theta_hat); self.s_trace.append(0.0)
            return np.zeros_like(q)

        e_dot = 0.0 if self._prev_e is None else (e - self._prev_e) / max(dt, 1e-6)
        self._prev_e = e
        sld = e_dot + self.LAMBDA * e

        # residual estimate, with a leak so it cannot wind up without excitation
        self.theta_hat += dt * (self.GAMMA_TH * sld - self.SIGMA * self.theta_hat)
        self.theta_trace.append(self.theta_hat)
        self.s_trace.append(sld)

        # Demanded lateral acceleration, m/s^2. theta_hat lands near 3 m/s^2 here,
        # the right order for this robot (g * offset/height ~ 1.6), so the ESTIMATOR
        # is sound -- but an acceleration cannot be handed to a law that expects a
        # joint error in radians. F * J / ||J||^2 has units (m/s^2)/(m/rad) = rad/s^2,
        # which at dt = 0.02 is ~2500x too large and destabilised the policy
        # (survival 741 -> 147, box dropped). Convert to the CoM DISPLACEMENT that
        # acceleration produces over the sliding surface's own time constant 1/LAMBDA:
        #     dx ~ F / LAMBDA^2      (m)
        # which is then a position error the Jacobian maps to radians correctly.
        F = -(self.KD_S * sld + self.theta_hat)
        dx = F / (self.LAMBDA ** 2)
        if abs(dx) < 1e-9:
            return np.zeros_like(q)
        self._engaged += 1

        J = np.zeros_like(q)
        for j in range(len(q)):
            qp = q.copy(); qp[j] += self.EPS
            ep = self._lateral_offset(qp, jn, rp, rq)
            J[j] = 0.0 if ep is None else (ep - e) / self.EPS
        n2 = float(J @ J)
        if n2 < 1e-12:
            return np.zeros_like(q)
        e_joint = -dx * J / n2
        self.ejoint_trace.append(float(np.linalg.norm(e_joint)))
        sc = self._spectral_scale() if hasattr(self, "_spectral_scale") else 1.0
        return -(sc * self.P_diag * e_joint)
