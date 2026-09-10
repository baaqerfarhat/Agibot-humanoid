"""Deterministic LIBERO scenario resets, including reset-randomized fixture geometry.

LIBERO's set_init_state restores MuJoCo data, but reset can also randomize fixture
body_pos/body_quat in the MODEL. Seed each reset by scenario, then apply the stored
initial state. Hard-reset model rebuilding also appends generated property samplers;
clear those before rebuilding so their number cannot change the seeded RNG sequence.
This does not pin policy sampling and does not retroactively validate old
paired rollouts. The caller should still compare complete physical fingerprints.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


MODEL_FIELDS = ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos", "geom_quat",
                "geom_friction", "geom_solref", "geom_solimp", "jnt_range", "dof_damping",
                "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                "actuator_forcerange", "eq_data")


def scenario_seed(suite, task, init, base_seed=7):
    """Stable uint32; independent of Python's process-randomized hash()."""
    payload = json.dumps(["libero-reset-v1", int(base_seed), str(suite), int(task), int(init)],
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _sim(env):
    return (env.env if hasattr(env, "env") else env).sim


def _digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def physics_arrays(env):
    """Relevant model arrays plus full simulation data state, available for numeric checks."""
    sim = _sim(env)
    result = {"model."+name: np.asarray(getattr(sim.model, name)).copy()
              for name in MODEL_FIELDS if hasattr(sim.model, name)}
    result.update({"data."+name: np.asarray(getattr(sim.data, name)).copy()
                   for name in ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied",
                                "mocap_pos", "mocap_quat") if hasattr(sim.data, name)})
    return result


def physics_fingerprint(env):
    arrays = physics_arrays(env)
    digests = {name: _digest(value) for name, value in arrays.items()}
    model = {k: v for k, v in digests.items() if k.startswith("model.")}
    return dict(fields=digests,
                model_sha256=hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest(),
                note="Listed model arrays and data fields only; policy RNG is not included.")


def reset_libero(env, init_state, *, suite, task, init, base_seed=7):
    """Seed cached-env reset by scenario, restore stored init, return obs and provenance.

    Clears external forces before reset and after state restoration. Apply the requested
    experiment fault AFTER this function returns. Other model changes (gain/friction etc.)
    remain the caller's responsibility to restore before switching arms.
    """
    sim = _sim(env)
    sim.data.qfrc_applied[:] = 0.0
    sim.data.xfrc_applied[:] = 0.0
    rs = env.env if hasattr(env, "env") else env
    cleared_properties = 0
    if getattr(rs, "hard_reset", False) and not getattr(rs, "deterministic_reset", False):
        # Stock LIBERO's _add_placement_initializer appends these on EVERY model
        # rebuild, but only initializes the list in __init__. They are regenerated
        # from the BDDL immediately below, so retaining them duplicates RNG draws.
        previous = getattr(rs, "object_property_initializers", ())
        supported = {"OpenCloseSampler", "TurnOnOffSampler"}
        if any(type(item).__name__ not in supported for item in previous):
            raise ValueError("scenario reset requires stock LIBERO property initializers")
        cleared_properties = len(previous)
        rs.object_property_initializers = []
    seed = scenario_seed(suite, task, init, base_seed)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    sim = _sim(env)  # hard reset may construct a different simulation object
    sim.data.qfrc_applied[:] = 0.0
    sim.data.xfrc_applied[:] = 0.0
    return obs, dict(protocol="libero-reset-v1", suite=str(suite), task=int(task), init=int(init),
                     base_seed=int(base_seed), scenario_seed=seed,
                     discarded_generated_property_initializers=cleared_properties,
                     regenerated_property_initializers=len(getattr(rs, "object_property_initializers", ())),
                     physics=physics_fingerprint(env))
