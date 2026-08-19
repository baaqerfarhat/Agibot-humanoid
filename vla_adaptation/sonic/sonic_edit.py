"""SONIC decoder edit harness — apply b6/w6 layer edits to the shipped ONNX, verified.

The validated adaptation method edits ONE layer of a frozen policy:
    b6 : output bias shift        b <- b + db          (constant action offset)
    w6 : output row rescale       W[:,j] <- (1+t_j)W[:,j], b_j <- (1+t_j)b_j
         (multiplicative action rescale — the inverse class for gain-type faults)

On SONIC these live in the decoder ONNX as plain graph initializers:
    module.decoders.g1_dyn.module.12.bias   [29]
    onnx::MatMul_142                         [512, 29]

edit_decoder() loads the ONNX, mutates the initializers, and writes a variant file.
Any consumer (onnxruntime, the C++ TensorRT deploy, the sim2sim stack) executes the
edited layer — no gradients, no retraining, no model surgery beyond the two tensors.

Verification (run this file): for random obs,
    b6: action(edited) - action(base) == db          (exactly, to float32 eps)
    w6: action(edited) == (1+t) * action(base)       (exactly — final layer is affine)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

BASE = Path.home() / "vla_ws/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
DECODER = BASE / "model_decoder.onnx"
BIAS_NAME = "module.decoders.g1_dyn.module.12.bias"
WEIGHT_NAME = "onnx::MatMul_142"


def edit_decoder(db: np.ndarray | None = None, theta: np.ndarray | None = None,
                 src: Path = DECODER, dst: Path | None = None) -> Path:
    """Write a decoder variant with bias shift `db` [29] and/or row rescale `theta` [29]."""
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    W = numpy_helper.to_array(inits[WEIGHT_NAME]).copy()      # [512, 29]
    b = numpy_helper.to_array(inits[BIAS_NAME]).copy()        # [29]
    if theta is not None:
        scale = 1.0 + np.asarray(theta, np.float32)
        W *= scale[None, :]
        b *= scale
    if db is not None:
        b += np.asarray(db, np.float32)
    inits[WEIGHT_NAME].CopyFrom(numpy_helper.from_array(W.astype(np.float32), WEIGHT_NAME))
    inits[BIAS_NAME].CopyFrom(numpy_helper.from_array(b.astype(np.float32), BIAS_NAME))
    dst = dst or src.with_name("model_decoder_edited.onnx")
    onnx.save(m, str(dst))
    return dst


def _run(path: Path, obs: np.ndarray) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    return s.run(None, {iname: obs})[0][0]


def _verify():
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, (1, 994)).astype(np.float32)
    a0 = _run(DECODER, obs)
    print(f"base action: shape {a0.shape}  |a| mean {np.abs(a0).mean():.3f}")

    db = rng.normal(0, 0.05, 29).astype(np.float32)
    p = edit_decoder(db=db)
    a1 = _run(p, obs)
    err_b = np.max(np.abs((a1 - a0) - db))
    print(f"b6 edit:  max|delta - db|      = {err_b:.2e}  "
          f"{'PASS' if err_b < 1e-5 else 'FAIL'}")

    theta = rng.normal(0, 0.1, 29).astype(np.float32)
    p = edit_decoder(theta=theta)
    a2 = _run(p, obs)
    err_w = np.max(np.abs(a2 - (1.0 + theta) * a0))
    print(f"w6 edit:  max|a - (1+t)*a0|    = {err_w:.2e}  "
          f"{'PASS' if err_w < 1e-4 else 'FAIL'}")

    # combined edit, second random obs — order of operations check
    obs2 = rng.normal(0, 1, (1, 994)).astype(np.float32)
    a0b = _run(DECODER, obs2)
    p = edit_decoder(db=db, theta=theta)
    a3 = _run(p, obs2)
    err_c = np.max(np.abs(a3 - ((1.0 + theta) * a0b + db)))
    print(f"combined: max|a - ((1+t)a0+db)| = {err_c:.2e}  "
          f"{'PASS' if err_c < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    _verify()
