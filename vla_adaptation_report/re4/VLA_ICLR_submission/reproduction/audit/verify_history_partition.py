#!/usr/bin/env python3
"""Reconstruct the Appendix B.4 history-state counterexample.

This is a constructed modeling check, not an additional robot experiment.
The extra-input bound is checked on independent finite samples; its proof
is the Euclidean triangle inequality given in the accompanying memo.
"""
from pathlib import Path
import json
import math
import random


def main():
    fault = 0.3
    p = healthy_p = 0.7
    v = healthy_v = 0.0
    initial_distance = math.hypot(p - healthy_p, v - healthy_v)
    maximum_physical_error = 0.0
    for _ in range(25):
        nominal_action = 0.0
        issued_command = -fault
        q = issued_command + fault - nominal_action
        p = 0.8 * p + 0.2 * (issued_command + fault)
        healthy_p = 0.8 * healthy_p + 0.2 * nominal_action
        v, healthy_v = issued_command, nominal_action
        maximum_physical_error = max(maximum_physical_error, abs(p - healthy_p))
        assert q == 0.0
    augmented_distance = math.hypot(p - healthy_p, v - healthy_v)
    assert maximum_physical_error == 0.0
    assert initial_distance == 0.0
    assert augmented_distance == abs(fault) > 0.0

    rng = random.Random(20260910)
    maximum_violation = 0.0
    count = 1000
    for _ in range(count):
        sp, sv, hp, hv, sq, sh = [rng.uniform(-3.0, 3.0) for _ in range(6)]
        # F((p,v),q,h)=(.8p+.2q,h), F0((hp,hv))=(.8hp,0).
        actual_distance = math.hypot(0.8 * (sp - hp) + 0.2 * sq, sh)
        bound = 0.8 * math.hypot(sp - hp, sv - hv) + 0.2 * abs(sq) + abs(sh)
        maximum_violation = max(maximum_violation, actual_distance - bound)
    assert maximum_violation <= 1e-12

    results = {
        "scope": "Constructed history partition counterexample; no VLA evaluation.",
        "fault": fault,
        "physical_steps": 25,
        "initial_augmented_distance": initial_distance,
        "maximum_physical_tracking_error": maximum_physical_error,
        "raw_command_memory_difference": v - healthy_v,
        "augmented_distance_under_perfect_physical_correction": augmented_distance,
        "zero_forcing_q_only_tube_valid_for_raw_augmented_state": False,
        "extra_input_bound": {
            "nominal_amplification": 0.8,
            "physical_input_gain": 0.2,
            "history_input_gain": 1.0,
            "sample_checks": count,
            "maximum_positive_violation": maximum_violation,
        },
    }
    path = Path(__file__).with_name("history_partition_checks.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
