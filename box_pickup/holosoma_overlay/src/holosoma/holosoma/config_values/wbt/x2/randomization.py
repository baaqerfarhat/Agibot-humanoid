"""Whole Body Tracking randomization presets for the AgiBot X2 robot.

Base robot DR terms are robot-agnostic, so we reuse the G1 WBT sets. The object
physics DR is X2-specific: G1's ranges (friction 0.1-0.6, restitution up to
1.0, mass 1-4 kg) make a large fraction of episodes physically unsolvable for
X2, whose wrist actuators peak at 4.8 Nm. Holding a 4 kg box at friction 0.1
between the palm spheres requires ~200 N of squeeze per hand -- impossible --
so the policy learned to ignore the box entirely. The ranges below keep the
grasp learnable while still randomizing enough for transfer; widen them again
for robustness fine-tuning once the pickup is mastered.
"""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

from holosoma.config_values.wbt.g1.randomization import (
    base_reset_terms,
    base_setup_terms,
    base_step_terms,
)

# The v6 run showed the policy squeezing at the wrists' 4.8 Nm hardware limit
# and still losing the box after a ~4 cm lift: the grasp is friction-limited at
# the torque ceiling. Grippier box surface + lighter mass floor make the hold
# achievable; PhysX combines robot/object friction, so the robot-side floor is
# raised below as well.
x2_object_state_dr_at_setup = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.8, 1.4],
            "dynamic_friction_range": [0.7, 1.2],
            "restitution_range": [0.0, 0.2],
        },
    ),
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_startup",
        params={
            # v27: cover light cardboard up to denser demo boxes
            "mass_distribution_params": [0.3, 2.0],
        },
    ),
    "randomize_object_rigid_body_inertia_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_inertia_startup",
        params={
            "inertia_distribution_params_dict": {
                "Ixx": [0.5, 1.5],
            },
        },
    ),
}

# Raise the robot-side friction floor (G1 base allows 0.3): with PhysX friction
# combining, a slippery-palm draw nullifies any object-side friction and makes
# the squeeze-grasp episodes unwinnable. Floor 0.6 keeps the grasp physical;
# the upper bound is unchanged for robustness.
x2_robot_material_dr = {
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
        params={
            "static_friction_range": [0.6, 1.6],
            "dynamic_friction_range": [0.5, 1.2],
            "restitution_range": [0.0, 0.5],
        },
    ),
}

# v28: stronger + more frequent pushes than the G1 base (0.5 m/s @ 1-3 s).
# Random shoves landing mid-bend and mid-stand-up force the policy to learn
# balance recovery in exactly the phases that wobble on hardware, instead of
# only ever seeing them unperturbed.
x2_push_dr = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            "push_interval_s": [1.0, 2.5],
            "max_push_vel": [0.7, 0.7, 0.3, 0.6, 0.6, 0.9],
            "enabled": True,
        },
    ),
}

# Hardware-transfer DR (v11, widened further in v27 after stand-up-with-box
# thrashing on the real robot): PD gains, control latency, and encoder bias
# must cover the real ROS loop + actuator mismatch, especially once a box is
# in the arms and the COM is no longer the training default.
# v29: kp/kd +/-30% -> +/-40% so the policy tolerates deployment gain
# retuning (softer or stiffer PD on the real robot) without retraining.
x2_hardware_robustness_setup = {
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [0.60, 1.40],
            "kd_range": [0.60, 1.40],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": True,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 3],  # 0-60 ms at 50 Hz
            "enabled": True,
        },
    ),
    # Encoder offset / imperfect initial pose: +/-0.07 rad (v27: 0.05).
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.07, 0.07],
            "enabled": True,
        },
    ),
}

# v27: wider torso COM -- payload + imperfect arm squeeze shift the CoM far
# more than empty-robot walking; stand-up after grasp is the failure mode.
x2_torso_com_dr = {
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {
                "x": [-0.08, 0.10],   # forward bias covers box on chest (v28: wider)
                "y": [-0.06, 0.06],
                "z": [-0.06, 0.06],
            },
        },
    ),
}

x2_31dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **x2_push_dr,       # overrides G1 push magnitude/frequency
        **x2_torso_com_dr,  # overrides G1 COM ranges
        **x2_robot_material_dr,
        **x2_hardware_robustness_setup,
    },
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

x2_31dof_wbt_randomization_w_object = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **x2_push_dr,       # overrides G1 push magnitude/frequency
        **x2_torso_com_dr,  # overrides G1 COM ranges
        **x2_robot_material_dr,
        **x2_hardware_robustness_setup,
        **x2_object_state_dr_at_setup,
    },
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

__all__ = ["x2_31dof_wbt_randomization", "x2_31dof_wbt_randomization_w_object"]
