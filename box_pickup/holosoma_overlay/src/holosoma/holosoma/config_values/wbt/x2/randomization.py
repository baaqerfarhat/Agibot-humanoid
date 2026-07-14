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
            "mass_distribution_params": [0.3, 1.5],
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

x2_31dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms, **x2_robot_material_dr},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

x2_31dof_wbt_randomization_w_object = RandomizationManagerCfg(
    setup_terms={**base_setup_terms, **x2_robot_material_dr, **x2_object_state_dr_at_setup},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

__all__ = ["x2_31dof_wbt_randomization", "x2_31dof_wbt_randomization_w_object"]
