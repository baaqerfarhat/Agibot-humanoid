"""Whole Body Tracking curriculum presets for the AgiBot X2 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

x2_31dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["x2_31dof_wbt_curriculum"]
