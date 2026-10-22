"""T1 flat-ground AMP dribbling configuration."""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from booster_mjlab.environment.ball.ball_constants import get_ball_cfg, get_ball_radius
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.config.t1.env_cfgs import (
    T1_AMP_LEG_JOINTS,
    booster_t1_flat_env_cfg,
)

from . import mdp


def booster_t1_dribble_env_cfg(runner_cfg, play=False):
    cfg = booster_t1_flat_env_cfg(play=play)
    cfg = with_amp_obs_group(
        cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        include_base_lin_vel=True,
        amp_joint_names=T1_AMP_LEG_JOINTS,
    )
    cfg.scene.entities["ball"] = get_ball_cfg(ball_pos=(0.5, 0.0, None))
    cfg.sim.mujoco.cone = "elliptic"
    cfg.commands = {
        "ball_velocity": mdp.BallVelocityCommandCfg(resampling_time_range=(3.0, 6.0))
    }
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["command"] = ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "ball_velocity"},
        )
        terms["ball_position"] = ObservationTermCfg(func=mdp.ball_position)
        terms["ball_velocity"] = ObservationTermCfg(func=mdp.ball_velocity)

    # Appended after AMP's robot reset so the ball follows the sampled robot pose.
    cfg.events["reset_ball"] = EventTermCfg(
        func=mdp.reset_ball, mode="reset", params={"radius": get_ball_radius()}
    )
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "standing_pose_l1",
    ):
        cfg.rewards.pop(name)
    cfg.rewards["track_ball_velocity"] = RewardTermCfg(
        func=mdp.track_ball_velocity, weight=4.25
    )
    cfg.rewards["ball_proximity"] = RewardTermCfg(func=mdp.ball_proximity, weight=0.5)

    upright = cfg.rewards["upright"]
    upright.func = velocity_mdp.upright
    upright.params = {
        "asset_cfg": upright.params["asset_cfg"],
        "std": upright.params["std_walking"],
    }
    posture = cfg.rewards["upper_body_posture"]
    posture.func = mdp.upper_body_posture
    posture.params = {
        "asset_cfg": posture.params["asset_cfg"],
        "std": posture.params["std_walking"],
    }
    slip = cfg.rewards["foot_slip"]
    slip.func = mdp.feet_slip
    slip.params.pop("command_name")
    slip.params.pop("command_threshold")
    cfg.rewards["soft_landing"].params.pop("command_name")
    cfg.rewards["soft_landing"].params.pop("command_threshold")
    cfg.curriculum.pop("command_vel")
    cfg.curriculum.pop("soft_landing_weight")
    # Same absolute-step convention as velocity tracking: 24 rollout steps per
    # default PPO iteration. Restored counters select the right stage on resume.
    stages = [
        {
            "step": iteration * 24,
            "speed_range": (0.0, speed),
            "heading_change_range": (-math.radians(angle), math.radians(angle)),
        }
        for iteration, speed, angle in (
            (0, 0.2, 15),
            (5000, 0.5, 45),
            (10000, 0.5, 90),
            (15000, 1.0, 90),
            (20000, 1.0, 180),
        )
    ]
    if play:
        command = cfg.commands["ball_velocity"]
        command.speed_range = stages[-1]["speed_range"]
        command.heading_change_range = stages[-1]["heading_change_range"]
    else:
        cfg.curriculum["ball_command"] = CurriculumTermCfg(
            func=mdp.ball_command_curriculum, params={"stages": stages}
        )
    cfg.terminations["ball_too_far"] = TerminationTermCfg(func=mdp.ball_too_far)
    return cfg
