"""T1 flat-ground AMP dribbling configuration."""

import math

from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.sensor import ContactMatch, ContactSensorCfg

from booster_mjlab.environment.ball.ball_constants import (
    get_ball_cfg,
    get_ball_radius,
    get_rolling_friction,
)
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.config.t1.env_cfgs import (
    T1_AMP_LEG_JOINTS,
    booster_t1_flat_env_cfg,
)

from . import mdp
from .perception import ACTOR_HISTORY_LENGTH, BallDetection


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
    cfg.scene.sensors += (
        ContactSensorCfg(
            name="nonfoot_ball_contact",
            primary=ContactMatch(
                mode="body",
                entity="robot",
                pattern=r".*",
                exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
            secondary=ContactMatch(mode="body", entity="ball", pattern="ball"),
            fields=("found",),
            reduce="netforce",
            num_slots=1,
        ),
    )
    cfg.sim.mujoco.cone = "elliptic"
    # Absolute environment steps keep the schedule aligned when resuming.
    stages = [
        {
            "step": iteration * runner_cfg.num_steps_per_env,
            "speed_range": (min_speed, max_speed),
            "heading_change_range": (-math.radians(angle), math.radians(angle)),
        }
        for iteration, min_speed, max_speed, angle in (
            (0, 0.2, 0.8, 45),
        )
    ]
    command_stage = stages[-1] if play else stages[0]
    cfg.commands = {
        "ball_velocity": mdp.BallVelocityCommandCfg(
            speed_range=command_stage["speed_range"],
            heading_change_range=command_stage["heading_change_range"],
            zero_speed_probability=0.0,
            resampling_time_range=(3.0, 6.0),
        )
    }
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["command"] = ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "ball_velocity"},
        )

    actor = cfg.observations["actor"]
    actor.history_length = ACTOR_HISTORY_LENGTH
    actor.flatten_history_dim = True
    # One packet keeps the detection flag aligned with the delayed position.
    # Perception remains enabled in play; it is the actor's sensor model.
    actor.terms["ball_detection"] = ObservationTermCfg(
        func=BallDetection,
        params={
            "position_noise_std": 0.03,
            "bias_range": 0.05,
            "horizontal_fov": math.radians(120),
            "max_distance": 4.0,
            "false_positive_probability": 0.02,
            "false_negative_probability": 0.1,
        },
        delay_min_lag=0,
        delay_max_lag=4,
        delay_hold_prob=0.3,
    )
    critic_terms = cfg.observations["critic"].terms
    critic_terms["ball_position"] = ObservationTermCfg(func=mdp.ball_position)
    critic_terms["ball_velocity"] = ObservationTermCfg(func=mdp.ball_velocity)

    # Appended after AMP's robot reset so the ball follows the sampled robot pose.
    cfg.events["reset_ball"] = EventTermCfg(
        func=mdp.reset_ball, mode="reset", params={"radius": get_ball_radius()}
    )
    # Resample physics per episode while keeping the ball's radius and COM fixed.
    # pseudo_inertia scales mass and inertia together by exp(2 * alpha).
    cfg.events["ball_inertia"] = EventTermCfg(
        func=velocity_mdp.dr.pseudo_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=("ball",)),
            "alpha_range": (0.5 * math.log(0.75), 0.5 * math.log(1.25)),
        },
    )
    cfg.events["ball_friction"] = EventTermCfg(
        func=velocity_mdp.dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
            "operation": "abs",
            "axes": [0, 1, 2],
            "ranges": {
                0: (0.25, 0.8),
                1: (0.0005, 0.006),
                2: (
                    get_rolling_friction(rolling_resistance=0.03),
                    get_rolling_friction(rolling_resistance=0.09),
                ),
            },
        },
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
    cfg.rewards["action_rate_l2"].weight = -0.2
    cfg.rewards["nonfoot_ball_contact"] = RewardTermCfg(
        func=mdp.nonfoot_ball_contact,
        weight=-5.0,
        params={"sensor_name": "nonfoot_ball_contact"},
    )

    upright = cfg.rewards["upright"]
    upright.func = velocity_mdp.upright
    upright.params = {
        "asset_cfg": upright.params["asset_cfg"],
        "std": upright.params["std_walking"],
    }
    posture = cfg.rewards["upper_body_posture"]
    posture.weight = -0.2
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
    if not play:
        cfg.curriculum["ball_command"] = CurriculumTermCfg(
            func=mdp.ball_command_curriculum, params={"stages": stages}
        )
    cfg.terminations["ball_too_far"] = TerminationTermCfg(func=mdp.ball_too_far)
    return cfg
