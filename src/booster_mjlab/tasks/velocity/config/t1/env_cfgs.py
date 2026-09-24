"""Booster T1 velocity tracking environment configurations."""

from booster_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.tasks.velocity import mdp
from booster_mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef


T1_AMP_LEG_JOINTS = (
    r".*_hip_.*_joint",
    r".*_knee_.*_joint",
    r".*_ankle_.*_joint",
)


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the full-23-DoF serial T1 velocity tracking task."""
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": get_t1_robot_cfg()}
    cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
        "robot/angular-velocity"
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left|right)_ankle_roll_link$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body", entity="robot", pattern=r".*",
            exclude=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",), reduce="none", num_slots=1,
    )
    foot_height_scan = FootClearanceSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=name, entity="robot")
            for name in ("left_foot_sole", "right_foot_sole")
        ),
        pattern=FootSoleGridPatternCfg(size=(0.20, 0.08)),
        ray_alignment="yaw", max_distance=1.0, exclude_parent_body=True,
        include_geom_groups=(0,), debug_vis=True,
    )
    cfg.scene.sensors = (
        feet_ground_cfg, nonfoot_ground_cfg, self_collision_cfg, foot_height_scan,
    )

    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.curriculum = True

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = T1_ACTION_SCALE

    cfg.viewer.body_name = "trunk"
    assert cfg.commands is not None
    twist = cfg.commands["twist"]
    assert isinstance(twist, mdp.UniformVelocityCommandCfg)
    twist.rel_standing_envs = 0.2
    twist.viz.z_offset = 1.25

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision", "right_foot_collision",
    )
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["limb_inertia"].params["asset_cfg"].body_names = (r"(?!trunk$).*",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = (
            "left_foot", "right_foot",
        )
    for metric_cfg in cfg.metrics.values():
        if "asset_cfg" in metric_cfg.params:
            metric_cfg.params["asset_cfg"].site_names = ("left_foot", "right_foot")

    cfg.rewards["upper_body_posture"].params["asset_cfg"].joint_names = (
        r"aahead_.*", r".*_(shoulder|elbow)_.*", r"waist_yaw_joint",
    )
    cfg.rewards["upper_body_posture"].params.update(
        std_standing={
            r"aahead_.*": 0.05,
            r".*_(shoulder|elbow)_.*": 0.05,
            r"waist_yaw_joint": 0.05,
        },
        std_walking={
            r"aahead_.*": 0.10,
            r".*_(shoulder|elbow)_.*": 0.15,
            r"waist_yaw_joint": 0.10,
        },
        std_running={
            r"aahead_.*": 0.20,
            r".*_(shoulder|elbow)_.*": 0.35,
            r"waist_yaw_joint": 0.25,
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("illegal_contact", None)
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0
    return cfg


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = booster_t1_rough_env_cfg(play=play)
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 60
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 50
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    assert cfg.curriculum is not None
    cfg.curriculum.pop("terrain_levels", None)
    if play:
        assert cfg.commands is not None
        twist = cfg.commands["twist"]
        assert isinstance(twist, mdp.UniformVelocityCommandCfg)
        twist.ranges.lin_vel_x = (-2.5, 2.5)
        twist.ranges.lin_vel_y = (-2.0, 2.0)
        twist.ranges.ang_vel_z = (-3.7, 3.7)
    return cfg
