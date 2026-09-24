"""Booster T1 full-body motion tracking configuration."""

import dataclasses as _dc
from copy import deepcopy

from booster_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.robots.booster_k1.torque_speed import (
    BoosterEntityCfg,
    BoosterPositionActuatorCfg,
    initialize_torque_speed_limits,
)
from booster_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from booster_mjlab.tasks.velocity.mdp import foot_height
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg

T1_TRACKING_BODIES = (
    "trunk", "aahead_yaw_link", "aahead_pitch_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_elbow_pitch_link", "left_elbow_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_elbow_pitch_link", "right_elbow_yaw_link", "waist_yaw_link", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_pitch_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_pitch_link", "right_ankle_pitch_link", "right_ankle_roll_link",
)


def _get_t1_tracking_robot_cfg() -> BoosterEntityCfg:
    """Use the K1 tracking torque-speed mechanism with T1 motor curves."""
    base_robot_cfg = get_t1_robot_cfg()
    robot_cfg = BoosterEntityCfg(
        **{
            f.name: getattr(base_robot_cfg, f.name)
            for f in _dc.fields(base_robot_cfg)
        }
    )
    robot_cfg.articulation = deepcopy(robot_cfg.articulation)
    assert robot_cfg.articulation is not None
    robot_cfg.articulation.actuators = tuple(
        BoosterPositionActuatorCfg(
            **{f.name: getattr(actuator, f.name) for f in _dc.fields(actuator)}
        )
        for actuator in robot_cfg.articulation.actuators
    )
    return robot_cfg


def booster_t1_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": _get_t1_tracking_robot_cfg()}
    self_collision = ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"), secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"), fields=("found",), reduce="none", num_slots=1)
    foot_scan = FootClearanceSensorCfg(name="foot_height_scan", frame=tuple(ObjRef(type="site", name=name, entity="robot") for name in ("left_foot_sole", "right_foot_sole")), pattern=FootSoleGridPatternCfg(size=(0.20, 0.08)), ray_alignment="yaw", max_distance=1.0, exclude_parent_body=True, include_geom_groups=(0,), debug_vis=True)
    cfg.scene.sensors = (self_collision, foot_scan)
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(func=foot_height, params={"sensor_name": "foot_height_scan"})
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = T1_ACTION_SCALE
    cfg.events["initialize_torque_speed_limits"] = EventTermCfg(
        mode="startup", func=initialize_torque_speed_limits
    )
    assert cfg.commands is not None
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.anchor_body_name = "trunk"
    motion.body_names = T1_TRACKING_BODIES
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot_collision", "right_foot_collision")
    cfg.events["body_friction"].params["asset_cfg"].geom_names = r"(?!(left|right)_foot_collision$).*_collision"
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["limb_inertia"].params["asset_cfg"].body_names = (r"(?!trunk$).*",)
    cfg.rewards["motion_foot_pos"] = RewardTermCfg(func=mdp.motion_relative_body_position_error_exp, weight=2.0, params={"command_name": "motion", "std": 0.1, "body_names": ("left_ankle_roll_link", "right_ankle_roll_link")})
    cfg.terminations["ee_body_pos"].params["body_names"] = ("left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_yaw_link", "right_elbow_yaw_link")
    cfg.viewer.body_name = "trunk"
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.sampling_mode = "start"
    return cfg
