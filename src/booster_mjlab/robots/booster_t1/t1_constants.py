"""Booster T1 serial-ankle model and actuator configuration."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.robots.booster_k1.actuators import (
    ActuatorConfig,
    MotorPositionActuatorCfg,
)

T1_XML: Path = BOOSTER_MJLAB_SRC_PATH / "robots" / "booster_t1" / "xmls" / "t1.xml"
assert T1_XML.exists()

T1_JOINT_ORDER: tuple[str, ...] = (
    "aahead_yaw_joint", "aahead_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


def get_spec() -> mujoco.MjSpec:
    """Load the official T1 model and label its primitive collision geoms."""
    spec = mujoco.MjSpec.from_file(str(T1_XML))
    collision_index = 0
    for geom in spec.geoms:
        if not geom.contype and not geom.conaffinity:
            continue
        if not geom.name:
            geom.name = f"t1_{collision_index}_collision"
            collision_index += 1
        geom.group = 3
    return spec


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.70),
    joint_pos={
        "left_shoulder_roll_joint": -1.4,
        "right_shoulder_roll_joint": 1.4,
        "left_hip_pitch_joint": -0.4,
        "left_knee_pitch_joint": 0.8,
        "left_ankle_pitch_joint": -0.4,
        "right_hip_pitch_joint": -0.4,
        "right_knee_pitch_joint": 0.8,
        "right_ankle_pitch_joint": -0.4,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",), contype=1, conaffinity=1,
    solref=(0.01, 1), condim=3, priority=0, solmix=0.001,
    friction={r"^(left|right)_foot_collision$": (1.0,), ".*": (0.45,)},
)

_ACTION_SCALE_FACTOR = 0.25
_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)


def _motor(names: tuple[str, ...], *, armature: float, effort_limit: float, velocity_limit: float, stiffness: float, damping: float) -> MotorPositionActuatorCfg:
    motor = ActuatorConfig(
        armature=armature, effort_limit=effort_limit, velocity_limit=velocity_limit,
        knee_point_velocity=velocity_limit * 0.5, stiffness=stiffness, damping=damping,
    )
    return MotorPositionActuatorCfg(
        motor=motor, target_names_expr=names, effort_limit=effort_limit,
        armature=armature, stiffness=stiffness, damping=damping, **_DELAY,
    )


# Limits and armatures come from T1_23dof.xml. PD/speed values are conservative
# simulation defaults and must be identified before real-robot deployment.
T1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _motor((r"aahead_.*",), armature=0.0018, effort_limit=7.0, velocity_limit=8.0, stiffness=4.0, damping=0.25),
        _motor((r".*_shoulder_.*", r".*_elbow_.*"), armature=0.0282528, effort_limit=38.3, velocity_limit=18.0, stiffness=10.0, damping=1.0),
        _motor((r"waist_yaw_joint",), armature=0.0478125, effort_limit=68.0, velocity_limit=14.0, stiffness=80.0, damping=4.0),
        _motor((r".*_hip_pitch_.*",), armature=0.0523908, effort_limit=98.8, velocity_limit=14.0, stiffness=80.0, damping=4.0),
        _motor((r".*_hip_(roll|yaw)_.*",), armature=0.0478125, effort_limit=68.0, velocity_limit=14.0, stiffness=80.0, damping=4.0),
        _motor((r".*_knee_pitch_.*",), armature=0.0636012, effort_limit=130.5, velocity_limit=12.0, stiffness=80.0, damping=4.0),
        _motor((r".*_ankle_pitch_.*",), armature=0.0679104, effort_limit=73.1, velocity_limit=14.0, stiffness=50.0, damping=2.0),
        _motor((r".*_ankle_roll_.*",), armature=0.02037312, effort_limit=17.2, velocity_limit=14.0, stiffness=50.0, damping=2.0),
    ), soft_joint_pos_limit_factor=0.9,
)

T1_ACTION_SCALE = {
    name: _ACTION_SCALE_FACTOR * actuator.effort_limit / actuator.stiffness
    for actuator in T1_ARTICULATION.actuators
    for name in actuator.target_names_expr
}


def get_t1_robot_cfg(default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME, default_collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,)) -> EntityCfg:
    return EntityCfg(init_state=default_keyframe, collisions=default_collisions, spec_fn=get_spec, articulation=T1_ARTICULATION)
