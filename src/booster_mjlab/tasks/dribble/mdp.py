"""Simple commands, observations, resets, and rewards for dribbling."""

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.lab_api.string import resolve_matching_names_values


def to_robot_yaw_frame(env, vector_w):
    return quat_apply_inverse(
        yaw_quat(env.scene["robot"].data.root_link_quat_w), vector_w
    )


def ball_position(env):
    """Ball position relative to the robot, in the robot's yaw frame."""
    delta = (
        env.scene["ball"].data.root_link_pos_w - env.scene["robot"].data.root_link_pos_w
    )
    return to_robot_yaw_frame(env, delta)


def ball_velocity(env):
    """Absolute planar ball velocity expressed in the robot's yaw frame."""
    return to_robot_yaw_frame(env, env.scene["ball"].data.root_link_lin_vel_w)[:, :2]


class BallVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return to_robot_yaw_frame(self._env, self.velocity_w)[:, :2]

    def _resample_command(self, env_ids):
        # Sample around the current forward direction, then hold fixed in world space.
        velocity = torch.zeros(len(env_ids), 3, device=self.device)
        velocity[:, 0].uniform_(*self.cfg.forward_range)
        velocity[:, 1].uniform_(*self.cfg.lateral_range)
        robot = self._env.scene["robot"]
        # Resampling also runs before forward() during a reset.
        pose = self._env.sim.data.qpos[:, robot.indexing.free_joint_q_adr][env_ids]
        quat = yaw_quat(pose[:, 3:7])
        self.velocity_w[env_ids] = quat_apply(quat, velocity)

    def _update_command(self, env_ids=None):
        pass  # The command property performs the current yaw-frame transform.

    def _update_metrics(self):
        error = (
            self.velocity_w[:, :2]
            - self._env.scene["ball"].data.root_link_lin_vel_w[:, :2]
        )
        self.metrics["velocity_error"] += (
            error.norm(dim=-1) * self._env.step_dt / self.cfg.resampling_time_range[1]
        )


@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
    forward_range: tuple[float, float] = (0.2, 0.8)
    lateral_range: tuple[float, float] = (-0.2, 0.2)

    def build(self, env):
        return BallVelocityCommand(self, env)


def reset_ball(env, env_ids, radius):
    """Run after the robot reset; place a stationary ball in front of its new pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    # Read qpos directly: derived body poses are stale until the reset's forward().
    pose = env.sim.data.qpos[:, robot.indexing.free_joint_q_adr][env_ids]
    offset = torch.zeros(len(env_ids), 3, device=env.device)
    offset[:, 0].uniform_(0.4, 0.6)
    offset[:, 1].uniform_(-0.1, 0.1)
    position = pose[:, :3] + quat_apply(yaw_quat(pose[:, 3:7]), offset)
    position[:, 2] = env.scene.env_origins[env_ids, 2] + radius
    rotation = torch.zeros(len(env_ids), 4, device=env.device)
    rotation[:, 0] = 1.0
    env.scene["ball"].write_root_link_pose_to_sim(
        torch.cat((position, rotation), dim=-1), env_ids=env_ids
    )
    env.scene["ball"].write_root_link_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )


def track_ball_velocity(env, std=0.3):
    error = ball_velocity(env) - env.command_manager.get_command("ball_velocity")
    return torch.exp(-error.square().sum(dim=-1) / std**2)


def ball_proximity(env, distance=0.4, std=0.5):
    excess = (ball_position(env)[:, :2].norm(dim=-1) - distance).clamp_min(0.0)
    return torch.exp(-(excess / std).square())


def ball_too_far(env, max_distance=3.0):
    delta = (
        env.scene["ball"].data.root_link_pos_w - env.scene["robot"].data.root_link_pos_w
    )
    return delta[:, :2].norm(dim=-1) > max_distance


def nonfoot_ball_contact(env, sensor_name):
    """Unit cost each timestep with any non-foot robot–ball contact."""
    return (env.scene[sensor_name].data.found > 0).any(dim=-1).float()


def feet_slip(env, sensor_name, asset_cfg):
    """Always-active foot slip cost, using terrain contacts only."""
    contact = env.scene[sensor_name].data.found > 0
    velocity = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    return (velocity.square().sum(dim=-1) * contact).sum(dim=-1)


class upper_body_posture:
    """Fixed per-joint posture tolerances, independent of any velocity command."""

    def __init__(self, cfg, env):
        asset_cfg = cfg.params["asset_cfg"]
        _, names = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names)
        _, _, values = resolve_matching_names_values(cfg.params["std"], names)
        self.std = torch.tensor(values, device=env.device)

    def __call__(self, env, asset_cfg, std):
        asset = env.scene[asset_cfg.name]
        error = (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
        return (error / self.std).square().mean(dim=-1)
