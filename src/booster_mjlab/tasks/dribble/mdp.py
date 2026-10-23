"""Simple commands, observations, resets, and rewards for dribbling."""

from dataclasses import dataclass
import math

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    wrap_to_pi,
    yaw_quat,
)
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
        self.heading_w = torch.zeros(self.num_envs, device=self.device)
        self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)
        self._joystick_enabled = None
        self._joystick_speed = None
        self._joystick_heading = None
        self._joystick_get_env_idx = None

    @property
    def command(self):
        return to_robot_yaw_frame(self._env, self.velocity_w)[:, :2]

    def _resample_command(self, env_ids):
        # Start each episode from the robot's heading. Later turns are relative
        # to the previous command, independent of the robot's current heading.
        robot = self._env.scene["robot"]
        initial_ids = env_ids[self.command_counter[env_ids] == 0]
        # Resampling also runs before forward() during a reset.
        pose = self._env.sim.data.qpos[:, robot.indexing.free_joint_q_adr][initial_ids]
        self.heading_w[initial_ids] = euler_xyz_from_quat(pose[:, 3:7])[2]
        speed = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.speed_range
        )
        turn = torch.empty_like(speed).uniform_(*self.cfg.heading_change_range)
        stop = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.zero_speed_probability
        )
        speed[stop] = 0.0
        # Retain direction through stops, where a zero velocity has no angle.
        turn[stop] = 0.0
        heading = wrap_to_pi(self.heading_w[env_ids] + turn)
        self.heading_w[env_ids] = heading
        self.velocity_w[env_ids, 0] = speed * torch.cos(heading)
        self.velocity_w[env_ids, 1] = speed * torch.sin(heading)

    def _update_command(self, env_ids=None):
        if self._joystick_enabled is None or not self._joystick_enabled.value:
            return
        idx = self._joystick_get_env_idx()
        # Partial resets must not modify the command of another environment.
        if env_ids is not None and not torch.any(env_ids == idx):
            return
        heading = math.radians(self._joystick_heading.value)
        speed = self._joystick_speed.value
        self.heading_w[idx] = heading
        self.velocity_w[idx, 0] = speed * math.cos(heading)
        self.velocity_w[idx, 1] = speed * math.sin(heading)

    def create_gui(
        self, name, server, get_env_idx, on_change=None, request_action=None
    ):
        """Viewer joystick sliders override the selected environment's ball target."""
        from viser import Icon

        with server.gui.add_folder(name.replace("_", " ").title()):
            enabled = server.gui.add_checkbox("Enable joystick", initial_value=False)
            speed = server.gui.add_slider(
                "Ball speed (m/s)",
                min=0.0,
                max=self.cfg.speed_range[1],
                step=0.01,
                initial_value=0.0,
            )
            heading = server.gui.add_slider(
                "World heading (degrees)",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=0.0,
            )
            stop = server.gui.add_button("Stop ball command", icon=Icon.SQUARE_X)

            @stop.on_click
            def _stop(_event):
                speed.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_speed = speed
        self._joystick_heading = heading
        self._joystick_get_env_idx = get_env_idx

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
    speed_range: tuple[float, float] = (0.0, 0.2)
    heading_change_range: tuple[float, float] = (-math.radians(15), math.radians(15))
    """Heading change per resample in radians (initially ±15 degrees)."""
    zero_speed_probability: float = 0.1

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


def ball_command_curriculum(env, env_ids, stages):
    """Expand speed and turn ranges at absolute environment-step milestones."""
    del env_ids
    cfg = env.command_manager.get_term("ball_velocity").cfg
    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            cfg.speed_range = stage["speed_range"]
            cfg.heading_change_range = stage["heading_change_range"]
    return {
        "speed_max": torch.tensor(cfg.speed_range[1]),
        "heading_change_max": torch.tensor(cfg.heading_change_range[1]),
    }


def track_ball_velocity(env, std=0.2):
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
