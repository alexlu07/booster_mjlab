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
        self._gui_controls = None
        self._gui_get_env_idx = None

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
        self._update_command(env_ids)

    def _update_command(self, env_ids=None):
        if self._gui_controls is None:
            return
        enabled, speed, direction = self._gui_controls
        if not enabled.value:
            return
        idx = self._gui_get_env_idx()
        if env_ids is not None and not (env_ids == idx).any():
            return
        heading = math.radians(direction.value)
        self.heading_w[idx] = heading
        self.velocity_w[idx, 0] = speed.value * math.cos(heading)
        self.velocity_w[idx, 1] = speed.value * math.sin(heading)
        self.velocity_w[idx, 2] = 0.0

    def create_gui(
        self, name, server, get_env_idx, on_change=None, request_action=None
    ):
        """Control the selected environment's ball velocity in the world frame.

        Callbacks change GUI state only; tensor writes happen during command
        updates on the simulation thread. Automatic sampling continues for all
        other environments and resumes at the next resample when disabled.
        """
        del request_action
        max_speed = max(0.1, self.cfg.speed_range[1])
        with server.gui.add_folder(name.replace("_", " ").capitalize()):
            enabled = server.gui.add_checkbox("Manual control", initial_value=False)
            speed = server.gui.add_slider(
                "Ball speed (m/s)", min=0.0, max=max_speed, step=0.01,
                initial_value=max(0.0, min(max_speed, sum(self.cfg.speed_range) / 2)),
            )
            direction = server.gui.add_slider(
                "Direction (world frame, deg)", min=-180.0, max=180.0,
                step=1.0, initial_value=0.0,
            )
            stop = server.gui.add_button("Stop ball command")

        @stop.on_click
        def _stop(_event):
            enabled.value = True
            speed.value = 0.0
            if on_change is not None:
                on_change()

        if on_change is not None:
            for control in (enabled, speed, direction):
                control.on_update(lambda _event: on_change())
        self._gui_controls = (enabled, speed, direction)
        self._gui_get_env_idx = get_env_idx

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
    speed_range: tuple[float, float] = (0.0, 0.8)
    heading_change_range: tuple[float, float] = (-math.radians(45), math.radians(45))
    """Heading change per resample in radians."""
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


def track_ball_velocity(env, std=0.3, allow_overspeed=False):
    """Track exact velocity, or a minimum forward speed with no lateral slip.

    Overspeed mode uses only the current command and velocity. For a stop
    command, retain the exact tracker's penalty on motion in every direction.
    """
    velocity = ball_velocity(env)
    command = env.command_manager.get_command("ball_velocity")
    if not allow_overspeed:
        error = (velocity - command).square().sum(dim=-1)
    else:
        command_speed = command.norm(dim=-1)
        direction = command / command_speed.clamp_min(1.0e-6)[:, None]
        forward_speed = (velocity * direction).sum(dim=-1)
        sideways_velocity = velocity - forward_speed[:, None] * direction
        shortfall = (command_speed - forward_speed).clamp_min(0.0)
        moving_error = shortfall.square() + sideways_velocity.square().sum(dim=-1)
        error = torch.where(
            command_speed > 1.0e-6,
            moving_error,
            velocity.square().sum(dim=-1),
        )
    return torch.exp(-error / std**2)


def ball_proximity(env, distance=0.4, std=0.5):
    excess = (ball_position(env)[:, :2].norm(dim=-1) - distance).clamp_min(0.0)
    return torch.exp(-(excess / std).square())


def stand_still_near_ball(
    env, distance=0.5, linear_std=0.1, yaw_std=0.2, command_threshold=1.0e-3
):
    """Reward a stationary robot near the ball only during stop commands.

    Ball stopping is handled independently by track_ball_velocity. This term
    discourages shuffling and turning while waiting beside the stopped ball.
    """
    command = env.command_manager.get_command("ball_velocity")
    is_stop = command.norm(dim=-1) < command_threshold
    near_ball = ball_position(env)[:, :2].norm(dim=-1) < distance
    robot = env.scene["robot"]
    linear_error = robot.data.root_link_lin_vel_w[:, :2].square().sum(dim=-1)
    yaw_error = robot.data.root_link_ang_vel_w[:, 2].square()
    stillness = torch.exp(-linear_error / linear_std**2 - yaw_error / yaw_std**2)
    return is_stop * near_ball * stillness


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
