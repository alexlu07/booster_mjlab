"""A position-only ball detector for the actor; the critic uses simulator state."""

import math

import torch

from .mdp import ball_position


ACTOR_HISTORY_LENGTH = 8


class BallDetection:
    """Return yaw-frame position (m) and a detection flag, never ball velocity.

    FOV is a robot-yaw-aligned horizontal cone, not a rendered camera or an
    occlusion model. A miss returns four zeros. A false positive reports a
    plausible ground-ball location inside the cone and sets the flag to one;
    the actor is not told which detections are false. The observation manager
    delays this entire packet and stores history after applying the delay.
    """

    def __init__(self, cfg, env):
        self.position_bias = torch.zeros(env.num_envs, 3, device=env.device)
        self.bias_range = cfg.params.get("bias_range", 0.05)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        bias = self.position_bias[env_ids]
        self.position_bias[env_ids] = torch.empty_like(bias).uniform_(
            -self.bias_range, self.bias_range
        )

    def __call__(
        self,
        env,
        position_noise_std=0.03,
        bias_range=0.05,
        horizontal_fov=math.radians(120),
        max_distance=4.0,
        false_positive_probability=0.02,
        false_negative_probability=0.1,
    ):
        del bias_range  # Sampled only on reset, never per observation.
        if position_noise_std < 0 or self.bias_range < 0:
            raise ValueError("Ball noise and bias magnitudes must be nonnegative.")
        if not 0 < horizontal_fov <= 2 * math.pi or max_distance <= 0:
            raise ValueError("Ball FOV must be in (0, 2*pi] and range positive.")
        if not all(
            0 <= probability <= 1
            for probability in (
                false_positive_probability, false_negative_probability
            )
        ):
            raise ValueError("Ball detection probabilities must be in [0, 1].")

        position = ball_position(env)
        distance = position[:, :2].norm(dim=-1)
        bearing = torch.atan2(position[:, 1], position[:, 0])
        visible = (bearing.abs() <= horizontal_fov / 2) & (distance <= max_distance)
        detected = visible & (
            torch.rand_like(distance) >= false_negative_probability
        )
        measured = (
            position
            + torch.randn_like(position) * position_noise_std
            + self.position_bias
        )

        # Spurious detections can also replace a real detection. Their height
        # is consistent with a ground ball without exposing its true x/y.
        false_positive = torch.rand_like(distance) < false_positive_probability
        fake_bearing = (torch.rand_like(distance) - 0.5) * horizontal_fov
        fake_range = max_distance * torch.rand_like(distance).sqrt()
        fake_position = torch.stack(
            (
                fake_range * fake_bearing.cos(),
                fake_range * fake_bearing.sin(),
                env.scene.env_origins[:, 2]
                + env.scene["ball"].data.default_root_state[:, 2]
                - env.scene["robot"].data.root_link_pos_w[:, 2],
            ),
            dim=-1,
        )
        measured = torch.where(false_positive[:, None], fake_position, measured)
        detected |= false_positive
        measured = torch.where(detected[:, None], measured, 0.0)
        return torch.cat((measured, detected[:, None].to(measured.dtype)), dim=-1)
