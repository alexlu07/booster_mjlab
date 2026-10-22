"""Left/right augmentation for the T1 dribble observation layout."""

import torch
from tensordict import TensorDict

from booster_mjlab.tasks.velocity.mdp.observations import (
    CRITIC_EXTRA_DIM,
    T1_POLICY_DIM,
    flip_t1_action_left_right,
    flip_t1_critic_obs_left_right,
    flip_t1_policy_obs_left_right,
)


def flip_t1_dribble_obs_left_right(
    obs: torch.Tensor, critic: bool = False
) -> torch.Tensor:
    """Mirror robot state, planar command, and ball state without changing inputs."""
    # Dribble replaces the 3D twist with a 2D command and appends ball position(3)
    # and planar velocity(2), after the critic extras when present.
    command_end = T1_POLICY_DIM - 1
    expected_dim = command_end + 5 + (CRITIC_EXTRA_DIM if critic else 0)
    if obs.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} T1 dribble observations, got {obs.shape[1]}."
        )

    # Insert a zero yaw command to reuse the tested robot and foot-state mirrors.
    robot_obs = torch.cat(
        (obs[:, :command_end], obs.new_zeros(len(obs), 1), obs[:, command_end:-5]),
        dim=-1,
    )
    mirror_robot = (
        flip_t1_critic_obs_left_right if critic else flip_t1_policy_obs_left_right
    )
    robot_obs = mirror_robot(robot_obs)
    ball_obs = obs[:, -5:] * obs.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0])
    return torch.cat(
        (robot_obs[:, :command_end], robot_obs[:, command_end + 1 :], ball_obs), dim=-1
    )


def augment_symmetries_t1_dribble(
    env, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append mirrored PPO samples; AMP histories use their existing separate path."""
    del env
    if obs is not None:
        actor, critic = obs["actor"], obs["critic"]
        actor_aug = torch.cat((actor, flip_t1_dribble_obs_left_right(actor)), dim=0)
        critic_aug = torch.cat(
            (critic, flip_t1_dribble_obs_left_right(critic, critic=True)), dim=0
        )
        obs = TensorDict(
            {"actor": actor_aug, "critic": critic_aug}, batch_size=(len(actor_aug),)
        )
    if actions is not None:
        actions = torch.cat((actions, flip_t1_action_left_right(actions)), dim=0)
    return obs, actions
