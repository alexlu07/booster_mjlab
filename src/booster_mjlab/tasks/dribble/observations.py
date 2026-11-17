"""Left/right augmentation for the T1 dribble observation layout."""

import torch
from tensordict import TensorDict

from booster_mjlab.tasks.velocity.mdp.observations import (
    CRITIC_EXTRA_DIM,
    T1_JOINT_DIM,
    T1_POLICY_DIM,
    flip_t1_action_left_right,
    flip_t1_critic_obs_left_right,
    flip_t1_policy_obs_left_right,
)


def flip_t1_dribble_obs_left_right(
    obs: torch.Tensor, critic: bool = False
) -> torch.Tensor:
    """Mirror term-major actor history or the critic's single privileged frame."""
    command_end = T1_POLICY_DIM - 1
    ball_dim = 5 if critic else 4
    frame_dim = command_end + ball_dim + (CRITIC_EXTRA_DIM if critic else 0)
    batch_size = len(obs)
    if critic:
        if obs.shape[1] != frame_dim:
            raise ValueError(
                f"Expected {frame_dim} critic observations, got {obs.shape[1]}."
            )
        history_length = 1
    else:
        if obs.shape[1] == 0 or obs.shape[1] % frame_dim:
            raise ValueError(
                f"Expected a multiple of {frame_dim} actor observations, "
                f"got {obs.shape[1]}."
            )
        history_length = obs.shape[1] // frame_dim
        # mjlab concatenates [term A's H frames, term B's H frames, ...].
        term_dims = (3, 3, T1_JOINT_DIM, T1_JOINT_DIM, T1_JOINT_DIM, 2, ball_dim)
        terms = obs.split([dim * history_length for dim in term_dims], dim=-1)
        obs = torch.cat(
            [
                term.reshape(batch_size, history_length, dim)
                for term, dim in zip(terms, term_dims, strict=True)
            ],
            dim=-1,
        ).reshape(batch_size * history_length, frame_dim)

    # Insert a zero yaw command to reuse the tested robot and foot-state mirrors.
    robot_obs = torch.cat(
        (
            obs[:, :command_end],
            obs.new_zeros(len(obs), 1),
            obs[:, command_end:-ball_dim],
        ),
        dim=-1,
    )
    mirror_robot = (
        flip_t1_critic_obs_left_right if critic else flip_t1_policy_obs_left_right
    )
    robot_obs = mirror_robot(robot_obs)
    ball_signs = (
        [1.0, -1.0, 1.0, 1.0, -1.0] if critic else [1.0, -1.0, 1.0, 1.0]
    )
    ball_obs = obs[:, -ball_dim:] * obs.new_tensor(ball_signs)
    mirrored = torch.cat(
        (robot_obs[:, :command_end], robot_obs[:, command_end + 1 :], ball_obs), dim=-1
    )
    if critic:
        return mirrored
    frames = mirrored.reshape(batch_size, history_length, frame_dim)
    return torch.cat(
        [term.reshape(batch_size, -1) for term in frames.split(term_dims, dim=-1)],
        dim=-1,
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
