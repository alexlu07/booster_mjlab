from __future__ import annotations

from functools import partial

from rsl_rl.env import VecEnv
from tensordict import TensorDict

import torch

K1_JOINT_DIM = 22
K1_ACTION_DIM = 22

# Joint slots whose axes change sign under a left/right mirror.
K1_INVERTED_JOINT_INDICES: tuple[int, ...] = (
    0,
    3,
    5,
    7,
    9,
    11,
    12,
    15,
    17,
    18,
    21,
)

# The parallel ankle's A/B crank axes both point along +y. Unlike serial ankle
# roll, neither crank changes sign under a left/right mirror.
K1_PARALLEL_INVERTED_JOINT_INDICES: tuple[int, ...] = tuple(
    index for index in K1_INVERTED_JOINT_INDICES if index not in (15, 21)
)
POLICY_DIM_NO_BASE_LIN_VEL = 75
CRITIC_EXTRA_DIM = 15


def _augment_symmetries(
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    inverted_indices: tuple[int, ...],
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply symmetry augmentations to observations and actions."""

    if obs is not None:
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]
        actor_dim = actor_obs.shape[1]
        critic_dim = critic_obs.shape[1]
        if not _is_supported_policy_dim(actor_dim):
            raise ValueError(
                f"Unsupported actor observation dim: {actor_dim}. "
                f"Expected one of {SUPPORTED_POLICY_DIMS}."
            )
        expected_critic_dim = actor_dim + CRITIC_EXTRA_DIM
        if critic_dim != expected_critic_dim:
            raise ValueError(
                f"Critic observation dim mismatch: got {critic_dim}, "
                f"expected {expected_critic_dim}."
            )

        n_envs = actor_obs.shape[0]

        # augment the batch size to 2 different symmetries
        actor_aug = torch.zeros(n_envs * 2, actor_obs.shape[1], device=actor_obs.device)
        actor_aug[:n_envs] = actor_obs
        actor_aug[n_envs : 2 * n_envs] = flip_k1_policy_obs_left_right(
            actor_obs, inverted_indices
        )

        critic_aug = torch.zeros(
            n_envs * 2, critic_obs.shape[1], device=critic_obs.device
        )
        critic_aug[:n_envs] = critic_obs
        critic_aug[n_envs : 2 * n_envs] = flip_k1_critic_obs_left_right(
            critic_obs, inverted_indices
        )

        obs = TensorDict(
            {"actor": actor_aug, "critic": critic_aug}, batch_size=(n_envs * 2,)
        )

    if actions is not None:
        if actions.shape[1] != K1_ACTION_DIM:
            raise ValueError(
                f"Unsupported action dim: {actions.shape[1]}. Expected {K1_ACTION_DIM}."
            )

        n_envs = actions.shape[0]

        # augment the batch size to 2 different symmetries
        actions_aug = torch.zeros(n_envs * 2, actions.shape[1], device=actions.device)
        actions_aug[:n_envs] = actions[:]
        actions_aug[n_envs : 2 * n_envs] = flip_k1_action_left_right(
            actions, inverted_indices
        )

        actions = actions_aug

    return obs, actions


def augment_symmetries(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the serial-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_INVERTED_JOINT_INDICES)


def augment_symmetries_parallel(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the parallel-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_PARALLEL_INVERTED_JOINT_INDICES)


def flip_k1_action_left_right(
    action: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    action = action.clone()
    # switch left and right joints
    action = _switch_k1_joints_left_right(action, inverted_indices)
    return action


def flip_k1_policy_obs_left_right(
    obs: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1]
    num_linear_obs_blocks = _get_num_linear_obs_blocks(policy_dim)
    if num_linear_obs_blocks is None:
        raise ValueError(
            f"Unsupported policy observation dim: {policy_dim}. "
            f"Expected one of {SUPPORTED_POLICY_DIMS}."
        )

    # No leading base linear terms in the current velocity policy layout.
    # Each 3D vector mirrors with [x, y, z] -> [x, -y, z].
    for i in range(num_linear_obs_blocks):
        start = i * 3
        obs[:, start : start + 3] = obs[:, start : start + 3] * obs.new_tensor(
            [1.0, -1.0, 1.0]
        )

    base_offset = num_linear_obs_blocks * 3

    # base ang vel
    obs[:, base_offset : base_offset + 3] = obs[
        :, base_offset : base_offset + 3
    ] * obs.new_tensor([-1.0, 1.0, -1.0])
    # projected gravity
    obs[:, base_offset + 3 : base_offset + 6] = obs[
        :, base_offset + 3 : base_offset + 6
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    joint_pos_start = base_offset + 6
    joint_vel_start = joint_pos_start + K1_JOINT_DIM
    last_actions_start = joint_vel_start + K1_JOINT_DIM
    command_start = last_actions_start + K1_ACTION_DIM

    # joint pos
    obs[:, joint_pos_start:joint_vel_start] = _switch_k1_joints_left_right(
        obs[:, joint_pos_start:joint_vel_start], inverted_indices
    )
    # joint vel
    obs[:, joint_vel_start:last_actions_start] = _switch_k1_joints_left_right(
        obs[:, joint_vel_start:last_actions_start], inverted_indices
    )
    # last actions
    obs[:, last_actions_start:command_start] = _switch_k1_joints_left_right(
        obs[:, last_actions_start:command_start], inverted_indices
    )
    # velocity command
    obs[:, command_start : command_start + 3] = obs[
        :, command_start : command_start + 3
    ] * obs.new_tensor([1.0, -1.0, -1.0])

    return obs


def flip_k1_critic_obs_left_right(
    obs: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1] - CRITIC_EXTRA_DIM
    if not _is_supported_policy_dim(policy_dim):
        raise ValueError(
            f"Unsupported critic observation dim: {obs.shape[1]}. "
            f"Expected policy part of one of {SUPPORTED_POLICY_DIMS} "
            f"plus {CRITIC_EXTRA_DIM} extra dims."
        )

    # policy obs
    obs[:, :policy_dim] = flip_k1_policy_obs_left_right(
        obs[:, :policy_dim], inverted_indices
    )

    # base lin vel: [x, y, z] -> [x, -y, z]
    base_lin_vel_start = policy_dim
    obs[:, base_lin_vel_start : base_lin_vel_start + 3] = obs[
        :, base_lin_vel_start : base_lin_vel_start + 3
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    foot_height_start = policy_dim + 3
    foot_air_time_start = foot_height_start + 2
    foot_contact_start = foot_air_time_start + 2
    foot_contact_forces_start = foot_contact_start + 2

    # foot height
    obs[:, [foot_height_start, foot_height_start + 1]] = obs[
        :, [foot_height_start + 1, foot_height_start]
    ]
    # foot air time
    obs[:, [foot_air_time_start, foot_air_time_start + 1]] = obs[
        :, [foot_air_time_start + 1, foot_air_time_start]
    ]
    # foot contact
    obs[:, [foot_contact_start, foot_contact_start + 1]] = obs[
        :, [foot_contact_start + 1, foot_contact_start]
    ]
    # foot contact forces
    obs[
        :,
        [
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
        ],
    ] = obs[
        :,
        [
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
        ],
    ] * obs.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    return obs


SUPPORTED_POLICY_DIMS = (POLICY_DIM_NO_BASE_LIN_VEL,)


def _is_supported_policy_dim(policy_dim: int) -> bool:
    return _get_num_linear_obs_blocks(policy_dim) is not None


def _get_num_linear_obs_blocks(policy_dim: int) -> int | None:
    """Infer number of leading 3D base linear vectors from policy dim.

    Layout is:
    [base_ang_vel(3), projected_gravity(3), joint_pos(22), joint_vel(22),
     last_action(22), command(3)]
    """
    if policy_dim != POLICY_DIM_NO_BASE_LIN_VEL:
        return None
    return 0


def _switch_k1_joints_left_right(
    joints: torch.Tensor,
    inverted_indices: tuple[int, ...] = K1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    """Switch left and right joints in the K1 joint tensor.

    Joint order:

    'Head_Yaw' (Inverted), 'Head_Pitch',
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll' (Inverted), 'Left_Elbow_Pitch', 'Left_Elbow_Yaw' (Inverted),
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll' (Inverted), 'Right_Elbow_Pitch', 'Right_Elbow_Yaw' (Inverted),
    'Left_Hip_Pitch', 'Left_Hip_Roll' (Inverted), 'Left_Hip_Yaw' (Inverted), 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll' (Inverted),
    'Right_Hip_Pitch', 'Right_Hip_Roll' (Inverted), 'Right_Hip_Yaw' (Inverted), 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll' (Inverted)

    Joints marked as "Inverted" need to have their sign flipped when calculating the left-right symmetry.

    Args:
        joint_tensor (torch.Tensor): The joint tensor of shape (..., 22).
    """
    if joints.shape[1] != K1_JOINT_DIM:
        raise ValueError(
            f"Unsupported joint tensor dim: {joints.shape[1]}. Expected {K1_JOINT_DIM}."
        )
    joints_flipped = torch.zeros_like(joints)

    # Head joints
    joints_flipped[:, :2] = joints[:, :2]
    # Shoulders and Elbows
    joints_flipped[:, 2:6] = joints[:, 6:10]
    joints_flipped[:, 6:10] = joints[:, 2:6]
    # Hips, Knees, Ankles
    joints_flipped[:, 10:16] = joints[:, 16:22]
    joints_flipped[:, 16:22] = joints[:, 10:16]

    joints_flipped[:, list(inverted_indices)] = -joints_flipped[
        :, list(inverted_indices)
    ]

    return joints_flipped


flip_k1_parallel_action_left_right = partial(
    flip_k1_action_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)
flip_k1_parallel_policy_obs_left_right = partial(
    flip_k1_policy_obs_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)
flip_k1_parallel_critic_obs_left_right = partial(
    flip_k1_critic_obs_left_right,
    inverted_indices=K1_PARALLEL_INVERTED_JOINT_INDICES,
)


# T1 has the K1 upper-body layout plus a central waist-yaw joint before the
# two six-DoF legs. Keep this alongside the K1 implementation until a third
# layout warrants a broader symmetry abstraction.
T1_JOINT_DIM = 23
T1_INVERTED_JOINT_INDICES: tuple[int, ...] = (
    0, 3, 5, 7, 9, 10, 12, 13, 16, 18, 19, 22,
)
T1_POLICY_DIM = 6 + 3 * T1_JOINT_DIM + 3


def _switch_t1_joints_left_right(
    joints: torch.Tensor,
    inverted_indices: tuple[int, ...] = T1_INVERTED_JOINT_INDICES,
) -> torch.Tensor:
    if joints.shape[1] != T1_JOINT_DIM:
        raise ValueError(f"Expected {T1_JOINT_DIM} T1 joints, got {joints.shape[1]}.")
    out = torch.zeros_like(joints)
    out[:, :2] = joints[:, :2]
    out[:, 2:6] = joints[:, 6:10]
    out[:, 6:10] = joints[:, 2:6]
    out[:, 10] = joints[:, 10]
    out[:, 11:17] = joints[:, 17:23]
    out[:, 17:23] = joints[:, 11:17]
    out[:, list(inverted_indices)] *= -1.0
    return out


def flip_t1_action_left_right(action: torch.Tensor) -> torch.Tensor:
    return _switch_t1_joints_left_right(action.clone())


def flip_t1_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[1] != T1_POLICY_DIM:
        raise ValueError(f"Expected {T1_POLICY_DIM} T1 actor observations, got {obs.shape[1]}.")
    out = obs.clone()
    out[:, :3] *= out.new_tensor([-1.0, 1.0, -1.0])
    out[:, 3:6] *= out.new_tensor([1.0, -1.0, 1.0])
    joint_pos = 6
    joint_vel = joint_pos + T1_JOINT_DIM
    last_action = joint_vel + T1_JOINT_DIM
    command = last_action + T1_JOINT_DIM
    out[:, joint_pos:joint_vel] = _switch_t1_joints_left_right(out[:, joint_pos:joint_vel])
    out[:, joint_vel:last_action] = _switch_t1_joints_left_right(out[:, joint_vel:last_action])
    out[:, last_action:command] = _switch_t1_joints_left_right(out[:, last_action:command])
    out[:, command:command + 3] *= out.new_tensor([1.0, -1.0, -1.0])
    return out


def flip_t1_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[1] != T1_POLICY_DIM + CRITIC_EXTRA_DIM:
        raise ValueError(f"Expected {T1_POLICY_DIM + CRITIC_EXTRA_DIM} T1 critic observations, got {obs.shape[1]}.")
    out = obs.clone()
    out[:, :T1_POLICY_DIM] = flip_t1_policy_obs_left_right(out[:, :T1_POLICY_DIM])
    out[:, T1_POLICY_DIM:T1_POLICY_DIM + 3] *= out.new_tensor([1.0, -1.0, 1.0])
    foot_height = T1_POLICY_DIM + 3
    foot_air_time = foot_height + 2
    foot_contact = foot_air_time + 2
    foot_forces = foot_contact + 2
    for start in (foot_height, foot_air_time, foot_contact):
        out[:, [start, start + 1]] = out[:, [start + 1, start]]
    out[:, foot_forces:foot_forces + 6] = out[:, [
        foot_forces + 3, foot_forces + 4, foot_forces + 5,
        foot_forces, foot_forces + 1, foot_forces + 2,
    ]] * out.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    return out


def augment_symmetries_t1(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry augmentation for serial T1."""
    del env
    if obs is not None:
        actor, critic = obs["actor"], obs["critic"]
        actor_aug = torch.cat((actor, flip_t1_policy_obs_left_right(actor)), dim=0)
        critic_aug = torch.cat((critic, flip_t1_critic_obs_left_right(critic)), dim=0)
        obs = TensorDict({"actor": actor_aug, "critic": critic_aug}, batch_size=(len(actor_aug),))
    if actions is not None:
        actions = torch.cat((actions, flip_t1_action_left_right(actions)), dim=0)
    return obs, actions
