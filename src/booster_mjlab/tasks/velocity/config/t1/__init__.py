"""Booster T1 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.config.t1.env_cfgs import (
    T1_AMP_LEG_JOINTS,
    booster_t1_flat_env_cfg,
    booster_t1_rough_env_cfg,
)
from booster_mjlab.tasks.velocity.config.t1.rl_cfg import (
    booster_t1_amp_ppo_runner_cfg,
    booster_t1_amp_ppo_symmetric_runner_cfg,
    booster_t1_ppo_runner_cfg,
    booster_t1_symmetric_ppo_runner_cfg,
)
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner


def _with_amp(env_cfg, runner_cfg):
    return with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        include_base_lin_vel=True,
        amp_joint_names=T1_AMP_LEG_JOINTS,
    )


_PPO_TASKS = {
    "Rough": (booster_t1_rough_env_cfg, booster_t1_ppo_runner_cfg),
    "Flat": (booster_t1_flat_env_cfg, booster_t1_ppo_runner_cfg),
    "Rough-DA": (booster_t1_rough_env_cfg, booster_t1_symmetric_ppo_runner_cfg),
    "Flat-DA": (booster_t1_flat_env_cfg, booster_t1_symmetric_ppo_runner_cfg),
}
_AMP_TASKS = {
    "Rough-Amp": (booster_t1_rough_env_cfg, booster_t1_amp_ppo_runner_cfg),
    "Flat-Amp": (booster_t1_flat_env_cfg, booster_t1_amp_ppo_runner_cfg),
    "Flat-Amp-DA": (booster_t1_flat_env_cfg, booster_t1_amp_ppo_symmetric_runner_cfg),
}

for use_muon in (False, True):
    optimizer = "-Muon" if use_muon else ""
    for name, (env_cfg_fn, runner_cfg_fn) in _PPO_TASKS.items():
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{optimizer}-Booster-T1",
            env_cfg=env_cfg_fn(), play_env_cfg=env_cfg_fn(play=True),
            rl_cfg=runner_cfg_fn(use_muon=use_muon), runner_cls=BoosterOnPolicyRunner,
        )
    for name, (env_cfg_fn, runner_cfg_fn) in _AMP_TASKS.items():
        runner_cfg = runner_cfg_fn(use_muon=use_muon)
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{optimizer}-Booster-T1",
            env_cfg=_with_amp(env_cfg_fn(), runner_cfg),
            play_env_cfg=_with_amp(env_cfg_fn(play=True), runner_cfg),
            rl_cfg=runner_cfg, runner_cls=VelocityAmpOnPolicyRunner,
        )
