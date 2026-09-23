"""RL configuration for Booster T1 velocity tasks."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    booster_k1_amp_ppo_runner_cfg,
    booster_k1_ppo_runner_cfg,
)

_T1_SYMMETRY = "booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries_t1"


def _set_t1_symmetry(cfg: RslRlOnPolicyRunnerCfg, enabled: bool) -> None:
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=enabled,
        use_mirror_loss=False,
        data_augmentation_func=_T1_SYMMETRY,
    )


def booster_t1_ppo_runner_cfg(use_muon: bool = False) -> RslRlOnPolicyRunnerCfg:
    cfg = booster_k1_ppo_runner_cfg(use_muon=use_muon)
    _set_t1_symmetry(cfg, enabled=False)
    cfg.experiment_name = cfg.experiment_name.replace("k1_velocity", "t1_velocity")
    return cfg


def booster_t1_symmetric_ppo_runner_cfg(use_muon: bool = False) -> RslRlOnPolicyRunnerCfg:
    cfg = booster_t1_ppo_runner_cfg(use_muon=use_muon)
    _set_t1_symmetry(cfg, enabled=True)
    cfg.experiment_name = cfg.experiment_name.replace("t1_velocity", "t1_velocity_symmetric")
    return cfg


def booster_t1_amp_ppo_runner_cfg(use_muon: bool = False) -> AmpOnPolicyRunnerCfg:
    cfg = booster_k1_amp_ppo_runner_cfg(use_muon=use_muon)
    cfg.dataset_root = ""
    cfg.dataset_augmentations = [
        {"name": "t1_mirror"},
        {"name": "speed", "percent": 10.0},
        {"name": "speed", "percent": -10.0},
        {"name": "speed", "percent": 20.0},
        {"name": "speed", "percent": -20.0},
    ]
    _set_t1_symmetry(cfg, enabled=False)
    cfg.experiment_name = cfg.experiment_name.replace("k1_velocity_amp", "t1_velocity_amp")
    return cfg


def booster_t1_amp_ppo_symmetric_runner_cfg(use_muon: bool = False) -> AmpOnPolicyRunnerCfg:
    cfg = booster_t1_amp_ppo_runner_cfg(use_muon=use_muon)
    _set_t1_symmetry(cfg, enabled=True)
    cfg.experiment_name = cfg.experiment_name.replace("t1_velocity_amp", "t1_velocity_amp_symmetric")
    return cfg
