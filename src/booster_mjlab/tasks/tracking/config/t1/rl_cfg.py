"""RL configuration for Booster T1 tracking."""

from booster_mjlab.tasks.tracking.config.k1.rl_cfg import booster_k1_tracking_ppo_runner_cfg


def booster_t1_tracking_ppo_runner_cfg():
    cfg = booster_k1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "t1_tracking"
    return cfg
