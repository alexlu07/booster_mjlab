"""T1 flat AMP dribbling, using the existing AMP runner and motion features."""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.config import RslRlSymmetryCfg
from booster_mjlab.tasks.dribble.env_cfg import booster_t1_dribble_env_cfg
from booster_mjlab.tasks.velocity.config.t1.rl_cfg import booster_t1_amp_ppo_runner_cfg
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

for use_muon in (False, True):
    optimizer = "-Muon" if use_muon else ""
    for use_da in (False, True):
        runner_cfg = booster_t1_amp_ppo_runner_cfg(use_muon=use_muon)
        runner_cfg.algorithm.symmetry_cfg = (
            RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=False,
                data_augmentation_func=(
                    "booster_mjlab.tasks.dribble.observations:augment_symmetries_t1_dribble"
                ),
            )
            if use_da
            else None
        )
        experiment = "t1_dribble_symmetric" if use_da else "t1_dribble"
        runner_cfg.experiment_name = runner_cfg.experiment_name.replace(
            "t1_velocity", experiment
        )
        augmentation = "-DA" if use_da else ""
        register_mjlab_task(
            task_id=f"Mjlab-Dribble-Flat-Amp{augmentation}{optimizer}-Booster-T1",
            env_cfg=booster_t1_dribble_env_cfg(runner_cfg),
            play_env_cfg=booster_t1_dribble_env_cfg(runner_cfg, play=True),
            rl_cfg=runner_cfg,
            runner_cls=VelocityAmpOnPolicyRunner,
        )
