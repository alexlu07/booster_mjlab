"""T1 flat AMP dribbling, using the existing AMP runner and motion features."""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.tasks.dribble.env_cfg import booster_t1_dribble_env_cfg
from booster_mjlab.tasks.velocity.config.t1.rl_cfg import booster_t1_amp_ppo_runner_cfg
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

for use_muon in (False, True):
    runner_cfg = booster_t1_amp_ppo_runner_cfg(use_muon=use_muon)
    # The velocity-task mirror also runs for logging when augmentation is off;
    # its observation layout does not include the ball.
    runner_cfg.algorithm.symmetry_cfg = None
    runner_cfg.experiment_name = runner_cfg.experiment_name.replace(
        "t1_velocity", "t1_dribble"
    )
    optimizer = "-Muon" if use_muon else ""
    register_mjlab_task(
        task_id=f"Mjlab-Dribble-Flat-Amp{optimizer}-Booster-T1",
        env_cfg=booster_t1_dribble_env_cfg(runner_cfg),
        play_env_cfg=booster_t1_dribble_env_cfg(runner_cfg, play=True),
        rl_cfg=runner_cfg,
        runner_cls=VelocityAmpOnPolicyRunner,
    )
