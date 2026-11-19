"""Thin wrapper around mjlab's train script.

Adds:
- FastSAC runner config support (RslRlFastSacRunnerCfg union)
- Unpicklable object cleanup before config YAML serialization
- Upload completed training videos to the active W&B run
"""

import sys
from dataclasses import asdict, dataclass, field
from typing import Literal

import tyro

from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.train import launch_training
from mjlab.scripts.train import run_train as _mjlab_run_train
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from booster_mjlab.rl import RslRlFastSacRunnerCfg, RslRlOnPolicyRunnerCfg
from booster_mjlab.scripts.utils import remove_unpicklable_objects


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg | RslRlFastSacRunnerCfg
    registry_name: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    log_root: str = "logs/rsl_rl"
    """Root directory under which experiment logs are written."""
    ball_command_curriculum: bool = False
    """Use minimum-speed dribbling with command stages at 15k/20k/25k iterations."""
    fixed_soft_landing: bool = False
    """Remove the soft-landing weight curriculum and use its final penalty from step zero."""
    torchrunx_log_dir: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, (RslRlOnPolicyRunnerCfg, RslRlFastSacRunnerCfg))
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def _patch_dump_yaml() -> None:
    """Patch mjlab's train module to clean unpicklable objects before YAML serialization."""
    import mjlab.scripts.train as train_mod

    original_dump_yaml = train_mod.dump_yaml

    def safe_dump_yaml(filename, data, sort_keys=False):
        data = remove_unpicklable_objects(data, filename.stem)
        original_dump_yaml(filename, data, sort_keys)

    train_mod.dump_yaml = safe_dump_yaml


def _sync_amp_dataset_root(args: TrainConfig) -> None:
    """Pass the CLI AMP dataset override into the registered environment."""
    dataset_root = getattr(args.agent, "dataset_root", None)
    reset_event = (
        args.env.events.get("reset_robot_from_motion")
        if args.env.events is not None
        else None
    )
    if dataset_root and reset_event is not None:
        reset_event.params["dataset_root"] = dataset_root


def _patch_wandb_resume_config() -> None:
    """Keep original W&B config when continuing an existing W&B run."""
    import wandb
    from rsl_rl.utils import WandbLogWriter

    original_store_config = WandbLogWriter.store_config

    def store_config(self, env_cfg, train_cfg):
        if wandb.run is not None and wandb.run.resumed:
            saved_env = env_cfg if isinstance(env_cfg, dict) else asdict(env_cfg)
            wandb.config.update(
                {"resume_train_cfg": train_cfg, "resume_env_cfg": saved_env},
                allow_val_change=True,
            )
        else:
            original_store_config(self, env_cfg, train_cfg)

    WandbLogWriter.store_config = store_config


def _run_train(task_id, args, log_dir):
    """Install integrations inside each worker, including torchrunx workers."""
    import mjlab.scripts.train as train_mod

    _patch_dump_yaml()
    if args.agent.resume:
        _patch_wandb_resume_config()
    if args.video:
        from booster_mjlab.scripts.video_recorder import WandbVideoRecorder

        train_mod.VideoRecorder = WandbVideoRecorder
    _mjlab_run_train(task_id, args, log_dir)


def main():
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=TYRO_FLAGS,
    )
    del remaining_args

    _sync_amp_dataset_root(args)
    if args.ball_command_curriculum:
        from booster_mjlab.tasks.dribble.env_cfg import configure_minimum_speed_curriculum

        configure_minimum_speed_curriculum(args.env, args.agent.num_steps_per_env)
    if args.fixed_soft_landing:
        from booster_mjlab.tasks.dribble.env_cfg import configure_fixed_soft_landing

        configure_fixed_soft_landing(args.env)
    import mjlab.scripts.train as train_mod

    train_mod.run_train = _run_train
    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
