"""Restore dribble experiment settings that differ between runs of one task."""

from copy import deepcopy
from pathlib import Path


_REWARD_PARAMS = {
    "track_ball_velocity": ("allow_overspeed", "std"),
    "ball_proximity": ("distance", "std"),
    "stand_still_near_ball": (
        "distance", "linear_std", "yaw_std", "command_threshold"
    ),
}
_COMMAND_PARAMS = (
    "zero_speed_probability", "speed_range", "heading_change_range",
    "resampling_time_range",
)


def load_saved_environment(cfg):
    """Read run config without importing any serialized Python objects."""
    if cfg.checkpoint_file is not None:
        path = Path(cfg.checkpoint_file).expanduser().resolve().parent / "params/env.yaml"
        if path.is_file():
            import yaml

            class ConfigLoader(yaml.SafeLoader):
                pass

            def metadata(loader, suffix, node):
                # mjlab YAML contains Python name/tuple/dataclass tags. Treat
                # them as ordinary data; never instantiate or import them.
                if isinstance(node, yaml.ScalarNode):
                    return loader.construct_scalar(node)
                if isinstance(node, yaml.SequenceNode):
                    return loader.construct_sequence(node, deep=True)
                return loader.construct_mapping(node, deep=True)

            ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", metadata)
            return yaml.load(path.read_text(), Loader=ConfigLoader)
    if cfg.wandb_run_path is not None:
        import wandb

        config = wandb.Api().run(str(cfg.wandb_run_path)).config
        return config.get("env_cfg")
    return None


def restore_dribble_settings(env_cfg, saved):
    """Restore reward weights, scalar dribble parameters, and command sampling.

    Keep play's sensor, terrain, termination, and entity settings. In particular,
    serialized scene-entity dictionaries must not replace resolved config objects.
    """
    if "curriculum" in saved and "soft_landing_weight" not in (saved["curriculum"] or {}):
        # A fixed-penalty experiment must not regain the default weight schedule
        # during playback resets and overwrite its restored reward weight.
        if getattr(env_cfg, "curriculum", None):
            env_cfg.curriculum.pop("soft_landing_weight", None)
    for name, term in env_cfg.rewards.items():
        source = saved.get("rewards", {}).get(name)
        if term is None or not isinstance(source, dict):
            continue
        if "weight" in source:
            term.weight = float(source["weight"])
        for param in _REWARD_PARAMS.get(name, ()):
            if param in source.get("params", {}):
                term.params[param] = deepcopy(source["params"][param])
        if name == "track_ball_velocity" and "allow_overspeed" not in source.get("params", {}):
            # Original checkpoints predate the experiment flag and used exact
            # velocity tracking, even if future task defaults change.
            term.params["allow_overspeed"] = False
    command = env_cfg.commands["ball_velocity"]
    source = saved.get("commands", {}).get("ball_velocity", {})
    for param in _COMMAND_PARAMS:
        if param in source:
            value = deepcopy(source[param])
            if isinstance(getattr(command, param), tuple):
                value = tuple(value)
            setattr(command, param, value)


def install_experiment_config(mjlab_play):
    """Wrap run_play once so parsed checkpoint selection is available to config loading."""
    original_run_play = mjlab_play.run_play

    def run_play(task_id, cfg):
        original_load_env_cfg = mjlab_play.load_env_cfg

        def load_env_cfg(*args, **kwargs):
            env_cfg = original_load_env_cfg(*args, **kwargs)
            if cfg.agent in {"zero", "random"} or "ball_velocity" not in (env_cfg.commands or {}):
                return env_cfg
            saved = load_saved_environment(cfg)
            if saved is None:
                print("[WARN] No saved experiment config found; dribble rewards use task defaults.")
                return env_cfg
            restore_dribble_settings(env_cfg, saved)
            minimum = env_cfg.rewards["track_ball_velocity"].params.get("allow_overspeed", False)
            mode = "minimum-speed (overspeed allowed)" if minimum else "exact-speed"
            stop_probability = env_cfg.commands["ball_velocity"].zero_speed_probability
            print(f"[INFO] Restored dribble experiment: {mode}, stop probability={stop_probability:g}")
            return env_cfg

        mjlab_play.load_env_cfg = load_env_cfg
        try:
            return original_run_play(task_id, cfg)
        finally:
            mjlab_play.load_env_cfg = original_load_env_cfg

    mjlab_play.run_play = run_play
