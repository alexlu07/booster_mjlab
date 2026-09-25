"""Thin wrapper around mjlab's play script.

Swaps the viser viewer for :class:`RecordingViserPlayViewer`, which
adds a record / stop-recording button that captures the browser view to an mp4
under ``--record-dir``. Frames are timed by the sim clock, so the video plays
back in real time even when the machine can't step the policy at full speed.
"""

import functools
import sys

import mjlab.scripts.play as mjlab_play

from booster_mjlab.viewer import DEFAULT_VIDEO_DIR, RecordingViserPlayViewer


def _pop_flag(argv: list[str], flag: str) -> str | None:
    """Remove ``--flag VALUE`` or ``--flag=VALUE`` from argv; return VALUE."""
    for i, tok in enumerate(argv):
        if tok == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value")
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if tok.startswith(flag + "="):
            value = tok.split("=", 1)[1]
            del argv[i]
            return value
    return None


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        # mjlab's help won't list --record-dir (we pop it before mjlab parses),
        # so surface it here.
        print(
            "booster_mjlab play adds:\n"
            "  --record-dir STR  (viser viewer: where recorded mp4s are written, "
            f"default {DEFAULT_VIDEO_DIR})\n"
            "  --dataset-root STR  (AMP reference-motion dataset for environment resets)\n",
            file=sys.stderr,
        )
    record_dir = _pop_flag(argv, "--record-dir")
    dataset_root = _pop_flag(argv, "--dataset-root")
    sys.argv = [sys.argv[0], *argv]

    if dataset_root is not None:
        original_load_env_cfg = mjlab_play.load_env_cfg

        def load_env_cfg_with_dataset_root(*args, **kwargs):
            env_cfg = original_load_env_cfg(*args, **kwargs)
            reset_event = (
                env_cfg.events.get("reset_robot_from_motion")
                if env_cfg.events is not None
                else None
            )
            if reset_event is not None:
                reset_event.params["dataset_root"] = dataset_root
            return env_cfg

        mjlab_play.load_env_cfg = load_env_cfg_with_dataset_root

    # Viser viewer with a record button.
    mjlab_play.ViserPlayViewer = functools.partial(  # type: ignore[assignment]
        RecordingViserPlayViewer,
        video_dir=record_dir or DEFAULT_VIDEO_DIR,
    )

    mjlab_play.main()


if __name__ == "__main__":
    main()
