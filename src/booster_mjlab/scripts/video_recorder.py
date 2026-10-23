"""Training video recording with W&B media logging."""

import warnings

import wandb

from mjlab.utils.wrappers import VideoRecorder


class WandbVideoRecorder(VideoRecorder):
    """Save clips locally and queue completed MP4s in the active W&B run."""

    def _finish_recording(self) -> None:
        path = self.current_video_path
        has_frames = bool(self.current_video_frames)
        super()._finish_recording()
        if not has_frames or path is None or wandb.run is None:
            return
        try:
            # Leave the W&B step to the training logger. Committing here would
            # advance its step in the middle of a PPO iteration.
            wandb.run.log(
                {"Videos/train": wandb.Video(str(path), format="mp4", caption=path.stem)},
                commit=False,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not log video to W&B; local recording is at {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
