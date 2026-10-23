<img src="./assets/banner.webp" alt="Project banner">

# booster_mjlab

booster_mjlab is an [mjlab](https://github.com/mujocolab/mjlab) integration for the Booster K1 and T1.
It provides velocity and motion-tracking tasks, and an [Adversarial Motion Priors](https://arxiv.org/abs/2104.02180) (AMP) training pipeline that learns natural-looking gaits from motion capture data.

Real-robot and simulation demos are on the [project page](https://intelligentroboticslab.github.io/booster_mjlab/),
which also runs the velocity policy live in the browser (MuJoCo WebAssembly + the exported network).

## Getting Started

booster_mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Install from source:**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/IntelligentRoboticsLab/booster-mjlab.git && cd booster-mjlab
uv run list_envs
```

The last command syncs the environment and prints every registered task.

## Training Examples

### 1. Velocity Tracking

Train a Booster K1 to follow velocity commands on flat terrain. The reward has been tuned to work together with AMP reference motions.
The default [motion dataset](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1) contains a few locomotion clips retargeted from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --env.scene.num-envs 4096
```

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

To train on your own motions, pass a Hugging Face dataset (`namespace/repo`) or a local motion file or directory
with `--agent.dataset-root`.

The `-DA-` tasks train with left/right symmetry data augmentation. Drop `DA` from the task ID to train without it.

The `-Muon-` tasks use the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) (Jordan et al., 2024) for the
actor and critic weight matrices. Drop `Muon` from the task ID for the Adam variant with otherwise identical settings.

### T1 velocity tracking

The serial-ankle T1 uses a full 23-joint policy. Its AMP discriminator observes the 12 leg joints plus base state, matching the K1 design: arms, head, and waist remain policy-controlled but are not AMP-scored. Provide T1 motion data explicitly; there is no bundled T1 AMP dataset.

```bash
uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-T1 \
    --agent.dataset-root /path/to/gmr_t1_motions \
    --env.scene.num-envs 4096
```

`Flat`, `Rough`, Adam/Muon, AMP/non-AMP, and symmetry-augmentation variants use the same task naming as K1 with `Booster-T1` in place of `Booster-K1`.

### T1 dribbling

Track a commanded planar ball velocity on flat ground with T1 AMP. The task uses
the existing size-4 ball, spawns it 0.4–0.6 m ahead of the robot, and resets when
the ball is more than 3 m away. Each command samples a speed and a heading change
relative to the previous command direction, then stays fixed in world coordinates
for 3–6 seconds. Episode resets initialize the direction from the robot's yaw;
10% of commands are exactly zero and retain the previous direction.

| PPO iteration (24 steps/iteration) | Speed (m/s) | Heading change |
|---:|---:|---:|
| 0 | 0–0.2 | ±15° |
| 5,000 | 0–0.5 | ±45° |
| 10,000 | 0–0.5 | ±90° |
| 15,000 | 0–1.0 | ±90° |
| 20,000 | 0–1.0 | ±180° |

The schedule uses absolute environment steps and continues on resume. Play mode
uses the final ranges. Ball velocity tracking tolerance is half the commanded
speed, clamped between 0.05 and 0.2 m/s, including a 0.05 m/s tolerance for stops.

In the Viser play viewer, open **Commands → Ball Velocity** and enable the
joystick to set ball speed and world heading for the selected environment.
Heading is absolute (0° = world +X, 90° = world +Y). **Stop ball command** sets
the target speed to zero; disabling the joystick lets automatic sampling resume.

```bash
uv run train Mjlab-Dribble-Flat-Amp-Booster-T1 \
    --agent.dataset-root /path/to/gmr_t1_motions \
    --env.scene.num-envs 4096
```

Use `Mjlab-Dribble-Flat-Amp-DA-Muon-Booster-T1` for left/right data augmentation
with Muon, or omit `DA` for Muon alone. Dribble augmentation mirrors the robot,
ball state, and lateral ball command. To use the combined local walking/dribble
dataset prepared in `data/t1_amp_mix`:

```bash
uv run train Mjlab-Dribble-Flat-Amp-DA-Muon-Booster-T1 \
    --agent.dataset-root ./data/t1_amp_mix \
    --env.scene.num-envs 4096
```

AMP motion features and action regularization match the T1 velocity task.
Posture and upright tolerances
are fixed at walking values; foot slip and landing penalties are always active.
The landing penalty retains the velocity task's weight curriculum: −0.0001
initially, −0.001 at 1,000 PPO iterations, and −0.005 at 7,000 (24 steps/iteration).
Airtime, clearance, swing-height, and standing-pose rewards are omitted.
Ball contact with non-foot robot bodies (including shins) incurs a −1.0 reward
rate while contact persists, regardless of contact count or force. Contacts with
the two foot bodies are allowed. The penalty does not terminate the episode.

### Training videos

Add `--video --video-length 500 --video-interval 12000` to a training command
to record 500-frame clips every 12,000 environment steps. For T1 dribble this
is 10 seconds of simulation every 500 PPO iterations with the default rollout
length. Only the first environment on rank 0 is recorded.

Completed clips are saved under the run's `videos/train/` directory and logged
to **Videos/train** in the active W&B run. After syncing, open that video panel
on the run page from your computer or phone. Without an active W&B run, clips
are saved locally. Video encoding requires ffmpeg; headless rendering requires
an appropriate graphics backend, typically `MUJOCO_GL=egl` on NVIDIA machines.

### 2. Motion Imitation

Train a Booster K1 to track motion capture clips on flat terrain. Motions are managed as WandB artifacts;
see [Preparing Motions for Tracking](docs/motion_preparation.md) for converting and uploading GMR (`.pkl`)
or BeyondMimic (`.csv`) clips.

Once a motion is uploaded, train and evaluate with:

```bash
uv run train Mjlab-Tracking-Flat-Booster-K1 --registry-name your-org/motion_upload/motion-tracking:v0 --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Booster-K1 --wandb-run-path your-org/mjlab/run-id
```

For a motion prepared with `csv_to_npz`, pass `--registry-name your-org/motions/motion_name` instead.

For a serial T1 GMR clip, upload and train with:

```bash
uv run upload-motion --robot t1 --input-file /path/to/t1_motion.pkl --entity your-org
uv run train Mjlab-Tracking-Flat-Booster-T1 --registry-name your-org/motion_upload/t1_motion-tracking:v0 --env.scene.num-envs 4096
```

### 3. Browse Motion Data

Inspect, play back and edit motion clips in the browser before training on them:

```bash
uv run visualize-motions --help
```

Clips from the [lafan_locomotion_k1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
dataset, retargeted onto the parallel-ankle K1:

<picture>
    <source
    srcset="./assets/motion_grid_dark.webp"
    media="(prefers-color-scheme: dark)">
    <source
    srcset="./assets/motion_grid_light.webp"
    media="(prefers-color-scheme: light)">
    <img
    src="./assets/motion_grid_light.webp"
    alt="Retargeted lafan_locomotion_k1 clips playing on the K1"
    width="720">
</picture>

## Tasks

Velocity tasks come in `Flat` and `Rough` terrain variants, optionally with `-Amp-` (motion prior) and
`-DA-` (symmetry data augmentation). Tasks suffixed `-Parallel` use the parallel-linkage ankle model of the K1; the first T1 pass is serial-ankle only.
Tracking tasks run on flat terrain. List them all with:

```bash
uv run list_envs
```

## Deployment

Example code for deploying trained policies on the real Booster K1 is coming soon.

## Hungry for more?

booster_mjlab is a whIRLwind project. Our mission is to push the boundaries of robotics and AI in robot football. We are always looking for interested students or collaboration partners to join us. Learn more at [whirlwind.team](https://whirlwind.team).

<picture>
    <source
    srcset="./assets/whirlwind_logo_light.png"
    media="(prefers-color-scheme: dark)"
    height="125"
    />
    <img
    src="./assets/whirlwind_logo_dark.png"
    alt="whIRLwind logo"
    height="125"
    />
</picture>
