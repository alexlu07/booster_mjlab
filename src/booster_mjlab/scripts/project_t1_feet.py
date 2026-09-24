"""Project serial T1 motion clips away from sole-to-sole collisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from booster_mjlab.motion import MotionFile
from booster_mjlab.scripts.visualize_motions import (
    FootGeometry,
    STANCE_MAX_CLEARANCE,
    STANCE_SPEED_THRESHOLD,
    create_play_env,
    save_motion_file,
)

T1_LEG_COLUMNS = np.arange(11, 23)
DISTANCE_MARGIN = 0.005
FINITE_DIFFERENCE = 1e-4
MAX_ITERATIONS = 12
MAX_JOINT_STEP = 0.04

# Prefer the lateral joints that naturally widen the stance over knee and
# ankle changes that noticeably distort the gait.
JOINT_PREFERENCE = np.array(
    [0.15, 1.0, 0.8, 0.15, 0.15, 0.25] * 2, dtype=np.float64
)


def _set_qpos(
    data: mujoco.MjData, motion: MotionFile, frame: int, dof_pos: np.ndarray
) -> None:
    data.qpos[:3] = motion.root_pos[frame]
    quat = motion.root_rot[frame]
    data.qpos[3:7] = (quat[3], quat[0], quat[1], quat[2])
    data.qpos[7:] = dof_pos


def _stance_mask(
    motion: MotionFile, model: mujoco.MjModel, geometry: FootGeometry
) -> np.ndarray:
    """Classify each foot as planted from its clearance and planar speed."""
    data = mujoco.MjData(model)
    positions = np.empty((motion.num_frames, 2, 3))
    clearances = np.empty((motion.num_frames, 2))
    for frame, q in enumerate(motion.dof_pos):
        _set_qpos(data, motion, frame, q)
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[geometry.body_ids]
        clearances[frame] = geometry.clearances(data)
    speeds = np.zeros_like(clearances)
    if motion.num_frames > 1:
        speeds[1:] = (
            np.linalg.norm(np.diff(positions[:, :, :2], axis=0), axis=2) / motion.dt
        )
        speeds[0] = speeds[1]
    return (speeds <= STANCE_SPEED_THRESHOLD) & (clearances <= STANCE_MAX_CLEARANCE)


def project_motion(motion: MotionFile, model: mujoco.MjModel) -> MotionFile:
    """Return a copy with overlapping T1 soles separated in joint space."""
    if motion.dof_pos.shape[1] != 23 or model.nq != 30:
        raise ValueError("This projector supports the serial 23-DoF T1 only.")

    geometry = FootGeometry.create(model)
    if geometry is None or any(len(ids) != 1 for ids in geometry.geom_ids):
        raise ValueError("Expected one collision geom on each serial T1 foot.")
    left_geom, right_geom = (ids[0] for ids in geometry.geom_ids)
    data = mujoco.MjData(model)
    fromto = np.empty(6)
    output = motion.dof_pos.astype(np.float64, copy=True)
    stance = _stance_mask(motion, model, geometry)
    changed = 0
    unresolved_double_stance = 0

    def distance(frame: int, q: np.ndarray) -> float:
        _set_qpos(data, motion, frame, q)
        mujoco.mj_forward(model, data)
        return float(
            mujoco.mj_geomDistance(model, data, left_geom, right_geom, 1e6, fromto)
        )

    for frame, q in enumerate(output):
        if stance[frame].all():
            # Both soles are planted: moving either leg makes it slide. Keep
            # the physically meaningful pose and report it for clip curation.
            if distance(frame, q) < DISTANCE_MARGIN:
                unresolved_double_stance += 1
            continue
        allowed = (
            T1_LEG_COLUMNS[6:]
            if stance[frame, 0]
            else T1_LEG_COLUMNS[:6]
            if stance[frame, 1]
            else T1_LEG_COLUMNS
        )
        preference = JOINT_PREFERENCE[np.isin(T1_LEG_COLUMNS, allowed)]
        for _ in range(MAX_ITERATIONS):
            current = distance(frame, q)
            if current >= DISTANCE_MARGIN:
                break
            gradient = np.empty(len(allowed))
            for index, column in enumerate(allowed):
                q[column] += FINITE_DIFFERENCE
                plus = distance(frame, q)
                q[column] -= 2 * FINITE_DIFFERENCE
                minus = distance(frame, q)
                q[column] += FINITE_DIFFERENCE
                gradient[index] = (plus - minus) / (2 * FINITE_DIFFERENCE)
            weighted = gradient * preference
            denominator = float(np.dot(gradient, weighted))
            if denominator < 1e-10:
                break
            step = (DISTANCE_MARGIN - current) * weighted / denominator
            q[allowed] += np.clip(step, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            changed += 1

    print(
        f"[project] adjusted {changed} collision iterations; "
        f"left {unresolved_double_stance} double-stance frames unchanged"
    )
    return MotionFile(
        fps=motion.fps,
        root_pos=motion.root_pos,
        root_rot=motion.root_rot,
        dof_pos=output.astype(motion.dof_pos.dtype),
        local_body_pos=motion.local_body_pos,
        link_body_list=motion.link_body_list,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    context = create_play_env("Mjlab-Velocity-Flat-Booster-T1", "cpu")
    paths = sorted(args.input_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl clips in {args.input_dir}")
    for path in paths:
        save_motion_file(project_motion(MotionFile.load(path), context.sim.mj_model), args.output_dir / path.name)


if __name__ == "__main__":
    main()
