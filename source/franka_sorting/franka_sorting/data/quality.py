# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks an episode must pass before it is saved to a dataset. Pure numpy, no simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PANDA_JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
"""Franka Panda joint lower limits [rad]."""
PANDA_JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
"""Franka Panda joint upper limits [rad]."""


@dataclass(frozen=True)
class QualityLimits:
    """Bounds of the episode gates."""

    min_steps: int = 30
    """Shortest episode [control steps]."""
    max_steps: int = 3000
    """Longest episode [control steps]."""
    joint_margin_rad: float = 0.05
    """Slack [rad] beyond the Panda joint limits that state and target may reach."""
    max_joint_step_rad: float = 0.30
    """Largest change [rad] of one joint between two control steps, in state and in target."""
    min_frame_std: float = 3.0
    """Smallest standard deviation of the pixel values (0-255) of a camera video; below it the frames are blank."""


def check_episode(
    state: np.ndarray,
    action: np.ndarray,
    frames: dict[str, np.ndarray] | None = None,
    limits: QualityLimits | None = None,
) -> list[str]:
    """Reasons an episode must not be saved. An empty list means it passes.

    Args:
        state: ``[T, 8]`` joint positions [rad] and gripper closure (0..1).
        action: ``[T, 8]`` joint targets [rad] and gripper command (0 or 1).
        frames: ``{camera: uint8 [T, H, W, 3]}``.
        limits: Gate bounds.

    Returns:
        Short reason codes: ``too_short``, ``too_long``, ``nan``, ``joint_limit``, ``joint_jump``, ``gripper_range``
        and ``blank_frames:<camera>``.
    """
    limits = limits or QualityLimits()
    state, action = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
    reasons: list[str] = []
    steps = len(action)
    if steps < limits.min_steps:
        reasons.append("too_short")
    if steps > limits.max_steps:
        reasons.append("too_long")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        reasons.append("nan")
        return reasons
    lower = np.asarray(PANDA_JOINT_LOWER) - limits.joint_margin_rad
    upper = np.asarray(PANDA_JOINT_UPPER) + limits.joint_margin_rad
    for values in (state[:, :7], action[:, :7]):
        if (values < lower).any() or (values > upper).any():
            reasons.append("joint_limit")
            break
    for values in (state[:, :7], action[:, :7]):
        if len(values) > 1 and np.abs(np.diff(values, axis=0)).max() > limits.max_joint_step_rad:
            reasons.append("joint_jump")
            break
    if (state[:, 7] < -0.05).any() or (state[:, 7] > 1.05).any() or not np.isin(action[:, 7], (0.0, 1.0)).all():
        reasons.append("gripper_range")
    for name, images in (frames or {}).items():
        sample = np.asarray(images)[:: max(1, len(images) // 16)]
        if sample.size == 0 or float(sample.astype(np.float32).std()) < limits.min_frame_std:
            reasons.append(f"blank_frames:{name}")
    return reasons
