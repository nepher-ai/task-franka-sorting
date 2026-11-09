# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool-pose and gripper command terms for the low-level tracker."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from franka_sorting.frames import grip_down_quat, project_to_reach, tcp_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TcpPoseCommand(CommandTerm):
    """Tool pose ``[N, 7]`` (position [m], xyzw), gripper down. Some resamples are short steps."""

    cfg: TcpPoseCommandCfg

    def __init__(self, cfg: TcpPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 2] = 0.2
        self.pose_command_b[:, 3:] = grip_down_quat(torch.zeros(1, device=self.device))[0]
        self._low = torch.tensor(cfg.ranges.low, device=self.device)
        self._high = torch.tensor(cfg.ranges.high, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pos, quat = tcp_pose_b(self.robot, self.body_idx)
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_b[:, :3] - pos, dim=-1)
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(quat, self.pose_command_b[:, 3:])

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        count = ids.numel()
        uniform = self._low + (self._high - self._low) * torch.rand(count, 3, device=self.device)
        step = (torch.rand(count, 3, device=self.device) * 2.0 - 1.0) * self.cfg.local_step_m
        local = torch.clamp(self.pose_command_b[ids, :3] + step, self._low, self._high)
        use_local = torch.rand(count, 1, device=self.device) < self.cfg.local_prob
        self.pose_command_b[ids, :3] = project_to_reach(torch.where(use_local, local, uniform))
        ranges = self.cfg.ranges
        yaw = (torch.rand(count, device=self.device) * 2.0 - 1.0) * ranges.yaw
        tilt = (torch.rand(count, 2, device=self.device) * 2.0 - 1.0) * ranges.tilt
        if self.cfg.wide_prob > 0.0:
            self._widen(ids, yaw, tilt)
        self.pose_command_b[ids, 3:] = math_utils.quat_unique(grip_down_quat(yaw, tilt[:, 0], tilt[:, 1]))

    def _widen(self, ids: torch.Tensor, yaw: torch.Tensor, tilt: torch.Tensor) -> None:
        """Give a share of the targets a wider orientation than gripper-down: level and closing vertically, leaning,
        or pitched about the closing axis. The sorting planner leans outward near the edge of the workspace.
        ``tilt`` is ``[N, 2]`` (about world X, then Y) and is changed in place, as are the positions of ``ids``.
        """
        count = ids.numel()
        dev = self.device
        wide = torch.rand(count, device=dev) < self.cfg.wide_prob
        if not bool(wide.any()):
            return
        low, high = torch.tensor(self.cfg.wide_low, device=dev), torch.tensor(self.cfg.wide_high, device=dev)
        box = low + (high - low) * torch.rand(count, 3, device=dev)
        self.pose_command_b[ids[wide], :3] = project_to_reach(box[wide])
        family = torch.rand(count, device=dev)
        sign = torch.where(torch.rand(count, device=dev) < 0.5, -1.0, 1.0)
        noise = (torch.rand(count, 2, device=dev) * 2.0 - 1.0) * 0.15
        level = wide & (family < 0.5)
        lean = wide & (family >= 0.5) & (family < 0.8)
        pitched = wide & (family >= 0.8)
        tilt[level, 0] = sign[level] * math.pi / 2 + noise[level, 0]
        tilt[level, 1] = noise[level, 1]
        tilt[lean, 0] = (torch.rand(int(lean.sum()), device=dev) * 2.0 - 1.0) * 1.2
        tilt[lean, 1] = (torch.rand(int(lean.sum()), device=dev) * 2.0 - 1.0) * 0.5
        tilt[pitched, 0] = noise[pitched, 0]
        tilt[pitched, 1] = sign[pitched] * (0.3 + 1.3 * torch.rand(int(pitched.sum()), device=dev))

    def _update_command(self):
        pass


@configclass
class TcpPoseCommandCfg(CommandTermCfg):
    @configclass
    class Ranges:
        """Sampling box in the robot root frame."""

        low: tuple[float, float, float] = (0.30, -0.45, 0.03)
        """Lower corner [m]."""
        high: tuple[float, float, float] = (0.80, 0.45, 0.45)
        """Upper corner [m], then pulled inside the tracked reach."""
        yaw: float = math.pi / 2
        """Largest yaw [rad] about world Z."""
        tilt: float = 0.15
        """Largest lean [rad] of the approach axis about world X and Y."""

    class_type: type = TcpPoseCommand
    asset_name: str = "robot"
    body_name: str = "base_link"
    ranges: Ranges = Ranges()
    local_prob: float = 0.5
    """Share of resamples that stay within ``local_step_m`` of the previous target."""
    local_step_m: float = 0.12
    """Largest local step [m] per axis."""
    wide_prob: float = 0.0
    """Share of resamples with a level, leaning, or pitched orientation instead of gripper-down."""
    wide_low: tuple[float, float, float] = (0.36, -0.32, 0.10)
    """Lower corner [m] of the box those targets are drawn from."""
    wide_high: tuple[float, float, float] = (0.70, 0.32, 0.30)


class GripperCommand(CommandTerm):
    """Gripper target ``[N, 1]``: 0 open, 1 closed."""

    cfg: GripperCommandCfg

    def __init__(self, cfg: GripperCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._target

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._target[ids, 0] = (torch.rand(ids.numel(), device=self.device) < self.cfg.close_prob).float()

    def _update_command(self):
        pass


@configclass
class GripperCommandCfg(CommandTermCfg):
    class_type: type = GripperCommand
    close_prob: float = 0.5
    """Probability a resample asks for a closed gripper."""
