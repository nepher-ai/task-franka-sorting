# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env task state. Env ``i`` runs task ``i % T`` and resets on its own.

Progress uses the latched benchmark predicates. The success expression has to hold on the current
state, and then it has to keep holding for :data:`SUCCESS_HOLD_S`. Every episode can run the published layout
or a fresh random one on the same bodies, and an episode whose parts moved before the arm touched them is
flagged as an invalid start.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from franka_sorting.benchmark.episode import resample_layout
from franka_sorting.benchmark.predicates import task_progress
from franka_sorting.benchmark.stage_plan import StagePlan, entity_pose, surface_body_pose
from franka_sorting.benchmark.success import note_placements, task_satisfied
from franka_sorting.benchmark.task import TaskLayout, world_position
from franka_sorting.layout.catalog import AssetLibrary
from franka_sorting.layout.sampler import LayoutError
from franka_sorting.scene.contact import slots_in_contact
from franka_sorting.scene.robot import GRIPPER_CLOSED_RAD

# A finished pose has to hold this long before the episode ends.
SUCCESS_HOLD_S = 0.5
# An object this far below the table or this far from the env origin is gone.
LOST_Z_M = -0.15
LOST_RADIUS_M = 1.5
# Settled parts must stay put for this long [s] after an episode starts, or the start is invalid.
START_CHECK_S = 0.4
# A part that moved this far [m] while settling was pushed by an overlap or fell off its support.
START_DRIFT_M = 0.005


def _commit_hold(
    stored_latched: dict[str, bool],
    stored_entered: list[str],
    latched: dict[str, bool],
    entered: list[str],
    score: float,
    hold_s: float,
    dt: float,
) -> tuple[dict[str, bool], list[str], float, bool]:
    """Commit a finished pose to the latch only after :data:`SUCCESS_HOLD_S`.

    A score of 1 leaves the stored latch unchanged and keeps counting. A lower score stores partial
    progress and clears the hold. Returns the latch, entry order, hold time [s], and whether the
    latch was committed. The episode outcome comes from ``task_satisfied`` instead.
    """
    if score >= 1.0 - 1e-9:
        hold_s += dt
        if hold_s >= SUCCESS_HOLD_S:
            return latched, entered, hold_s, True
        return stored_latched, stored_entered, hold_s, False
    return latched, entered, 0.0, False


def _host(values) -> np.ndarray:
    values = values.torch if hasattr(values, "torch") else values
    return values.detach().cpu().numpy()


class TaskTracker:
    """Scene layout, benchmark scoring, and latched outcomes.

    Args:
        env: Manager-based env that owns the bodies of ``plan``.
        layouts: Published layout of every task of the plan.
        plan: Stage plan.
        library: Asset library of the benchmark. It gives the tray cell geometry and the random layouts.
        randomize: Draw a new random layout on the same bodies at every episode start.
        seed: Base seed of the random layouts.
    """

    def __init__(
        self,
        env,
        layouts: Sequence[TaskLayout],
        plan: StagePlan,
        library: AssetLibrary,
        randomize: bool = False,
        seed: int = 1,
    ):
        self.env = env
        self.plan = plan
        self.task_ids = tuple(plan.task_ids)
        self.task_layouts = tuple(layouts)
        self.num_envs = env.num_envs
        if plan.num_envs != self.num_envs:
            raise ValueError(f"plan covers {plan.num_envs} envs, the env has {self.num_envs}")
        if len(self.task_layouts) != len(self.task_ids):
            raise ValueError("layouts and plan.task_ids must name the same tasks")
        self.specs = tuple(layout.spec for layout in self.task_layouts)
        self.phase = 0
        self.task_index = np.array(plan.assignment[0], dtype=int)
        self.entities = plan.entities
        self.library = library
        self.randomize = bool(randomize)
        self.device = env.device
        self._seed = int(seed)
        self._env_episodes = np.zeros(env.num_envs, dtype=int)
        self.episode_keys: list[list[int]] = [[self._seed, index, 0] for index in range(env.num_envs)]
        """``[seed, env, episode]`` that drew each env's current layout."""
        self.last_episode_keys: list[list[int]] = [list(key) for key in self.episode_keys]
        self._active = False
        self.env_layouts: list[TaskLayout] = [self.task_layouts[int(task)] for task in self.task_index]
        self.last_layouts: list[TaskLayout] = list(self.env_layouts)
        self.required: list[tuple[str, ...]] = [self._required_slots(layout) for layout in self.env_layouts]
        self.held = {entity: np.zeros(self.num_envs, dtype=bool) for entity in self.entities}
        self.latched: list[dict[str, bool]] = [{} for _ in range(self.num_envs)]
        self.entered: list[list[str]] = [[] for _ in range(self.num_envs)]
        self.hold_s = np.zeros(self.num_envs, dtype=np.float64)
        self.success_hold_s = np.zeros(self.num_envs, dtype=np.float64)
        self.first_placed: list[dict[str, int]] = [{} for _ in range(self.num_envs)]
        self.progress = np.zeros(self.num_envs, dtype=np.float64)
        self.lost = np.zeros(self.num_envs, dtype=bool)
        self.invalid_start = np.zeros(self.num_envs, dtype=bool)
        self.last_invalid_start = np.zeros(self.num_envs, dtype=bool)
        self._start_checked = np.ones(self.num_envs, dtype=bool)
        self.success_now = np.zeros(self.num_envs, dtype=bool)
        self.last_success = np.zeros(self.num_envs, dtype=bool)
        self.last_lost = np.zeros(self.num_envs, dtype=bool)
        self.last_progress = np.zeros(self.num_envs, dtype=np.float64)
        self.last_task_ids = [self.task_id_of(index) for index in range(self.num_envs)]
        self.finished_steps = np.full(self.num_envs, -1, dtype=int)
        self._steps = np.zeros(self.num_envs, dtype=int)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.roots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.contact: dict[str, np.ndarray] = {}
        self.gripper_open = np.ones(self.num_envs, dtype=bool)
        self._origins = _host(env.scene.env_origins).reshape(-1, 3)
        self._updated_at = -1
        self._finger = int(env.scene["robot"].find_joints("finger_joint")[0][0])

    @staticmethod
    def _required_slots(layout: TaskLayout) -> tuple[str, ...]:
        names: list[str] = []

        def walk(check) -> None:
            for goal in check.goals:
                if goal.slot not in names:
                    names.append(goal.slot)
            for child in check.children:
                walk(child)

        spec = layout.spec
        if spec is not None and spec.check is not None:
            walk(spec.check)
        placed = {item.slot for item in layout.placements if item.asset.kind != "anchor"}
        return tuple(slot for slot in names if slot in placed)

    def task_id_of(self, index: int) -> str:
        return self.task_ids[int(self.task_index[index])]

    def instruction(self, index: int) -> str:
        return self.env_layouts[index].instruction

    def entity(self, index: int, slot: str) -> str:
        return self.plan.phase_slots[self.phase][index][slot]

    def sample_env(self, task: int) -> tuple[int, int]:
        """``(phase, env)`` whose body carries task ``task``, for mesh sampling."""
        for env, assigned in enumerate(self.plan.assignment[0]):
            if assigned == task:
                return 0, env
        raise RuntimeError(f"no env runs task {self.task_ids[task]}")

    def root(self, index: int, slot: str) -> tuple[np.ndarray, np.ndarray]:
        """Root position [m] relative to the env origin and xyzw quaternion of a slot."""
        entity = self.entity(index, slot)
        return self.roots[entity][0][index], self.roots[entity][1][index]

    def touching(self, index: int, slot: str) -> bool:
        return bool(self.contact[self.entity(index, slot)][index])

    def note_step(self, lengths) -> None:
        """Store episode length [steps]. Finish counts are read from the env."""
        self._steps = np.asarray(lengths, dtype=int).reshape(-1)

    def prepare_reset(self) -> None:
        """Clear finish marks before an explicit evaluator reset."""
        self.finished_steps[:] = -1
        self._active = False

    def begin(self, env_ids: torch.Tensor | slice | None, seed: int | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = list(range(self.num_envs))
        else:
            ids = [int(i) for i in torch.as_tensor(env_ids).reshape(-1).tolist()]
        if self._active:
            for index in ids:
                length = self.env.episode_length_buf[index]
                self.finished_steps[index] = int(length.item() if hasattr(length, "item") else length)
                self.last_task_ids[index] = self.task_id_of(index)
                self.last_success[index] = self.success_now[index]
                self.last_progress[index] = self.progress[index]
                self.last_invalid_start[index] = self.invalid_start[index]
                self.last_lost[index] = self.lost[index]
                self.last_episode_keys[index] = list(self.episode_keys[index])
                self.last_layouts[index] = self.env_layouts[index]
        self._active = True
        for index in ids:
            # The draw depends only on the base seed, the env, and that env's episode count, so a rerun with the
            # same seed reproduces every layout whatever order the envs finish in.
            key = [int(seed if seed is not None else self._seed), index, int(self._env_episodes[index])]
            self._env_episodes[index] += 1
            self.episode_keys[index] = key
            published = self.task_layouts[int(self.task_index[index])]
            self.env_layouts[index] = self._episode_layout(published, key)
            self.required[index] = self._required_slots(self.env_layouts[index])
            self.invalid_start[index] = False
            self._start_checked[index] = False
            self.latched[index] = {}
            self.entered[index] = []
            for held in self.held.values():
                held[index] = False
            self.hold_s[index] = 0.0
            self.success_hold_s[index] = 0.0
            self.first_placed[index] = {}
            self.progress[index] = 0.0
            self.lost[index] = False
            self.success_now[index] = False
        self._write_layout(ids)
        self._updated_at = -1

    def _episode_layout(self, published: TaskLayout, key: list[int]) -> TaskLayout:
        """The published layout, or a random one on its bodies. An unsolvable draw keeps the published layout."""
        if not self.randomize:
            return published
        try:
            return resample_layout(published, self.library, np.random.default_rng(np.random.SeedSequence(key)))
        except LayoutError:
            return published

    def episode_finished(self):
        """``None``. Each env resets on its own."""
        return

    def episode_steps(self) -> np.ndarray:
        """Steps each env had run when it finished. ``-1`` means it has not finished."""
        return self.finished_steps.copy()

    def _write_layout(self, ids: list[int]) -> None:
        env_ids = torch.tensor(ids, device=self.device)
        for entity_index, entity in enumerate(self.entities):
            poses = [
                self._world_pose(index, *entity_pose(self.plan, self.env_layouts, index, entity_index, self.phase))
                for index in ids
            ]
            self._write(self.env.scene[entity], poses, env_ids)
        for surface_index, name in enumerate(self.plan.surface_entities):
            poses = [
                self._world_pose(index, *surface_body_pose(self.plan, index, surface_index, self.phase))
                for index in ids
            ]
            self._write(self.env.scene[name], poses, env_ids)

    def _world_pose(self, index: int, position, quat_wxyz) -> list[float]:
        origin = tuple(float(value) for value in self._origins[index])
        w, x, y, z = quat_wxyz
        return [*world_position(position, origin), x, y, z, w]

    def _write(self, entity, poses: list[list[float]], env_ids: torch.Tensor) -> None:
        entity.write_root_pose_to_sim(torch.tensor(poses, device=self.device, dtype=torch.float32), env_ids=env_ids)
        entity.write_root_velocity_to_sim(
            torch.zeros((len(poses), 6), device=self.device, dtype=torch.float32), env_ids=env_ids
        )

    def read_roots(self) -> None:
        """Copy root poses to the host. Positions are relative to the env origin [m]."""
        self.roots = {}
        for entity in self.entities:
            data = self.env.scene[entity].data
            pos = _host(data.root_pos_w).reshape(-1, 3) - self._origins
            self.roots[entity] = (pos, _host(data.root_quat_w).reshape(-1, 4))

    def update(self) -> None:
        step = int(self.env.common_step_counter)
        if step == self._updated_at:
            return
        self._updated_at = step
        self.read_roots()
        self.contact = slots_in_contact(self.env.scene, self.num_envs, self.entities)
        for entity in self.entities:
            self.held[entity] |= self.contact[entity]
        finger = _host(self.env.scene["robot"].data.joint_pos)[:, self._finger]
        self.gripper_open = (finger / GRIPPER_CLOSED_RAD) < 0.5
        dt = float(self.env.step_dt)
        for index, layout in enumerate(self.env_layouts):
            names = self.plan.phase_slots[self.phase][index]
            held = {slot: bool(self.held[entity][index]) for slot, entity in names.items()}
            touching = {slot: bool(self.contact[entity][index]) for slot, entity in names.items()}
            spec = layout.spec
            if spec is None:
                continue
            poses = self._poses(index, layout)
            score, latched, entered = task_progress(
                spec,
                poses,
                held,
                touching,
                self.latched[index],
                self.entered[index],
                side=layout.side,
                gripper_open=bool(self.gripper_open[index]),
            )
            self.progress[index] = score
            committed, order, hold, _ = _commit_hold(
                self.latched[index],
                self.entered[index],
                latched,
                entered,
                score,
                float(self.hold_s[index]),
                dt,
            )
            self.latched[index] = committed
            self.entered[index] = order
            self.hold_s[index] = hold
            step_index = int(self.env.episode_length_buf[index])
            note_placements(spec, poses, touching, self.first_placed[index], step_index, side=layout.side)
            satisfied = task_satisfied(spec, poses, touching, self.first_placed[index], side=layout.side)
            self.success_hold_s[index] = self.success_hold_s[index] + dt if satisfied else 0.0
            self.success_now[index] = bool(self.success_hold_s[index] >= SUCCESS_HOLD_S)
            self.lost[index] = any(self.is_lost(self.root(index, slot)[0]) for slot in self.required[index])
            if not self._start_checked[index] and int(self.env.episode_length_buf[index]) * dt >= START_CHECK_S:
                self._start_checked[index] = True
                self.invalid_start[index] = self._start_moved(index, layout)
        done = torch.as_tensor(self.success_now, device=self.device)
        self.completed |= done
        self.failed |= torch.as_tensor(self.lost, device=self.device) & ~done

    def _start_moved(self, index: int, layout: TaskLayout) -> bool:
        """Whether a body is further than :data:`START_DRIFT_M` from its layout pose.

        The arm cannot reach a part this early, so a part that moved was pushed by an overlap or lost its support.
        """
        for placement in layout.placements:
            if placement.asset.kind == "anchor":
                continue
            drift = np.linalg.norm(self.root(index, placement.slot)[0] - np.asarray(placement.position_m))
            if drift > START_DRIFT_M:
                return True
        return False

    @staticmethod
    def is_lost(pos: np.ndarray) -> bool:
        return bool(pos[2] < LOST_Z_M or np.linalg.norm(pos[:2]) > LOST_RADIUS_M)

    def _poses(self, index: int, layout: TaskLayout) -> dict:
        poses = {}
        for placement in layout.placements:
            if placement.asset.kind == "anchor":
                pos = np.asarray(placement.position_m, dtype=np.float64)
                quat = np.asarray(placement.quat_wxyz, dtype=np.float64)
            else:
                root_pos, root_quat = self.root(index, placement.slot)
                pos = root_pos.astype(np.float64)
                x, y, z, w = root_quat.astype(np.float64)
                quat = np.array([w, x, y, z])
            poses[placement.slot] = {
                "pos": pos.reshape(1, 3),
                "quat": quat.reshape(1, 4),
                "half": np.array(placement.asset.size_m, dtype=np.float64) * 0.5,
            }
            if placement.asset.kind == "tray":
                tray = self.library.trays[placement.asset.variant]
                poses[placement.slot].update(
                    cell_centers=np.asarray(tray.cell_centers, dtype=np.float64),
                    cell_half=np.asarray(tray.cell_half_m, dtype=np.float64),
                    rim_z=float(tray.rim_z),
                )
        return poses

    def origin(self, index: int) -> np.ndarray:
        """Env origin [m] in the simulation frame."""
        return self._origins[index]

    def clear_outcomes(self) -> None:
        self.completed.zero_()
        self.failed.zero_()
