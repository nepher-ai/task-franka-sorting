# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sorting planner. Each step is a tool position, yaw, and gripper target.

The planner picks one part at a time out of the container and sets it into its tray cell. Motion is planned as
rest-to-rest moves along waypoint polylines (:mod:`.motion`), so the tool cruises at the speed the evaluator
rewards and only stops where it has to: at the grasp and at the release. Every goal is checked against the
benchmark's ``in_cell`` predicate, so a part that left its cell after the release is picked up again.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from franka_sorting.benchmark.predicates import in_cell
from franka_sorting.benchmark.stage import resolve_surface
from franka_sorting.frames import REACH_HIGH_M, REACH_LOW_M, REACH_Z_HIGH_M, REACH_Z_LOW_M

from ..tracker import TaskTracker
from .cells import (
    ALIGN_ASPECT,
    CellGoal,
    cell_axis,
    cell_goals,
    free_spot,
    pick_score,
    principal_axes,
    release_turn,
    rotation,
    wrap_half_pi,
    yaw_of_wxyz,
)
from .grasp import (
    PAD_HALF_LENGTH_M,
    PAD_HALF_WIDTH_M,
    TOOL_Z_MIN_M,
    Grasp,
    fallback_grasp,
    find_grasps,
    neighbor_height,
    quat_wxyz_to_matrix,
    sample_prim_points,
    tool_yaw,
)
from .motion import YAW_RATE_RAD_S, Move

DT_S = 1.0 / 15.0
# The tracker is commanded this far ahead along the reference, which cancels the joint-drive lag.
LEAD_S = 0.10
# The reference clock slows when the tool falls this far behind [m] and stops at the second value, so a
# lagging tracker cannot cut corners into objects.
LAG_SLOW_M = 0.035
LAG_STOP_M = 0.08
COMMAND_REACH_M = 0.036
SPEED_CAP_M_S = 0.45
# Moves with a part in the hand ramp no harder than this [m/s^2]: parts held by friction slip out of the pads
# under the empty-hand profile's acceleration.
CARRY_A_MAX_M_S2 = 2.0
# Motion profile (cruise [m/s], acceleration [m/s^2]). A clear pick has to finish in about 5 s, so four
# picks fit the 40 s episode. Grasp tolerances are unchanged.
PROFILE = (0.40, 2.0)
# Final vertical approach above a grasp [m].
APPROACH_H_M = 0.07
# Lowest transit tool height [m], and the fingertip clearance kept above other objects [m].
TRAVEL_Z_M = 0.16
TRANSIT_CLEAR_M = 0.06
PATH_MARGIN_M = 0.08
CARRY_CLEAR_M = 0.045
MAX_TOOL_Z_M = 0.36
SAFE_Z_M = 0.30
HOME = np.array([0.45, 0.0, 0.26])
# Gap [m] between a part's bottom and the support when it is let go.
RELEASE_CLEARANCE_M = 0.012
# Control steps between checks, once done, that every part is still in its cell.
RECHECK_STEPS = 8
RETRACT_M = 0.05
GRASP_TOL_M = 0.006
GRASP_Z_TOL_M = 0.007
LIFT_MIN_M = 0.03
MISSED_CLOSURE = 0.92
# Closure of a settled grip with no pad contact reported. The closing lags and plateaus near 0.1-0.2 at its start.
GRIPPED_MIN_CLOSURE = 0.25
CARRY_SLIP_M = 0.025
OPEN_CLOSURE = 0.3
MAX_ATTEMPTS = 6
TILT_RESEARCH_RAD = 0.3
# Grasps kept per part. At each attempt the ones whose fingers or palm would hit another object are dropped.
GRASP_LIMIT = 24
# A grasp may move up this far [m] so that the fingertips clear what the part rests on.
NEIGHBOR_RAISE_M = 0.03
# Other objects within this distance [m] of the grasped part are checked against the fingers.
NEIGHBOR_RADIUS_M = 0.2
# A grasp whose contact slope is at least this steep can squeeze its part out; it is tried last when that
# would send the part toward a table edge closer than this [m].
SQUEEZE_SLOPE = 0.8
SQUEEZE_EDGE_M = 0.15
CORRECT_MIN_M = 0.004
CORRECT_MAX_M = 0.02
# A move whose reference clock has been held for this many steps (3 s) is abandoned.
STALL_STEPS = 45
REPLAN_SHIFT_M = 0.015
REPLAN_TURN_RAD = 0.2
MAX_REPLANS = 6
# A part thinner than this [m] is pinched by its edge, so the pads go down to THIN_PAD_CLEAR_M above its support.
THIN_PART_M = 0.008
THIN_PAD_CLEAR_M = 0.0015
PAD_CLEAR_M = TOOL_Z_MIN_M - PAD_HALF_LENGTH_M
# Parts near a release spot [m] that the open fingers must clear.
RELEASE_NEIGHBOR_M = 0.15
# Leaning the gripper about a tangential closing axis reaches past the gripper-down workspace
# (devkit/pose_probe.py): 15-20 degrees outward track within 3 mm at radius 0.80-0.85 m, where gripper-down
# misses by 2-4 cm. The lean ramps in over this range of radius [m].
NEAR_REACH_M = 0.285
NEAR_REACH_X_M = 0.22
LEAN_RAD = math.radians(20.0)
LEAN_FAR_M = (0.70, 0.78)
LEAN_BONUS_M = 0.07
# A closing axis within this angle [rad] of tangential can carry the lean as a pitch.
LEAN_AXIS_RAD = math.radians(35.0)
# Table used when a stage does not name a work surface [m].
_TABLE_CENTER = np.array([0.547, 0.0, 0.003])
_TABLE_HALF = np.array([0.35, 0.5, 0.02])


def reach_limit(xy: np.ndarray, z: float, bonus: float = 0.0) -> np.ndarray:
    """Pull a horizontal tool position [m] inside the tracked reach at height ``z`` [m].

    ``bonus`` [m] widens the far reach, for a gripper that leans outward.
    """
    blend = float(np.clip((z - REACH_Z_LOW_M) / (REACH_Z_HIGH_M - REACH_Z_LOW_M), 0.0, 1.0))
    radius_max = REACH_LOW_M + (REACH_HIGH_M - REACH_LOW_M) * blend + bonus
    limited = np.asarray(xy, dtype=np.float64).copy()
    radius = max(float(np.hypot(*limited)), 1e-6)
    # Near the base the limit is a radius, not a line: gripper-down tracks within 1 cm down to 0.28 m from the base.
    limited *= float(np.clip(radius, NEAR_REACH_M, radius_max)) / radius
    limited[0] = max(limited[0], NEAR_REACH_X_M)
    return limited


def lean_angle(point: np.ndarray) -> float:
    """Outward lean [rad] of the approach axis for a tool at ``point`` [m], 0 inside the core workspace."""
    radius = float(np.hypot(point[0], point[1]))
    return LEAN_RAD * float(np.clip((radius - LEAN_FAR_M[0]) / (LEAN_FAR_M[1] - LEAN_FAR_M[0]), 0.0, 1.0))


def lean_pitch(point: np.ndarray, yaw: float) -> float | None:
    """Pitch [rad] about the closing axis that leans the tool as :func:`lean_angle` asks, or None.

    None when the closing axis is too far from tangential for a pitch to lean the tool radially.
    """
    angle = lean_angle(point)
    if angle == 0.0:
        return 0.0
    radial = np.array([point[0], point[1]], dtype=np.float64)
    radial /= max(float(np.linalg.norm(radial)), 1e-9)
    closing = np.array([-math.sin(yaw), math.cos(yaw)])
    if abs(float(closing @ np.array([-radial[1], radial[0]]))) < math.cos(LEAN_AXIS_RAD):
        return None
    # The approach axis leans along ``-sin(pitch) * (closing x z)``.
    side = float(np.array([closing[1], -closing[0]]) @ radial)
    return -angle * math.copysign(1.0, side)


def table_box(stage: dict | None, layout) -> tuple[np.ndarray, np.ndarray]:
    """Work-surface center [m] and half sizes [m]. A task ``surface_pose`` overrides the stage."""
    if not stage:
        return _TABLE_CENTER.copy(), _TABLE_HALF.copy()
    record = resolve_surface(stage, layout.surface)
    half = np.asarray(record.get("table_m") or (0.7, 1.0, 0.04), dtype=np.float64) / 2.0
    table_z = float(record.get("table_z_m", 0.0))
    pose = layout.surface_pose if isinstance(getattr(layout, "surface_pose", None), dict) else None
    if pose and pose.get("pos"):
        pos = pose["pos"]
        # ``table_z_m`` is the top with the surface at its stage pose. A task pose higher than that raises the top too.
        stage_z = record.get("pos")
        if stage_z is not None and len(stage_z) > 2:
            table_z += float(pos[2]) - float(stage_z[2])
        center = np.array([float(pos[0]), float(pos[1]), table_z])
    else:
        raw = record.get("center_xy") or tuple(record.get("pos", (0.547, 0.0)))[:2]
        center = np.array([float(raw[0]), float(raw[1]), table_z])
    return center, half


def _segment_distance(point: np.ndarray, a, b) -> float:
    """Distance [m] from an xy point to the segment ``a``-``b``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    span = b - a
    length2 = float(span @ span)
    t = 0.0 if length2 < 1e-12 else float(np.clip((point - a) @ span / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * span)))


class Stage(IntEnum):
    INIT = 0
    APPROACH = 1
    CLOSE = 2
    CARRY = 3
    OPEN = 4
    RETRACT = 5
    RECOVER = 6
    DONE = 7


@dataclass
class _Context:
    stage: Stage = Stage.INIT
    age: int = 0
    timer: int = 0
    pos: np.ndarray = field(default_factory=lambda: HOME.copy())
    yaw: float = 0.0
    pitch: float = 0.0
    grip: float = 0.0
    ref: np.ndarray = field(default_factory=lambda: HOME.copy())
    ref_yaw: float = 0.0
    ref_pitch: float = 0.0
    move: Move | None = None
    move_t: float = 0.0
    goals: list[CellGoal] = field(default_factory=list)
    current: CellGoal | None = None
    quota: int = 0
    quota_stop: bool = False
    attempts: dict[str, int] = field(default_factory=dict)
    candidates: list[Grasp] = field(default_factory=list)
    candidate_index: int = 0
    grasp_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasp_yaw: float = 0.0
    grasp_pitch: float = 0.0
    carry_z: float = TRAVEL_Z_M
    release: np.ndarray = field(default_factory=lambda: np.zeros(3))
    release_yaw: float = 0.0
    spot: np.ndarray = field(default_factory=lambda: np.zeros(2))
    retract_z: float = TRAVEL_Z_M
    below: float = 0.05
    previous_closure: float = 0.0
    rest_z: float = 0.0
    corrected: bool = False
    parked: bool = False
    # Part position relative to the tool when the carry began [m], for slip detection without contact.
    carry_offset: np.ndarray | None = None
    # Moves still to run in this stage, each ``(points, yaw, turn_by_end)``.
    queue: list = field(default_factory=list)
    reach_bonus: float = 0.0
    # Control steps the reference clock has been held back because the tool fell far behind.
    stalled: int = 0
    last_tcp: np.ndarray | None = None
    # Tool position two steps back [m]: the governor reads a two-step speed, which a period-two swing cancels.
    older_tcp: np.ndarray | None = None
    v_max: float = PROFILE[0]
    a_max: float = PROFILE[1]
    # Part pose when its grasp was chosen, and how often the approach started over.
    planned_pose: tuple | None = None
    replans: int = 0


class SortingPlanner:
    """Per-env state machine. ``points`` are surface points [m] in each slot's root frame.

    ``rest`` is ``{slot: (position [m], wxyz)}`` at the published layout, one dict per task.
    """

    def __init__(
        self,
        tracker: TaskTracker,
        points: Sequence[dict[str, np.ndarray]],
        rest: Sequence[dict[str, tuple]],
    ):
        self.tracker = tracker
        self.points = points
        self.rest = rest
        self.num_envs = tracker.num_envs
        self.v_max = np.full(self.num_envs, PROFILE[0])
        self.a_max = np.full(self.num_envs, PROFILE[1])
        self.contexts = [_Context() for _ in range(self.num_envs)]
        self._grasps: dict[tuple[int, str], list[Grasp]] = {}
        self.tables: list[tuple[np.ndarray, np.ndarray]] | None = None
        # Pitch [rad] about the closing axis of the last step, ``[N]``. Zero keeps the gripper pointing down.
        self.pitches = np.zeros(self.num_envs)
        # The command tilt is never used: the gripper only leans through the pitch.
        self.tilts = np.zeros(self.num_envs)

    # ------------------------------------------------------------------ setup

    def _task(self, index: int) -> int:
        return int(self.tracker.task_index[index])

    def _layout(self, index: int):
        return self.tracker.env_layouts[index]

    def _table(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.tables:
            return _TABLE_CENTER.copy(), _TABLE_HALF.copy()
        return self.tables[self._task(index)]

    def _table_z(self, index: int) -> float:
        return float(self._table(index)[0][2])

    def reset(self, env_ids) -> None:
        for index in np.asarray(env_ids).reshape(-1).tolist():
            index = int(index)
            context = _Context(v_max=float(self.v_max[index]), a_max=float(self.a_max[index]))
            layout = self._layout(index)
            context.goals = cell_goals(layout)
            check = layout.spec.check if layout.spec is not None else None
            context.quota_stop = check is not None and check.logical == "at_least"
            context.quota = check.k if context.quota_stop else len(context.goals)
            if not context.goals:
                context.stage = Stage.DONE
                print(f"[WARN] task {self.tracker.task_id_of(index)} has no in_cell goal; the planner idles")
            self.contexts[index] = context

    def stage_ids(self) -> np.ndarray:
        """:class:`Stage` of every env as ints, ``[N]``."""
        return np.fromiter((int(context.stage) for context in self.contexts), dtype=np.int64, count=self.num_envs)

    def retry_count(self, index: int) -> int:
        """Grasp attempts beyond the first, summed over the parts of the running episode."""
        return sum(max(count - 1, 0) for count in self.contexts[index].attempts.values())

    # ------------------------------------------------------------------ geometry

    def _pose(self, slot: str, index: int) -> tuple[np.ndarray, np.ndarray]:
        root_pos, root_quat = self.tracker.root(index, slot)
        x, y, z, w = root_quat.astype(np.float64)
        return root_pos.astype(np.float64), quat_wxyz_to_matrix(np.array([w, x, y, z]))

    def _heading(self, slot: str, index: int) -> float:
        x, y, z, w = self.tracker.root(index, slot)[1].astype(np.float64)
        return yaw_of_wxyz(np.array([w, x, y, z]))

    def _world_points(self, slot: str, index: int) -> np.ndarray:
        pos, rot = self._pose(slot, index)
        return self.points[self._task(index)][slot][::3] @ rot.T + pos

    def _top_z(self, slot: str, index: int) -> float:
        return float(self._world_points(slot, index)[:, 2].max())

    def _bottom_z(self, slot: str, index: int) -> float:
        return float(self._world_points(slot, index)[:, 2].min())

    def _half(self, index: int, slot: str) -> np.ndarray:
        return np.asarray(self._layout(index).by_slot()[slot].asset.size_m, dtype=np.float64) / 2

    def _radius(self, index: int, slot: str) -> float:
        half = self._half(index, slot)
        return float(np.hypot(half[0], half[1]))

    def _tray_of(self, index: int, goal: CellGoal):
        """Library asset of the goal's tray."""
        return self.tracker.library.trays[self._layout(index).by_slot()[goal.tray].asset.variant]

    def _parts(self, index: int) -> list[str]:
        """Slots of the parts that this episode sorts."""
        return [goal.slot for goal in self.contexts[index].goals]

    def _obstacle_top(self, index: int, exclude: tuple[str, ...], path=None) -> float:
        """Highest point [m] of the spawned objects except ``exclude``, or the table height.

        With ``path`` (two xy points [m]) only objects whose footprint comes within
        :data:`PATH_MARGIN_M` of that segment count.
        """
        top = float(self._table(index)[0][2])
        for placement in self._layout(index).placements:
            if placement.slot in exclude:
                continue
            root = self.tracker.root(index, placement.slot)[0]
            if self.tracker.is_lost(root):
                continue
            if path is not None:
                reach = 0.5 * float(np.hypot(*placement.asset.size_m[:2])) + PATH_MARGIN_M
                if _segment_distance(np.asarray(root[:2], dtype=np.float64), *path) > reach:
                    continue
            top = max(top, self._top_z(placement.slot, index))
        return top

    @staticmethod
    def _tool_z_min(height: float) -> float:
        """Lowest tool height [m] above a part's support: the pads end 19 mm below the tool.

        A thin part is pinched by its edge, so its pads may come closer to the support.
        """
        clearance = THIN_PAD_CLEAR_M if height < THIN_PART_M else PAD_CLEAR_M
        return PAD_HALF_LENGTH_M + clearance

    def _grasp_search(self, points: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> list[Grasp]:
        """Grasps of a part at ``pos`` with orientation ``rot``, in the part\'s own (object) frame."""
        aligned = points @ rot.T
        bottom = float(pos[2] + aligned[:, 2].min())
        tool_z_min = self._tool_z_min(float(np.ptp(aligned[:, 2])))
        found = find_grasps(aligned, float(pos[2]), limit=GRASP_LIMIT, table_z=bottom, tool_z_min=tool_z_min)
        if not found:
            found = [fallback_grasp(aligned, float(pos[2]), bottom, tool_z_min)]
        return [Grasp(rot.T @ g.center_local, rot.T @ g.axis_local, g.width_m, g.score, g.slope) for g in found]

    def _candidates(self, slot: str, index: int) -> list[Grasp]:
        task = self._task(index)
        points = self.points[task][slot]
        pos, rot = self._pose(slot, index)
        pos0, quat0 = self.rest[task][slot]
        rot0 = quat_wxyz_to_matrix(np.asarray(quat0))
        rel = rot @ rot0.T
        tilt = math.acos(float(np.clip(rel[2, 2], -1.0, 1.0)))
        if tilt > TILT_RESEARCH_RAD:
            return self._grasp_search(points, pos, rot)
        if (task, slot) not in self._grasps:
            self._grasps[(task, slot)] = self._grasp_search(points, np.asarray(pos0, dtype=np.float64), rot0)
        # The stored grasps are in the object frame, so a part that turned about the vertical keeps its grasps.
        return list(self._grasps[(task, slot)])

    def _squeezes_off(self, index: int, pos: np.ndarray, rot: np.ndarray, grasp: Grasp) -> bool:
        """Whether a steep grasp would squeeze its part toward a table edge closer than SQUEEZE_EDGE_M."""
        if grasp.slope < SQUEEZE_SLOPE:
            return False
        center = pos + rot @ grasp.center_local
        axis = rot @ grasp.axis_local
        across = np.array([-axis[1], axis[0]])
        across /= max(float(np.linalg.norm(across)), 1e-9)
        # The part leaves on the side of the chord where its middle is.
        away = across if float((pos[:2] - center[:2]) @ across) >= 0.0 else -across
        table_center, table_half = self._table(index)
        reach = np.inf
        for axis_index in (0, 1):
            if abs(away[axis_index]) < 1e-6:
                continue
            edge = table_center[axis_index] + math.copysign(table_half[axis_index], away[axis_index])
            reach = min(reach, (edge - pos[axis_index]) / away[axis_index])
        return bool(reach < SQUEEZE_EDGE_M)

    def _neighbor_points(
        self, slot: str, index: int, around: np.ndarray | None = None, radius: float = NEIGHBOR_RADIUS_M
    ):
        """Surface points [m] of the other objects within ``radius`` of ``around`` (default ``slot``), ``[M, 3]``."""
        center = self._pose(slot, index)[0][:2] if around is None else np.asarray(around, dtype=np.float64)[:2]
        found = []
        for placement in self._layout(index).placements:
            other = placement.slot
            if other == slot:
                continue
            o_pos, o_rot = self._pose(other, index)
            if self.tracker.is_lost(o_pos):
                continue
            if float(np.hypot(*(o_pos[:2] - center))) > radius + float(np.max(placement.asset.size_m)):
                continue
            points = self.points[self._task(index)][other] @ o_rot.T + o_pos
            found.append(points[np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) < radius])
        return np.concatenate(found) if found else np.zeros((0, 3))

    def _clear_candidates(self, slot: str, index: int, grasps: list[Grasp]) -> list[Grasp]:
        """Grasps whose open fingers and palm miss the other objects, raised where the part sits on one.

        Keeps every grasp when none is clear, so a crowded part is still attempted.
        """
        neighbors = self._neighbor_points(slot, index)
        if not len(neighbors):
            return grasps
        pos, rot = self._pose(slot, index)
        top, bottom = self._top_z(slot, index), self._bottom_z(slot, index)
        # Only what the part rests on may push a grasp up; a neighbour beside it blocks the grasp.
        under = neighbors[neighbors[:, 2] < bottom + 0.01]
        beside = neighbors[neighbors[:, 2] >= bottom + 0.01]
        clear = []
        for grasp in grasps:
            center = pos + rot @ grasp.center_local
            axis = rot @ grasp.axis_local
            if len(beside) and neighbor_height(center, axis, beside) > center[2]:
                continue
            need = neighbor_height(center, axis, under) if len(under) else -np.inf
            if need <= center[2]:
                clear.append(grasp)
            elif need - center[2] <= NEIGHBOR_RAISE_M and need - PAD_HALF_LENGTH_M < top - 0.01:
                raised = center.copy()
                raised[2] = need
                clear.append(Grasp(rot.T @ (raised - pos), grasp.axis_local, grasp.width_m, grasp.score, grasp.slope))
        return clear or grasps

    @staticmethod
    def _reachable(point: np.ndarray) -> bool:
        limited = reach_limit(point[:2], float(point[2]))
        return bool(np.linalg.norm(limited - np.asarray(point[:2], dtype=np.float64)) < 1e-4)

    # ------------------------------------------------------------------ goals

    def _in_specific(self, index: int, slot: str, tray_slot: str, cell: int) -> bool:
        """Whether the part's center is inside ``cell`` of ``tray_slot``."""
        tray = self.tracker.library.trays[self._layout(index).by_slot()[tray_slot].asset.variant]
        tray_pos = self.tracker.root(index, tray_slot)[0].astype(np.float64)
        x, y, z, w = self.tracker.root(index, tray_slot)[1].astype(np.float64)
        pos = self.tracker.root(index, slot)[0].astype(np.float64)
        return bool(
            in_cell(
                pos.reshape(1, 3),
                tray_pos,
                np.array([w, x, y, z]),
                tray.cell_centers[cell],
                np.asarray(tray.cell_half_m, dtype=np.float64),
                float(tray.rim_z),
            )[0]
        )

    def _in_cell(self, index: int, slot: str, goal: CellGoal) -> bool:
        """Whether the part's center is inside its assigned cell, whatever the gripper does."""
        return self._in_specific(index, slot, goal.tray, goal.cell)

    def _place_of(self, index: int, goal: CellGoal) -> tuple[str, int] | None:
        """The cell of the part's type that currently holds it, if any."""
        places = goal.cells or ((goal.tray, goal.cell),)
        for tray_slot, cell in places:
            if self._in_specific(index, goal.slot, tray_slot, cell):
                return tray_slot, cell
        return None

    def _sorted(self, index: int, goal: CellGoal) -> bool:
        return self._place_of(index, goal) is not None and not self.tracker.touching(index, goal.slot)

    def _goal_met(self, index: int, goal: CellGoal) -> bool:
        """The benchmark's rule: in the cell and let go."""
        return self._in_cell(index, goal.slot, goal) and not self.tracker.touching(index, goal.slot)

    def _lost(self, index: int, goal: CellGoal) -> bool:
        return bool(self.tracker.is_lost(self.tracker.root(index, goal.slot)[0]))

    def _clearance(self, index: int, goal: CellGoal, tcp_xy: np.ndarray) -> tuple[float, float, float]:
        """``(travel, clearance, height)`` of a part, for :func:`pick_score`."""
        slot = goal.slot
        pos = self.tracker.root(index, slot)[0].astype(np.float64)
        radius = self._radius(index, slot)
        gap = np.inf
        for other in self.contexts[index].goals:
            if other.slot == slot or self._in_cell(index, other.slot, other):
                continue
            o_pos = self.tracker.root(index, other.slot)[0].astype(np.float64)
            gap = min(gap, float(np.hypot(*(o_pos[:2] - pos[:2]))) - radius - self._radius(index, other.slot))
        height = self._top_z(slot, index) - self._bottom_z(slot, index)
        return float(np.hypot(*(pos[:2] - tcp_xy))), float(gap), height

    # ------------------------------------------------------------------ motion

    def _go(
        self,
        context: _Context,
        points,
        yaw: float,
        pitch: float = 0.0,
        turn_by_end: float = 0.0,
        a_max: float | None = None,
        reach_bonus: float = 0.0,
    ) -> None:
        """Start a rest-to-rest move from the current reference through ``points``, ending at ``yaw`` and ``pitch``.

        ``turn_by_end`` [m] finishes the turn that far before the end of the path. ``reach_bonus`` [m] widens the
        reach limit for a pose that leans toward the edge of the workspace.
        """
        path = np.vstack([context.ref.reshape(1, 3), np.asarray(points, dtype=np.float64).reshape(-1, 3)])
        path[1:, :2] = np.array([reach_limit(p[:2], float(p[2]), reach_bonus) for p in path[1:]])
        start = np.array([context.ref_yaw, context.ref_pitch, 0.0])
        context.move = Move(
            path,
            start,
            np.array([float(yaw), float(pitch), 0.0]),
            v_max=context.v_max,
            a_max=context.a_max if a_max is None else min(a_max, context.a_max),
            turn_by_end=turn_by_end,
            yaw_rate=YAW_RATE_RAD_S,
        )
        context.move_t = 0.0

    def _track(self, context: _Context, tcp: np.ndarray) -> None:
        if context.move is None:
            context.pos = context.ref.copy()
            context.yaw = context.ref_yaw
            context.pitch = context.ref_pitch
            context.older_tcp = context.last_tcp
            context.last_tcp = tcp.copy()
            return
        reach_cap = COMMAND_REACH_M
        lag = float(np.linalg.norm(tcp - context.ref))
        rate = float(np.clip((LAG_STOP_M - lag) / (LAG_STOP_M - LAG_SLOW_M), 0.0, 1.0))
        if context.move_t >= context.move.duration:
            rate = 1.0
        context.stalled = context.stalled + 1 if rate < 0.2 else 0
        context.move_t += DT_S * rate
        context.ref, (context.ref_yaw, context.ref_pitch, _) = context.move.sample(context.move_t)
        context.pos, (context.yaw, context.pitch, _) = context.move.sample(context.move_t + LEAD_S)
        # Speed governor: the tracker moves faster the farther its target is, so a tool that fell behind would
        # sprint to catch up. Capping the command distance keeps the catch-up near cruise speed. A governor fed
        # the one-step speed can lock into a two-step cycle, so it reads the speed over two steps.
        if context.last_tcp is not None:
            if context.older_tcp is not None:
                speed = float(np.linalg.norm(tcp - context.older_tcp)) / (2.0 * DT_S)
            else:
                speed = float(np.linalg.norm(tcp - context.last_tcp)) / DT_S
            if speed > SPEED_CAP_M_S:
                reach_cap = min(reach_cap, max(0.005, reach_cap * (SPEED_CAP_M_S / speed) ** 4))
        context.older_tcp = context.last_tcp
        context.last_tcp = tcp.copy()
        delta = context.pos - tcp
        reach = float(np.linalg.norm(delta))
        if reach > reach_cap:
            context.pos = tcp + delta * (reach_cap / reach)

    @staticmethod
    def _moved(context: _Context) -> bool:
        return context.move is None or context.move_t >= context.move.duration

    @staticmethod
    def _overtime(context: _Context) -> float:
        """Seconds since the running move ended [s]."""
        if context.move is None:
            return 0.0
        return context.move_t - context.move.duration

    @staticmethod
    def _near(tcp: np.ndarray, point: np.ndarray, tol_xy: float, tol_z: float) -> bool:
        return bool(np.hypot(*(tcp[:2] - point[:2])) < tol_xy and abs(tcp[2] - point[2]) < tol_z)

    @staticmethod
    def _switch(context: _Context, stage: Stage) -> None:
        context.stage = stage
        context.timer = 0

    # ------------------------------------------------------------------ state machine

    def step(self, tcp: np.ndarray, closure: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position [m] ``[N, 3]``, yaw [rad] ``[N]``, and grip ``[N]`` (0 open, 1 close)."""
        positions = np.zeros((self.num_envs, 3))
        yaws = np.zeros(self.num_envs)
        grips = np.zeros(self.num_envs)
        for index, context in enumerate(self.contexts):
            self._advance(context, index, tcp[index], float(closure[index]))
            positions[index] = context.pos
            yaws[index] = context.yaw
            grips[index] = context.grip
            self.pitches[index] = context.pitch
        return positions, yaws, grips

    def _advance(self, context: _Context, index: int, tcp: np.ndarray, closure: float) -> None:
        context.age += 1
        if context.age <= 2:
            context.pos = tcp.copy()
            context.ref = tcp.copy()
            context.move = None
            return
        context.timer += 1
        handler = {
            Stage.INIT: self._init,
            Stage.APPROACH: self._approach,
            Stage.CLOSE: self._close,
            Stage.CARRY: self._carry,
            Stage.OPEN: self._open,
            Stage.RETRACT: self._retract,
            Stage.RECOVER: self._recover,
            Stage.DONE: self._done,
        }[context.stage]
        if context.stalled > STALL_STEPS and context.stage not in (Stage.RECOVER, Stage.DONE):
            # The tracker cannot follow this move: give it up instead of waiting for the episode to run out.
            context.stalled = 0
            self._fail(context)
        else:
            handler(context, index, tcp, closure)
        self._track(context, tcp)
        context.previous_closure = closure

    def _init(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        placed = [goal for goal in context.goals if self._sorted(index, goal)]
        if context.quota_stop and len(placed) >= max(context.quota, 1):
            self._switch(context, Stage.DONE)
            return
        occupied = {place for goal in context.goals if (place := self._place_of(index, goal)) is not None}
        placed_slots = {goal.slot for goal in placed}
        remaining = []
        for goal in context.goals:
            if goal.slot in placed_slots or self._lost(index, goal):
                continue
            if context.attempts.get(goal.slot, 0) >= MAX_ATTEMPTS:
                continue
            free = [place for place in (goal.cells or ((goal.tray, goal.cell),)) if place not in occupied]
            if free:
                remaining.append((goal, free))
        if not remaining:
            self._switch(context, Stage.DONE)
            return
        if context.quota_stop:
            goal, free = max(remaining, key=lambda item: self._top_z(item[0].slot, index))
            tray_slot, cell = free[0]
            context.current = CellGoal(goal.slot, tray_slot, cell, goal.cells)
        else:

            def score(item: tuple[CellGoal, list]) -> float:
                travel, clearance, height = self._clearance(index, item[0], tcp[:2])
                return pick_score(travel, clearance, height, context.attempts.get(item[0].slot, 0))

            goal, _ = min(remaining, key=score)
            context.current = goal
        slot = context.current.slot
        context.candidates = []
        context.candidate_index = context.attempts.get(slot, 0)
        context.attempts[slot] = context.attempts.get(slot, 0) + 1
        context.corrected = False
        context.queue = []
        context.grasp_pitch = 0.0
        context.reach_bonus = 0.0
        self._begin_grasp(context, index)

    def _target_moved(self, context: _Context, index: int) -> bool:
        if context.planned_pose is None:
            return False
        pos, rot = self._pose(context.current.slot, index)
        pos0, rot0 = context.planned_pose
        turn = math.acos(float(np.clip((np.trace(rot @ rot0.T) - 1.0) / 2.0, -1.0, 1.0)))
        return bool(np.linalg.norm(pos - pos0) > REPLAN_SHIFT_M or turn > REPLAN_TURN_RAD)

    def _begin_grasp(self, context: _Context, index: int) -> None:
        slot = context.current.slot
        pos, rot = self._pose(slot, index)
        if not context.candidates:
            context.candidates = self._clear_candidates(slot, index, self._candidates(slot, index))
            # Reachable grasps first, keeping the search order inside each group. At the edge of the workspace
            # a grasp that can lean counts as reachable.
            flags = []
            for g in context.candidates:
                center = pos + rot @ g.center_local
                leans = lean_angle(center) != 0.0 and lean_pitch(center, tool_yaw(rot @ g.axis_local)) is not None
                flags.append(self._reachable(center) or leans)
            # A steep chord squeezes the part out sideways. Pointed at a nearby table edge, that throws it off the
            # table, which ends the episode: try those grasps last.
            risky = [self._squeezes_off(index, pos, rot, g) for g in context.candidates]
            ranked = sorted(range(len(context.candidates)), key=lambda k: (not flags[k], risky[k]))
            context.candidates = [context.candidates[k] for k in ranked]
        grasp = context.candidates[context.candidate_index % len(context.candidates)]
        center = pos + rot @ grasp.center_local
        # A part that settled lower than where it was published must not push the pads into its support.
        bottom = self._bottom_z(slot, index)
        height = self._top_z(slot, index) - bottom
        center[2] = max(center[2], bottom + self._tool_z_min(height))
        context.grasp_center = center
        context.grasp_yaw = tool_yaw(rot @ grasp.axis_local)
        pitch = lean_pitch(center, context.grasp_yaw)
        context.grasp_pitch = 0.0 if pitch is None else pitch
        context.reach_bonus = LEAN_BONUS_M if context.grasp_pitch else 0.0
        if context.grasp_pitch:
            # A leaning pad drops one lower corner by its half width times the sine of the lean.
            floor = bottom + self._tool_z_min(height) + PAD_HALF_WIDTH_M * abs(math.sin(context.grasp_pitch))
            center[2] = max(center[2], floor + 0.003)
        context.rest_z = float(pos[2])
        self._approach_path(context, index)

    def _approach_path(self, context: _Context, index: int) -> None:
        """Transit high enough to clear every object, then descend straight onto the grasp."""
        pos, rot = self._pose(context.current.slot, index)
        context.planned_pose = (pos.copy(), rot.copy())
        center = context.grasp_center
        transit = float(
            np.clip(
                max(
                    TRAVEL_Z_M,
                    center[2] + APPROACH_H_M,
                    self._obstacle_top(index, (), (context.ref[:2], center[:2])) + TRANSIT_CLEAR_M,
                ),
                0.0,
                MAX_TOOL_Z_M,
            )
        )
        ref = context.ref
        path = []
        if np.hypot(*(ref[:2] - center[:2])) > 0.02:
            if ref[2] < transit - 0.03:
                path.append([ref[0], ref[1], transit])
            path.append([center[0], center[1], transit])
        path.append(center)
        # The fingers are turned before they descend around the part.
        descent = float(path[-2][2] - center[2]) if len(path) >= 2 else 0.0
        self._go(
            context,
            path,
            context.grasp_yaw,
            pitch=context.grasp_pitch,
            turn_by_end=max(descent - 0.01, 0.0),
            reach_bonus=context.reach_bonus,
        )
        self._switch(context, Stage.APPROACH)

    def _approach(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        if context.timer % 3 == 0 and self._target_moved(context, index):
            # The part shifted or tipped since the grasp was chosen: choose again, without spending an attempt.
            slot = context.current.slot
            context.attempts[slot] = max(context.attempts.get(slot, 1) - 1, 0)
            context.replans += 1
            if context.replans <= MAX_REPLANS:
                self._switch(context, Stage.INIT)
                return
        if not self._moved(context):
            return
        if self._near(tcp, context.grasp_center, GRASP_TOL_M, GRASP_Z_TOL_M):
            self._switch(context, Stage.CLOSE)
        elif self._overtime(context) > 2.5:
            if self._near(tcp, context.grasp_center, 2.5 * GRASP_TOL_M, 2.0 * GRASP_Z_TOL_M):
                self._switch(context, Stage.CLOSE)
            else:
                self._fail(context)

    def _close(self, context, index, tcp, closure) -> None:
        slot = context.current.slot
        context.grip = 1.0
        if context.timer < 3:
            return
        settled = abs(closure - context.previous_closure) < 0.01
        # The pads carry the contact sensors. A settled closure short of empty means the fingers stopped on
        # something too, which the knuckles can do without any pad contact.
        touching = self.tracker.touching(index, slot)
        gripped = touching or (settled and GRIPPED_MIN_CLOSURE < closure < MISSED_CLOSURE)
        if closure > MISSED_CLOSURE and not touching:
            self._fail(context)
        elif settled and gripped:
            context.carry_offset = None
            self._plan_carry(context, index)
            self._switch(context, Stage.CARRY)
        elif context.timer > 20:
            self._fail(context)

    # ------------------------------------------------------------------ carrying to the cell

    def _occupants(self, index: int, goal: CellGoal) -> list[tuple[np.ndarray, float, float]]:
        """``(center xy [m], radius [m], top z [m])`` of the other parts that already rest in the goal's cell.

        Parts dropped into the cell before the episode are not goals. They still occupy the cell.
        """
        found = []
        seen = {goal.slot}
        for other in self.contexts[index].goals:
            if other.slot in seen or other.tray != goal.tray or other.cell != goal.cell:
                continue
            if not self._in_cell(index, other.slot, goal):
                continue
            seen.add(other.slot)
            pos = self.tracker.root(index, other.slot)[0].astype(np.float64)
            found.append((pos[:2], self._radius(index, other.slot), self._top_z(other.slot, index)))
        for placement in self._layout(index).placements:
            if placement.asset.kind != "item" or placement.slot in seen:
                continue
            if not self._in_specific(index, placement.slot, goal.tray, goal.cell):
                continue
            pos = self.tracker.root(index, placement.slot)[0].astype(np.float64)
            found.append((pos[:2], self._radius(index, placement.slot), self._top_z(placement.slot, index)))
        return found

    def _plan_carry(self, context: _Context, index: int) -> None:
        goal = context.current
        slot = goal.slot
        g = context.grasp_center
        o_pos, _ = self._pose(slot, index)
        context.below = float(max(g[2] - self._bottom_z(slot, index), 0.0))
        tray = self._tray_of(index, goal)
        tray_pos = self.tracker.root(index, goal.tray)[0].astype(np.float64)
        tray_yaw = self._heading(goal.tray, index)
        cell_world = tray_pos[:2] + rotation(tray_yaw) @ np.asarray(tray.cell_centers[goal.cell], dtype=np.float64)
        axis = cell_axis(tray_yaw, tray.cell_half_m)
        # Turn the held part so that its long side lies along the cell, then find the emptiest spot of the cell.
        angle, long_half, short_half = principal_axes(self._world_points(slot, index)[:, :2])
        context.release_yaw = wrap_half_pi(context.grasp_yaw + release_turn(angle, long_half, short_half, axis))
        turn = context.release_yaw - context.grasp_yaw
        if long_half < ALIGN_ASPECT * short_half:
            extent = (long_half, long_half)
        elif tray.cell_half_m[0] >= tray.cell_half_m[1]:
            extent = (long_half, short_half)
        else:
            extent = (short_half, long_half)
        occupants = self._occupants(index, goal)
        spot, _ = free_spot(
            cell_world,
            tray_yaw,
            tray.cell_half_m,
            extent,
            [(center, radius) for center, radius, _ in occupants],
            float(np.hypot(long_half, short_half)),
        )
        context.spot = spot
        # Tool position that puts the held part's root on the spot after the turn about the tool axis.
        release_xy = spot + rotation(turn) @ (g[:2] - o_pos[:2])
        floor_z = float(tray_pos[2] + tray.floor_z)
        piled = [top for center, radius, top in occupants if np.linalg.norm(center - spot) < radius + long_half]
        support = max([floor_z, *piled])
        height = support + context.below + RELEASE_CLEARANCE_M
        release = np.array([release_xy[0], release_xy[1], height])
        # The open fingers must clear the dividers and the parts beside the spot, so the part is let go from above.
        closing = np.array([-math.sin(context.release_yaw), math.cos(context.release_yaw), 0.0])
        obstacles = [self._world_points(goal.tray, index), self._neighbor_points(slot, index, spot, RELEASE_NEIGHBOR_M)]
        release[2] = float(max(height, neighbor_height(release, closing, np.concatenate(obstacles))))
        release[2] = min(release[2], MAX_TOOL_Z_M)
        context.release = release
        context.retract_z = float(release[2] + RETRACT_M)
        clear = self._obstacle_top(index, (slot,), (g[:2], release[:2])) + context.below + CARRY_CLEAR_M
        context.carry_z = float(np.clip(max(TRAVEL_Z_M, clear, release[2] + 0.03), 0.0, MAX_TOOL_Z_M))
        # Lift first, so that turning the part never sweeps it through its neighbors, then turn on the way over.
        self._go(context, [[g[0], g[1], context.carry_z]], context.grasp_yaw, a_max=CARRY_A_MAX_M_S2)
        descent = context.carry_z - release[2]
        over = [[release[0], release[1], context.carry_z], release]
        context.queue = [(over, context.release_yaw, max(descent, 0.0))]

    def _held_part(self, context: _Context, index: int, tcp: np.ndarray) -> bool:
        """Whether the part is still in the hand: touching the pads, or staying with the tool as it lifts."""
        slot = context.current.slot
        pos, _ = self._pose(slot, index)
        if self.tracker.touching(index, slot):
            return True
        near = float(np.linalg.norm(pos[:2] - tcp[:2])) < 0.12 and abs(pos[2] - tcp[2]) < 0.15
        if not near:
            return False
        offset = pos - tcp
        if context.carry_offset is None:
            context.carry_offset = offset
            return True
        return bool(np.linalg.norm(offset - context.carry_offset) < CARRY_SLIP_M)

    def _carry(self, context, index, tcp, closure) -> None:
        context.grip = 1.0
        if context.timer > 4 and not self._held_part(context, index, tcp):
            self._fail(context)
            return
        if context.timer == 8:
            pos, _ = self._pose(context.current.slot, index)
            if pos[2] - context.rest_z < LIFT_MIN_M and context.move_t < context.move.duration:
                if tcp[2] - context.grasp_center[2] > 2 * LIFT_MIN_M:
                    self._fail(context)
                    return
        if not self._moved(context):
            return
        if context.queue:
            points, yaw, turn_by_end = context.queue.pop(0)
            self._go(context, points, yaw, turn_by_end=turn_by_end, a_max=CARRY_A_MAX_M_S2)
            return
        near = self._near(tcp, context.release, 0.012, 0.012)
        if not (near or self._overtime(context) > 0.6):
            return
        if not context.corrected and self._correct(context, index):
            return
        self._switch(context, Stage.OPEN)

    def _correct(self, context: _Context, index: int) -> bool:
        """Shift the release so the held part lands on its spot. True when a move started."""
        context.corrected = True
        o_pos, _ = self._pose(context.current.slot, index)
        error = context.spot - o_pos[:2]
        size = float(np.linalg.norm(error))
        if size < CORRECT_MIN_M:
            return False
        target = context.ref.copy()
        target[:2] += error * min(1.0, CORRECT_MAX_M / size)
        self._go(context, [target], context.release_yaw, a_max=CARRY_A_MAX_M_S2)
        return True

    def _open(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        if closure < OPEN_CLOSURE and context.timer >= 2 or context.timer > 15:
            up = np.array([context.ref[0], context.ref[1], max(context.retract_z, context.ref[2] + 0.03)])
            self._go(context, [up], context.ref_yaw)
            self._switch(context, Stage.RETRACT)

    def _retract(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        if self._moved(context):
            self._switch(context, Stage.INIT)

    def _recover(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        if context.timer == 1 or context.move is None:
            context.move = None
        if context.timer >= 3 and context.move is None:
            up = np.array([context.ref[0], context.ref[1], max(TRAVEL_Z_M, context.ref[2] + 0.05)])
            self._go(context, [up], context.ref_yaw, pitch=0.0)
        if context.move is not None and self._moved(context) and closure < OPEN_CLOSURE or context.timer > 45:
            self._switch(context, Stage.INIT)

    def _done(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        # A placed part can still slide or tip after the hand has left: try it again.
        if context.goals and context.timer % RECHECK_STEPS == 0:
            if not all(self._goal_met(index, goal) for goal in context.goals):
                self._switch(context, Stage.INIT)
                return
        if context.parked:
            return
        context.parked = True
        rest = np.array([HOME[0], HOME[1], SAFE_Z_M])
        if np.linalg.norm(context.ref - rest) > 0.02:
            path = []
            if context.ref[2] < SAFE_Z_M - 0.03:
                path.append([context.ref[0], context.ref[1], SAFE_Z_M])
            path.append(rest)
            self._go(context, path, 0.0)

    @staticmethod
    def _fail(context: _Context) -> None:
        """Open the gripper and recover so the next grasp can be tried."""
        context.grip = 0.0
        context.move = None
        context.queue = []
        context.parked = False
        context.stage = Stage.RECOVER
        context.timer = 0


def _stage(env):
    import omni.usd

    return omni.usd.get_context().get_stage()


def build_planner(
    env, tracker: TaskTracker, stage: dict | None = None, points_per_object: int = 6000
) -> SortingPlanner:
    """Sample each USD variant once and build the planner. ``stage`` supplies the work-surface boxes.

    A point's root-frame position is ``R^T (p - root)`` at that body's spawn pose.
    """
    usd = _stage(env)
    plan = tracker.plan
    points: list[dict[str, np.ndarray]] = []
    rest: list[dict[str, tuple]] = []
    # One surface sample per USD variant. Tasks that share a mesh reuse it.
    sampled: dict[tuple, np.ndarray] = {}
    for task, published in enumerate(tracker.task_layouts):
        phase, sample = tracker.sample_env(task)
        origin = tracker.origin(sample)
        task_points: dict[str, np.ndarray] = {}
        task_rest: dict[str, tuple] = {}
        for placement in published.placements:
            variant = (placement.asset.usd, tuple(placement.scale or ()))
            if variant not in sampled:
                entity = plan.phase_slots[phase][sample][placement.slot]
                # place_planned_layout put this prim at its task's placement.
                world = sample_prim_points(usd, f"/World/envs/env_{sample}/{entity}", points_per_object)
                spawn_rot = quat_wxyz_to_matrix(np.asarray(placement.quat_wxyz, dtype=np.float64))
                spawn_pos = np.asarray(placement.position_m, dtype=np.float64)
                sampled[variant] = (world - (origin + spawn_pos)) @ spawn_rot
            task_points[placement.slot] = sampled[variant]
            task_rest[placement.slot] = (
                np.asarray(placement.position_m, dtype=np.float64),
                np.asarray(placement.quat_wxyz, dtype=np.float64),
            )
        points.append(task_points)
        rest.append(task_rest)
    planner = SortingPlanner(tracker, points, rest)
    planner.tables = [table_box(stage, layout) for layout in tracker.task_layouts]
    return planner
