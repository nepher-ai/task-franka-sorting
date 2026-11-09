# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success expression of one benchmark task."""

from __future__ import annotations

from dataclasses import dataclass, replace

_MOVED = ("in_container", "in_cell", "in_typed_cell", "on_side", "on_top", "stacked", "lifted", "centered")


@dataclass(frozen=True)
class Goal:
    """One object's geometric stage.

    Attributes:
        slot: Scene slot this stage scores.
        op: Predicate name. One of ``in_container``, ``in_cell``, ``in_typed_cell``, ``on_side``, ``on_top``,
            ``stacked``, ``upright``, ``lifted``, ``picked``, ``resting``, ``outside``, ``off``, or ``centered``.
        target: Slot of the container, support, tray, or reference. Empty when the stage has none.
        side: Used by ``on_side``. One of left, right, front, behind.
        margin: Table height [m] for ``resting``, lift height [m] for ``picked``, and distance [m] for ``centered``.
        cell: Cell index of the target tray for ``in_cell`` (row-major, ``row * cols + col``). ``-1`` otherwise.
        places: ``(tray slot, cell)`` pairs accepted by ``in_typed_cell``.
    """

    slot: str
    op: str
    target: str = ""
    side: str = "left"
    margin: float = 0.0
    cell: int = -1
    places: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Check:
    """Success expression. ``all`` averages object scores, ``any`` takes the max, ``choose`` the top ``k``.

    ``at_least`` succeeds when at least ``k`` children are met and scores the mean of the top ``k``.

    Attributes:
        logical: How to combine ``goals`` or ``children``. One of all, any, choose, at_least.
        goals: Object stages scored by this node.
        children: Nested expressions. Used instead of ``goals`` when the block groups checks.
        grasp: Slots that must be grasped and then released before their stage counts.
        hold: Slots that count only while the gripper is still holding them.
        k: How many objects ``choose`` averages, and the quota of ``at_least``.
        order: Slot order for ``in_order``. Empty for every other check.
    """

    logical: str
    goals: tuple[Goal, ...] = ()
    children: tuple[Check, ...] = ()
    grasp: tuple[str, ...] = ()
    hold: tuple[str, ...] = ()
    k: int = 1
    order: tuple[str, ...] = ()


@dataclass(frozen=True)
class SortingTaskSpec:
    """One benchmark success expression.

    Attributes:
        task_id: Authored task id.
        roles: Slot name to itself, kept so instruction helpers can name objects.
        success: Expression op.
        failure: Unused. Episode failure is an object leaving the table.
        time_limit_s: Episode budget [s]. The evaluator may override it.
        side: Default side for ``on_side``. One of left, right, front, behind.
        check: Expression the evaluator scores.
    """

    task_id: str
    roles: dict[str, str]
    success: str
    failure: str
    time_limit_s: float
    side: str = "left"
    check: Check | None = None


def spec_from_success(
    task_id: str,
    success: dict,
    time_limit_s: float,
    name_to_slot: dict[str, str],
) -> SortingTaskSpec:
    """Build a spec from an ``op`` block. ``require_grasp`` overrides who must be grasped and released.

    Raises:
        ValueError: The block is the old short ``kind`` form.
    """
    if "kind" in success and "op" not in success and "checks" not in success:
        raise ValueError(f"task {task_id} uses a short-form success kind; write it as an op")
    side = str(success.get("side", "left"))
    check = _parse(success, name_to_slot, side)
    if "require_grasp" in success:
        grasp = _named(success["require_grasp"], name_to_slot)
    else:
        grasp = _default_grasp(check)
    hold = _named(success["hold"], name_to_slot) if "hold" in success else ()
    return SortingTaskSpec(
        task_id=task_id,
        roles=_role_map(check),
        success=str(success.get("op", "")),
        failure="",
        time_limit_s=float(time_limit_s),
        side=side,
        check=replace(check, grasp=grasp, hold=hold),
    )


def _parse(block: dict, name_to_slot: dict[str, str], side: str) -> Check:
    op = str(block.get("op") or "")
    side = str(block.get("side", side))
    if op in ("all", "any", "choose", "at_least") and "checks" in block:
        children = tuple(_parse(child, name_to_slot, side) for child in block["checks"])
        return Check(logical=op, children=children, k=int(block.get("k", 1)))
    if op == "in_order":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(slot, "in_container", target, side, block) for slot in slots)
        return Check(logical="all", goals=goals, order=slots)
    if op == "choose":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(slot, "in_container", target, side, block) for slot in slots)
        return Check(logical="choose", goals=goals, k=int(block.get("k", 1)))
    if op == "in_typed_cell":
        slot = _slots(block.get("object", block.get("objects")), name_to_slot)[0]
        raw = block.get("cells") or []
        places = tuple((str(item["container"]), int(item["cell"])) for item in raw)
        if not places:
            raise ValueError(f"in_typed_cell for {slot} needs cells of its type")
        goal = Goal(slot, op, places[0][0], side, float(block.get("margin", 0.0)), places[0][1], places)
        return Check(logical="all", goals=(goal,))
    if op in ("all", "any", "at_least"):
        raise ValueError(f"success op {op!r} needs a checks list")
    logical = str(block.get("logical", "all"))
    if logical not in ("all", "any", "choose", "at_least"):
        raise ValueError(f"logical must be all, any, choose, or at_least, got {logical!r}")
    slots = _slots(block.get("object", block.get("objects")), name_to_slot)
    target = _target(block, name_to_slot)
    goals = tuple(_goal(slot, op, target, side, block) for slot in slots)
    return Check(logical=logical, goals=goals, k=int(block.get("k", 1)))


def _goal(slot: str, op: str, target: str, side: str, block: dict) -> Goal:
    if op == "in_cell" and (not target or int(block.get("cell", -1)) < 0):
        raise ValueError(f"in_cell for {slot} needs a tray (container) and a cell index")
    return Goal(slot, op, target, side, float(block.get("margin", 0.0)), int(block.get("cell", -1)))


def _named(value, name_to_slot: dict[str, str]) -> tuple[str, ...]:
    """Map object names to slots. An empty list is no slots."""
    if value is None or value == []:
        return ()
    return _slots(value, name_to_slot)


def _target(block: dict, name_to_slot: dict[str, str]) -> str:
    for key in ("container", "reference", "support", "target"):
        if key in block:
            return _slots(block[key], name_to_slot)[0]
    return ""


def _slots(value, name_to_slot: dict[str, str]) -> tuple[str, ...]:
    names = value if isinstance(value, list) else [value]
    slots = []
    for name in names:
        if name not in name_to_slot:
            raise KeyError(f"success names {name!r}, which the task does not place")
        slots.append(name_to_slot[str(name)])
    if not slots:
        raise ValueError("success names no objects")
    return tuple(slots)


def _default_grasp(check: Check) -> tuple[str, ...]:
    found: list[str] = []
    for goal in check.goals:
        if goal.op in _MOVED and goal.slot not in found:
            found.append(goal.slot)
    for child in check.children:
        for slot in _default_grasp(child):
            if slot not in found:
                found.append(slot)
    return tuple(found)


def _role_map(check: Check) -> dict[str, str]:
    roles = {}
    for goal in check.goals:
        roles.setdefault(goal.slot, goal.slot)
    for child in check.children:
        for role, slot in _role_map(child).items():
            roles.setdefault(role, slot)
    return roles
