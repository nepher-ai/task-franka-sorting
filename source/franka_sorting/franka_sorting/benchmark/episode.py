# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A fresh random layout for one episode of a published sorting task."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..layout.catalog import AssetLibrary
from ..layout.sampler import PartInstance, SamplerConfig, SortingLayout, _sample_furniture, quat_from_yaw, quat_multiply
from ..layout.task_record import CONTAINER_SLOT, part_slot, task_payload
from .task import TaskLayout, layout_from_payload


def part_slots(layout: TaskLayout) -> list[tuple[str, str]]:
    """``(slot, part type)`` of every part body of a task, in slot order."""
    return [
        (placement.slot, placement.asset.variant)
        for placement in layout.placements
        if placement.asset.kind == "item"
    ]


def resample_layout(
    published: TaskLayout,
    library: AssetLibrary,
    rng: np.random.Generator,
    config: SamplerConfig | None = None,
) -> TaskLayout:
    """New container pose, tray pose, and part poses for the bodies of ``published``.

    The scene keeps one body per published part, so the new layout has at most as many parts of a type as the
    published one, and at least one. Bodies that the new layout does not use are left out of it and stay parked.
    The container, tray, part types, and cell mapping do not change. The instruction is rebuilt, because the
    tray may face another way.

    Args:
        published: Task as the benchmark publishes it.
        library: Asset library of the benchmark.
        rng: Random generator. The result is a function of its state.
        config: Sampler limits. Its table center follows the published table.

    Raises:
        LayoutError: No valid layout exists within the sampler budget.
    """
    config = config or SamplerConfig()
    if published.surface_pose is not None:
        config = replace(config, table_center=tuple(published.surface_pose["pos"][:2]))
    bodies = part_slots(published)
    container_placement = next(item for item in published.placements if item.slot == CONTAINER_SLOT)
    tray_placements = [item for item in published.placements if item.asset.kind == "tray"]
    trays = [library.trays[item.asset.variant] for item in tray_placements]
    container_pose, tray_poses = _sample_furniture(rng, library.containers[published.container], trays, config)
    old_yaw = _yaw(container_placement.quat_wxyz)
    old_xy = np.asarray(container_placement.position_m[:2])
    new_yaw = container_pose[2]
    new_xy = np.asarray(container_pose[:2])
    parts = []
    for placement in published.placements:
        if placement.asset.kind != "item":
            continue
        local = _rotate(np.asarray(placement.position_m[:2]) - old_xy, -old_yaw)
        world = _rotate(local, new_yaw) + new_xy
        yaw = _yaw(placement.quat_wxyz) + new_yaw - old_yaw
        turned = quat_multiply(quat_from_yaw(new_yaw - old_yaw), placement.quat_wxyz)
        parts.append(
            PartInstance(
                name=placement.asset.variant,
                pose_index=0,
                position_m=(float(world[0]), float(world[1]), float(placement.position_m[2])),
                yaw=float(yaw),
                quat_wxyz=turned,
                cell=-1,
            )
        )
    check = published.spec.check if published.spec is not None else None
    quota = check.k if check is not None and check.logical == "at_least" else 0
    layout = SortingLayout(
        container=published.container,
        tray=trays[0].name,
        container_pose=container_pose,
        tray_pose=tray_poses[0],
        parts=tuple(parts),
        cell_types=published.cell_map,
        table_top_z=float(container_placement.position_m[2]),
        extra={"pile": quota > 0, "settled": quota > 0},
        trays=tuple(tray.name for tray in trays) if quota or len(trays) > 1 else (),
        tray_poses=tray_poses if quota or len(trays) > 1 else (),
        quota=quota,
        population=len(parts),
    )
    payload = task_payload(
        layout,
        library,
        task_id=published.task_id,
        background=published.background,
        surface=published.surface,
        table_center_xy=config.table_center,
    )
    free: dict[str, list[str]] = {}
    for slot, variant in bodies:
        free.setdefault(variant, []).append(slot)
    renamed = {part_slot(index): free[part.name].pop(0) for index, part in enumerate(layout.parts)}
    for item in payload["placements"]:
        item["slot"] = item["name"] = renamed.get(item["slot"], item["slot"])
    for check in payload["success"]["checks"]:
        check["object"] = renamed[check["object"]]
    time_limit_s = published.spec.time_limit_s if published.spec is not None else None
    result = layout_from_payload(payload, **({} if time_limit_s is None else {"time_limit_s": time_limit_s}))
    result.surface_pose = published.surface_pose
    return result


def _yaw(quat_wxyz: tuple[float, ...]) -> float:
    w, _, _, z = quat_wxyz
    return float(2.0 * np.arctan2(z, w))


def _rotate(point: np.ndarray, yaw: float) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([point[0] * cos - point[1] * sin, point[0] * sin + point[1] * cos])
