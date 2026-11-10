# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The ``tasks/<task_id>.json`` record of a sorting layout. Benchmark builds and procedural data generation share it."""

from __future__ import annotations

import numpy as np

from .catalog import AssetLibrary
from .instruction import build_instruction
from .sampler import PartInstance, SortingLayout, quat_from_yaw

CONTAINER_SLOT = "container"
TRAY_SLOT = "tray"


def tray_slot(index: int) -> str:
    """Scene slot of the ``index``-th tray. A one-tray task still uses ``tray``."""
    return TRAY_SLOT if index == 0 else f"tray_{index}"
SURFACE_Z_M = -0.347
"""Pose z [m] of the work table. Its top is then ``SamplerConfig.table_top_z`` above the environment origin."""


def part_slot(index: int) -> str:
    """Scene slot of the ``index``-th part. A slot is unique even when two parts share a type."""
    return f"part_{index:02d}"


def short_tray_name(tray: str) -> str:
    return tray.removeprefix("parts_tray_")


def default_task_id(index: int, layout: SortingLayout) -> str:
    names = "+".join(short_tray_name(name) for name in layout.tray_list())
    return f"{index:03d}_{layout.container}_{names}"


def task_payload(
    layout: SortingLayout,
    library: AssetLibrary,
    *,
    task_id: str,
    background: str,
    surface: str,
    table_center_xy: tuple[float, float],
    seed: int | None = None,
) -> dict:
    """Task file content: placements, success expression, instruction, and the type to cell mapping.

    Args:
        layout: Sampled layout.
        library: Library the layout was sampled from.
        task_id: Task id. The file is named ``<task_id>.json``.
        background: Dome name from ``stage.json``.
        surface: Work-table name from ``stage.json``.
        table_center_xy: Work-table center [m] in the environment frame.
        seed: Sampler seed, kept for provenance.
    """
    container = library.containers[layout.container]
    trays = [(library.trays[name], pose) for name, pose in zip(layout.tray_list(), layout.tray_pose_list())]
    placements = [
        {
            "slot": CONTAINER_SLOT,
            "name": CONTAINER_SLOT,
            "variant": container.name,
            "kind": "container",
            "usd": f"assets/{container.usd}",
            "size_m": [round(v, 6) for v in container.size_m],
            "position_m": [*layout.container_pose[:2], layout.table_top_z],
            "quat_wxyz": list(quat_from_yaw(layout.container_pose[2])),
            "kinematic": True,
        }
    ]
    multi = bool(layout.quota) or len(trays) > 1
    for index, (tray, pose) in enumerate(trays):
        slot = f"tray_{index}" if multi else TRAY_SLOT
        placements.append(
            {
                "slot": slot,
                "name": slot,
                "variant": tray.name,
                "kind": "tray",
                "usd": f"assets/{tray.usd}",
                "size_m": [round(v, 6) for v in tray.size_m],
                "position_m": [*pose[:2], layout.table_top_z],
                "quat_wxyz": list(quat_from_yaw(pose[2])),
                "kinematic": True,
            }
        )
    places_of = _places_by_type(layout, trays)
    checks = []
    for index, part in enumerate(layout.parts):
        asset = library.parts[part.name]
        slot = part_slot(index)
        placements.append(
            {
                "slot": slot,
                "name": slot,
                "variant": part.name,
                "kind": "item",
                "usd": f"assets/{asset.usd}",
                "size_m": [round(v, 6) for v in asset.size_m],
                "position_m": list(part.position_m),
                "quat_wxyz": list(part.quat_wxyz),
                "pose_index": part.pose_index,
                "yaw": part.yaw,
                "cell": part.cell,
            }
        )
        if layout.quota:
            checks.append({"op": "in_typed_cell", "object": slot, "cells": places_of.get(part.name, [])})
        else:
            checks.append({"op": "in_cell", "object": slot, "container": TRAY_SLOT, "cell": part.cell})
    names = "+".join(short_tray_name(tray.name) for tray, _ in trays)
    success = {"op": "at_least", "k": int(layout.quota), "checks": checks} if layout.quota else {"op": "all", "checks": checks}
    payload = {
        "task_id": task_id,
        "layout": f"{layout.container}__{names}",
        "background": background,
        "surface": surface,
        "surface_pose": {"pos": [*table_center_xy, SURFACE_Z_M], "rot": [1.0, 0.0, 0.0, 0.0]},
        "instruction": build_instruction(layout, trays[0][0], library),
        "success": success,
        "container": layout.container,
        "tray": layout.tray,
        "trays": [tray.name for tray, _ in trays],
        "cell_map": list(layout.cell_types),
        "n_parts": len(layout.parts),
        "quota": int(layout.quota),
        "population": int(layout.population or len(layout.parts)),
        "settled": bool(layout.extra.get("settled")),
        "placements": placements,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return _plain(payload)


def _places_by_type(layout: SortingLayout, trays) -> dict[str, list[dict]]:
    """Cells of each assigned type, addressed by the tray slot written into the task."""
    places: dict[str, list[dict]] = {}
    offset = 0
    multi = bool(layout.quota) or len(trays) > 1
    for index, (tray, _) in enumerate(trays):
        slot = f"tray_{index}" if multi else TRAY_SLOT
        for cell in range(tray.cells):
            name = layout.cell_types[offset + cell]
            if name:
                places.setdefault(name, []).append({"container": slot, "cell": cell})
        offset += tray.cells
    return places


def layout_of_payload(payload: dict, table_top_z: float | None = None) -> SortingLayout:
    """The :class:`SortingLayout` that a task file records. The inverse of :func:`task_payload`."""
    by_slot = {item["slot"]: item for item in payload["placements"]}

    def pose(slot: str) -> tuple[float, float, float]:
        item = by_slot[slot]
        w, _, _, z = item["quat_wxyz"]
        return (float(item["position_m"][0]), float(item["position_m"][1]), float(2.0 * np.arctan2(z, w)))

    parts = tuple(
        PartInstance(
            name=item["variant"],
            pose_index=int(item["pose_index"]),
            position_m=tuple(float(v) for v in item["position_m"]),
            yaw=float(item["yaw"]),
            quat_wxyz=tuple(float(v) for v in item["quat_wxyz"]),
            cell=int(item["cell"]),
        )
        for item in payload["placements"]
        if item["kind"] == "item" and item.get("home") != "tray"
    )
    container = by_slot[CONTAINER_SLOT]
    tray_names = tuple(payload.get("trays") or (payload["tray"],))
    tray_slots = []
    for index in range(len(tray_names)):
        slot = f"tray_{index}"
        if slot not in by_slot:
            slot = TRAY_SLOT
        tray_slots.append(slot)
    tray_poses = tuple(pose(slot) for slot in tray_slots)
    quota = int(payload.get("quota") or 0)
    return SortingLayout(
        container=payload["container"],
        tray=payload["tray"],
        container_pose=pose(CONTAINER_SLOT),
        tray_pose=tray_poses[0],
        parts=parts,
        cell_types=tuple(payload["cell_map"]),
        table_top_z=float(container["position_m"][2]) if table_top_z is None else table_top_z,
        extra={"pile": quota > 0, "settled": bool(payload.get("settled"))},
        trays=tray_names if quota or len(tray_names) > 1 else (),
        tray_poses=tray_poses if quota or len(tray_names) > 1 else (),
        quota=quota,
        population=int(payload.get("population") or len(parts)),
    )


def _plain(value):
    """JSON-ready copy: numpy scalars become Python numbers and tuples become lists."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
