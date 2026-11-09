# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent check of a sorting layout: nothing overlaps, everything is reachable, the mapping is complete."""

from __future__ import annotations

import numpy as np

from . import geometry as geo
from .catalog import AssetLibrary
from .part_scale import MASS_MAX_KG
from .sampler import (
    SamplerConfig,
    SortingLayout,
    _tray_outline,
    container_pick_region,
    fits_cell,
    part_quat,
    tray_cell_world,
)

_TOLERANCE = 1e-9


def part_polygon(library: AssetLibrary, part) -> np.ndarray:
    """World footprint [m] of a placed part."""
    pose = library.parts[part.name].poses[part.pose_index]
    return geo.transform(pose.footprint, part.yaw, part.position_m[:2])


def _tray_entries(layout: SortingLayout, library: AssetLibrary):
    names, poses = layout.tray_list(), layout.tray_pose_list()
    if len(names) != len(poses):
        return None
    entries = []
    for name, pose in zip(names, poses):
        tray = library.trays.get(name)
        if tray is None:
            return None
        entries.append((tray, pose))
    return entries


def validate_layout(layout: SortingLayout, library: AssetLibrary, config: SamplerConfig | None = None) -> list[str]:
    """Problems of ``layout``. An empty list means the layout is valid."""
    config = config or SamplerConfig()
    problems: list[str] = []
    entries = _tray_entries(layout, library)
    if layout.container not in library.containers or not entries:
        return [f"unknown container {layout.container} or tray {layout.tray}"]
    container = library.containers[layout.container]
    pile = bool(layout.extra.get("pile"))

    table_center = np.asarray(config.table_center)
    table_half = np.asarray(config.table_half) - config.table_margin
    container_outline = geo.transform(container.outer_polygon, layout.container_pose[2], layout.container_pose[:2])
    tray_outlines = [
        geo.transform(_tray_outline(tray), pose[2], pose[:2]) for tray, pose in entries
    ]
    for name, outline in (("container", container_outline), *((f"tray {i}", outline) for i, outline in enumerate(tray_outlines))):
        if np.any(np.abs(outline - table_center) > table_half + _TOLERANCE):
            problems.append(f"{name} leaves the table")
    outlines = [container_outline, *tray_outlines]
    crowded = False
    for i in range(len(outlines)):
        for j in range(i + 1, len(outlines)):
            if geo.separation(outlines[i], outlines[j]) < config.object_gap - _TOLERANCE:
                crowded = True
    if crowded:
        problems.append("container and trays are closer than the object gap")

    # Reach of every pick region corner and every cell.
    region = container_pick_region(container, config)
    if region[0] <= 0.0 or region[1] <= 0.0:
        problems.append("container has no usable floor")
    corners = geo.transform(geo.rectangle(*region), layout.container_pose[2], layout.container_pose[:2])
    cell_points = []
    cell_owners = []
    for tray, pose in entries:
        world = geo.transform(tray.cell_centers, pose[2], pose[:2])
        cell_points.append(world)
        cell_owners.extend([tray] * tray.cells)
    cells = np.vstack(cell_points) if cell_points else np.zeros((0, 2))
    for name, points in (("container floor", corners), ("tray cells", cells)):
        if len(points) == 0:
            continue
        if np.any(np.linalg.norm(points, axis=1) > config.reach_radius + _TOLERANCE):
            problems.append(f"{name} is beyond the reach radius")
        if np.any(points[:, 0] < config.reach_min_x - _TOLERANCE):
            problems.append(f"{name} is closer to the robot than the minimum reach")

    total_cells = sum(tray.cells for tray, _ in entries)
    if len(layout.cell_types) != total_cells:
        problems.append(f"{len(layout.cell_types)} cell types for {total_cells} cells")
    assigned = [name for name in layout.cell_types if name]
    if pile:
        if layout.quota and not (4 <= layout.quota <= 6):
            problems.append(f"quota {layout.quota} is outside 4..6")
        if len(set(assigned)) < 3 or len(set(assigned)) > 6:
            problems.append(f"{len(set(assigned))} part types, expected 3..6")
        if len(assigned) < max(len(set(assigned)), layout.quota):
            problems.append("fewer assigned cells than the types and the quota")
    elif len(entries) == 1:
        if len(set(layout.cell_types)) != len(layout.cell_types):
            problems.append("a part type is mapped to two cells")
    for index, name in enumerate(layout.cell_types):
        if not name:
            continue
        asset = library.parts.get(name)
        tray = cell_owners[index] if index < len(cell_owners) else None
        if asset is None or not asset.usable:
            problems.append(f"{name} is not a usable part")
        elif tray is not None and not fits_cell(asset, tray):
            problems.append(f"{name} does not fit a cell of {tray.name}")
    present = {part.name for part in layout.parts}
    for name in assigned:
        if name not in present:
            problems.append(f"{name} has no part in the container")
    if len(layout.parts) > config.max_parts:
        problems.append(f"{len(layout.parts)} parts exceed the limit {config.max_parts}")
    if pile:
        return problems + _pile_parts(layout, library, config, container)
    tray_outline = tray_outlines[0]

    # Parts: pose, support height, footprint inside the pick region, pairwise gaps, payload.
    polygons = []
    cos, sin = np.cos(layout.container_pose[2]), np.sin(layout.container_pose[2])
    for index, part in enumerate(layout.parts):
        asset = library.parts.get(part.name)
        if asset is None:
            problems.append(f"part {index}: unknown type {part.name}")
            continue
        if layout.cell_types[part.cell] != part.name:
            problems.append(f"part {index}: cell {part.cell} is mapped to {layout.cell_types[part.cell]}")
        pose = asset.poses[part.pose_index]
        if not pose.usable:
            problems.append(f"part {index}: rest pose {part.pose_index} is not usable")
        if asset.mass_kg > MASS_MAX_KG:
            problems.append(f"part {index}: {asset.mass_kg} kg is above the payload limit")
        expected_z = config.table_top_z + container.floor_z + pose.rest_z + config.settle_gap
        if abs(part.position_m[2] - expected_z) > 1e-6:
            problems.append(f"part {index}: z {part.position_m[2]} is not the rest height {expected_z}")
        if not np.allclose(part.quat_wxyz, part_quat(pose, part.yaw), atol=1e-6):
            problems.append(f"part {index}: orientation does not match its rest pose and yaw")
        polygon = part_polygon(library, part)
        polygons.append(polygon)
        local = (polygon - layout.container_pose[:2]) @ np.array([[cos, -sin], [sin, cos]])
        if np.any(np.abs(local) > np.asarray(region) + _TOLERANCE):
            problems.append(f"part {index} ({part.name}) leaves the pick region of the container")
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if geo.separation(polygons[i], polygons[j]) < config.part_gap - _TOLERANCE:
                problems.append(f"parts {i} and {j} are closer than the part gap")
    for i, polygon in enumerate(polygons):
        if geo.separation(polygon, tray_outline) < config.object_gap - _TOLERANCE:
            problems.append(f"part {i} is too close to the tray")
    return problems


def _pile_parts(layout, library, config, container) -> list[str]:
    """Checks for a piled bin. Overlap is allowed. A settled drop must land inside the walls and below the rim."""
    problems = []
    assigned = {name for name in layout.cell_types if name}
    settled = bool(layout.extra.get("settled"))
    yaw = layout.container_pose[2]
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos, -sin], [sin, cos]])
    rim = config.table_top_z + container.rim_z - 0.005
    for index, part in enumerate(layout.parts):
        asset = library.parts.get(part.name)
        if asset is None:
            problems.append(f"part {index}: unknown type {part.name}")
            continue
        if part.name not in assigned:
            problems.append(f"part {index}: {part.name} has no cell")
        if asset.mass_kg > MASS_MAX_KG:
            problems.append(f"part {index}: {asset.mass_kg} kg is above the payload limit")
        if not settled:
            continue
        local = (np.asarray(part.position_m[:2]) - layout.container_pose[:2]) @ rotation
        if abs(local[0]) > container.floor_half_m[0] + _TOLERANCE or abs(local[1]) > container.floor_half_m[1] + _TOLERANCE:
            problems.append(f"part {index} center is outside the bin")
        if part.position_m[2] > rim:
            problems.append(f"part {index} center is above the rim")
    return problems


def cell_world_centers(layout: SortingLayout, library: AssetLibrary) -> list[tuple[float, float]]:
    """World xy [m] of every cell center, walking the trays in order."""
    centers = []
    for tray_name, pose in zip(layout.tray_list(), layout.tray_pose_list()):
        tray = library.trays[tray_name]
        centers.extend(tray_cell_world(tray, pose, cell) for cell in range(tray.cells))
    return centers
