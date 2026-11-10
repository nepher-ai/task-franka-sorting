# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlap-free random layouts: where the container and the tray stand, and where each part lies in the container.

Everything is in the environment frame (robot base at the origin, x forward, y left, z up). Lengths are [m],
angles [rad]. A layout is valid by construction. ``validate.py`` re-checks it independently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from . import geometry as geo
from .catalog import AssetLibrary, ContainerAsset, PartAsset, PartPose, TrayAsset
from .part_scale import CELL_MARGIN_M, WIDE_VARIANTS


class LayoutError(RuntimeError):
    """No valid layout was found within the attempt budget."""


@dataclass(frozen=True)
class SamplerConfig:
    """Geometry and randomization limits. Defaults match the 0.7 x 1.0 m work table of the stage."""

    table_center: tuple[float, float] = (0.5471368, 0.0)
    table_half: tuple[float, float] = (0.35, 0.50)
    table_top_z: float = 0.003
    """Table surface height [m] in the environment frame (stage pose z = -0.347)."""
    table_margin: float = 0.03
    """Distance [m] every container and tray keeps from the table edge."""
    reach_radius: float = 0.76
    """Largest horizontal distance [m] from the robot base of a pick or place point (arm limit 0.80)."""
    reach_min_x: float = 0.32
    """Smallest x [m] of a pick or place point (the arm folds into its joint limits below 0.30)."""
    wall_clearance: float = 0.05
    """Distance [m] from a part footprint to the edge of the flat container floor: finger and hand room."""
    part_gap: float = 0.015
    """Smallest gap [m] between two part footprints: the fingers close between neighbours."""
    object_gap: float = 0.06
    """Smallest gap [m] between the container and the tray."""
    settle_gap: float = 0.0005
    """Height [m] a part starts above its support, so no start penetrates."""
    fill_range: tuple[float, float] = (0.20, 0.45)
    """Share of the usable floor area that the part footprints cover."""
    max_parts: int = 40
    yaw_jitter: float = np.deg2rad(15.0)
    """Half-range [rad] added to a quarter-turn heading of a container or tray."""
    """Random yaw [rad] of the container and tray on top of the quarter turn."""
    placement_tries: int = 200
    layout_tries: int = 60
    type_budget: float = 0.45
    """Share of the usable floor that one footprint of each part type may take."""
    pair_tries: int = 1500
    pair_pools: int = 4
    pair_samples: int = 20000


@dataclass(frozen=True)
class PartInstance:
    """One part of a layout. ``position_m`` is the center of mass; ``quat_wxyz`` includes the yaw."""

    name: str
    pose_index: int
    position_m: tuple[float, float, float]
    yaw: float
    quat_wxyz: tuple[float, float, float, float]
    cell: int


@dataclass(frozen=True)
class SortingLayout:
    """Container, trays, and parts. Poses are ``(x, y, yaw)``. ``cell_types[cell]`` is the part type of a cell.

    ``tray`` and ``tray_pose`` are the first tray, so a one-tray layout still reads them. ``trays`` lists every
    tray. An empty string in ``cell_types`` is a cell with no type. ``quota`` is how many parts must be sorted.
    ``population`` is how many parts the bin was asked to hold.
    """

    container: str
    tray: str
    container_pose: tuple[float, float, float]
    tray_pose: tuple[float, float, float]
    parts: tuple[PartInstance, ...]
    cell_types: tuple[str, ...]
    table_top_z: float = 0.003
    extra: dict = field(default_factory=dict, compare=False)
    trays: tuple[str, ...] = ()
    tray_poses: tuple[tuple[float, float, float], ...] = ()
    quota: int = 0
    population: int = 0

    def tray_list(self) -> tuple[str, ...]:
        return self.trays or (self.tray,)

    def tray_pose_list(self) -> tuple[tuple[float, float, float], ...]:
        return self.tray_poses or (self.tray_pose,)


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))


def quat_multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Hamilton product ``a * b`` of two ``wxyz`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def part_quat(pose: PartPose, yaw: float) -> tuple[float, float, float, float]:
    """World orientation of a part that rests in ``pose`` and is turned by ``yaw`` about the vertical."""
    return quat_multiply(quat_from_yaw(yaw), pose.quat_wxyz)


def cell_inner_size(tray: TrayAsset) -> tuple[float, float]:
    """Inner (long, short) size [m] of one cell."""
    hx, hy = tray.cell_half_m
    return (2.0 * max(hx, hy), 2.0 * min(hx, hy))


def fits_cell(part: PartAsset, tray: TrayAsset) -> bool:
    """Whether every usable rest pose of ``part`` fits a cell of ``tray``.

    The wide variants may fill the cell. Every other part keeps ``CELL_MARGIN_M`` on each side. A pose taller
    than the cell interior does not fit, so a 60 mm part is seated only in a deeper tray.
    """
    cell_long, cell_short = cell_inner_size(tray)
    margin = 0.0 if part.name in WIDE_VARIANTS else CELL_MARGIN_M
    interior = tray.rim_z - tray.floor_z - 0.004
    for pose in part.usable_poses:
        long_side, short_side, _ = geo.min_area_rectangle(pose.footprint)
        if long_side > cell_long - 2.0 * margin or short_side > cell_short - 2.0 * margin:
            return False
        if pose.height > interior:
            return False
    return True


def tray_cell_world(tray: TrayAsset, pose: tuple[float, float, float], cell: int) -> tuple[float, float]:
    """World xy [m] of the center of ``cell``."""
    local = tray.cell_centers[cell]
    return tuple(float(v) for v in geo.transform(local[None, :], pose[2], (pose[0], pose[1]))[0])


def container_pick_region(container: ContainerAsset, config: SamplerConfig) -> tuple[float, float]:
    """Half sizes [m] of the rectangle (container frame) that part footprints must stay inside."""
    return (
        container.floor_half_m[0] - config.wall_clearance,
        container.floor_half_m[1] - config.wall_clearance,
    )


def choose_types(
    rng: np.random.Generator, library: AssetLibrary, tray: TrayAsset, budget: float = np.inf
) -> list[PartAsset]:
    """One distinct, usable part type per cell of ``tray``, in cell order.

    Types come from as many different part classes as possible, and every one fits a cell. Together the types
    take at most ``budget`` [m^2] of footprint (largest rest pose of each), so that one of each can fit a small
    container. When random draws exceed the budget, the smallest types are taken.
    """
    by_class: dict[str, list[PartAsset]] = {}
    for cls, variants in library.families().items():
        fitting = [v for v in variants if fits_cell(v, tray)]
        if fitting:
            by_class[cls] = fitting
    if sum(len(v) for v in by_class.values()) < tray.cells:
        raise LayoutError(f"{tray.name} needs {tray.cells} part types, but fewer fit its cells")

    def area(part: PartAsset) -> float:
        return max(p.footprint_area for p in part.usable_poses)

    def draw(pool_by_class: dict[str, list[PartAsset]]) -> list[PartAsset]:
        classes = list(pool_by_class)
        rng.shuffle(classes)
        chosen: list[PartAsset] = []
        while len(chosen) < tray.cells:
            progressed = False
            for cls in classes:
                pool = [v for v in pool_by_class[cls] if v not in chosen]
                if pool and len(chosen) < tray.cells:
                    chosen.append(pool[int(rng.integers(len(pool)))])
                    progressed = True
            if not progressed:
                raise LayoutError(f"{tray.name}: not enough part types")
        return chosen

    chosen = draw(by_class)
    for _ in range(60):
        if sum(area(p) for p in chosen) <= budget:
            break
        chosen = draw(by_class)
    else:
        # Fall back to the smallest variant of every class, then the smallest remaining types.
        smallest = sorted((min(v, key=area) for v in by_class.values()), key=area)
        rest = sorted((p for v in by_class.values() for p in v if p not in smallest), key=area)
        chosen = (smallest + rest)[: tray.cells]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def _pose_weights(part: PartAsset) -> tuple[list[int], np.ndarray]:
    indices = [i for i, pose in enumerate(part.poses) if pose.usable]
    weights = np.array([part.poses[i].prob for i in indices])
    return indices, weights / weights.sum()


def _pack(
    rng: np.random.Generator,
    parts: list[PartAsset],
    region: tuple[float, float],
    config: SamplerConfig,
    target_area: float | None,
    cells: list[int],
    capacity: Sequence[int] | None = None,
) -> list[tuple[PartAsset, int, float, float, float, int]] | None:
    """Place one instance of each of ``parts`` (and extras up to ``target_area`` [m^2]) inside ``region``.

    ``capacity[i]`` is the most instances of ``parts[i]`` there may be. ``None`` means no limit.

    Returns ``(part, pose_index, x, y, yaw, cell)`` in the container frame, or ``None`` when a required part
    does not fit.
    """
    placed: list[tuple[PartAsset, int, float, float, float, int]] = []
    polygons: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    radii: list[float] = []

    def try_place(part: PartAsset, cell: int) -> bool:
        indices, weights = _pose_weights(part)
        for _ in range(config.placement_tries):
            pose_index = int(rng.choice(indices, p=weights))
            pose = part.poses[pose_index]
            yaw = float(rng.uniform(-np.pi, np.pi))
            polygon0 = geo.transform(pose.footprint, yaw)
            hx = region[0] - float(np.abs(polygon0[:, 0]).max())
            hy = region[1] - float(np.abs(polygon0[:, 1]).max())
            if hx <= 0.0 or hy <= 0.0:
                continue
            x, y = float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))
            polygon = polygon0 + np.array([x, y])
            if centers:
                near = np.linalg.norm(np.asarray(centers) - [x, y], axis=1) < (
                    np.asarray(radii) + pose.radius + config.part_gap
                )
                if any(not geo.polygons_clear(polygon, polygons[k], config.part_gap) for k in np.nonzero(near)[0]):
                    continue
            placed.append((part, pose_index, x, y, yaw, cell))
            polygons.append(polygon)
            centers.append(np.array([x, y]))
            radii.append(pose.radius)
            return True
        return False

    # Largest first: the big parts are hardest to place.
    order = sorted(range(len(parts)), key=lambda i: -max(p.footprint_area for p in parts[i].usable_poses))
    for index in order:
        if not try_place(parts[index], cells[index]):
            return None
    if target_area is None:
        return placed
    covered = sum(geo.area(p) for p in polygons)
    counts = [1] * len(parts)
    limit = [config.max_parts] * len(parts) if capacity is None else list(capacity)
    while covered < target_area and len(placed) < config.max_parts:
        open_cells = [i for i in range(len(parts)) if counts[i] < limit[i]]
        if not open_cells:
            break
        cell = open_cells[int(rng.integers(len(open_cells)))]
        if not try_place(parts[cell], cell):
            break
        counts[cell] += 1
        covered += geo.area(polygons[-1])
    return placed


def _feasible_poses(
    rng: np.random.Generator,
    outline: np.ndarray,
    reach_points: np.ndarray,
    config: SamplerConfig,
    count: int,
) -> np.ndarray:
    """``[n, 3]`` poses ``(x, y, yaw)`` whose outline fits the table and whose ``reach_points`` are reachable.

    Candidates are drawn uniformly over the table and tested in one batch. A large tray leaves only a small
    region, so a scalar rejection loop would rarely find it.
    """
    center = np.asarray(config.table_center)
    span = np.asarray(config.table_half) - config.table_margin
    xy = rng.uniform(center - span, center + span, size=(count, 2))
    yaw = rng.integers(4, size=count) * np.pi / 2.0 + rng.uniform(-config.yaw_jitter, config.yaw_jitter, count)
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def place(points: np.ndarray) -> np.ndarray:
        x = points[None, :, 0] * cos - points[None, :, 1] * sin + xy[:, :1]
        y = points[None, :, 0] * sin + points[None, :, 1] * cos + xy[:, 1:]
        return np.stack([x, y], axis=-1)

    corners = place(outline)
    inside = np.all(np.abs(corners - center) <= span, axis=(1, 2))
    reach = place(reach_points)
    ok = np.all(np.linalg.norm(reach, axis=-1) <= config.reach_radius, axis=1) & np.all(
        reach[..., 0] >= config.reach_min_x, axis=1
    )
    keep = inside & ok
    return np.column_stack([xy[keep], yaw[keep]])


def _grid_poses(outline: np.ndarray, reach_points: np.ndarray, config: SamplerConfig, step: float = 0.04) -> np.ndarray:
    """Feasible quarter-turn poses on a table grid. Used when random samples miss a tight row."""
    center = np.asarray(config.table_center)
    span = np.asarray(config.table_half) - config.table_margin
    xs = np.arange(center[0] - span[0], center[0] + span[0] + 1e-9, step)
    ys = np.arange(center[1] - span[1], center[1] + span[1] + 1e-9, step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    base = np.column_stack([xx.ravel(), yy.ravel()])
    xy = np.repeat(base, 4, axis=0)
    yaw = np.tile(np.arange(4) * (np.pi / 2.0), len(base))
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def place(points: np.ndarray) -> np.ndarray:
        x = points[None, :, 0] * cos - points[None, :, 1] * sin + xy[:, :1]
        y = points[None, :, 0] * sin + points[None, :, 1] * cos + xy[:, 1:]
        return np.stack([x, y], axis=-1)

    corners = place(outline)
    inside = np.all(np.abs(corners - center) <= span, axis=(1, 2))
    reach = place(reach_points)
    ok = np.all(np.linalg.norm(reach, axis=-1) <= config.reach_radius, axis=1) & np.all(
        reach[..., 0] >= config.reach_min_x, axis=1
    )
    keep = inside & ok
    return np.column_stack([xy[keep], yaw[keep]])


def _sample_pose_pair(
    rng: np.random.Generator,
    container: ContainerAsset,
    tray: TrayAsset,
    config: SamplerConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Table poses of the container and the tray that fit the table and the reach, and keep a gap."""
    region = container_pick_region(container, config)
    if region[0] <= 0.0 or region[1] <= 0.0:
        raise LayoutError(f"{container.name} has no usable floor")
    for _ in range(config.pair_pools):
        container_poses = _feasible_poses(
            rng, container.outer_polygon, geo.rectangle(*region), config, config.pair_samples
        )
        tray_poses = _feasible_poses(rng, _tray_outline(tray), tray.cell_centers, config, config.pair_samples)
        if len(container_poses) == 0:
            raise LayoutError(f"no table pose for {container.name} within reach")
        if len(tray_poses) == 0:
            raise LayoutError(f"no table pose for {tray.name} within reach")
        for _ in range(config.pair_tries):
            c = container_poses[int(rng.integers(len(container_poses)))]
            t = tray_poses[int(rng.integers(len(tray_poses)))]
            outline_c = geo.transform(container.outer_polygon, c[2], c[:2])
            outline_t = geo.transform(_tray_outline(tray), t[2], t[:2])
            if geo.polygons_clear(outline_c, outline_t, config.object_gap):
                return tuple(float(v) for v in c), tuple(float(v) for v in t)
    raise LayoutError(f"no table pose for {tray.name} beside {container.name}")


def _tray_outline(tray: TrayAsset) -> np.ndarray:
    hx, hy = tray.outer_half_m
    return geo.rectangle(hx, hy)


def sample_layout(
    rng: np.random.Generator,
    library: AssetLibrary,
    container: str,
    tray: str,
    config: SamplerConfig | None = None,
    cell_types: Sequence[str] | None = None,
    capacity: Sequence[int] | None = None,
) -> SortingLayout:
    """Random layout for one container and one tray.

    The tray has one distinct part type per cell. The container holds at least one part of every type and more
    until the footprints fill a random share of the usable floor. No two footprints, and neither the container
    and the tray, ever overlap.

    Args:
        rng: Random generator. The layout is a function of its state.
        library: Asset library with the parts, containers, and trays.
        container: Container name.
        tray: Tray name.
        config: Sampler limits.
        cell_types: Fixed part type of every cell, in cell order. Random when ``None``.
        capacity: Most parts of each cell's type, in cell order (the bodies a stage has). ``None`` is no limit.

    Raises:
        LayoutError: No valid layout exists within the attempt budget.
    """
    config = config or SamplerConfig()
    container_asset, tray_asset = library.containers[container], library.trays[tray]
    region = container_pick_region(container_asset, config)
    area = 4.0 * region[0] * region[1]
    # The packing is in the container frame, so it does not depend on where the container stands.
    container_pose, tray_pose = _sample_pose_pair(rng, container_asset, tray_asset, config)
    for _ in range(config.layout_tries):
        if cell_types is None:
            types = choose_types(rng, library, tray_asset, budget=config.type_budget * area)
        else:
            types = _fixed_types(library, tray_asset, cell_types)
        target = float(rng.uniform(*config.fill_range)) * area
        packed = _pack(rng, types, region, config, target, list(range(tray_asset.cells)), capacity)
        if packed is not None:
            return _assemble(container_asset, tray_asset, container_pose, tray_pose, types, packed, config)
    raise LayoutError(f"no packing of {tray_asset.cells} part types on the floor of {container}")


def _fixed_types(library: AssetLibrary, tray: TrayAsset, cell_types: Sequence[str]) -> list[PartAsset]:
    if len(cell_types) != tray.cells or len(set(cell_types)) != len(cell_types):
        raise LayoutError(f"{tray.name} needs {tray.cells} distinct part types, got {list(cell_types)}")
    parts = [library.parts[name] for name in cell_types]
    for part in parts:
        if not part.usable or not fits_cell(part, tray):
            raise LayoutError(f"{part.name} is not usable in a cell of {tray.name}")
    return parts


def _assemble(
    container: ContainerAsset,
    tray: TrayAsset,
    container_pose: tuple[float, float, float],
    tray_pose: tuple[float, float, float],
    types: list[PartAsset],
    packed: list[tuple[PartAsset, int, float, float, float, int]],
    config: SamplerConfig,
) -> SortingLayout:
    parts = []
    for part, pose_index, x, y, yaw, cell in packed:
        pose = part.poses[pose_index]
        wx, wy = geo.transform(np.array([[x, y]]), container_pose[2], container_pose[:2])[0]
        world_yaw = geo.wrap_angle(yaw + container_pose[2])
        z = config.table_top_z + container.floor_z + pose.rest_z + config.settle_gap
        parts.append(
            PartInstance(
                name=part.name,
                pose_index=pose_index,
                position_m=(float(wx), float(wy), float(z)),
                yaw=world_yaw,
                quat_wxyz=part_quat(pose, world_yaw),
                cell=cell,
            )
        )
    return SortingLayout(
        container=container.name,
        tray=tray.name,
        container_pose=container_pose,
        tray_pose=tray_pose,
        parts=tuple(parts),
        cell_types=tuple(part.name for part in types),
        table_top_z=config.table_top_z,
    )


# Cells of each grid. The name is the plan's name: 1x2 is two cells on the 150 mm side and one on the 200 mm side.
GRID_CELLS = {"1x1": 1, "1x2": 2, "2x2": 4, "2x3": 6}
TYPE_COUNT = (3, 6)
QUOTA_COUNT = (4, 6)
POPULATION_COUNT = (20, 40)
COMBO_TRIES = 40


def tray_grid(name: str) -> str:
    """Grid name of a tray key, such as ``1x2`` from ``parts_tray_1x2_green``."""
    return name.removeprefix("parts_tray_").rsplit("_", 1)[0]


def trays_by_grid(library: AssetLibrary) -> dict[str, list[str]]:
    groups = {grid: [] for grid in GRID_CELLS}
    for name in sorted(library.trays):
        grid = tray_grid(name)
        if grid in groups:
            groups[grid].append(name)
    return groups


def _world_aabbs(outline: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds ``(min, max)`` of ``outline`` at each ``(x, y, yaw)`` pose."""
    yaw = poses[:, 2]
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    local = np.asarray(outline, dtype=np.float64)
    x = local[None, :, 0] * cos - local[None, :, 1] * sin + poses[:, None, 0]
    y = local[None, :, 0] * sin + local[None, :, 1] * cos + poses[:, None, 1]
    corners = np.stack([x, y], axis=-1)
    return corners.min(axis=1), corners.max(axis=1)


def _aabb_clear(cmin: np.ndarray, cmax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, gap: float) -> np.ndarray:
    """Which candidate boxes are at least ``gap`` [m] from every blocker. A separated box is a separated rectangle."""
    dx = np.maximum(bmin[None, :, 0] - cmax[:, None, 0], cmin[:, None, 0] - bmax[None, :, 0])
    dy = np.maximum(bmin[None, :, 1] - cmax[:, None, 1], cmin[:, None, 1] - bmax[None, :, 1])
    distance = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return np.all(distance >= gap - 1e-9, axis=1)


def _pack_axis_aligned(
    rng: np.random.Generator,
    container_poses: np.ndarray,
    container_bounds: tuple[np.ndarray, np.ndarray],
    tray_poses: Sequence[np.ndarray],
    tray_bounds: Sequence[tuple[np.ndarray, np.ndarray]],
    gap: float,
    *,
    container_limit: int = 160,
    window: int = 12,
    node_budget: int = 1200,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """One container pose and one pose per tray. Trays sit in the first gap that clears the bin and each other."""
    cmin, cmax = container_bounds
    order = rng.permutation(len(container_poses))
    budget = [node_budget]
    for container_index in order[:container_limit]:
        if budget[0] <= 0:
            break
        blocks_min = [cmin[container_index]]
        blocks_max = [cmax[container_index]]
        chosen: list[np.ndarray] = []
        if _seat_trays(rng, tray_poses, tray_bounds, blocks_min, blocks_max, gap, 0, chosen, budget, window):
            return container_poses[container_index], chosen
    return None


def _seat_trays(rng, tray_poses, tray_bounds, blocks_min, blocks_max, gap, index, chosen, budget, window) -> bool:
    if index == len(tray_poses):
        return True
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    tmin, tmax = tray_bounds[index]
    clear = _aabb_clear(tmin, tmax, np.stack(blocks_min), np.stack(blocks_max), gap)
    options = np.flatnonzero(clear)
    if len(options) == 0:
        return False
    center = (np.stack(blocks_min) + np.stack(blocks_max)).mean(axis=0)
    tray_center = (tmin[options] + tmax[options]) * 0.5
    nearest = options[np.argsort(np.linalg.norm(tray_center - center, axis=1))[:window]]
    rng.shuffle(nearest)
    for pick in nearest:
        blocks_min.append(tmin[pick])
        blocks_max.append(tmax[pick])
        chosen.append(tray_poses[index][pick])
        if _seat_trays(rng, tray_poses, tray_bounds, blocks_min, blocks_max, gap, index + 1, chosen, budget, window):
            return True
        chosen.pop()
        blocks_min.pop()
        blocks_max.pop()
    return False


def _sample_furniture(
    rng: np.random.Generator,
    container: ContainerAsset,
    trays: Sequence[TrayAsset],
    config: SamplerConfig,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Table poses of the container and every tray. A combination that does not pack is an error.

    The row is packed on quarter turns, where a box gap is the real gap. Each body then takes its own heading
    offset inside ``yaw_jitter``, so the published pose is not left parallel to the table edge.
    """
    region = container_pick_region(container, config)
    if region[0] <= 0.0 or region[1] <= 0.0:
        raise LayoutError(f"{container.name} has no usable floor")
    packing = replace(config, yaw_jitter=0.0)
    samples = min(config.pair_samples, 4000)
    floor = geo.rectangle(*region)
    for _ in range(config.pair_pools):
        container_poses = _feasible_poses(rng, container.outer_polygon, floor, packing, samples)
        tray_pose_sets = [
            _feasible_poses(rng, _tray_outline(tray), tray.cell_centers, packing, samples) for tray in trays
        ]
        if len(container_poses) == 0 or any(len(poses) == 0 for poses in tray_pose_sets):
            continue
        tray_bounds = [_world_aabbs(_tray_outline(tray), poses) for tray, poses in zip(trays, tray_pose_sets)]
        packed = _pack_axis_aligned(
            rng,
            container_poses,
            _world_aabbs(container.outer_polygon, container_poses),
            tray_pose_sets,
            tray_bounds,
            config.object_gap,
        )
        if packed is not None:
            container_pose, chosen = packed
            return _with_yaw_jitter(rng, container, trays, container_pose, chosen, config, region)
    packed = _pack_flush(rng, container, trays, packing, floor)
    if packed is not None:
        container_pose, chosen = packed
        return _with_yaw_jitter(rng, container, trays, container_pose, chosen, config, region)
    names = ", ".join(tray.name for tray in trays)
    raise LayoutError(f"no packing of {names} beside {container.name}")


def _filter_poses(
    xy: np.ndarray, yaw: np.ndarray, outline: np.ndarray, reach_points: np.ndarray, config: SamplerConfig
) -> np.ndarray:
    """Mask of poses whose outline is on the table and whose reach points the arm can touch."""
    center = np.asarray(config.table_center)
    span = np.asarray(config.table_half) - config.table_margin
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def place(points: np.ndarray) -> np.ndarray:
        x = points[None, :, 0] * cos - points[None, :, 1] * sin + xy[:, :1]
        y = points[None, :, 0] * sin + points[None, :, 1] * cos + xy[:, 1:]
        return np.stack([x, y], axis=-1)

    corners = place(outline)
    inside = np.all(np.abs(corners - center) <= span, axis=(1, 2))
    reach = place(reach_points)
    ok = np.all(np.linalg.norm(reach, axis=-1) <= config.reach_radius, axis=1) & np.all(
        reach[..., 0] >= config.reach_min_x, axis=1
    )
    return inside & ok


def _pack_flush(
    rng: np.random.Generator,
    container: ContainerAsset,
    trays: Sequence[TrayAsset],
    config: SamplerConfig,
    floor: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Place each tray against the bin. The gap is the box gap, so a thin strip beside a wide bin is not missed."""
    container_poses = _grid_poses(container.outer_polygon, floor, config, step=0.01)
    if len(container_poses) == 0:
        return None
    pairs = _flush_pairs(container_poses, container.outer_polygon, trays[0], config)
    if len(pairs) == 0:
        return None
    order = rng.permutation(len(pairs))
    budget = [360]
    for index in order[:240]:
        if budget[0] <= 0:
            break
        container_pose, first = pairs[index]
        chosen = [first]
        blocks = [
            _world_aabbs(container.outer_polygon, container_pose[None, :]),
            _world_aabbs(_tray_outline(trays[0]), first[None, :]),
        ]
        if _seat_flush(rng, trays[1:], blocks, chosen, config, budget):
            return container_pose, chosen
    return None


def _seat_flush(rng, trays, blocks, chosen, config, budget) -> bool:
    """Add a pose for every remaining tray in a gap beside the furniture already placed."""
    if not trays:
        return True
    if budget[0] <= 0:
        return False
    options = _flush_next(blocks, trays[0], config)
    if len(options) > 8:
        picks = rng.choice(len(options), 8, replace=False)
        options = [options[int(i)] for i in picks]
    for pose in options:
        budget[0] -= 1
        chosen.append(pose)
        blocks.append(_world_aabbs(_tray_outline(trays[0]), pose[None, :]))
        if _seat_flush(rng, trays[1:], blocks, chosen, config, budget):
            return True
        chosen.pop()
        blocks.pop()
    return False


def _flush_pairs(
    container_poses: np.ndarray,
    container_outline: np.ndarray,
    tray: TrayAsset,
    config: SamplerConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Container poses paired with a tray sitting on one side, slid along that side."""
    outline = _tray_outline(tray)
    cmin, cmax = _world_aabbs(container_outline, container_poses)
    cx = (cmin[:, 0] + cmax[:, 0]) * 0.5
    cy = (cmin[:, 1] + cmax[:, 1]) * 0.5
    shifts = np.linspace(-0.48, 0.48, 17)
    gap = config.object_gap
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for yaw in np.arange(4) * (np.pi / 2.0):
        probe = _world_aabbs(outline, np.array([[0.0, 0.0, yaw]]))
        thx = float(probe[1][0, 0] - probe[0][0, 0]) * 0.5
        thy = float(probe[1][0, 1] - probe[0][0, 1]) * 0.5
        sides = (
            (cmin[:, 0] - gap - thx, cy[None, :] + shifts[:, None]),
            (cmax[:, 0] + gap + thx, cy[None, :] + shifts[:, None]),
            (cx[None, :] + shifts[:, None], cmin[:, 1] - gap - thy),
            (cx[None, :] + shifts[:, None], cmax[:, 1] + gap + thy),
        )
        shape = (len(shifts), len(cx))
        for x_row, y_row in sides:
            x_grid = np.broadcast_to(np.asarray(x_row), shape)
            y_grid = np.broadcast_to(np.asarray(y_row), shape)
            for row in range(len(shifts)):
                xy = np.column_stack([x_grid[row], y_grid[row]])
                yaw_row = np.full(len(cx), yaw)
                kept = np.flatnonzero(_filter_poses(xy, yaw_row, outline, tray.cell_centers, config))
                keep = _even(kept, 4)
                for i in keep:
                    pairs.append((container_poses[i], np.array([xy[i, 0], xy[i, 1], yaw])))
    return pairs


def _even(indices: np.ndarray, count: int) -> np.ndarray:
    """``count`` entries spread through ``indices``, so a later shift is not dropped for an earlier one."""
    if len(indices) <= count:
        return indices
    return indices[np.linspace(0, len(indices) - 1, count, dtype=int)]


def _flush_next(
    blocks: list[tuple[np.ndarray, np.ndarray]], tray: TrayAsset, config: SamplerConfig
) -> list[np.ndarray]:
    """Tray poses beside any placed box that also clear the others. A pocket between two boxes stays usable."""
    outline = _tray_outline(tray)
    found: list[np.ndarray] = []
    for bounds in blocks:
        umin, umax = bounds[0][0], bounds[1][0]
        center = (umin + umax) * 0.5
        half = (umax - umin) * 0.5
        box = geo.rectangle(float(half[0]), float(half[1]))
        pose = np.array([[float(center[0]), float(center[1]), 0.0]])
        found.extend(tray_pose for _, tray_pose in _flush_pairs(pose, box, tray, config))
    if not found:
        return []
    poses = np.stack(found)
    tmin, tmax = _world_aabbs(outline, poses)
    bmin = np.stack([bounds[0][0] for bounds in blocks])
    bmax = np.stack([bounds[1][0] for bounds in blocks])
    clear = np.flatnonzero(_aabb_clear(tmin, tmax, bmin, bmax, config.object_gap))
    return [poses[i] for i in clear[:32]]


def _with_yaw_jitter(rng, container, trays, container_pose, chosen, config, region):
    """Quarter-turn packing plus one heading offset per body, of at most ``config.yaw_jitter`` [rad]."""
    outlines = [container.outer_polygon, *[_tray_outline(tray) for tray in trays]]
    reach = [geo.rectangle(*region), *[np.asarray(tray.cell_centers, dtype=np.float64) for tray in trays]]
    jittered = _jitter_headings(rng, outlines, reach, [container_pose, *chosen], config)
    return _pose_tuple(jittered[0], jittered[1:])


def _jitter_headings(rng, outlines, reach, poses, config) -> list[np.ndarray]:
    """Apply each body's own offset, sliding it a few centimetres if the seated spot has no room to turn."""
    poses = [np.asarray(pose, dtype=np.float64).copy() for pose in poses]
    worlds = [
        geo.transform(outline, float(pose[2]), (float(pose[0]), float(pose[1])))
        for outline, pose in zip(outlines, poses)
    ]
    for index in rng.permutation(len(poses)):
        base = poses[index]
        sign = float(rng.choice(np.array([-1.0, 1.0])))
        offset = sign * float(config.yaw_jitter)
        chosen = _fit_heading(base, offset, index, outlines, reach, worlds, config)
        if chosen is None:
            continue
        poses[index] = chosen
        worlds[index] = geo.transform(outlines[index], float(chosen[2]), (float(chosen[0]), float(chosen[1])))
    return poses


def _fit_heading(base, offset, index, outlines, reach, worlds, config) -> np.ndarray | None:
    """Largest fraction of ``offset`` that stays on the table and clear of the other bodies.

    The seated row is against the table edge, so the same heading is also tried a few centimetres away.
    """
    shifts = np.linspace(-0.18, 0.18, 10)
    pairs = sorted(
        ((float(dx), float(dy)) for dx in shifts for dy in shifts),
        key=lambda item: item[0] * item[0] + item[1] * item[1],
    )
    for scale in (1.0, 0.8, 0.6):
        yaw = geo.wrap_angle(float(base[2]) + offset * scale)
        for dx, dy in pairs:
            trial = np.array([float(base[0]) + float(dx), float(base[1]) + float(dy), yaw], dtype=np.float64)
            if _heading_clear(trial, index, outlines, reach, worlds, config):
                return trial
    return None


def _heading_clear(trial, index, outlines, reach, worlds, config) -> bool:
    if not _filter_poses(trial[:2][None, :], np.array([trial[2]]), outlines[index], reach[index], config)[0]:
        return False
    world = geo.transform(outlines[index], float(trial[2]), (float(trial[0]), float(trial[1])))
    return all(geo.polygons_clear(world, other, config.object_gap) for j, other in enumerate(worlds) if j != index)


def _pose_tuple(
    container_pose: np.ndarray, chosen: list[np.ndarray]
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    return tuple(float(v) for v in container_pose), tuple(tuple(float(v) for v in pose) for pose in chosen)


def _candidate_types(library: AssetLibrary) -> dict[str, list[PartAsset]]:
    """Usable variants grouped by class. A variant qualifies when it fits at least one tray cell."""
    trays = list(library.trays.values())
    by_class: dict[str, list[PartAsset]] = {}
    for cls, variants in library.families().items():
        fitting = [variant for variant in variants if any(fits_cell(variant, tray) for tray in trays)]
        if fitting:
            by_class[cls] = fitting
    return by_class


def _draw_types(rng: np.random.Generator, library: AssetLibrary, count: int) -> list[PartAsset]:
    by_class = _candidate_types(library)
    if sum(len(variants) for variants in by_class.values()) < count:
        raise LayoutError(f"need {count} part types, but fewer fit a cell")
    classes = list(by_class)
    rng.shuffle(classes)
    chosen: list[PartAsset] = []
    while len(chosen) < count:
        progressed = False
        for cls in classes:
            pool = [variant for variant in by_class[cls] if variant not in chosen]
            if pool and len(chosen) < count:
                chosen.append(pool[int(rng.integers(len(pool)))])
                progressed = True
        if not progressed:
            raise LayoutError(f"need {count} part types")
    return chosen


def _draw_trays(rng: np.random.Generator, library: AssetLibrary, cells_needed: int) -> list[str] | None:
    """A random tray combination whose cell count is between ``cells_needed`` and 6."""
    groups = trays_by_grid(library)
    grids = [grid for grid, names in groups.items() if names]
    if not grids:
        return None
    for _ in range(80):
        chosen: list[str] = []
        total = 0
        for _ in range(6):
            if cells_needed <= total <= 6 and (total == 6 or rng.random() < 0.55):
                break
            options = [grid for grid in grids if total + GRID_CELLS[grid] <= 6]
            if cells_needed > total:
                growing = [grid for grid in options if total + GRID_CELLS[grid] >= cells_needed or GRID_CELLS[grid] < 6]
                options = growing or options
            if not options:
                break
            # Larger grids first, so a 6-cell task tries one 2x3 before a row of trays that will not pack.
            weights = np.array([GRID_CELLS[grid] for grid in options], dtype=np.float64)
            grid = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            pool = [name for name in groups[grid] if name not in chosen]
            if not pool:
                break
            chosen.append(pool[int(rng.integers(len(pool)))])
            total += GRID_CELLS[grid]
        if cells_needed <= total <= 6 and chosen:
            return chosen
    return None


def _assign_cells(
    rng: np.random.Generator, types: Sequence[PartAsset], trays: Sequence[TrayAsset], quota: int
) -> tuple[str, ...] | None:
    """Type of every cell, or ``None`` when the types do not fit. Empty cells are ``""``."""
    cells = [(index, cell, tray) for index, tray in enumerate(trays) for cell in range(tray.cells)]
    needed = max(len(types), quota)
    if len(cells) < needed:
        return None
    used: set[int] = set()
    assignment: dict[int, str] = {}

    def open_cells(part: PartAsset) -> list[int]:
        return [index for index, (_, _, tray) in enumerate(cells) if index not in used and fits_cell(part, tray)]

    order = sorted(range(len(types)), key=lambda index: -max(pose.footprint_area for pose in types[index].usable_poses))
    for index in order:
        options = open_cells(types[index])
        if not options:
            return None
        pick = options[int(rng.integers(len(options)))]
        used.add(pick)
        assignment[pick] = types[index].name
    extras = needed - len(types)
    for _ in range(extras):
        placed = False
        for index in rng.permutation(len(types)):
            options = open_cells(types[int(index)])
            if not options:
                continue
            pick = options[int(rng.integers(len(options)))]
            used.add(pick)
            assignment[pick] = types[int(index)].name
            placed = True
            break
        if not placed:
            return None
    return tuple(assignment.get(index, "") for index in range(len(cells)))


def _population(
    rng: np.random.Generator,
    types: Sequence[PartAsset],
    container: ContainerAsset,
    container_pose: tuple[float, float, float],
    config: SamplerConfig,
    count: int,
) -> tuple[PartInstance, ...]:
    """``count`` parts of the given types, dropped in name only. Physics replaces these poses."""
    names = [part.name for part in types]
    by_name = {part.name: part for part in types}
    multiset = list(names)
    while len(multiset) < count:
        multiset.append(names[int(rng.integers(len(names)))])
    rng.shuffle(multiset)
    region = container_pick_region(container, config)
    parts = []
    for name in multiset:
        part = by_name[name]
        indices, weights = _pose_weights(part)
        pose_index = int(rng.choice(indices, p=weights))
        pose = part.poses[pose_index]
        yaw = float(rng.uniform(-np.pi, np.pi))
        hx = max(region[0] * 0.6, 1e-3)
        hy = max(region[1] * 0.6, 1e-3)
        local = np.array([[float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy))]])
        wx, wy = geo.transform(local, container_pose[2], container_pose[:2])[0]
        world_yaw = geo.wrap_angle(yaw + container_pose[2])
        z = config.table_top_z + container.floor_z + pose.rest_z + config.settle_gap
        parts.append(
            PartInstance(
                name=name,
                pose_index=pose_index,
                position_m=(float(wx), float(wy), float(z)),
                yaw=world_yaw,
                quat_wxyz=part_quat(pose, world_yaw),
                cell=-1,
            )
        )
    return tuple(parts)


def sample_sorting_task(
    rng: np.random.Generator,
    library: AssetLibrary,
    container: str,
    config: SamplerConfig | None = None,
) -> SortingLayout:
    """One task: 3-6 types, quota 4-6, a tray combination, and 20-40 parts in the bin.

    The bin poses are a placeholder. The benchmark builder replaces them with settled drops. A combination
    that does not pack on the table, or that cannot seat the drawn types, is discarded and another is drawn.

    Raises:
        LayoutError: No combination packs within the attempt budget.
    """
    config = config or SamplerConfig()
    container_asset = library.containers[container]
    type_count = int(rng.integers(TYPE_COUNT[0], TYPE_COUNT[1] + 1))
    quota = int(rng.integers(QUOTA_COUNT[0], QUOTA_COUNT[1] + 1))
    population = int(rng.integers(POPULATION_COUNT[0], POPULATION_COUNT[1] + 1))
    types = _draw_types(rng, library, type_count)
    needed = max(type_count, quota)
    for _ in range(COMBO_TRIES):
        names = _draw_trays(rng, library, needed)
        if not names:
            continue
        trays = [library.trays[name] for name in names]
        cell_types = _assign_cells(rng, types, trays, quota)
        if cell_types is None:
            continue
        try:
            container_pose, tray_poses = _sample_furniture(rng, container_asset, trays, config)
        except LayoutError:
            continue
        parts = _population(rng, types, container_asset, container_pose, config, population)
        return SortingLayout(
            container=container,
            tray=names[0],
            container_pose=container_pose,
            tray_pose=tray_poses[0],
            parts=parts,
            cell_types=cell_types,
            table_top_z=config.table_top_z,
            extra={"pile": True, "settled": False},
            trays=tuple(names),
            tray_poses=tray_poses,
            quota=quota,
            population=population,
        )
    raise LayoutError(f"no tray combination for {type_count} types and quota {quota} on {container}")
