# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout sampler: geometry against shapely, scale rules, and overlap-free layouts for every container and tray.

``SORTING_LAYOUT_SEEDS`` sets the seeds per container and tray pair (default 3). Run with 200 or more for a
soak test. ``SORTING_ASSETS_BUNDLE`` points at another bundle than the master library.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

shapely_geometry = pytest.importorskip("shapely.geometry")
Polygon, box, Point = shapely_geometry.Polygon, shapely_geometry.box, shapely_geometry.Point

from franka_sorting.layout import geometry as geo  # noqa: E402
from franka_sorting.layout import part_scale  # noqa: E402
from franka_sorting.layout.catalog import load_library  # noqa: E402
from franka_sorting.layout.instruction import (  # noqa: E402
    build_instruction,
    container_name as spoken_container,
    display_name,
    view_cells,
    view_shape,
)
from franka_sorting.layout.sampler import (  # noqa: E402
    GRID_CELLS,
    SamplerConfig,
    choose_types,
    container_pick_region,
    fits_cell,
    sample_sorting_task,
    tray_grid,
)
from franka_sorting.layout.validate import part_polygon, validate_layout  # noqa: E402

ENVHUB = Path(__file__).resolve().parents[2] / "envhub" / "environments"
BUNDLE = Path(os.environ.get("SORTING_ASSETS_BUNDLE", ENVHUB / "franka-sorting-parts-v2"))
SEEDS = int(os.environ.get("SORTING_LAYOUT_SEEDS", "3"))


@pytest.fixture(scope="module")
def library():
    if not (BUNDLE / "assets" / "catalog.json").is_file():
        pytest.skip(f"no asset bundle at {BUNDLE}")
    return load_library(BUNDLE)


def _random_convex(rng: np.random.Generator, count: int = 12) -> np.ndarray:
    return geo.convex_hull(rng.normal(size=(count, 2)) * rng.uniform(0.01, 0.05, size=2))


def test_separation_matches_shapely_distance():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = _random_convex(rng)
        b = geo.transform(_random_convex(rng), rng.uniform(-np.pi, np.pi), rng.uniform(-0.12, 0.12, size=2))
        gap = geo.separation(a, b)
        reference = Polygon(a).distance(Polygon(b))
        if reference > 0.0:
            assert gap == pytest.approx(reference, abs=1e-9)
        else:
            assert gap <= 1e-12


def test_enclosing_polygon_contains_its_input_and_is_small():
    rng = np.random.default_rng(1)
    for _ in range(100):
        hull = _random_convex(rng, 60)
        outer = geo.enclosing_polygon(hull, 10)
        assert len(outer) <= 10
        assert Polygon(outer).buffer(1e-9).contains(Polygon(hull))


def test_min_width_and_area_match_shapely():
    rng = np.random.default_rng(2)
    for _ in range(50):
        hull = _random_convex(rng)
        assert geo.area(hull) == pytest.approx(Polygon(hull).area)
        long_side, short_side, _ = geo.min_area_rectangle(hull)
        reference = Polygon(hull).minimum_rotated_rectangle
        sides = sorted(np.linalg.norm(np.diff(np.asarray(reference.exterior.coords), axis=0), axis=1)[:2], reverse=True)
        assert long_side * short_side <= sides[0] * sides[1] * 1.02


def test_scale_rules_reject_unfit_poses():
    ok = part_scale.NativePose(prob=1.0, height=0.01, grip_width=0.02, rect_long=0.04, rect_short=0.02)
    assert part_scale.pose_usable(ok, 1.0)
    assert not part_scale.pose_usable(ok, 8.0)  # too tall, and the footprint leaves the cell
    thin = part_scale.NativePose(prob=1.0, height=0.002, grip_width=0.02, rect_long=0.02, rect_short=0.01)
    assert not part_scale.pose_usable(thin, 1.0)
    assert part_scale.pose_usable(thin, 3.0)
    assert part_scale.variant_usable([ok], 1.0, native_mass_kg=1.0)
    assert not part_scale.variant_usable([ok], 1.0, native_mass_kg=-1.0)
    scale, flags = part_scale.choose_family_scale({"thin": [thin]})
    assert scale == pytest.approx(2.5)  # the smallest grid scale with a 5 mm thickness
    assert flags == {"thin": True}
    wide = part_scale.NativePose(prob=1.0, height=0.004, grip_width=0.05, rect_long=0.07, rect_short=0.05)
    margin, height_max = part_scale.pose_limits("plate_cutout_large_v3")
    assert part_scale.pose_usable(wide, 2.5, margin=margin, height_max=height_max)


def test_library_shape(library):
    assert len(library.containers) == 5
    assert len(library.trays) == 32
    assert {tray_grid(name) for name in library.trays} == set(GRID_CELLS)
    usable = [part for part in library.parts.values() if part.usable]
    assert len(usable) >= 16
    for part in usable:
        assert part.mass_kg <= part_scale.MASS_MAX_KG + 0.05
        assert part.usable_poses
        _, height_max = part_scale.pose_limits(part.name)
        for pose in part.usable_poses:
            assert part_scale.THICKNESS_MIN_M <= pose.height <= height_max + 1e-9
            def _inside(value: float) -> bool:
                return part_scale.GRIP_WIDTH_MIN_M <= value <= part_scale.CLOSING_WIDTH_MAX_M

            assert _inside(pose.grip_width) or _inside(pose.height)
            assert len(pose.footprint) <= 10


def test_every_tray_has_enough_fitting_types(library):
    for tray in library.trays.values():
        fitting = {part.name for part in library.parts.values() if part.usable and fits_cell(part, tray)}
        classes = {library.parts[name].cls for name in fitting}
        assert len(fitting) >= tray.cells, tray.name
        assert len(classes) >= 2, tray.name


def test_choose_types_are_distinct_and_fit(library):
    rng = np.random.default_rng(3)
    for tray in library.trays.values():
        for _ in range(5):
            types = choose_types(rng, library, tray, budget=0.45 * 0.2)
            names = [part.name for part in types]
            assert len(names) == len(set(names)) == tray.cells
            assert all(fits_cell(part, tray) for part in types)


def test_cell_view_is_a_permutation_for_every_tray_and_yaw(library):
    for tray in library.trays.values():
        for quarter in range(4):
            yaw = quarter * np.pi / 2.0 + 0.05
            cells = view_cells(tray, yaw)
            assert len({tuple(cell) for cell in cells}) == tray.cells
            rows, columns = view_shape(tray, yaw)
            assert rows * columns == tray.cells
            assert cells.min() == 1


def _all_pairs(library):
    return [(c, t) for c in library.containers for t in library.trays]


def test_sorting_tasks_pack_and_state_the_quota(library):
    """A task draws 3-6 types, a quota of 4-6, enough cells, and 20-40 parts."""
    config = SamplerConfig()
    table = box(
        config.table_center[0] - config.table_half[0] + config.table_margin,
        config.table_center[1] - config.table_half[1] + config.table_margin,
        config.table_center[0] + config.table_half[0] - config.table_margin,
        config.table_center[1] + config.table_half[1] - config.table_margin,
    )
    for container_name in library.containers:
        for seed in range(SEEDS):
            layout = sample_sorting_task(np.random.default_rng(seed), library, container_name, config)
            context = f"{container_name} seed {seed}"
            assert validate_layout(layout, library, config) == [], context
            types = {name for name in layout.cell_types if name}
            assert 3 <= len(types) <= 6, context
            assert 4 <= layout.quota <= 6, context
            cells = len(layout.cell_types)
            assert max(len(types), layout.quota) <= cells <= 6, context
            assert layout.population == len(layout.parts)
            assert 20 <= len(layout.parts) <= 40, context
            outline_c = Polygon(
                geo.transform(library.containers[container_name].outer_polygon, layout.container_pose[2], layout.container_pose[:2])
            )
            assert table.contains(outline_c), context
            for name, pose in zip(layout.tray_list(), layout.tray_pose_list()):
                tray = library.trays[name]
                outline = Polygon(geo.transform(geo.rectangle(*tray.outer_half_m), pose[2], pose[:2]))
                assert table.contains(outline), context
                assert outline.distance(outline_c) >= config.object_gap - 1e-9, context
            text = build_instruction(layout, library.trays[layout.tray], library)
            assert f"at least {layout.quota}" in text
            assert spoken_container(layout.container) in text
            assert re.search(r"\bv\d+\b", text) is None
            assert "1 rows" not in text and "1 columns" not in text
            for name in types:
                assert display_name(name) in text


def _off_cardinal(yaw: float) -> float:
    folded = abs(float(yaw)) % (np.pi / 2.0)
    return min(folded, np.pi / 2.0 - folded)


def test_heading_jitter_is_about_fifteen_degrees_and_keeps_large_trays(library):
    """The heading offset is the configured jitter, and a task still draws the wider tray grids."""
    grids = set()
    for seed in range(6):
        layout = sample_sorting_task(np.random.default_rng(seed), library, "bin_a06")
        grids.update(tray_grid(name) for name in layout.tray_list())
    room = sample_sorting_task(np.random.default_rng(2), library, "container_f24")
    turned = [_off_cardinal(room.container_pose[2])]
    turned.extend(_off_cardinal(pose[2]) for pose in room.tray_pose_list())
    assert max(turned) > np.deg2rad(8.0)
    assert max(turned) <= SamplerConfig().yaw_jitter + 1e-6
    assert grids & {"1x2", "2x2", "2x3"}


def test_layout_is_deterministic_and_seed_dependent(library):
    first = sample_sorting_task(np.random.default_rng(7), library, "bin_b04")
    again = sample_sorting_task(np.random.default_rng(7), library, "bin_b04")
    other = sample_sorting_task(np.random.default_rng(8), library, "bin_b04")
    assert first == again
    assert first != other


def test_rejects_a_container_that_leaves_the_table(library):
    from dataclasses import replace

    layout = sample_sorting_task(np.random.default_rng(0), library, "grey_bin")
    assert validate_layout(layout, library) == []
    off_table = replace(layout, container_pose=(5.0, 0.0, 0.0))
    assert any("leaves the table" in problem for problem in validate_layout(off_table, library))
    assert Point(0, 0).distance(Point(1, 0)) == 1.0
