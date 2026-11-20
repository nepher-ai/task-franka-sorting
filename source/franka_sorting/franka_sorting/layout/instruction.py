# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Names and the instruction text of a sorting task. Cells are named as the robot sees the tray.

The instruction tells the robot where to put each part. It names parts and containers by how they look, and
compartments by where they sit (near, far, left, right). It does not cite a USD file stem or explain the grid.
"""

from __future__ import annotations

import re

import numpy as np

from . import geometry as geo
from .catalog import TrayAsset
from .sampler import SortingLayout

# Spoken name of a part class. Variants of one class differ by a few millimeters and never share a task, so the
# variant index in the file name is not something the scene shows.
_PART_NAMES = {
    "ball_stud": "ball stud",
    "barbed_fitting_small": "small hose barb",
    "bracket_square_window": "window bracket",
    "cap_screw_long": "socket screw",
    "drilled_tube_long": "long drilled tube",
    "drilled_tube_slotted": "short drilled tube",
    "ferrule_small": "flanged sleeve",
    "flat_washer": "washer",
    "hex_collar": "hex collar",
    "hose_barb_hex": "large hose barb",
    "perforated_disc": "perforated disc",
    "pin_anodized_tip": "purple-tipped pin",
    "pin_brass": "brass pin",
    "plate_cutout_large": "cutout plate",
    "probe_needle_tip": "needle probe",
    "stepped_pin": "stepped pin",
    "union_body": "grooved fitting",
    "union_nut_ring": "hex nut",
}
# The third hex-collar variant is turned round. The file name still says hex.
_PART_OVERRIDES = {"hex_collar_v3": "round collar"}
_VARIANT_SUFFIX = re.compile(r"_v\d+$")

# One container is on the table. ``bin_a06`` and ``bin_b04`` are grey plastic bins; ``container_f24`` is blue.
_CONTAINERS = {
    "bin_a06": "grey bin",
    "bin_b04": "grey bin",
    "container_f24": "blue bin",
    "grey_bin": "grey bin",
    "purple_crate": "purple crate",
}


def display_name(variant: str) -> str:
    """Name of a part as it looks.

    ``ball_stud_v1`` is ``ball stud``. ``pin_anodized_tip_v2`` is ``purple-tipped pin``. The round collar
    published as ``hex_collar_v3`` is ``round collar``.

    Args:
        variant: Part variant id, the USD file stem.

    Returns:
        A short name with no file stem and no variant index.
    """
    if variant in _PART_OVERRIDES:
        return _PART_OVERRIDES[variant]
    cls = _VARIANT_SUFFIX.sub("", variant)
    if cls in _PART_NAMES:
        return _PART_NAMES[cls]
    return cls.replace("_", " ")


def container_name(container: str) -> str:
    """Name of a container as it looks.

    Args:
        container: Container id, the USD file stem.

    Returns:
        A short name. Catalog codes such as ``a06`` are not included.
    """
    return _CONTAINERS.get(container, container.replace("_", " "))


def _article(name: str) -> str:
    return "an" if name[0].lower() in "aeiou" else "a"


def _plural(name: str) -> str:
    return name + "s"


def _finish_word(name: str) -> str:
    finish = name.removeprefix("parts_tray_").rsplit("_", 1)[-1]
    return "wooden" if finish == "wood" else finish


def _tray_kind(rows: int, cols: int) -> str:
    """A short shape word, used only when two trays share a finish."""
    if rows == 1 and cols == 1:
        return "small"
    if rows == cols:
        return "square"
    if rows == 1:
        return "wide"
    if cols == 1:
        return "narrow"
    return "large"


def _place(rows: int, cols: int, row: int, col: int) -> tuple[str, str, str]:
    """``(preposition, place, shared noun)``. ``place`` has no article.

    A side-by-side pair is ``on the left``. A deeper grid is ``in the near-left compartment``.
    """
    if rows == 1 and cols == 2:
        return "on", "left" if col == 1 else "right", ""
    if rows == 2 and cols == 1:
        return "in", "nearer" if row == 1 else "farther", "compartment"
    if rows == 1 and cols >= 3:
        side = "left" if col == 1 else "right" if col == cols else "middle"
        return ("in" if side == "middle" else "on"), side, ""
    depth = "near" if row == 1 else "far" if row == rows else "middle"
    if cols == 1:
        return "in", depth, "compartment"
    side = "left" if col == 1 else "right" if col == cols else "middle"
    return "in", f"{depth}-{side}", "compartment"


def _places(prep: str, names: list[str], noun: str) -> str:
    tail = f" {noun}" if noun and len(names) == 1 else ""
    plural_noun = f" {noun}s" if noun and len(names) > 1 else ""
    if len(names) == 1:
        return f"{prep} the {names[0]}{tail}"
    if len(names) == 2:
        return f"{prep} the {names[0]} and {names[1]}{plural_noun}"
    listed = ", ".join(names[:-1])
    return f"{prep} the {listed}, and {names[-1]}{plural_noun}"


def _put_clause(rows: int, cols: int, items: list[tuple[str, int, int]]) -> str:
    """Parts in reading order. Neighbors of the same kind are named once."""
    groups: list[tuple[str, str, str, list[str]]] = []
    for spoken, row, col in items:
        prep, place, noun = _place(rows, cols, row, col)
        if groups and groups[-1][0] == spoken:
            groups[-1][3].append(place)
        else:
            groups.append((spoken, prep, noun, [place]))
    pieces = []
    for spoken, prep, noun, spots in groups:
        where = _places(prep, spots, noun)
        name = f"{_article(spoken)} {spoken}" if len(spots) == 1 else _plural(spoken)
        pieces.append(f"{name} {where}")
    if len(pieces) == 1:
        return pieces[0]
    if len(pieces) == 2:
        return f"{pieces[0]} and {pieces[1]}"
    return ", ".join(pieces[:-1]) + ", and " + pieces[-1]


def view_cells(tray: TrayAsset, tray_yaw: float) -> np.ndarray:
    """``[cells, 2]`` (row, column) of every cell, 1-based, as seen from the robot.

    Rows count from the side of the tray nearest the robot (small x) to the far side. Columns count from the
    robot's left (large y) to its right. The tray's own yaw is snapped to a quarter turn.
    """
    quarter = int(np.round(tray_yaw / (np.pi / 2.0))) % 4
    centers = geo.transform(tray.cell_centers, quarter * np.pi / 2.0)
    rows = _rank(centers[:, 0]) + 1
    columns = _rank(-centers[:, 1]) + 1
    return np.stack([rows, columns], axis=1)


def _rank(values: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Rank of each value among the distinct ones, where values within ``tolerance`` [m] are the same."""
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=int)
    rank = 0
    for previous, current in zip(order[:-1], order[1:]):
        if values[current] - values[previous] > tolerance:
            rank += 1
        ranks[current] = rank
    return ranks


def view_shape(tray: TrayAsset, tray_yaw: float) -> tuple[int, int]:
    """``(rows, columns)`` of the tray as seen from the robot."""
    cells = view_cells(tray, tray_yaw)
    return int(cells[:, 0].max()), int(cells[:, 1].max())


def cell_label(tray: TrayAsset, tray_yaw: float, cell: int) -> str:
    row, column = view_cells(tray, tray_yaw)[cell]
    return f"row {row}, column {column}"


def _assignments(
    cell_types: tuple[str, ...], cells: np.ndarray, order: list[int], offset: int
) -> list[tuple[str, int, int]]:
    items = []
    for cell in order:
        variant = cell_types[offset + cell]
        if variant:
            items.append((display_name(variant), int(cells[cell, 0]), int(cells[cell, 1])))
    return items


def _tray_label(name: str, rows: int, cols: int, finishes: list[str]) -> str:
    finish = _finish_word(name)
    if finishes.count(finish) > 1:
        return f"{_tray_kind(rows, cols)} {finish} tray"
    return f"{finish} tray"


def build_instruction(layout: SortingLayout, tray: TrayAsset, library=None) -> str:
    """Tell the robot which part goes in which compartment."""
    if layout.trays and library is not None:
        return _multi_instruction(layout, library)
    cells = view_cells(tray, layout.tray_pose[2])
    rows, columns = view_shape(tray, layout.tray_pose[2])
    order = sorted(range(tray.cells), key=lambda cell: (cells[cell, 0], cells[cell, 1]))
    items = _assignments(layout.cell_types, cells, order, 0)
    label = _tray_label(tray.name, rows, columns, [_finish_word(tray.name)])
    source = container_name(layout.container)
    if rows == 1 and columns == 1 and items:
        spoken = items[0][0]
        opening = f"Sort at least {layout.quota} parts from the {source}."
        return f"{opening} Put {_article(spoken)} {spoken} in the {label}."
    opening = f"Sort at least {layout.quota} parts from the {source} into the {label}."
    if not items:
        return opening
    return f"{opening} Put {_put_clause(rows, columns, items)}."


def _multi_instruction(layout: SortingLayout, library) -> str:
    """One spoken direction per tray, skipping a tray that has nothing to place."""
    described = []
    offset = 0
    for name, pose in zip(layout.tray_list(), layout.tray_pose_list()):
        tray = library.trays[name]
        cells = view_cells(tray, pose[2])
        rows, columns = view_shape(tray, pose[2])
        order = sorted(range(tray.cells), key=lambda cell: (cells[cell, 0], cells[cell, 1]))
        items = _assignments(layout.cell_types, cells, order, offset)
        described.append((name, rows, columns, items))
        offset += tray.cells
    finishes = [_finish_word(name) for name, _, _, items in described if items]
    active = [(name, rows, columns, items) for name, rows, columns, items in described if items]
    source = container_name(layout.container)
    opening = f"Sort at least {layout.quota} parts from the {source}."
    if len(active) == 1 and not (active[0][1] == 1 and active[0][2] == 1):
        name, rows, columns, items = active[0]
        label = _tray_label(name, rows, columns, finishes)
        opening = f"Sort at least {layout.quota} parts from the {source} into the {label}."
        return f"{opening} Put {_put_clause(rows, columns, items)}."
    sentences = [opening]
    index = 0
    while index < len(active):
        name, rows, columns, items = active[index]
        if rows == 1 and columns == 1:
            bits = []
            while index < len(active) and active[index][1] == 1 and active[index][2] == 1:
                cup_name, _, _, cup_items = active[index]
                spoken = cup_items[0][0]
                label = _tray_label(cup_name, 1, 1, finishes)
                bits.append(f"{_article(spoken)} {spoken} in the {label}")
                index += 1
            sentences.append("Put " + _join(bits) + ".")
            continue
        label = _tray_label(name, rows, columns, finishes)
        sentences.append(f"In the {label}, put {_put_clause(rows, columns, items)}.")
        index += 1
    return " ".join(sentences)


def _join(pieces: list[str]) -> str:
    if len(pieces) == 1:
        return pieces[0]
    if len(pieces) == 2:
        return f"{pieces[0]} and {pieces[1]}"
    return ", ".join(pieces[:-1]) + ", and " + pieces[-1]
