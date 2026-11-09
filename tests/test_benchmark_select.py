# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "franka_sorting"))

from franka_sorting.benchmark.bundle import ALL_TASKS  # noqa: E402
from franka_sorting.benchmark_select import parse_task_indices, select_tasks, shared_background  # noqa: E402


class _Benchmark:
    task_ids = ("a", "b", "c", "d")
    backgrounds = {"a": "dome1", "b": "dome1", "c": "dome2", "d": "dome2"}

    def select(self, task_id):
        if task_id == ALL_TASKS:
            return self.task_ids
        if task_id not in self.task_ids:
            raise FileNotFoundError(task_id)
        return (task_id,)

    def task(self, task_id):
        return SimpleNamespace(background=self.backgrounds[task_id])


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        (["3"], [3]),
        (["16~19"], [16, 17, 18, 19]),
        (["16-18"], [16, 17, 18]),
        (["5,0,16~17"], [0, 5, 16, 17]),
        (["4", "2~4", "4"], [2, 3, 4]),
        (["383"], [383]),
    ],
)
def test_parse_task_indices(specs, expected):
    assert parse_task_indices(specs, 384) == expected


@pytest.mark.parametrize("spec", ["384", "10~384", "31~16", "x", "1~", "-3", ""])
def test_parse_task_indices_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_task_indices([spec], 384)


def test_parse_task_indices_names_the_valid_range():
    with pytest.raises(ValueError, match="0..383"):
        parse_task_indices(["500"], 384)


def test_select_tasks_keeps_benchmark_order_without_repeats():
    benchmark = _Benchmark()
    assert select_tasks(benchmark, "all") == benchmark.task_ids
    assert select_tasks(benchmark, "c") == ("c",)
    assert select_tasks(benchmark, "d,a,d") == ("a", "d")
    with pytest.raises(FileNotFoundError):
        select_tasks(benchmark, "a,missing")


def test_shared_background_rejects_mixed_domes():
    benchmark = _Benchmark()
    assert shared_background(benchmark, ("a", "b")) == "dome1"
    with pytest.raises(ValueError, match="2 backgrounds"):
        shared_background(benchmark, ("b", "c"))
