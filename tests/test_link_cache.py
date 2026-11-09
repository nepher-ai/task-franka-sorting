# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``link_cache.py`` makes idempotent junctions and never replaces a real directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import link_cache  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows feature")


def _envhub(tmp_path: Path, *ids: str) -> Path:
    envhub = tmp_path / "envhub"
    for env_id in ids:
        (envhub / env_id).mkdir(parents=True)
        (envhub / env_id / "benchmark.yaml").write_text("tasks: []\n")
    return envhub


def test_link_is_idempotent_and_repairs_a_wrong_target(tmp_path):
    envhub, cache = _envhub(tmp_path, "a", "b"), tmp_path / "cache"
    assert link_cache.link(cache, "a", envhub) == "linked"
    assert link_cache.link(cache, "a", envhub) == "ok"
    assert (cache / "a" / "benchmark.yaml").is_file()
    os.rmdir(cache / "a")
    link_cache.link(cache, "b", envhub)
    os.rename(cache / "b", cache / "a")  # a junction that points at b
    assert link_cache.link(cache, "a", envhub) == "repaired"
    assert Path(os.path.realpath(cache / "a")) == (envhub / "a").resolve()
    assert (envhub / "b" / "benchmark.yaml").is_file()  # the repair never touches the target


def test_link_refuses_a_real_directory_and_a_missing_benchmark(tmp_path):
    envhub, cache = _envhub(tmp_path, "a"), tmp_path / "cache"
    (cache / "a").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        link_cache.link(cache, "a", envhub)
    with pytest.raises(FileNotFoundError):
        link_cache.link(cache, "missing", envhub)
