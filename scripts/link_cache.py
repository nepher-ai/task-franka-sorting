# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Link the sorting benchmarks into the nepher cache, so ``nepher`` finds them without a download.

Each ``<cache>/franka-sorting-parts-v{0,1,2}`` becomes a junction (``mklink /J``, no admin rights needed) to the
folder under ``source/envhub/environments``. The script is idempotent. It repairs a junction that points
elsewhere, and it refuses to replace a real directory or file::

    python source\\task-franka-sorting\\scripts\\link_cache.py [--cache <dir>] [--ids id1 id2 ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ENVHUB = Path(__file__).resolve().parents[2] / "envhub" / "environments"
DEFAULT_CACHE = Path.home() / ".nepher" / "cache"
DEFAULT_IDS = ("franka-sorting-parts-v0", "franka-sorting-parts-v1", "franka-sorting-parts-v2")


def is_junction(path: Path) -> bool:
    return os.path.isjunction(path) if hasattr(os.path, "isjunction") else bool(path.is_symlink())


def link(cache: Path, env_id: str, envhub: Path = ENVHUB) -> str:
    """Make ``cache/env_id`` a junction to ``envhub/env_id``. Returns what happened.

    Raises:
        FileNotFoundError: The benchmark folder does not exist.
        FileExistsError: ``cache/env_id`` is a real directory or file.
    """
    target, path = (envhub / env_id).resolve(), cache / env_id
    if not (target / "benchmark.yaml").is_file():
        raise FileNotFoundError(f"{target} is not a built benchmark (no benchmark.yaml)")
    if is_junction(path):
        if Path(os.path.realpath(path)) == target:
            return "ok"
        os.rmdir(path)  # removes the junction only, never its contents
        action = "repaired"
    elif path.exists():
        raise FileExistsError(f"{path} is a real directory or file. Move it away first.")
    else:
        action = "linked"
    cache.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(target)], capture_output=True, text=True)
    if result.returncode != 0:
        raise OSError(f"mklink failed for {path}: {result.stderr.strip() or result.stdout.strip()}")
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()
    for env_id in args.ids:
        print(f"{env_id}: {link(args.cache, env_id)}")


if __name__ == "__main__":
    main()
