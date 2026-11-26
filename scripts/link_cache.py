# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Link the sorting benchmarks into the nepher cache, so ``nepher`` finds them without a download.

Each ``<cache>/franka-sorting-parts-v{0,1,2}`` becomes a directory symlink to the folder under
``source/envhub/environments``. The script is idempotent. It repairs a link that points elsewhere, and it
refuses to replace a real directory or file::

    python source\\task-franka-sorting\\scripts\\link_cache.py [--cache <dir>] [--ids id1 id2 ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ENVHUB = Path(__file__).resolve().parents[2] / "envhub" / "environments"
DEFAULT_CACHE = Path.home() / ".nepher" / "cache"
DEFAULT_IDS = ("franka-sorting-parts-v0", "franka-sorting-parts-v1", "franka-sorting-parts-v2")


def is_junction(path: Path) -> bool:
    """Whether ``path`` is a symlink or a Windows junction, and not a real directory."""
    if path.is_symlink():
        return True
    return os.path.isjunction(path) if hasattr(os.path, "isjunction") else False


def link(cache: Path, env_id: str, envhub: Path = ENVHUB) -> str:
    """Make ``cache/env_id`` a directory symlink to ``envhub/env_id``. Returns what happened.

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
        os.rmdir(path)  # removes the link only, never its contents
        action = "repaired"
    elif path.exists():
        raise FileExistsError(f"{path} is a real directory or file. Move it away first.")
    else:
        action = "linked"
    cache.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path, target_is_directory=True)
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
