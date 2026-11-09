# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the Gym ids this project registers.

./isaaclab.sh -p source/task-franka-sorting-parts/scripts/list_envs.py
"""

import franka_sorting.tasks  # noqa: F401
import gymnasium as gym


def main() -> None:
    for env_id in sorted(gym.registry):
        if env_id.startswith("Nepher-FrankaSorting"):
            spec = gym.registry[env_id]
            print(f"{env_id:44s} {spec.kwargs.get('env_cfg_entry_point', '')}")


if __name__ == "__main__":
    main()
