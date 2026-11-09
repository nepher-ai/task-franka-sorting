# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level tool-pose tracker.

Registered environments:
  Nepher-FrankaSorting-LL-v0       fine-tuning (4096 envs, observation noise, 35% wide orientations)
  Nepher-FrankaSorting-LL-Play-v0  evaluation (32 envs, no noise)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-FrankaSorting-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-FrankaSorting-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
