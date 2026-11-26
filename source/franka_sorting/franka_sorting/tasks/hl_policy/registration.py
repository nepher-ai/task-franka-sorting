# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the high-level sorting envs.

Imported by the package init. The entry-point strings stay on ``hl_policy`` so they name the env module,
not this file.
"""

import gymnasium as gym

_ENV = "franka_sorting.tasks.hl_policy.env:HLSortingEnv"
_CFG = "franka_sorting.tasks.hl_policy.hl_env_cfg:HLEnvCfg_PLAY"
_RUNNER = "franka_sorting.tasks.ll_policy.agents.ppo_cfg:LLPPORunnerCfg"
_PUBLISHED = "franka_sorting.tasks.ll_policy.agents.ppo_cfg:LLPPORunnerCfg_ENVHUB_PLAY"

gym.register(
    id="Nepher-FrankaSorting-HL-Play-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CFG,
        "rsl_rl_cfg_entry_point": _RUNNER,
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-FrankaSorting-HL-EnvhubPlay-v0",
    entry_point=_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CFG,
        "rsl_rl_cfg_entry_point": _PUBLISHED,
        "default_agent": "rsl_rl",
    },
)
