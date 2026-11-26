# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner plus tracker.

Registered from :mod:`franka_sorting.tasks.hl_policy.registration` when the package is imported:

  Nepher-FrankaSorting-HL-Play-v0        every task in one stage (``task_id="all"``)
  Nepher-FrankaSorting-HL-EnvhubPlay-v0  eval-nav entry

Both load the tracker checkpoint and the nepher cache of the cfg's ``env_id``.
"""
