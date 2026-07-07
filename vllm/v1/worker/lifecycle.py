# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker lifecycle roles shared by worker and executor code."""

from enum import Enum


class WorkerRole(str, Enum):
    ACTIVE = "active"
    STANDBY = "standby"
