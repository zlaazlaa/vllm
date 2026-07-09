# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.interface import Platform


class DummyGraphPoolPlatform(Platform):
    _global_graph_pool = None

    def __init__(self) -> None:
        self.next_pool = 0

    def graph_pool_handle(self) -> str:
        self.next_pool += 1
        return f"pool-{self.next_pool}"


def test_reset_global_graph_pool_forces_new_pool_handle():
    platform = DummyGraphPoolPlatform()

    first_pool = platform.get_global_graph_pool()
    assert platform.get_global_graph_pool() == first_pool

    platform.reset_global_graph_pool()

    assert platform.get_global_graph_pool() == "pool-2"
