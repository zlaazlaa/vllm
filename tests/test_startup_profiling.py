# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from vllm.envs import environment_variables
from vllm.startup_profiling import (
    get_startup_profile_events,
    reset_startup_profile_events,
    startup_profile,
)


class StartupProfilingTest(unittest.TestCase):

    def tearDown(self) -> None:
        os.environ.pop("VLLM_STARTUP_PROFILING", None)
        os.environ.pop("VLLM_STARTUP_PROFILE_DIR", None)
        reset_startup_profile_events()

    def test_startup_profile_default_disabled(self) -> None:
        os.environ.pop("VLLM_STARTUP_PROFILING", None)
        os.environ.pop("VLLM_STARTUP_PROFILE_DIR", None)
        reset_startup_profile_events()

        with startup_profile("default_process_group", world_size=2):
            pass

        self.assertEqual(get_startup_profile_events(), [])

    def test_startup_profile_records_event_when_enabled(self) -> None:
        os.environ["VLLM_STARTUP_PROFILING"] = "1"
        os.environ.pop("VLLM_STARTUP_PROFILE_DIR", None)
        reset_startup_profile_events()

        with patch("vllm.startup_profiling.time.perf_counter") as perf_counter:
            perf_counter.side_effect = [10.0, 10.125]
            with startup_profile(
                "model_parallel_group", group_name="tp", world_size=2
            ):
                pass

        events = get_startup_profile_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["name"], "model_parallel_group")
        self.assertAlmostEqual(event["duration_s"], 0.125)
        self.assertEqual(event["metadata"], {"group_name": "tp", "world_size": 2})

    def test_startup_profile_writes_jsonl_when_output_dir_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["VLLM_STARTUP_PROFILING"] = "1"
            os.environ["VLLM_STARTUP_PROFILE_DIR"] = tmp_dir
            reset_startup_profile_events()

            with patch("vllm.startup_profiling.time.perf_counter") as perf_counter:
                perf_counter.side_effect = [20.0, 20.5]
                with startup_profile("kv_cache_initialize", worker_count=2):
                    pass

            profile_files = list(os.scandir(tmp_dir))
            self.assertEqual(len(profile_files), 1)
            self.assertTrue(profile_files[0].name.startswith("startup_profile_"))
            self.assertTrue(profile_files[0].name.endswith(".jsonl"))

            with open(profile_files[0].path) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["name"], "kv_cache_initialize")
            self.assertAlmostEqual(record["duration_s"], 0.5)
            self.assertEqual(record["metadata"], {"worker_count": 2})

    def test_startup_profile_envs_are_registered(self) -> None:
        os.environ.pop("VLLM_STARTUP_PROFILING", None)
        os.environ.pop("VLLM_STARTUP_PROFILE_DIR", None)
        self.assertIs(environment_variables["VLLM_STARTUP_PROFILING"](), False)
        self.assertEqual(environment_variables["VLLM_STARTUP_PROFILE_DIR"](), "")

        os.environ["VLLM_STARTUP_PROFILING"] = "1"
        os.environ["VLLM_STARTUP_PROFILE_DIR"] = "/tmp/vllm-startup"
        self.assertIs(environment_variables["VLLM_STARTUP_PROFILING"](), True)
        self.assertEqual(
            environment_variables["VLLM_STARTUP_PROFILE_DIR"](),
            "/tmp/vllm-startup",
        )


if __name__ == "__main__":
    unittest.main()


class StartupProfilingIntegrationTest(unittest.TestCase):

    def tearDown(self) -> None:
        os.environ.pop("VLLM_STARTUP_PROFILING", None)
        os.environ.pop("VLLM_STARTUP_PROFILE_DIR", None)
        reset_startup_profile_events()

    def test_prepare_communication_buffer_records_active_groups(self) -> None:
        from vllm.distributed import parallel_state

        class FakeGroup:

            def __init__(self) -> None:
                self.calls = 0
                self.world_size = 1

            def prepare_communication_buffer_for_model(self, model) -> None:
                self.calls += 1

        old_groups = (
            parallel_state._TP,
            parallel_state._PCP,
            parallel_state._PP,
            parallel_state._DP,
            parallel_state._EP,
            parallel_state._EPLB,
        )
        tp_group = FakeGroup()
        pp_group = FakeGroup()

        try:
            os.environ["VLLM_STARTUP_PROFILING"] = "1"
            reset_startup_profile_events()
            parallel_state._TP = tp_group
            parallel_state._PCP = None
            parallel_state._PP = pp_group
            parallel_state._DP = None
            parallel_state._EP = None
            parallel_state._EPLB = None

            parallel_state.prepare_communication_buffer_for_model(object())

            self.assertEqual(tp_group.calls, 1)
            self.assertEqual(pp_group.calls, 1)
            events = get_startup_profile_events()
            self.assertEqual(
                [event["metadata"]["group_name"] for event in events],
                ["tp", "pp"],
            )
            self.assertTrue(
                all(event["name"] == "communication_buffer_prepare" for event in events)
            )
        finally:
            (
                parallel_state._TP,
                parallel_state._PCP,
                parallel_state._PP,
                parallel_state._DP,
                parallel_state._EP,
                parallel_state._EPLB,
            ) = old_groups

    def test_process_weights_after_loading_records_event(self) -> None:
        try:
            from vllm.model_executor.model_loader import utils
        except ImportError as e:
            self.skipTest(f"model loader dependencies unavailable: {e}")

        class FakeModelConfig:
            dtype = torch.float16
            quantization = None

        os.environ["VLLM_STARTUP_PROFILING"] = "1"
        reset_startup_profile_events()

        utils.process_weights_after_loading(
            torch.nn.Module(), FakeModelConfig(), torch.device("cpu")
        )

        events = get_startup_profile_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["name"], "process_weights_after_loading")
