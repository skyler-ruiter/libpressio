#!/usr/bin/env python3
"""FZGM configuration transactions must preserve values and live pipelines on rejection."""
import sys
import unittest
binding_dir = sys.argv.pop(1)
sys.path.insert(0, binding_dir)
import libpressio as lp
import numpy as np


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.comp = (lp.FZGPUModulesChain()
            .add("quantizer:float:uint32", linear_mode=True)
            .add("lorenzo:int32", from_port="codes", block_size=32)
            .add("adaptive_bitpack:int32", block_size=32)
            .configure(memory_strategy="preallocate", specialization="off")
            .compressor(eb=1e-3))
        self.x = np.sin(np.arange(131, dtype=np.float32))
        self.baseline = self.comp.encode(self.x)
        self.before = self.comp.get_options()

    def assert_rollback(self, options):
        with self.assertRaises(lp.PressioException): self.comp.set_options(options)
        after = self.comp.get_options()
        for key in self.before:
            if key.startswith("fzgpumodules:") or key == "pressio:abs":
                np.testing.assert_equal(after[key], self.before[key], err_msg=key)
        np.testing.assert_array_equal(self.comp.encode(self.x), self.baseline)
        self.assertFalse(self.comp.get_metrics()["fzgpumodules:rebuilt"])

    def test_rollback_after_top_level_failure(self):
        for key, value in (("memory_strategy", "bogus"), ("error_bound_mode", "bogus"),
                           ("specialization", "bogus"), ("fusion", "auto")):
            with self.subTest(key=key):
                self.assert_rollback({"pressio:abs": 0.25,
                    "fzgpumodules:stages": ["bitshuffle"],
                    "fzgpumodules:connections": [], "fzgpumodules:"+key: value})

    def test_rollback_after_earlier_stage_parameter_changed(self):
        self.assert_rollback({"pressio:abs": 0.25,
                              "fzgpumodules:s0:dither_seed": 42,
                              "fzgpumodules:s1:block_size": "wrong type"})

    def test_top_level_wrong_types_and_numeric_domains(self):
        for key, value in (("num_streams", "two"), ("num_streams", 0),
                           ("memory_multiplier", float("nan")), ("memory_multiplier", -1.0), ("memory_multiplier", 1e300),
                           ("dims", np.array([-1, 1, 1], dtype=np.int64)),
                           ("dims", np.array([1.5, 1, 1], dtype=np.float64)),
                           ("graph_mode", "true"), ("dims", "invalid")):
            with self.subTest(key=key):
                self.assert_rollback({"pressio:abs": 0.25, "fzgpumodules:"+key: value})
        for bound in (-1.0, float("inf")):
            with self.subTest(bound=bound): self.assert_rollback({"pressio:abs": bound})

    def test_successful_update_rebuilds_once_and_noop_preserves_pipeline(self):
        self.comp.set_options({"pressio:abs": 5e-4, "fzgpumodules:s0:dither_seed": 42})
        self.assertEqual(self.comp.get_options()["fzgpumodules:s0:dither_seed"], 42)
        self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])
        self.comp.set_options({"pressio:abs": 5e-4, "fzgpumodules:s0:dither_seed": 42})
        self.comp.encode(self.x)
        self.assertFalse(self.comp.get_metrics()["fzgpumodules:rebuilt"])

    def test_inactive_stage_settings_survive_transactions(self):
        self.comp.set_options({"fzgpumodules:s0:dither_seed": 42})
        self.comp.set_options({"fzgpumodules:stages": ["bitshuffle"], "fzgpumodules:connections": []})
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.comp.set_options({"fzgpumodules:stages": self.before["fzgpumodules:stages"],
                               "fzgpumodules:connections": self.before["fzgpumodules:connections"]})
        self.assertEqual(self.comp.get_options()["fzgpumodules:s0:dither_seed"], 42)

    def test_rejection_preserves_pending_rebuild_and_decode_state(self):
        self.comp.set_options({"pressio:abs": 5e-4})
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.assertEqual(self.comp.get_options()["pressio:abs"], 5e-4)
        encoded = self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])
        self.comp.decode(encoded)
        with self.assertRaises(lp.PressioException):
            self.comp.set_options({"pressio:abs": 0.25, "fzgpumodules:memory_strategy": "bogus"})
        self.assertEqual(self.comp.get_options()["pressio:abs"], 5e-4)
        self.comp.encode(self.x)
        self.assertTrue(self.comp.get_metrics()["fzgpumodules:rebuilt"])


if __name__ == "__main__": unittest.main()
