#!/usr/bin/env python3
"""FZGM archive tests: fresh processes, output schema, and malformed metadata."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

binding_dir = sys.argv.pop(1)
sys.path.insert(0, binding_dir)
import libpressio as lp
import numpy as np


def encoder(policy="auto"):
    return (lp.FZGPUModulesChain()
            .add("quantizer:float:uint32", linear_mode=True)
            .add("lorenzo:int32", from_port="codes", block_size=32)
            .add("adaptive_bitpack:int32", block_size=32)
            .configure(memory_strategy="preallocate", specialization=policy)
            .compressor(eb=1e-3))


def decoder(policy="off"):
    return lp.PressioCompressor.from_config({
        "compressor_id": "fzgpumodules",
        "compressor_config": {"fzgpumodules:specialization": policy,
                              "fzgpumodules:memory_strategy": "preallocate"}})


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.x = np.sin(np.arange(131, dtype=np.float32)).reshape(1, 131)
        self.encoded = encoder().encode(self.x)

    def assert_restored(self, result, x):
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.shape, x.shape)
        self.assertLessEqual(float(np.max(np.abs(result-x))), 1e-3 + 1e-6)

    def test_fresh_decoder_without_config_or_output(self):
        for policy in ("off", "auto"):
            with self.subTest(policy=policy):
                comp = decoder(policy)
                self.assert_restored(comp.decode(self.encoded), self.x)
                self.assertEqual(comp.get_metrics()["fzgpumodules:specialization_inverse_groups"],
                                 1 if policy == "auto" else 0)

    def test_saved_bytes_in_separate_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/"array.fzlp"
            result = Path(tmp)/"restored.npy"
            archive.write_bytes(self.encoded.tobytes())
            script = """import sys
sys.path.insert(0, sys.argv[1])
import libpressio as lp
import numpy as np
from pathlib import Path
comp = lp.PressioCompressor.from_config({'compressor_id': 'fzgpumodules'})
np.save(sys.argv[3], comp.decode(Path(sys.argv[2]).read_bytes()))
"""
            subprocess.run([sys.executable, "-I", "-c", script, binding_dir,
                            str(archive), str(result)], check=True, env=os.environ.copy())
            self.assert_restored(np.load(result), self.x)

    def test_old_payload_survives_later_encodes_and_config_changes(self):
        comp = encoder()
        old = comp.encode(self.x)
        other = np.cos(np.arange(4097, dtype=np.float32))
        new = comp.encode(other)
        comp.set_options({"fzgpumodules:stages": ["bitshuffle"],
                          "fzgpumodules:connections": []})
        self.assert_restored(comp.decode(old), self.x)
        self.assert_restored(comp.decode(new), other)
        self.assert_restored(decoder().decode(old), self.x)

    def test_lossless_single_segment_restores_float64(self):
        x = np.linspace(-2, 3, 21, dtype=np.float64).reshape(3, 7)
        comp = lp.FZGPUModulesChain().add("bitshuffle", element_width=8).compressor()
        encoded = comp.encode(x)
        restored = decoder().decode(encoded)
        self.assertEqual(restored.dtype, x.dtype)
        self.assertEqual(restored.shape, x.shape)
        np.testing.assert_array_equal(restored, x)

    def test_rejects_wrong_output_schema(self):
        for out in (np.empty(self.x.shape, dtype=np.float64),
                    np.empty((131,), dtype=np.float32)):
            with self.subTest(shape=out.shape, dtype=out.dtype):
                with self.assertRaises(lp.PressioException):
                    decoder().decode(self.encoded, out)

    def test_rejects_malformed_envelopes(self):
        data = self.encoded.tobytes()
        bad_magic = bytearray(data); bad_magic[0] ^= 1
        bad_shape = bytearray(data); bad_shape[40:48] = (2**64-1).to_bytes(8,"little")
        bad_header = bytearray(data); bad_header[((40+8*self.x.ndim+15)//16)*16+80] ^= 1
        bad_dtype = bytearray(data); bad_dtype[12:16] = (99).to_bytes(4,"little")
        concat_data = (lp.FZGPUModulesChain().add("lorenzo:float:uint16")
            .add("diff:uint16", from_port="codes").compressor().encode(self.x).tobytes())
        metadata_base = ((40+8*self.x.ndim+15)//16)*16
        count = int.from_bytes(concat_data[metadata_base+6:metadata_base+8], "little")
        self.assertGreater(count, 1)  # Quantizer outlier ports make a multi-output payload.
        payload_offset = metadata_base + int.from_bytes(concat_data[24:32], "little")
        bad_concat_count = bytearray(concat_data)
        bad_concat_count[payload_offset:payload_offset+4] = (99).to_bytes(4, "little")
        bad_concat_size = bytearray(concat_data)
        bad_concat_size[payload_offset+4:payload_offset+12] = (2**64-1).to_bytes(8, "little")
        for payload in (data[:8], data[:-1], data+b"x", bytes(bad_magic),
                        bytes(bad_shape), bytes(bad_header), bytes(bad_dtype),
                        bytes(bad_concat_count), bytes(bad_concat_size)):
            with self.subTest(length=len(payload)):
                with self.assertRaises(lp.PressioException):
                    decoder().decode(payload)
        # A rejected decode must not poison the next valid operation.
        comp = decoder()
        with self.assertRaises(lp.PressioException): comp.decode(bytes(bad_header))
        self.assert_restored(comp.decode(data), self.x)


if __name__ == "__main__":
    unittest.main()
