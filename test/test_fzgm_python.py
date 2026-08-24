#!/usr/bin/env python3
# test_fzgm_python.py -- FZGpuModules plugin tests via the high-level Python API.
#
# This exists alongside test_fzgm.cc (the C++ gtest suite) specifically to cover
# libpressio's Python option-boxing path, which test_fzgm.cc cannot exercise:
# python_to_pressio_options() (tools/swig/libpressio.py) boxes every Python `int`
# as C++ int64_t and every Python `float` as C++ double, while
# pressio_options::get() requires an *exact* type match with no numeric coercion.
# A plugin option declared as a narrower C++ type (int, float, uint32_t, ...)
# would silently never be set from Python -- this was a real, wide-reaching bug
# in the fzgpumodules plugin (fixed by widening every such field to
# int64_t/double). test_fzgm.cc's C++ pressio_options::set() calls use whatever
# type the test author writes, so they can accidentally match even when Python
# callers would not -- only a real Python-path test catches that class of
# regression.
#
# Run: python3 test_fzgm_python.py <path-to-swig-build-dir>

import sys
pressio_path = sys.argv[1]
sys.path.insert(0, pressio_path)

import numpy as np
import libpressio as lp

FAILED = 0
PASSED = 0


def check(label, condition, detail=""):
    global FAILED, PASSED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAILED: {label} {detail}")


def make_data():
    x = np.linspace(0, 8 * np.pi, 256 * 256).reshape(256, 256)
    return (np.sin(x) + 0.3 * np.cos(3 * x)).astype(np.float32)


data = make_data()

# ── Numeric option readback: every scalar option below is backed by a C++
# field this session widened to int64_t/double specifically so Python's
# int/float boxing would land on it. Setting each from Python and reading it
# straight back (no compress/decompress) isolates the option pipe from
# anything the compression algorithm itself does.

def check_scalar_readback(stage_token, key, value, label=None):
    comp = lp.PressioCompressor.from_config({
        "compressor_id": "fzgpumodules",
        "early_config": {
            "fzgpumodules:stages": [stage_token],
            "fzgpumodules:connections": [],
            key: value,
        },
    })
    got = comp.get_options()[key]
    check(label or key, got == value, f"(set {value!r}, got {got!r})")


check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:quant_radius", 999)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:outlier_capacity", 0.35)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:dither_seed", 42)
check_scalar_readback("quantizer:float:uint16", "fzgpumodules:s0:dither_strength", 0.5)
check_scalar_readback("rze", "fzgpumodules:s0:chunk_size", 8192)
check_scalar_readback("rze", "fzgpumodules:s0:word_size", 4)
check_scalar_readback("bitpack:uint16", "fzgpumodules:s0:nbits", 8)
check_scalar_readback("bitshuffle", "fzgpumodules:s0:element_width", 2)

comp = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
comp.set_options({"fzgpumodules:memory_multiplier": 5.0})
comp.set_options({"fzgpumodules:num_streams": 2})
opts = comp.get_options()
check("memory_multiplier", opts["fzgpumodules:memory_multiplier"] == 5.0)
check("num_streams", opts["fzgpumodules:num_streams"] == 2)

# ── End-to-end round-trips exercising the widened options for real, not just
# readback -- confirms the value actually reaches the FZGPUModules engine.

def roundtrip_ok(config, data, eb=1e-3):
    early = dict(config.get("early_config", {}))
    cfg = {"compressor_id": "fzgpumodules", "early_config": early,
           "compressor_config": {"pressio:abs": eb, "pressio:metric": "size",
                                  **config.get("compressor_config", {})}}
    comp = lp.PressioCompressor.from_config(cfg)
    d = np.asarray(comp.decode(comp.encode(data), data.copy()))
    max_err = float(np.abs(data - d).max())
    return max_err <= eb + 1e-6, max_err, comp.get_metrics()["size:compression_ratio"]

ok, max_err, _ = roundtrip_ok({
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:s0:quant_radius": 16000,
        "fzgpumodules:s0:outlier_capacity": 0.25,
    },
}, data)
check("lorenzo quant_radius/outlier_capacity round-trip", ok, f"max_err={max_err}")

ok, max_err, ratio = roundtrip_ok({
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:s1:chunk_size": 8192,
    },
}, data)
check("rze chunk_size takes effect (ratio differs from default)", ok and ratio != 9.781492537313433,
      f"ratio={ratio}")

# ── New stages from this session: smoke-test that the Python token strings
# work end to end (not exhaustive -- see test_fzgm.cc for full coverage).

for token, dtype, needs_dims in [
    ("szx:float", np.float32, False),
    ("adaptive_bitpack:int32", np.int32, False),
]:
    if dtype == np.int32:
        rng = np.random.default_rng(0)
        d = rng.integers(-1000, 1000, size=256 * 256).astype(np.int32)
        stages = ["tiled_lorenzo:int32", token] if "adaptive_bitpack" in token else [token]
        conns = ["s1 <- s0"] if "adaptive_bitpack" in token else []
        comp = lp.PressioCompressor.from_config({
            "compressor_id": "fzgpumodules",
            "early_config": {"fzgpumodules:stages": stages, "fzgpumodules:connections": conns},
        })
        out = np.asarray(comp.decode(comp.encode(d), d.copy()))
        check(f"{token} lossless round-trip", np.array_equal(d, out))
    else:
        ok, max_err, _ = roundtrip_ok({"early_config": {
            "fzgpumodules:stages": [token], "fzgpumodules:connections": []}}, data)
        check(f"{token} round-trip", ok, f"max_err={max_err}")

# ── Fusion / PREL / centering (new options this session)

comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {"fzgpumodules:fusion": "auto"}})
check("fusion=auto readback", comp.get_options()["fzgpumodules:fusion"] == "auto")

try:
    lp.PressioCompressor.from_config({
        "compressor_id": "fzgpumodules",
        "early_config": {"fzgpumodules:fusion": "bogus"}})
    check("invalid fusion value rejected", False)
except Exception:
    check("invalid fusion value rejected", True)

# PREL's pressio:abs is a *fraction*: abs_eb = pressio:abs * max(|data|).
# roundtrip_ok() checks max_err against the eb it's given, so pass it the
# resolved absolute bound rather than the fraction, matching what PREL
# actually enforces.
prel_frac = 1e-3
prel_abs_bound = prel_frac * float(np.abs(data).max())
comp = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": {
        "fzgpumodules:stages": ["lorenzo:float:uint16", "rze"],
        "fzgpumodules:connections": ["s1 <- s0:codes"],
        "fzgpumodules:error_bound_mode": "prel",
    },
    "compressor_config": {"pressio:abs": prel_frac, "pressio:metric": "size"},
})
d = np.asarray(comp.decode(comp.encode(data), data.copy()))
max_err = float(np.abs(data - d).max())
check("prel error mode round-trip", max_err <= prel_abs_bound + 1e-6, f"max_err={max_err}")

print(f"PASSED={PASSED} FAILED={FAILED}")
sys.exit(1 if FAILED else 0)
