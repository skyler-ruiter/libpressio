# FZGPUModules Python builder

Use an FZGM SDK that provides `Pipeline::fromHeader` (not released v2.0.0).
Build libpressio with `LIBPRESSIO_HAS_FZGPUMODULES=ON` and
`BUILD_PYTHON_WRAPPER=ON` to use the optional plugin through Python.
The builder is part of the `libpressio` module; no separate helper import is needed.

```python
import libpressio as lp
import numpy as np

compressor = (
    lp.FZGPUModulesChain()
    .add("quantizer:float:uint32", linear_mode=True)
    .add("lorenzo:int32", from_port="codes", block_size=32)
    .add("adaptive_bitpack:int32", block_size=32)
    .configure(memory_strategy="preallocate", specialization="auto")
    .compressor(eb=1e-3)
)
x = np.linspace(0, 1, 131, dtype=np.float32)
compressed = compressor.encode(x)
decoder = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
restored = decoder.decode(compressed)
assert np.max(np.abs(x - restored)) <= 1e-3
```

`add()` numbers stages as `s0`, `s1`, etc. and connects each new stage to
the preceding stage. `from_port` selects a named output on that preceding
stage; stage keyword arguments become its plugin options. `configure()`
sets top-level FZGM options. `early_config()` returns the structural options
for callers who want to construct a `PressioCompressor` themselves.
For branching pipelines, use the plugin's stages/connections options directly.

This replaces `from fzgm_helper import Chain` with
`lp.FZGPUModulesChain()` on new installations.

## Self-contained output

Each encoded buffer is an FZLP v1 envelope containing the original libpressio
array dtype/dimensions, an FZM metadata header, and the compressed payload.
A fresh compressor can decode it without a prior encode, chain options, TOML
file, or caller-provided output buffer. The installed SDK must support the
stages used by the archive. Decoder specialization policy and memory strategy are local
execution choices; stage topology and parameters come from the archive.

```python
from pathlib import Path

Path("array.fzlp").write_bytes(compressed.tobytes())
# In another Python process with the same compatible installation:
decoder = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
restored = decoder.decode(Path("array.fzlp").read_bytes())
```

A supplied output array must match the archived dtype and shape. NumPy inputs produce a host-resident encoded array that can be saved directly;
GPU inputs keep encoded output on the GPU. Decoded host output is returned by default; supplying a GPU output preserves GPU residency. Previously
encoded buffers remain independently decodable after later encodes or option
changes. Metadata bytes count toward libpressio's reported compressed size and
compression ratio. Reconstruction currently creates a fresh pipeline per decode;
a decoder cache is a separate performance improvement. Decoder reconstruction
currently uses the core default pool multiplier and stream setup, without CUDA
Graph capture. Execution time, peak memory, rebuild, outlier-count, and stage-output
metrics describe the last encode; specialization metrics describe the active pipeline.

This changes the plugin's output format from raw buffers to versioned envelopes.
Legacy raw buffers are rejected because they lack reconstruction metadata.
FZLP is a libpressio array envelope, not byte-identical to the core `.fzm` file
format. Neither a Python helper nor temporary files are involved in decoding.

### FZLP v1 layout

All integers are little endian. The 40-byte prefix contains the eight bytes
`FZLP0001`, a uint32 version (1), a uint32 dtype code, uint32 rank (1–32),
uint32 reserved flags (0), uint64 FZM metadata length, and uint64 payload length.
It is followed by `rank` uint64 dimensions in libpressio dimension order, zero
padding to a 16-byte boundary, the FZM metadata bytes, and the payload. Dtype codes 0–9 respectively denote
float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
The dimensions must describe exactly the uncompressed byte count in the FZM
header. The FZM header checksum validates its metadata; FZLP v1 does not add a
payload checksum. Array dimensions and dtype are validated against byte counts.
