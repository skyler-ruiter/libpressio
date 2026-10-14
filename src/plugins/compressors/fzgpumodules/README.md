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

## Execution and ownership

Use one GPU with the standard CUDA runtime primary context. Do not switch the
current device, use custom CUDA contexts, or reset the device while FZGM objects
are alive. `pressio:thread_safe=serialized` requires external serialization of
calls across **all** FZGM instances, including clones; the plugin adds no lock.

A clone copies options and stage parameters, then builds its own encoder and CUDA
stream on first execution. It does not share a live pipeline, buffers, or execution
metrics. Stream creation failures are reported through libpressio's error API.

NumPy input is uploaded for compression, and the resulting archive is returned to
host memory for ordinary Python save/load. GPU input keeps the archive on the GPU;
a GPU output request also keeps it there. Decoding returns host output by default
or GPU output when requested. Libpressio's domain manager owns these transfers;
the adapter copies core pool-owned buffers into independently owned output buffers.
CUDA Graph mode accelerates encoding only: its archives remain independently
decodable, and archive decoding builds a fresh pipeline without graph capture.

`fzgpumodules:execution_time_us` measures wall time around the core compression
call, excluding pipeline construction, input upload, archive assembly, and host
output transfer. It is not a CUDA-event kernel time or end-to-end Python latency.
`fzgpumodules:peak_memory` reports the core pipeline pool's peak usage, excluding
external input/archive allocations and host memory. Host `rusage` metrics measure
a different quantity and cannot replace this pool measurement.

## Configuration updates

FZGM option updates are transactional: an error restores all top-level values,
per-stage parameters (including inactive stage settings), and the pipeline's
previous rebuild state. Successful changes invalidate the cached encoder;
unchanged options retain it. Type errors and invalid top-level numeric values
are reported rather than silently ignored. Dimensions require one to three
integer values; only the first may be zero to select automatic sizing.

Stage-specific ranges, stage tokens, connection topology, and kernel constraints
are still validated lazily when the pipeline is constructed. A successful
`set_options()` does not guarantee that a new pipeline can execute.

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

## Adapter layout

- `fzgpumodules.cc`: libpressio options, pipeline lifecycle, execution, and metrics.
- `configuration.h`: option values, validation, and stage parameter ownership. Updates
  are parsed on a candidate and committed together after validation.
- `pipeline_builder.h`: manual/TOML graph construction, named-port wiring, output
  references, and graph capture. Returns a complete encoder before replacing live state.
- `stage_registry.h`: SDK-gated handler catalog, token lookup, and parameter snapshots.
- `stages/stage_kind.h`: common handler contract and value-copy support.
- `stages/stage_*.h`: stage-specific options, documentation, and construction.
- `archive.h`: the self-contained array envelope format.

To add a stage family, implement a handler in `stages/`, include it and register
it in `stage_registry.h`, and add FZGM integration tests. Use the existing SDK
capability guards for optional families. Registration order retains the previous
first-match behavior; individual handlers own their token aliases. The main
plugin requires no stage-specific edits.
