# FZGPUModules Python builder

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
restored = compressor.decode(compressed, np.empty_like(x))
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

## Decode lifetime

Decode immediately using the same compressor instance that encoded the input,
before another encode or pipeline reconfiguration. The plugin retains pipeline
and reconstruction state from compression and returns raw compressed buffers,
not a self-contained FZM archive. A fresh compressor currently rejects decode
with `Decompression requires a pipeline built by a prior compress call`.
Saving these bytes alone does not provide a supported decoder in another process.

This restriction belongs to the libpressio adapter, not the core library's FZM
archive interface. Restoring the adapter in a fresh process requires a separate
archive/metadata integration change.
