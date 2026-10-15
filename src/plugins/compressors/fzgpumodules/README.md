# FZGPUModules Python builder

```python
import libpressio as lp
import numpy as np

chain = (
    lp.FZGPUModulesChain()
    .add("quantizer:float:uint32", linear_mode=True)
    .add("lorenzo:int32", from_port="codes", block_size=32)
    .add("adaptive_bitpack:int32", block_size=32)
    .configure(memory_strategy="preallocate", specialization="auto")
)
compressor = chain.compressor(eb=1e-3)
x = np.linspace(0, 1, 131, dtype=np.float32)
compressed = compressor.encode(x)
decoder = lp.PressioCompressor.from_config({"compressor_id": "fzgpumodules"})
restored = decoder.decode(compressed)
assert np.max(np.abs(x - restored)) <= 1e-3
```

`add()` numbers stages as `s0`, `s1`, etc. and connects each new stage to the
preceding stage. `from_port` selects an output on that preceding stage; other
keyword arguments set parameters on the new stage. For branching pipelines,
use the plugin's stages/connections options directly.

## Global options with configure()

`configure()` stores pipeline-wide options and returns the builder for chaining.
Use unprefixed names: `specialization="auto"` becomes
`fzgpumodules:specialization`. Other examples are `memory_strategy`, `dims`,
`num_streams`, and `graph_mode`. Parameters for individual stages belong in
`add()`. Set the error bound with `compressor(eb=...)`.

This call does not create or update a live compressor. Options are checked when
constructing or using the compressor; some stage constraints are checked on
first encode. To change an existing compressor, use its `set_options()` method.

## Explicit construction with early_config()

Most callers only need `chain.compressor()`. For explicit libpressio construction:

```python
compressor = lp.PressioCompressor.from_config({
    "compressor_id": "fzgpumodules",
    "early_config": chain.early_config(),
    "compressor_config": {"pressio:abs": 1e-3, "pressio:metric": "size"},
})
```

`early_config()` returns a dictionary containing the stages, connections,
global options, and per-stage parameters. It does not construct a compressor.
Libpressio applies this dictionary before discovering the selected stages'
option types and applying `compressor_config`. `chain.compressor()` performs
these steps automatically.

Encoded arrays contain the information needed by a fresh decoder. Save them
with `compressed.tobytes()` and pass the saved bytes to `decoder.decode()`;
no chain configuration or output allocation is required. A supplied output
array must match the archived dtype and shape.

Use one GPU with the standard CUDA runtime primary context and serialize calls
across all FZGM instances. Keep that device/context alive while objects exist.
