# Whole-graph D3D12 GPU backend

GPU inference for **every GGUF model** (llama.cpp path) on Xbox Series S|X, including the 1-bit
`Q1_0` models such as Bonsai 27B.

## What it is

The earlier D3D12 path (`gguf_gpu_layers.txt`, see `gguf-gpu-decode.md`) only ran *weight matmuls* on the
GPU. Attention, the KV cache, norms and every other op stayed on the CPU, so each token made 50–70
CPU↔GPU round trips and the measured gain was ~1.6× (and it was off by default after failing its RAM gate).

This backend runs the **whole transformer graph on the GPU**: matmuls, RMS/L2 norm, RoPE (incl. the
interleaved multi-section RoPE of Qwen3.5/3.6), flash attention, softmax, the KV cache, the gated
delta-net and SSM-conv ops of the hybrid architectures (Qwen3.5/3.6, LFM2), MoE routing
(`MUL_MAT_ID`), embeddings. A token is one GPU submission. llama.cpp offloads layers the ordinary way.

* Source: `third_party/ggml-d3d12-full/` — a port of
  [koza-coder/llama.cpp-directx12](https://github.com/koza-coder/llama.cpp-directx12) v0.2.7 (MIT, as
  llama.cpp). The diff against upstream is `scripts/port-ggml-d3d12-full.py` (applied to the verbatim vendor
  commit), so it can be re-vendored.
* Glue: `src/bridge/ggml_d3d12_full.cpp`, `src/bridge/llama_gpu.h`.

## Shaders are baked, not compiled, on the console

A retail Xbox UWP package cannot load a shader compiler. The backend normally compiles HLSL at run time
with `dxcompiler.dll`; here that path is removed (`XLLAMA_D3D12_UWP`) and every pipeline is loaded from a
**precompiled DXIL cache** shipped in the package (`d3d12-shader-cache\*.dxil`).

The cache is produced in CI (`.github/workflows/d3d12-full-release.yml`): a desktop build of the same
backend runs real graphs of every catalogue model (and Bonsai 27B) on the runner's software D3D12 adapter
(WARP), with `GGML_D3D12_FAKE_VENDOR=1002` so it requests the pipelines an AMD UMA GPU requests. Each
pipeline is compiled by DXC and saved under a hash of *(shader source + compiler arguments)* — the same key
the console looks up. Kernel selection depends only on 16-bit support, the AMD+UMA fusion path and each
pipeline's own fixed minimum shader model (cs_6_0 / cs_6_2); wave sizes only affect logging.

If a pipeline is ever missing the device is flagged failed, `graph_compute` returns an error, `llama_decode`
fails cleanly, and the log names the missing key. It never aborts the app.

## Behaviour

| Situation | Result |
|---|---|
| GGUF model, backend usable | all layers that fit go to the GPU (`d3d12_full_fit_layers`) |
| Model larger than the GPU budget | partial offload sized from `ggml_backend_dev_memory`, minus a 1 GiB reserve |
| GPU model load fails | retry with half the layers, then CPU |
| Runtime LoRA adapter | CPU (not validated on the GPU path) |
| ORT/ONNX models | unchanged (DirectML, separate path) |

## Knobs (all files in `LocalState`)

| File | Effect |
|---|---|
| `gguf_gpu_layers.txt` | `0` forces CPU; `N` offloads at most N layers (always wins) |
| `d3d12f_disable.txt` | (any content) disable the backend |
| `d3d12f_reserve_mb.txt` | GPU MB kept free for KV/compute (default 1024) |
| `d3d12f_env.txt` | `KEY=VALUE` lines applied as environment before init, e.g. `GGML_D3D12_DISABLE_OPS=FLASH_ATTN_EXT` |
| `d3d12_backend.txt` | `legacy` selects the older matmul-only backend |
| `d3d12f_force.txt` | ignore the crash sentinel (below) |

## Crash safety

`d3d12f_attempt.txt` is written just before the GPU is first touched and deleted after the first successful
decode. If a launch finds it, the previous run died on the GPU: that launch runs on the CPU (and clears
the flag, so the next launch tries the GPU again). A GPU fault therefore cannot lock the app out.

## What is verified, and what is not

Verified in CI (software D3D12 adapter, DXC-compiled DXIL, compared against the CPU backend):
`test-backend-ops` for the ops above, including `MUL_MAT` with `Q1_0` weights; greedy generation
CPU-vs-GPU for several models; the real Bonsai 27B `Q1_0` graph.

**Not verified in CI: speed, and behaviour on the console GPU/driver.** A software adapter proves the
kernels are correct, not that they are fast, and a retail Xbox can differ in memory budget, driver and
watchdog behaviour. Please report numbers from your console (`xllama.log` has the
`[xllama] gguf gpu layers (full d3d12): ...` line with the budget it saw).
