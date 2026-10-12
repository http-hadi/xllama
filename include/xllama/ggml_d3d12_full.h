// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Glue for the whole-graph ggml D3D12 backend (third_party/ggml-d3d12-full, ported from
// koza-coder/llama.cpp-directx12). Unlike the older matmul-only backend in ggml_d3d12.h,
// this one runs the entire transformer graph on the GPU (norms, RoPE, flash attention,
// KV cache, the gated delta-net of Qwen3.5/3.6, ...), so llama.cpp offloads layers the
// ordinary way and a token needs one GPU submission instead of 50-70 CPU<->GPU round trips.
//
// UWP only: shaders are *precompiled* DXIL baked in CI and shipped in the package
// (d3d12-shader-cache\), because a retail AppContainer cannot load a shader compiler.
// Everywhere else (Linux host build, tests) the functions are inline no-ops.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace xllama {

#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)
// Registers the backend with ggml once. False when it cannot be used: disabled by
// LocalState\d3d12f_disable.txt, a previous GPU run that never finished a decode
// (crash sentinel), no baked shaders in the package, or no D3D12 device.
bool d3d12_full_register();
// Cheap policy check (no D3D12 work): should GGUF models default to the GPU?
bool d3d12_full_default_on();
const char* d3d12_full_device_name(); // e.g. "D3D12F0", valid after a successful register
// Called after every successful llama_decode: the first one clears the crash sentinel.
void d3d12_full_note_decode_ok();
// Logs the first few llama_decode calls (token count) so a death mid-generation is placed to the step.
void d3d12_full_note_decode_begin(int n_tokens);
// Step marker for the first generate of a launch (limited count), to place a silent death.
void d3d12_full_trace(const char* what);
#else
inline bool d3d12_full_register() {
    return false;
}
inline bool d3d12_full_default_on() {
    return false;
}
inline const char* d3d12_full_device_name() {
    return nullptr;
}
inline void d3d12_full_note_decode_ok() {}
inline void d3d12_full_note_decode_begin(int) {}
inline void d3d12_full_trace(const char*) {}
#endif

// How many layers to put on the GPU. Pure, so it is host-tested.
//   requested      layers asked for (999 = "as many as fit")
//   n_layer        block_count from the GGUF header (0 = unknown -> trust the request)
//   file_bytes     size of the .gguf, a good proxy for the weight bytes
//   free_bytes     GPU memory still available (0 = unknown -> trust the request)
//   reserve_bytes  kept free for KV cache, compute buffers and the UI
// The "+1" is the output layer (final norm + lm_head), which llama.cpp counts as a layer.
inline int d3d12_full_fit_layers(int requested, int n_layer, std::uint64_t file_bytes,
                                 std::uint64_t free_bytes, std::uint64_t reserve_bytes) {
    if (requested <= 0)
        return 0;
    if (n_layer <= 0)
        return requested;
    const int want = std::min(requested, n_layer + 1);
    if (free_bytes == 0 || file_bytes == 0)
        return want;
    if (free_bytes <= reserve_bytes)
        return 0;
    const std::uint64_t usable = free_bytes - reserve_bytes;
    if (file_bytes <= usable)
        return want;
    const double per_layer = static_cast<double>(file_bytes) / static_cast<double>(n_layer + 1);
    const int fit = static_cast<int>(static_cast<double>(usable) / per_layer);
    return std::max(0, std::min(want, fit));
}

} // namespace xllama
