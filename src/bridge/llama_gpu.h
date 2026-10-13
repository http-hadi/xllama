// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The one place that turns an n_gpu_layers request into llama.cpp params for
// the d3d12 backend (docs/gguf-gpu-decode.md, D2b). Used by run_inference_llama
// (CLI, headless bench) and the persistent LlamaSession (GUI, LAN API).
#pragma once

#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/ggml_d3d12_full.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/speculative.h"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace xllama {

// llama/ggml log lines worth keeping in xllama.log when layers run on d3d12:
// placement and buffer sizes, graph splits, and every warning or error.
// ggml prints GGML_ABORT / GGML_ASSERT messages straight to stderr, which a UWP app never sees: route them to the log.
inline void gguf_gpu_abort_log(const char* msg) {
    log_output(std::string("[xllama] GGML ABORT: ") + (msg ? msg : "(null)") + "\n");
}

inline void gguf_gpu_log(ggml_log_level level, const char* text, void*) {
    if (!text)
        return;
    const bool keep = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR ||
                      std::strstr(text, "buffer size") || std::strstr(text, "offload") ||
                      std::strstr(text, "graph splits") || std::strstr(text, "D3D12") ||
                      std::strstr(text, "ggml_d3d12") || std::strstr(text, "load_tensors");
    if (keep)
        log_output(std::string("[llama] ") + text);
}

// Working set at a load stage, for every GGUF load: the CPU and d3d12 runs log
// the same stages, so a peak difference can be attributed (#309).
inline void log_gguf_ws(const char* stage) {
    log_output(std::string("[xllama] ws ") + stage + ": " + std::to_string(working_set_mb()) +
               " MB (avail phys " + std::to_string(avail_phys_mb()) + " MB)\n");
}

// token_embd's type and whether the model has its own output.weight, read from
// the GGUF header only (no tensor data). False when the header cannot be read.
inline bool gguf_embedding_info(const std::string& path, ggml_type* embd_type, bool* has_output) {
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g)
        return false;
    const int64_t embd = gguf_find_tensor(g, "token_embd.weight");
    const bool ok = embd >= 0;
    if (ok) {
        *embd_type = gguf_get_tensor_type(g, embd);
        *has_output = gguf_find_tensor(g, "output.weight") >= 0;
    }
    gguf_free(g);
    return ok;
}


// True while the whole-graph backend (ggml_d3d12_full.h) owns the offload, so the context must
// NOT use the legacy workarounds (KV on the CPU, host buffers, tied-embedding override).
inline bool g_gpu_full_active = false;
// Size of the model on the whole-graph path and the device budget, so the context can be sized to what is left.
inline std::uint64_t g_gpu_full_file_mb = 0;
inline std::uint64_t g_gpu_full_total_mb = 0;

// Per-layer numeric trace for the first decode(s) of a large GPU model. A tensor that first turns NaN/inf (or
// explodes) names the layer, and with it the op type and the CPU/GPU side, that the kernel bug lives in.
inline std::atomic<int> g_trace_left{0};

inline bool gguf_gpu_eval_cb(struct ggml_tensor* t, bool ask, void*) {
    if (g_trace_left.load() <= 0)
        return false;
    const char* n = t->name;
    const bool want = std::strncmp(n, "l_out-", 6) == 0 || std::strcmp(n, "result_norm") == 0 ||
                      std::strcmp(n, "result_output") == 0 || std::strcmp(n, "inp_embd") == 0;
    if (ask)
        return want;
    if (!want || t->type != GGML_TYPE_F32 || ggml_nbytes(t) > (size_t(16) << 20))
        return true;
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    double sum_abs = 0.0;
    float mn = 3.4e38f, mx = -3.4e38f;
    size_t n_nan = 0, n_inf = 0;
    for (float v : buf) {
        if (v != v) { ++n_nan; continue; }
        if (v > 3.0e38f || v < -3.0e38f) { ++n_inf; continue; }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum_abs += std::fabs(v);
    }
    const size_t n_ok = buf.size() - n_nan - n_inf;
    log_output(std::string("[xllama] trace ") + n + " [" + ggml_backend_buffer_name(t->buffer) + "] n=" +
               std::to_string(buf.size()) + " nan=" + std::to_string(n_nan) + " inf=" + std::to_string(n_inf) +
               " min=" + std::to_string(mn) + " max=" + std::to_string(mx) + " mean|x|=" +
               std::to_string(n_ok ? sum_abs / n_ok : 0.0) + "\n");
    g_trace_left.fetch_sub(1);
    return true;
}

// Logs the working set once a second while a GPU generate runs (first 40 ticks). A silent process death
// otherwise leaves nothing to tell a memory kill (jump, then nothing) from a GPU stall (ticks, then nothing).
class GpuHeartbeat {
  public:
    GpuHeartbeat() {
        if (!g_gpu_full_active)
            return;
        t_ = std::thread([this] {
            for (int i = 1; i <= 40 && !stop_.load(); ++i) {
                for (int k = 0; k < 10 && !stop_.load(); ++k)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (stop_.load())
                    break;
                log_output("[xllama] hb " + std::to_string(i) + "s ws=" + std::to_string(working_set_mb()) +
                           " MB avail=" + std::to_string(avail_phys_mb()) + " MB\n");
            }
        });
    }
    GpuHeartbeat(const GpuHeartbeat&) = delete;
    GpuHeartbeat& operator=(const GpuHeartbeat&) = delete;
    ~GpuHeartbeat() {
        stop_.store(true);
        if (t_.joinable())
            t_.join();
    }

  private:
    std::atomic<bool> stop_{false};
    std::thread t_;
};

// Small text knob from LocalState (trimmed, empty when absent).
inline std::string read_local_text(const char* name) {
    std::ifstream f(resolve_local_path(name));
    std::string s;
    if (f)
        std::getline(f, s);
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    return s;
}

// block_count of the GGUF (0 when unreadable), from the header only.
inline int gguf_block_count(const std::string& path) {
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g)
        return 0;
    int n = 0;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(g, arch)) + ".block_count";
        const int64_t k = gguf_find_key(g, key.c_str());
        if (k >= 0) {
            const gguf_type t = gguf_get_kv_type(g, k);
            if (t == GGUF_TYPE_UINT32)
                n = static_cast<int>(gguf_get_val_u32(g, k));
            else if (t == GGUF_TYPE_INT32)
                n = gguf_get_val_i32(g, k);
        }
    }
    gguf_free(g);
    return n;
}

// Whole-graph path: llama.cpp offloads layers the ordinary way (KV cache, attention, norms and
// the delta-net all execute on the device). Returns the layers actually offloaded, 0 = CPU.
inline int apply_gguf_gpu_layers_full(int requested, llama_model_params& mparams,
                                      const std::string& model_path) {
    if (!d3d12_full_register()) {
        log_output("[xllama] gguf gpu layers: full d3d12 backend unavailable, using the CPU\n");
        return 0;
    }
    const char* name = d3d12_full_device_name();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name ? name : "D3D12F0");
    if (!dev) {
        log_output("[xllama] gguf gpu layers: no D3D12F device registered, using the CPU\n");
        return 0;
    }
    const int n_layer = gguf_block_count(model_path);
    std::error_code ec;
    const std::uint64_t file_bytes = std::filesystem::file_size(model_path, ec);
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);

    // Room kept for the KV cache, compute buffers and the system's own use of the GPU.
    std::uint64_t reserve = 1024ull << 20;
    const std::string rmb = read_local_text("d3d12f_reserve_mb.txt");
    if (!rmb.empty())
        reserve = static_cast<std::uint64_t>(std::max(0, std::atoi(rmb.c_str()))) << 20;

    const int layers = d3d12_full_fit_layers(requested, n_layer, ec ? 0 : file_bytes, free_b, reserve);
    char b[256];
    std::snprintf(b, sizeof(b),
                  "[xllama] gguf gpu layers (full d3d12): requested=%d n_layer=%d file=%llu MB "
                  "gpu_free=%zu MB gpu_total=%zu MB reserve=%llu MB -> %d\n",
                  requested, n_layer, static_cast<unsigned long long>(ec ? 0 : file_bytes >> 20),
                  free_b >> 20, total_b >> 20, static_cast<unsigned long long>(reserve >> 20), layers);
    log_output(b);
    if (layers <= 0) {
        log_output("[xllama] gguf gpu layers: the model does not fit the GPU budget, using the CPU\n");
        return 0;
    }
    static ggml_backend_dev_t devs[] = {nullptr, nullptr};
    devs[0] = dev;
    mparams.devices = devs;
    mparams.n_gpu_layers = layers;
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.main_gpu = 0;
    g_gpu_full_active = true;
    g_gpu_full_file_mb = ec ? 0 : (file_bytes >> 20);
    g_gpu_full_total_mb = total_b >> 20;
    return layers;
}

// Sets mparams.devices / n_gpu_layers and returns the layers actually
// offloaded: 0 when none were asked for or the device is unavailable, in
// which case the model loads exactly as before (CPU, no GPU device listed).
inline int apply_gguf_gpu_layers(int requested, llama_model_params& mparams,
                                 const std::string& model_path) {
    static ggml_backend_dev_t no_devices[] = {nullptr};
    mparams.n_gpu_layers = 0;
    mparams.devices = no_devices;
    g_gpu_full_active = false;
    if (requested <= 0)
        return 0;
    log_gguf_ws("before d3d12");
    llama_log_set(gguf_gpu_log, nullptr);
    ggml_set_abort_callback(gguf_gpu_abort_log);
    // Default: the whole-graph backend. LocalState\d3d12_backend.txt containing "legacy" selects the
    // older matmul-only backend below (kept for A/B comparison).
    if (read_local_text("d3d12_backend.txt") != "legacy")
        return apply_gguf_gpu_layers_full(requested, mparams, model_path);
    if (!ggml_d3d12_register()) {
        log_output("[xllama] gguf gpu layers: d3d12 backend unavailable, using the CPU\n");
        return 0;
    }
    static ggml_backend_dev_t d3d12[] = {ggml_backend_dev_by_name("D3D12"), nullptr};
    if (!d3d12[0]) {
        log_output("[xllama] gguf gpu layers: no D3D12 device registered, using the CPU\n");
        return 0;
    }
    mparams.devices = d3d12;
    mparams.n_gpu_layers = requested;
    // D3D12_Host is also the device's host buft, so llama.cpp puts the CPU
    // backend's compute buffer there and splits share activations in place.
    // Keep it out of the CPU *weight* list: weights left on the CPU stay in the
    // CPU / repack bufts exactly as on a CPU-only load.
    mparams.no_host = true;
    // A tied token_embd also serves as the lm_head: llama.cpp would keep the
    // input copy on the CPU and duplicate it into D3D12_Weights (244 MiB on
    // Coder-3B). Placing it in D3D12_Weights, where GET_ROWS also runs, leaves
    // one copy: the duplicate finds the original in the same buffer context.
    ggml_type embd_type = GGML_TYPE_COUNT;
    bool has_output = true;
    if (gguf_embedding_info(model_path, &embd_type, &has_output) &&
        d3d12_place_tied_embedding(embd_type, has_output)) {
        static llama_model_tensor_buft_override embd_on_gpu[] = {{"^token_embd\\.weight$", nullptr},
                                                                 {nullptr, nullptr}};
        embd_on_gpu[0].buft = ggml_d3d12_weights_buft();
        mparams.tensor_buft_overrides = embd_on_gpu;
        log_output("[xllama] gguf gpu layers: tied token_embd in D3D12_Weights\n");
    }
    log_gguf_ws("after d3d12 init");
    llama_log_set(gguf_gpu_log, nullptr);
    ggml_set_abort_callback(gguf_gpu_abort_log);
    log_output("[xllama] gguf gpu layers: " + std::to_string(requested) + " on D3D12\n");
    return requested;
}

// Most logits rows a context will request per ubatch, for n_outputs_max: one
// for prefill and decode, 1 + draft for a prompt-lookup verify batch
// (decode_loop.h), unlimited (0 = llama.cpp's default) for embeddings, which
// output every token. llama.cpp asserts in output_reserve when a batch asks
// for more, so this must cover every batch the context will see.
inline uint32_t gguf_gpu_outputs_max(bool embeddings, bool prompt_lookup) {
    if (embeddings)
        return 0;
    return 1u + (prompt_lookup ? static_cast<uint32_t>(kSpecDraftKDefault) : 0u);
}

// KV cache and attention stay on the CPU in ordinary memory: the backend only
// runs weight matmuls. Flash Attention AUTO would resolve to disabled here:
// llama.cpp compares each layer's device (D3D12) with the FA node's (the CPU,
// beside the KV cache), and any mismatch turns FA off. The CPU supports FA and
// the CPU-only path resolves AUTO to enabled, so the GPU path asks for it
// explicitly and keeps the same attention (#310).
// n_outputs_max also sizes the compute reserve: llama.cpp
// reserves the prefill graph for that many logits rows (n_ubatch by default:
// 512 x 151936 x 4 B = 297 MiB on Coder-3B), and on the GPU path the reserve
// is a committed D3D12_Host buffer, where the CPU's malloc'd one only counts
// the pages it touches (#309).
inline void apply_gguf_gpu_context(int applied_layers, llama_context_params& cparams,
                                   uint32_t outputs_max) {
    // Whole-graph backend: keep llama.cpp's defaults (KV cache on the device, flash attention AUTO), except the
    // logits reserve. The prefill graph is reserved for n_ubatch logits rows by default: 512 x 248320 x 4 B =
    // 485 MiB on Bonsai 27B, one D3D12 default-heap allocation that failed on the Xbox (HRESULT 0x8007000e)
    // after the weights were already resident. Generation only reads the last row (plus the draft rows when
    // prompt lookup is on), exactly what the legacy path already limits it to.
    if (g_gpu_full_active) {
        if (applied_layers > 0 && outputs_max > 0)
            cparams.n_outputs_max = outputs_max;
        // The compute buffers scale with n_ubatch and are committed at the first decode. A model that fills most of
        // the device budget (Bonsai 27B: 3627 of 4147 MB) has no room for the 512-row default, so cap it.
        if (applied_layers > 0 && g_gpu_full_total_mb > 0 && outputs_max > 0) {
            uint32_t cap = 512;
            if (g_gpu_full_file_mb * 100 > g_gpu_full_total_mb * 75)
                cap = 64;
            else if (g_gpu_full_file_mb * 100 > g_gpu_full_total_mb * 40)
                cap = 128;
            if (cparams.n_ubatch > cap)
                cparams.n_ubatch = cap;
        }
        // Trace the first decode of a model that fills the device.
        if (applied_layers > 0 && g_gpu_full_total_mb > 0 && g_gpu_full_file_mb * 100 > g_gpu_full_total_mb * 40) {
            g_trace_left.store(160);
            cparams.cb_eval = gguf_gpu_eval_cb;
            cparams.cb_eval_user_data = nullptr;
        }
        return;
    }
    if (applied_layers > 0) {
        cparams.offload_kqv = false;
        if (cparams.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO)
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (outputs_max > 0)
            cparams.n_outputs_max = outputs_max;
    }
}

// Persistent CPU threadpools for a context that alternates CPU and d3d12
// splits. Without one attached, ggml-cpu builds and joins a disposable pool on
// every graph_compute (ggml-cpu.c) — once per token on a CPU-only graph, but
// ~50-70 times per token once matmuls run on the GPU: lfm25-350m decoded at
// 26.7 tok/s with 6 threads and 66.9 with 1 (D2b smoke, CI 1.6.0.1130). The
// CPU-only path keeps llama's default so its baseline does not move. Must
// outlive every context it is attached to (declare it before the context).
class GgufCpuThreadpools {
  public:
    GgufCpuThreadpools() = default;
    GgufCpuThreadpools(const GgufCpuThreadpools&) = delete;
    GgufCpuThreadpools& operator=(const GgufCpuThreadpools&) = delete;
    ~GgufCpuThreadpools() {
        reset();
    }

    void attach(int applied_layers, llama_context* ctx, int n_threads, int n_threads_batch) {
        if (applied_layers <= 0 || !ctx || n_threads <= 0)
            return;
        if (!tp_) {
            ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
            tp_ = ggml_threadpool_new(&p);
            if (n_threads_batch > 0 && n_threads_batch != n_threads) {
                ggml_threadpool_params pb = ggml_threadpool_params_default(n_threads_batch);
                tp_batch_ = ggml_threadpool_new(&pb);
            }
        }
        if (tp_)
            llama_attach_threadpool(ctx, tp_, tp_batch_ ? tp_batch_ : tp_);
    }

    void reset() {
        if (tp_batch_)
            ggml_threadpool_free(tp_batch_);
        if (tp_)
            ggml_threadpool_free(tp_);
        tp_ = tp_batch_ = nullptr;
    }

  private:
    ggml_threadpool_t tp_ = nullptr;
    ggml_threadpool_t tp_batch_ = nullptr;
};

} // namespace xllama
