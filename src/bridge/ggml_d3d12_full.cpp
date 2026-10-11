// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// UWP glue for the whole-graph D3D12 backend; see include/xllama/ggml_d3d12_full.h.

#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)

    #include "xllama/ggml_d3d12_full.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"

    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
// clang-format off
    // windows.h must precede unknwn.h; unknwn.h must precede winrt/ headers.
    #include <windows.h>
    #include <unknwn.h>
    #include <winrt/Windows.ApplicationModel.h>
// clang-format on

    #include "ggml-backend.h"
    #include "ggml-d3d12.h" // third_party/ggml-d3d12-full

    #include <atomic>
    #include <cstdio>
    #include <cstdlib>
    #include <mutex>
    #include <string>

namespace xllama {
namespace {

std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 1)
        return std::wstring();
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

std::wstring local_wide(const char* name) {
    return widen(resolve_local_path(name));
}

bool exists(const std::wstring& path) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    return !path.empty() && GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &d) != 0;
}

std::wstring baked_shader_dir() {
    using namespace winrt::Windows::ApplicationModel;
    std::wstring p(Package::Current().InstalledPath().c_str());
    return p + L"\\d3d12-shader-cache";
}

// True when the package carries at least one baked shader (a build without the bake step
// ships none, and the backend must then stay off rather than fail every pipeline).
bool has_baked_shaders(const std::wstring& dir) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileExW((dir + L"\\*.dxil").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch,
                                nullptr, 0);
    if (h == INVALID_HANDLE_VALUE)
        return false;
    FindClose(h);
    return true;
}

// LocalState\d3d12f_env.txt: one KEY=VALUE per line, applied before the backend initialises.
// Lets a console owner try GGML_D3D12_* tuning/diagnostic switches without a rebuild, e.g.
//   GGML_D3D12_DISABLE_OPS=FLASH_ATTN_EXT
void apply_env_file() {
    FILE* f = nullptr;
    if (_wfopen_s(&f, local_wide("d3d12f_env.txt").c_str(), L"rb") != 0 || !f)
        return;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
            s.pop_back();
        const size_t eq = s.find('=');
        if (s.empty() || s[0] == '#' || eq == std::string::npos || eq == 0)
            continue;
        _putenv_s(s.substr(0, eq).c_str(), s.substr(eq + 1).c_str());
        log_output("[xllama] d3d12f: env " + s + "\n");
    }
    std::fclose(f);
}

std::string g_device_name;
std::atomic<bool> g_sentinel_armed{false};

} // namespace

// Decided once per launch, *before* d3d12_full_register() arms the crash sentinel (re-reading the sentinel
// later would make a second model loaded before the first decode finished fall back to the CPU), and always
// logged, so a silent CPU fallback can be told apart from "the GPU never ran".
bool d3d12_full_default_on() {
    static std::once_flag once;
    static bool on = false;
    std::call_once(once, [] {
        const char* why = nullptr;
        if (exists(local_wide("d3d12f_disable.txt")))
            why = "d3d12f_disable.txt is present";
        else if (exists(local_wide("d3d12f_attempt.txt")) && exists(local_wide("d3d12f_safe.txt")))
            why = "the previous launch did not finish a GPU decode and d3d12f_safe.txt asks for the CPU in that case";
        else if (!has_baked_shaders(baked_shader_dir()))
            why = "this package carries no baked shaders (d3d12-shader-cache)";
        on = (why == nullptr);
        log_output(on ? std::string("[xllama] d3d12f: GPU default ON for GGUF models (baked shaders present)\n")
                      : std::string("[xllama] d3d12f: GPU default OFF, models stay on the CPU: ") + why + "\n");
    });
    return on;
}

bool d3d12_full_register() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        if (exists(local_wide("d3d12f_disable.txt"))) {
            log_output("[xllama] d3d12f: disabled by d3d12f_disable.txt, using the CPU\n");
            return;
        }
        const std::wstring sentinel = local_wide("d3d12f_attempt.txt");
        if (exists(sentinel)) {
            if (exists(local_wide("d3d12f_safe.txt"))) {
                // Opt-in legacy behaviour: after a GPU crash run one launch on the CPU.
                DeleteFileW(sentinel.c_str());
                log_output("[xllama] d3d12f: previous GPU run did not finish a decode -> CPU for this "
                           "launch (d3d12f_safe.txt present)\n");
                return;
            }
            // Default: GPU-only. A crash must never silently move the user to the CPU.
            log_output("[xllama] d3d12f: WARNING previous GPU run did not finish a decode; staying on the GPU "
                       "(create d3d12f_disable.txt to turn it off, d3d12f_safe.txt for CPU-after-crash)\n");
        }
        const std::wstring dir = baked_shader_dir();
        if (!has_baked_shaders(dir)) {
            log_output("[xllama] d3d12f: no baked shaders in the package, using the CPU\n");
            return;
        }
        apply_env_file();
        xllama_d3d12f_set_shader_dir(dir.c_str());

        // Arm before touching the GPU; cleared by the first successful decode.
        if (FILE* f = nullptr; _wfopen_s(&f, sentinel.c_str(), L"wb") == 0 && f) {
            std::fputs("armed\n", f);
            std::fclose(f);
            g_sentinel_armed = true;
        }
        ggml_backend_reg_t reg = xllama_d3d12f_reg();
        const size_t n = reg ? ggml_backend_reg_dev_count(reg) : 0;
        if (n == 0) {
            log_output("[xllama] d3d12f: no D3D12 device, using the CPU\n");
            DeleteFileW(sentinel.c_str());
            g_sentinel_armed = false;
            return;
        }
        ggml_backend_register(reg);
        g_device_name = ggml_backend_dev_name(ggml_backend_reg_dev_get(reg, 0));
        log_output("[xllama] d3d12f: registered " + g_device_name + " (" + std::to_string(n) +
                   " device(s)), shaders from the package\n");
        ok = true;
    });
    return ok;
}

const char* d3d12_full_device_name() {
    return g_device_name.empty() ? nullptr : g_device_name.c_str();
}

void d3d12_full_note_decode_ok() {
    if (!g_sentinel_armed.load(std::memory_order_relaxed))
        return;
    if (g_sentinel_armed.exchange(false)) {
        DeleteFileW(local_wide("d3d12f_attempt.txt").c_str());
        log_output("[xllama] d3d12f: first GPU decode completed, crash sentinel cleared\n");
    }
}

} // namespace xllama

#endif // XLLAMA_UWP && XLLAMA_USE_LLAMA
