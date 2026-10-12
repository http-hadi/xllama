// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <exception>
#include <csignal>
#include "xllama/platform.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef XLLAMA_UWP
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
// clang-format off
    #include <windows.h>
    #define PSAPI_VERSION 2
    #include <psapi.h>
    #include <unknwn.h>
    #include <winrt/Windows.Storage.h>
    #include <dxgi1_4.h>
// clang-format on
#endif

namespace xllama {

int detect_threads() noexcept {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

int detect_threads_llama() noexcept {
#ifdef XLLAMA_UWP
    return std::min(detect_threads(), 6); // ggml livelock at t7/t8 (see platform.h)
#else
    return detect_threads();
#endif
}

#ifdef XLLAMA_UWP
// A process that dies without a word was the reason every GPU failure had to be guessed at: record fatal
// exceptions, std::terminate and abort() with the working set, so the next crash names itself.
static LONG CALLBACK xllama_vectored_handler(EXCEPTION_POINTERS* ep) {
    const DWORD c = ep->ExceptionRecord->ExceptionCode;
    if (c == 0xC0000005 || c == 0xC00000FD || c == 0xC000001D || c == 0xC0000374 || c == 0xC0000409 ||
        c == 0xC0000094 || c == 0xC000008C || c == 0x80000003 || c == 0xC0000017 || c == 0xC0000142) {
        char b[256];
        std::snprintf(b, sizeof(b), "[xllama] CRASH: exception 0x%08lx at %p (ws=%zu MB, avail=%zu MB)\n",
                      static_cast<unsigned long>(c), ep->ExceptionRecord->ExceptionAddress, working_set_mb(),
                      avail_phys_mb());
        log_output(b);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
static void xllama_terminate_handler() {
    log_output("[xllama] CRASH: std::terminate\n");
    std::abort();
}
static void xllama_abort_handler(int) {
    log_output("[xllama] CRASH: abort()\n");
}
static void install_crash_logging() {
    static bool done = false;
    if (done)
        return;
    done = true;
    AddVectoredExceptionHandler(1, xllama_vectored_handler);
    std::set_terminate(xllama_terminate_handler);
    std::signal(SIGABRT, xllama_abort_handler);
}
#endif

void log_output(const char* msg) noexcept {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
    // Also mirror to LocalFolder/xllama.log for post-mortem (bridge errors invisible otherwise).
    static std::mutex s_mtx;
    static FILE* s_fp = []() -> FILE* {
        try {
            auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
            std::wstring path = std::wstring(folder.Path().c_str()) + L"\\xllama.log";
            FILE* fp = _wfopen(path.c_str(), L"a");
            if (fp)
                setvbuf(fp, nullptr, _IONBF, 0); // unbuffered: writes survive hard kill
            return fp;
        } catch (...) {
            return nullptr;
        }
    }();
    static const bool s_crash_logging = (install_crash_logging(), true);
    (void)s_crash_logging;
    if (s_fp) {
        std::lock_guard<std::mutex> g(s_mtx);
        std::fputs(msg, s_fp); // _IONBF: no CRT buffer, write reaches kernel directly
    }
#else
    std::fputs(msg, stderr);
#endif
}

void log_output(const std::string& msg) noexcept {
    log_output(msg.c_str());
}

std::size_t peak_working_set_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    return 0;
#else
    // Linux host: peak RSS from /proc (VmHWM is in kB). Was a hard 0 before —
    // host coding/campaign benches then reported peak=0MB and looked broken.
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t working_set_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024 * 1024);
    return 0;
#else
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t avail_phys_mb() noexcept {
#ifdef XLLAMA_UWP
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<std::size_t>(ms.ullAvailPhys / (1024 * 1024));
    return 0;
#else
    // MemAvailable, not MemFree: the kernel's own estimate of what a new
    // allocation can claim without swapping (includes reclaimable page cache).
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 13, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

GpuMemInfo gpu_mem_info() noexcept {
#ifdef XLLAMA_UWP
    // Adapter cached for process lifetime: gpu_mem_info() is called per phase
    // (pre-load / post-load / post-decode) and must stay cheap.
    static IDXGIAdapter3* s_adapter = []() -> IDXGIAdapter3* {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            return nullptr;
        IDXGIAdapter1* adapter1 = nullptr;
        IDXGIAdapter3* adapter3 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter1))) {
            adapter1->QueryInterface(IID_PPV_ARGS(&adapter3));
            adapter1->Release();
        }
        factory->Release();
        return adapter3;
    }();

    GpuMemInfo info;
    if (!s_adapter)
        return info;
    DXGI_QUERY_VIDEO_MEMORY_INFO vmi{};
    if (FAILED(s_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vmi)))
        return info;
    info.current_mb = static_cast<std::size_t>(vmi.CurrentUsage / (1024 * 1024));
    info.budget_mb = static_cast<std::size_t>(vmi.Budget / (1024 * 1024));
    info.available = true;
    return info;
#else
    return {};
#endif
}

void set_cwd_to_local_folder() noexcept {
#ifdef XLLAMA_UWP
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        SetCurrentDirectoryW(folder.Path().c_str());
    } catch (...) {
    }
#endif
}

} // namespace xllama
