#!/usr/bin/env python3
"""Adapt the vendored koza-coder ggml-d3d12 backend for xllama.

Input : third_party/ggml-d3d12-full/ggml-d3d12.cpp + ggml-d3d12.h as vendored
        verbatim (commit "vendor: koza-coder/llama.cpp-directx12 ... verbatim").
Output: the same files, patched in place.

What the port does
  1. Renames the exported API (xllama_d3d12f_*) and the backend name ("D3D12F") so it
     can sit next to xllama's older single-purpose d3d12 backend without symbol clashes.
  2. XLLAMA_D3D12_UWP: Xbox UWP build. No dxcompiler.dll / d3dcompiler (not loadable in an
     AppContainer), d3d12.dll loaded dynamically, desktop-only Win32 calls removed.
     Shaders come from a *precompiled* cache (DXIL baked in CI) read from the package folder.
     A missing shader flags the device as failed (graph_compute -> GGML_STATUS_FAILED, so
     llama_decode errors out and the app can fall back to CPU) instead of aborting.
  3. Desktop "bake" support: GGML_D3D12_CACHE_DIR (where compiled DXIL goes),
     GGML_D3D12_MANIFEST (append "<file>\\t<pipeline key>" per pipeline requested) and
     GGML_D3D12_FAKE_VENDOR (hex vendor id; makes a CI software adapter request the same
     pipelines an AMD UMA console GPU does).
Every replacement must match exactly once, so a changed upstream fails loudly.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "third_party" / "ggml-d3d12-full"
cpp_path = root / "ggml-d3d12.cpp"
h_path = root / "ggml-d3d12.h"
src = cpp_path.read_text(encoding="utf-8")
hdr = h_path.read_text(encoding="utf-8")


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"port: expected exactly 1 match for [{what}], found {n}")
    return text.replace(old, new)


def between(text, start, end, what):
    """Return (i, j) such that text[i:j] spans from `start` (inclusive) to `end` (exclusive)."""
    i = text.find(start)
    if i < 0 or text.count(start) != 1:
        sys.exit(f"port: start marker not unique/found for [{what}]")
    j = text.find(end, i)
    if j < 0:
        sys.exit(f"port: end marker not found for [{what}]")
    return i, j


# ---------------------------------------------------------------- header + names
hdr = sub1(hdr, '#define GGML_D3D12_NAME "D3D12"', '#define GGML_D3D12_NAME "D3D12F"', "name")
hdr = sub1(hdr, "GGML_BACKEND_API ggml_backend_t ggml_backend_d3d12_init(int device);",
           "GGML_BACKEND_API ggml_backend_t xllama_d3d12f_init(int device);", "init decl")
hdr = sub1(hdr, "GGML_BACKEND_API ggml_backend_reg_t ggml_backend_d3d12_reg(void);",
           "GGML_BACKEND_API ggml_backend_reg_t xllama_d3d12f_reg(void);\n\n"
           "// Folder holding the precompiled DXIL cache (UWP: <package>\\\\d3d12-shader-cache; desktop: optional\n"
           "// override). Call before the first xllama_d3d12f_reg().\n"
           "GGML_BACKEND_API void xllama_d3d12f_set_shader_dir(const wchar_t * dir);",
           "reg decl")
hdr = hdr.replace('#include "ggml.h"', '#include <stddef.h>\n#include "ggml.h"', 1)

# exact-identifier rename of the two exported functions (static helpers such as
# ggml_backend_d3d12_reg_context keep their names: \b does not match before '_')
src = re.sub(r"\bggml_backend_d3d12_reg\b", "xllama_d3d12f_reg", src)
src = re.sub(r"\bggml_backend_d3d12_init\b", "xllama_d3d12f_init", src)

# ---------------------------------------------------------------- includes + shims
src = sub1(src, "#include <dxcapi.h>\n#ifdef GGML_D3D12_FXC_FA",
           "#ifndef XLLAMA_D3D12_UWP\n#include <dxcapi.h>\n#endif\n#ifdef GGML_D3D12_FXC_FA", "dxcapi include")

SHIM = r'''#include <vector>

// ===== xllama port ================================================================================
static std::wstring g_xllama_shader_dir;   // precompiled-DXIL folder (see xllama_d3d12f_set_shader_dir)

#ifdef XLLAMA_D3D12_UWP
// Xbox UWP / AppContainer: d3d12.dll is resolved at run time (as xllama's d3d12_dyn.h does) and there
// is no shader compiler, so every pipeline must come from the precompiled cache.
typedef void d3d12_compiler_t;
#define GGML_D3D12_DEV_COMPILER(dev) nullptr
typedef HRESULT (WINAPI * xllama_pfn_create_device)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
typedef HRESULT (WINAPI * xllama_pfn_serialize_rs)(const D3D12_ROOT_SIGNATURE_DESC *, D3D_ROOT_SIGNATURE_VERSION,
                                                    ID3DBlob **, ID3DBlob **);
static HMODULE xllama_d3d12_module() {
    static HMODULE m = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return m;
}
static HRESULT WINAPI xllama_d3d12_create_device(IUnknown * adapter, D3D_FEATURE_LEVEL fl, REFIID riid, void ** out) {
    HMODULE m = xllama_d3d12_module();
    auto    f = m ? (xllama_pfn_create_device) (void *) GetProcAddress(m, "D3D12CreateDevice") : nullptr;
    return f ? f(adapter, fl, riid, out) : E_NOTIMPL;
}
static HRESULT WINAPI xllama_d3d12_serialize_rs(const D3D12_ROOT_SIGNATURE_DESC * d, D3D_ROOT_SIGNATURE_VERSION v,
                                                ID3DBlob ** blob, ID3DBlob ** err) {
    HMODULE m = xllama_d3d12_module();
    auto    f = m ? (xllama_pfn_serialize_rs) (void *) GetProcAddress(m, "D3D12SerializeRootSignature") : nullptr;
    return f ? f(d, v, blob, err) : E_NOTIMPL;
}
#define D3D12CreateDevice           xllama_d3d12_create_device
#define D3D12SerializeRootSignature xllama_d3d12_serialize_rs
#define D3D12GetDebugInterface(...) (E_NOINTERFACE)   // debug layer / DRED do not exist in a retail AppContainer
#else
typedef IDxcCompiler3 d3d12_compiler_t;
#define GGML_D3D12_DEV_COMPILER(dev) (dev).compiler.get()
#endif

static bool ggml_d3d12_file_exists(const std::wstring & file) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    return !file.empty() && GetFileAttributesExW(file.c_str(), GetFileExInfoStandard, &data) != 0;
}

extern "C" void xllama_d3d12f_set_shader_dir(const wchar_t * dir) {
    g_xllama_shader_dir = dir ? dir : L"";
}

#ifdef XLLAMA_D3D12_UWP
static void ggml_d3d12_manifest_add(const std::wstring &, const std::string &) {}
#else
// bake support: GGML_D3D12_MANIFEST=<file> appends "<cache file name>\t<pipeline key>" per pipeline requested
static void ggml_d3d12_manifest_add(const std::wstring & cache_file, const std::string & key) {
    static const char * path = getenv("GGML_D3D12_MANIFEST");
    static std::mutex   mutex;
    if (!path || cache_file.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (FILE * f = fopen(path, "ab")) {
        const size_t slash = cache_file.find_last_of(L"\\/");
        const std::string name = ggml_d3d12_wide_to_utf8(cache_file.c_str() + (slash == std::wstring::npos ? 0 : slash + 1));
        fprintf(f, "%s\t%s\n", name.c_str(), key.c_str());
        fclose(f);
    }
}
#endif
// ===== end xllama port ============================================================================
'''
# ggml_d3d12_wide_to_utf8 is defined later than <vector>; forward-declare for the manifest helper
src = sub1(src, "#include <vector>\n", SHIM, "shim insertion")
src = sub1(src, "// ===== xllama port =====",
           "static std::string ggml_d3d12_wide_to_utf8(const wchar_t * w);\n// ===== xllama port =====", "fwd decl")

# ---------------------------------------------------------------- device ctx members
src = sub1(src,
    "    // shader compiler\n    HMODULE                 dxc_module = nullptr;\n"
    "    com_ptr<IDxcCompiler3>  compiler;\n    com_ptr<IDxcUtils>      utils;\n",
    "    bool                    failed = false;   // xllama: a precompiled shader was missing/rejected\n"
    "#ifndef XLLAMA_D3D12_UWP\n    // shader compiler\n    HMODULE                 dxc_module = nullptr;\n"
    "    com_ptr<IDxcCompiler3>  compiler;\n    com_ptr<IDxcUtils>      utils;\n#endif\n",
    "ctx members")

# ---------------------------------------------------------------- shader cache path
src = sub1(src,
    '    HMODULE module = nullptr;\n    wchar_t path[MAX_PATH];\n'
    '    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,\n'
    '                            (LPCWSTR) &ggml_d3d12_shader_cache_file, &module) ||\n'
    '        GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {   // 0 = error, MAX_PATH = truncated\n'
    '        return L"";\n    }\n    std::wstring dir = path;\n'
    '    dir = dir.substr(0, dir.find_last_of(L"\\\\/") + 1) + L"d3d12-shader-cache";\n'
    '    CreateDirectoryW(dir.c_str(), nullptr);\n',
    '#ifdef XLLAMA_D3D12_UWP\n'
    '    if (g_xllama_shader_dir.empty()) {\n        return L"";\n    }\n'
    '    std::wstring dir = g_xllama_shader_dir;\n'
    '    while (!dir.empty() && (dir.back() == L\'\\\\\' || dir.back() == L\'/\')) {\n        dir.pop_back();\n    }\n'
    '#else\n'
    '    std::wstring dir;\n'
    '    if (!g_xllama_shader_dir.empty()) {\n        dir = g_xllama_shader_dir;\n'
    '    } else if (const wchar_t * env = _wgetenv(L"GGML_D3D12_CACHE_DIR")) {\n        dir = env;\n'
    '    } else {\n'
    '        HMODULE module = nullptr;\n        wchar_t path[MAX_PATH];\n'
    '        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,\n'
    '                                (LPCWSTR) &ggml_d3d12_shader_cache_file, &module) ||\n'
    '            GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {   // 0 = error, MAX_PATH = truncated\n'
    '            return L"";\n        }\n        dir = path;\n'
    '        dir = dir.substr(0, dir.find_last_of(L"\\\\/") + 1) + L"d3d12-shader-cache";\n    }\n'
    '    CreateDirectoryW(dir.c_str(), nullptr);\n'
    '#endif\n',
    "cache dir")

# the cache file name starts with a backslash; make sure it is joined correctly in both modes (it already is)

# write_file: not available (and not wanted) in an AppContainer package folder
i, j = between(src, "// write to a per-process temp name and rename", "// counts shaders compiled from source", "write_file")
old_write = src[i:j]
src = src[:i] + "#ifdef XLLAMA_D3D12_UWP\nstatic void ggml_d3d12_write_file(const std::wstring &, const void *, size_t) {}\n#else\n" + \
      old_write.rstrip("\n") + "\n#endif\n\n" + src[j:]

# ---------------------------------------------------------------- build_pipeline
src = sub1(src, "                                               IDxcCompiler3 *                  compiler,",
           "                                               d3d12_compiler_t *               compiler,", "compiler param")

i, j = between(src,
    "    const std::wstring   cache_file = ggml_d3d12_shader_cache_file(source, args);\n    for (int attempt = 0;; attempt++) {",
    "// defines: list of \"NAME\" or \"NAME=VALUE\"", "dxc loop")
region = src[i:j]
UWP_BODY = r'''#ifdef XLLAMA_D3D12_UWP
    (void) compiler;
    const std::wstring   cache_file = ggml_d3d12_shader_cache_file(source, args);
    std::vector<uint8_t> dxil       = ggml_d3d12_read_file(cache_file);
    d3d12_pipeline       pipeline;
    pipeline.name = key;
    if (dxil.empty()) {
        GGML_LOG_ERROR("ggml_d3d12: precompiled shader missing for [%s] (cache %ls)\n", key.c_str(),
                       cache_file.empty() ? L"<no dir>" : cache_file.c_str());
        dev.failed = true;
        return pipeline;   // null PSO: ggml_d3d12_dispatch refuses it and graph_compute reports failure
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature                    = dev.root_sig.get();
    pso_desc.CS.pShaderBytecode                = dxil.data();
    pso_desc.CS.BytecodeLength                 = dxil.size();
    const HRESULT hr = dev.device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(pipeline.pso.put()));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d12: CreateComputePipelineState failed for [%s] (HRESULT 0x%08lx)\n", key.c_str(),
                       (unsigned long) hr);
        dev.failed = true;
        return pipeline;   // creation failed, so pso is still null
    }
    std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
    dev.n_compiles++;
    dev.t_compile_us += ggml_d3d12_time_us() - t_compile0;
    return pipeline;
}
#else
'''
region = region.replace(
    "    const std::wstring   cache_file = ggml_d3d12_shader_cache_file(source, args);\n",
    "    const std::wstring   cache_file = ggml_d3d12_shader_cache_file(source, args);\n"
    "    ggml_d3d12_manifest_add(cache_file, key);\n", 1)
src = src[:i] + UWP_BODY + region.rstrip("\n") + "\n#endif\n\n" + src[j:]

src = sub1(src, "ggml_d3d12_build_pipeline(dev, dev.compiler.get(), key, source, defines, min_sm);",
           "ggml_d3d12_build_pipeline(dev, GGML_D3D12_DEV_COMPILER(dev), key, source, defines, min_sm);", "get_pipeline call")

# ---------------------------------------------------------------- build_pipeline_jobs
src = sub1(src, "        n_missing += GetFileAttributesW(cf.c_str()) == INVALID_FILE_ATTRIBUTES;",
           "        n_missing += !ggml_d3d12_file_exists(cf);", "attr check")
i, j = between(src, "// builds the queued pipelines on worker threads, each with its own compiler instance",
               "/* Dispatch encoding */", "jobs fn")
old_jobs = src[i:j]
UWP_JOBS = r'''#ifdef XLLAMA_D3D12_UWP
// builds the queued pipelines from the precompiled cache on worker threads (PSO creation only, no compiler)
static void ggml_d3d12_build_pipeline_jobs(d3d12_device_ctx & dev) {
    std::vector<d3d12_device_ctx::pipeline_job> jobs;
    jobs.swap(dev.pipeline_jobs);
    if (jobs.empty()) {
        return;
    }
    const size_t        n_threads = std::min<size_t>(jobs.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<size_t> next{ 0 };
    auto worker = [&]() {
        for (size_t i; (i = next++) < jobs.size();) {
            const auto &   j = jobs[i];
            d3d12_pipeline p = ggml_d3d12_build_pipeline(dev, nullptr, j.key, j.source, j.defines, j.min_sm);
            std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
            dev.pipelines.emplace(j.key, std::move(p));
        }
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < n_threads; t++) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & th : threads) {
        th.join();
    }
}
#else
'''
src = src[:i] + UWP_JOBS + old_jobs.rstrip("\n") + "\n#endif\n\n" + src[j:]

# ---------------------------------------------------------------- dispatch choke point + failure report
src = sub1(src,
    "    GGML_ASSERT(uavs.size() <= D3D12_MAX_ROOT_UAVS);\n    if (dev.collecting) {\n        return;\n    }\n",
    "    GGML_ASSERT(uavs.size() <= D3D12_MAX_ROOT_UAVS);\n    if (dev.collecting) {\n        return;\n    }\n"
    "    if (!pipeline.pso) {   // xllama: shader missing or rejected, refuse instead of binding a null PSO\n"
    "        dev.failed = true;\n        return;\n    }\n",
    "dispatch guard")
src = sub1(src, "    ggml_d3d12_submit_and_wait(dev);\n    dev.n_graphs++;",
           "    ggml_d3d12_submit_and_wait(dev);\n"
           "    if (dev.failed) {\n        return GGML_STATUS_FAILED;\n    }\n    dev.n_graphs++;", "graph failure")

# ---------------------------------------------------------------- device init: no DXC on the console
i, j = between(src, "// Loads a DLL from the folder this backend DLL sits in",
               "static bool ggml_d3d12_init_device(", "load_beside")
src = src[:i] + "#ifndef XLLAMA_D3D12_UWP\n" + src[i:j].rstrip("\n") + "\n#endif\n\n" + src[j:]
i, j = between(src, "    // shader compiler: dxcompiler.dll + dxil.dll next to this DLL",
               "    // stats and GPU timestamp profiling", "dxc init")
src = src[:i] + "#ifndef XLLAMA_D3D12_UWP\n" + src[i:j].rstrip("\n") + "\n#endif\n\n" + src[j:]

# ---------------------------------------------------------------- bake: pretend to be an AMD UMA console GPU
src = sub1(src, "        dev->vendor_id     = desc.VendorId;\n",
           "        dev->vendor_id     = desc.VendorId;\n"
           "        if (const char * fv = getenv(\"GGML_D3D12_FAKE_VENDOR\")) {   // bake: see port script docstring\n"
           "            dev->vendor_id = (uint32_t) strtoul(fv, nullptr, 16);\n        }\n", "fake vendor")

# ---------------------------------------------------------------- ggml API drift
# The backend was written against a newer ggml whose buffer-type interface has two extra callbacks
# (alloc_buffer_n / get_alloc_size_n). xllama pins an older llama.cpp without them; both were NULL anyway.
src = sub1(src, "            /* .alloc_buffer_n = */ NULL,\n", "", "alloc_buffer_n")
src = sub1(src, "            /* .get_alloc_size_n = */ NULL,\n", "", "get_alloc_size_n")

cpp_path.write_text(src, encoding="utf-8")
h_path.write_text(hdr, encoding="utf-8")
print("port ok:", cpp_path, f"({src.count(chr(10))} lines)")
