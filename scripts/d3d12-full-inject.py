#!/usr/bin/env python3
"""Inject the whole-graph D3D12 backend into a pristine llama.cpp checkout.

Used by the CI *bake* build (.github/workflows/d3d12-full-release.yml): a desktop Windows build of
xllama's pinned llama.cpp that contains the same backend source the UWP package compiles, so it can
run real graphs on the runner's software D3D12 adapter, compile every HLSL variant those graphs
request with DXC, and save the DXIL for the console package.

    python scripts/d3d12-full-inject.py llama.cpp
    cmake -S llama.cpp -B build -DGGML_D3D12F=ON ...

It adds ggml/src/ggml-d3d12f/ (backend + shader header + CMakeLists), one ggml_add_backend() line
and the registry hook. It never touches the xllama UWP build, which compiles the files directly.
"""
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if len(sys.argv) != 2:
    sys.exit("usage: d3d12-full-inject.py <llama.cpp dir>")
ll = Path(sys.argv[1]).resolve()
if not (ll / "ggml" / "src" / "CMakeLists.txt").is_file():
    sys.exit(f"inject: {ll} is not a llama.cpp checkout")

src_dir = root / "third_party" / "ggml-d3d12-full"
dst = ll / "ggml" / "src" / "ggml-d3d12f"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_dir / "ggml-d3d12.cpp", dst / "ggml-d3d12.cpp")
shutil.copy2(src_dir / "ggml-d3d12.h", dst / "ggml-d3d12.h")
shutil.copy2(root / "shaders" / "generated" / "ggml-d3d12-shaders.hpp", dst / "ggml-d3d12-shaders.hpp")

(dst / "CMakeLists.txt").write_text(
    """ggml_add_backend_library(ggml-d3d12f
    ggml-d3d12.cpp
    ggml-d3d12.h
    ggml-d3d12-shaders.hpp
)
target_compile_features(ggml-d3d12f PRIVATE cxx_std_17)
# PUBLIC so ggml-backend-reg.cpp (in ggml) finds ggml-d3d12.h
target_include_directories(ggml-d3d12f PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ggml-d3d12f PRIVATE d3d12 dxgi dxguid)
target_compile_definitions(ggml-d3d12f PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
if (MSVC)
    target_compile_options(ggml-d3d12f PRIVATE /bigobj /utf-8)
endif()
""",
    encoding="utf-8",
)


def patch(path, anchor, add, before=True):
    text = path.read_text(encoding="utf-8")
    if add.strip() in text:
        return  # already injected
    if text.count(anchor) != 1:
        sys.exit(f"inject: anchor not unique/found in {path.name}: {anchor!r} ({text.count(anchor)})")
    text = text.replace(anchor, add + anchor if before else anchor + add)
    path.write_text(text, encoding="utf-8")


patch(ll / "ggml" / "src" / "CMakeLists.txt", "ggml_add_backend(Vulkan)\n", "ggml_add_backend(D3D12F)\n")
reg = ll / "ggml" / "src" / "ggml-backend-reg.cpp"
patch(reg, '#ifdef GGML_USE_WEBGPU\n#include "ggml-webgpu.h"',
      '#ifdef GGML_USE_D3D12F\n#include "ggml-d3d12.h"\n#endif\n\n')
patch(reg, '#ifdef GGML_USE_VULKAN\n    // Add runtime disable check',
      '#ifdef GGML_USE_D3D12F\n    register_backend(xllama_d3d12f_reg());\n#endif\n')
print("inject ok:", dst)
