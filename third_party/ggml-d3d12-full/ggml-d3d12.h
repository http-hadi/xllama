#pragma once

#include <stddef.h>
#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_D3D12_NAME "D3D12F"

GGML_BACKEND_API ggml_backend_t xllama_d3d12f_init(int device);

GGML_BACKEND_API ggml_backend_reg_t xllama_d3d12f_reg(void);

// Folder holding the precompiled DXIL cache (UWP: <package>\\d3d12-shader-cache; desktop: optional
// override). Call before the first xllama_d3d12f_reg().
GGML_BACKEND_API void xllama_d3d12f_set_shader_dir(const wchar_t * dir);

#ifdef  __cplusplus
}
#endif
