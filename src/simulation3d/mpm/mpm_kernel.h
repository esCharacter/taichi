/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math_simd.h>

TC_NAMESPACE_BEGIN

#define TC_MPM3D_PREPROCESS_STAGED_KERNELS\
    Vector4s w_stages[3][kernel_size]; \
    for (int k = 0; k < kernel_size; k++) { \
        w_stages[0][k] = Vector4s(dw_cache[0][k], w_cache[0][k], w_cache[0][k], w_cache[0][k]); \
        w_stages[1][k] = Vector4s(w_cache[1][k], dw_cache[1][k], w_cache[1][k], w_cache[1][k]); \
        w_stages[2][k] = Vector4s(w_cache[2][k], w_cache[2][k], dw_cache[2][k], w_cache[2][k]); \
    }

#define TC_MPM3D_KERNEL_ORDER 2

template <> const int MPM<3>::kernel_size = TC_MPM3D_KERNEL_ORDER + 1;
#if TC_MPM3D_KERNEL_ORDER == 2
// Quadratic Kernel

template <>
int MPM<3>::get_stencil_start(real x) const {
    return int(x - 0.5f);
}

#define TC_MPM3D_PREPROCESS_KERNELS\
    Vector4s w_cache[DIM]; \
    Vector4s dw_cache[DIM];\
    Vector p_fract = fract(p.pos - 0.5f); \
    for (int k = 0; k < DIM; k++) { \
        const Vector4s t = Vector4s(p_fract[k]) - Vector4s(-0.5f, 0.5f, 1.5f); \
        auto tt = t * t; \
        w_cache[k] = Vector4s(0.5, -1, 0.5) * tt + \
            Vector4s(-1.5, 0, 1.5) * t + \
            Vector4s(1.125f, 0.75f, 1.125f); \
        dw_cache[k] = Vector4s(-1, 2, -1) * t + Vector4s(1.5f, 0, -1.5f); \
    } \
    TC_MPM3D_PREPROCESS_STAGED_KERNELS

#elif TC_MPM3D_KERNEL_ORDER == 3
// Cubic Kernel

template <>
int MPM<3>::get_stencil_start(real x) const {
    return int(x) - 1;
}

#define TC_MPM3D_PREPROCESS_KERNELS\
    Vector4s w_cache[DIM]; \
    Vector4s dw_cache[DIM];\
    Vector p_fract = fract(p.pos); \
    for (int k = 0; k < DIM; k++) { \
        const Vector4s t = Vector4s(p_fract[k]) - Vector4s(-1, 0, 1, 2); \
        auto tt = t * t; \
        auto ttt = tt * t; \
        w_cache[k] = Vector4s(-1 / 6.0f, 0.5f, -0.5f, 1 / 6.0f) * ttt + \
        Vector4s(1, -1, -1, 1) * tt + \
        Vector4s(-2, 0, 0, 2) * t + \
        Vector4s(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f); \
        dw_cache[k] = Vector4s(-0.5f, 1.5f, -1.5f, 0.5f) * tt + \
        Vector4s(2, -2, -2, 2) * t + \
        Vector4s(-2, 0, 0, 2); \
    } \
    TC_MPM3D_PREPROCESS_STAGED_KERNELS

#else

#error "MPM 3D kernel order (TC_MPM3D_KERNEL_ORDER) should be specified as 2 or 3."

#endif

TC_NAMESPACE_END
