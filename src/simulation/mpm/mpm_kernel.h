/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

constexpr int mpm_kernel_order = 3;

//#define MPM_TRANSFER_OPT

template <int DIM, int ORDER>
struct MPMKernel {
    constexpr static int D = DIM;
    constexpr static int order = ORDER;
    constexpr static int kernel_size = ORDER + 1;

    using Vector = VectorND<DIM, real>;
    using VectorP = VectorND<DIM + 1, real>;
    using VectorI = VectorND<DIM, int>;

    VectorP w_stages[D][order];
    Vector4 w_cache[D];
    Vector4 dw_cache[D];

    MPMKernel(const Vector &pos) {
        calculate_kernel(pos);
        // Shuffle
        for (int k = 0; k < kernel_size; k++) {
            for (int j = 0; j < D; j++) {
                w_stages[j][k] = VectorP([&](int i) -> real {
                    if (k == i) {
                        return w_cache[j][k];
                    } else {
                        return dw_cache[j][k];
                    }
                });
            }
        }
    }

    void calculate_kernel(const Vector &pos);

    int get_stencil_start(real x) const;
};

template <int DIM>
int MPMKernel<DIM, 2>::get_stencil_start(real x) const {
    return int(x - 0.5f);
}

template <int DIM>
int MPMKernel<DIM, 3>::get_stencil_start(real x) const {
    return int(x) - 1;
}

// Quadratic kernel
template <int DIM>
void MPMKernel<DIM, 2>::calculate_kernel(const Vector &pos) {
    Vector p_fract = fract(pos - 0.5f);
    for (int k = 0; k < D; k++) {
        const Vector4 t = Vector4(p_fract[k]) - Vector4(-0.5f, 0.5f, 1.5f, 0.0f);
        auto tt = t * t;
        w_cache[k] = Vector4(0.5f, -1.0f, 0.5f, 0.0f) * tt +
                     Vector4(-1.5, 0, 1.5, 0.0f) * t +
                     Vector4(1.125f, 0.75f, 1.125f, 0.0f);
        dw_cache[k] = Vector4(-1.0f, 2.0f, -1.0f, 0.0f) * t + Vector4(1.5f, 0, -1.5f, 0.0f);
    }
}

// Cubic kernel
template <int DIM>
void MPMKernel<DIM, 3>::calculate_kernel(const Vector &pos) {
    Vector p_fract = fract(pos);
    for (int k = 0; k < DIM; k++) {
        const Vector4 t = Vector4(p_fract[k]) - Vector4(-1, 0, 1, 2);
        auto tt = t * t;
        auto ttt = tt * t;
        w_cache[k] = Vector4(-1 / 6.0f, 0.5f, -0.5f, 1 / 6.0f) * ttt +
                     Vector4(1, -1, -1, 1) * tt +
                     Vector4(-2, 0, 0, 2) * t +
                     Vector4(4 / 3.0f, 2 / 3.0f, 2 / 3.0f, 4 / 3.0f);
        dw_cache[k] = Vector4(-0.5f, 1.5f, -1.5f, 0.5f) * tt +
                      Vector4(2, -2, -2, 2) * t +
                      Vector4(-2, 0, 0, 2);
    }
}

TC_NAMESPACE_END
