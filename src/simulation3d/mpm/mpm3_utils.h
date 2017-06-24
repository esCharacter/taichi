/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <taichi/math/array_3d.h>
#include <taichi/math/levelset_3d.h>
#include <taichi/math/dynamic_levelset_3d.h>

TC_NAMESPACE_BEGIN

const int mpm3d_grid_block_size = 8;

// Note: assuming abs(x) <= 2!!
inline real w(real x) {
    x = abs(x);
#ifdef CV_ON
    assert(x <= 2);
#endif
    if (x < 1) {
        return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
    } else {
        return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
    }
}

// Note: assuming abs(x) <= 2!!
inline real dw(real x) {
    real s = x < 0.0f ? -1.0f : 1.0f;
    x *= s;
#ifdef CV_ON
    assert(x <= 2.0f);
#endif
    real val;
    real xx = x * x;
    if (x < 1.0f) {
        val = 1.5f * xx - 2.0f * x;
    } else {
        val = -0.5f * xx + 2.0f * x - 2.0f;
    }
    return s * val;
}

TC_NAMESPACE_END

