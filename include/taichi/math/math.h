/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <taichi/common/util.h>
#include <taichi/math/math.h>
#include <taichi/math/math_constants.h>
#include <taichi/math/geometry_util.h>

TC_NAMESPACE_BEGIN

template <typename T>
inline bool is_normal(Vector2 v) {
}

template <>
inline bool is_normal(Vector2d v) {
    return is_normal(v[0]) && is_normal(v[1]);
}

template <>
inline bool is_normal(Vector3 v) {
    return is_normal(v[0]) && is_normal(v[1]) && is_normal(v[2]);
}

template <>
inline bool is_normal(Vector3d v) {
    return is_normal(v[0]) && is_normal(v[1]) && is_normal(v[2]);
}

template <>
inline bool is_normal(Matrix2 m) {
    return is_normal(m[0][0]) && is_normal(m[0][1]) &&
           is_normal(m[1][0]) && is_normal(m[1][1]);
}

template <>
inline bool is_normal(Matrix3 m) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!is_normal(m[i][j])) return false;
        }
    }
    return true;
}

template <>
inline bool is_normal(Matrix4 m) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!is_normal(m[i][j])) return false;
        }
    }
    return true;
}

inline int64 get_largest_pot(int64 a) {
    assert_info(a > 0, "a should be positive, instead of " + std::to_string(a));
    // TODO: optimize
    int64 i = 1;
    while (i * 2 <= a) {
        i *= 2;
    }
    return i;
}

TC_NAMESPACE_END

