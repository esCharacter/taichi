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
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

const real pi{acosf(-1.0f)};
const real eps = 1e-6f;

#undef max
#undef min

template <typename T>
inline T abs(const T &a) {
    return std::abs(a);
}

template <typename T>
inline T clamp(T a, T min, T max) {
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

template <typename T>
inline T clamp(T a) {
    if (a < 0) return 0;
    if (a > 1) return 1;
    return a;
}

template <typename T, typename V>
inline V lerp(T a, V x_0, V x_1) {
    return (T(1) - a) * x_0 + a * x_1;
}

inline bool inside_unit_cube(const Vector3 &p) {
    return 0 <= p[0] && p[0] < 1 && 0 <= p[1] && p[1] < 1 && 0 <= p[2] && p[2] < 1;
}

template <typename T>
T sqr(const T &a) {
    return a * a;
}

template <typename T>
T cube(const T &a) {
    return a * a * a;
}

inline int sgn(float a) {
    if (a < -eps)
        return -1;
    else if (a > eps)
        return 1;
    return 0;
}

inline int sgn(double a) {
    if (a < -eps)
        return -1;
    else if (a > eps)
        return 1;
    return 0;
}

inline float fract(float x) {
    return x - std::floor(x);
}

inline Vector2 fract(const Vector2 &v) {
    return Vector2(fract(v.x), fract(v.y));
}

inline Vector3 fract(const Vector3 &v) {
    return Vector3(fract(v.x), fract(v.y), fract(v.z));
}

inline Vector4 fract(const Vector4 &v) {
    return Vector4(fract(v.x), fract(v.y), fract(v.z), fract(v.w));
}

// inline float frand() { return (float)rand() / (RAND_MAX + 1); }
inline float rand() {
    static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned int t = x ^(x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))) * (1.0f / 4294967296.0f);
}

inline Vector3 sample_sphere(float u, float v) {
    float x = u * 2 - 1;
    float phi = v * 2 * pi;
    float yz = sqrt(1 - x * x);
    return Vector3(x, yz * cos(phi), yz * sin(phi));
}


inline float catmull_rom(float f_m_1, float f_0, float f_1, float f_2,
                         float x_r) {
    float s = (f_1 - f_0);
    float s_0 = (f_1 - f_m_1) / 2.0f;
    float s_1 = (f_2 - f_0) / 2.0f;
    s_0 = s_0 * (s_1 * s > 0);
    s_1 = s_1 * (s_1 * s > 0);
    return f_0 + x_r * s_0 + (-3 * f_0 + 3 * f_1 - 2 * s_0 - s_1) * x_r * x_r +
           (2 * f_0 - 2 * f_1 + s_0 + s_1) * x_r * x_r * x_r;
}

inline float catmull_rom(float *pf_m_1, float x_r) {
    return catmull_rom(*pf_m_1, *(pf_m_1 + 1), *(pf_m_1 + 2), *(pf_m_1 + 3), x_r);
}

inline void print(std::string v) {
    printf("%s\n", v.c_str());
}

inline void print(float v) {
    printf("%f\n", v);
}

inline void print(int v) {
    printf("%d\n", v);
}

inline void print(unsigned int v) {
    printf("%u\n", v);
}

inline void print(long v) {
    printf("%ld\n", v);
}

#ifndef WIN32

inline void print(size_t v) {
    printf("%lld\n", (long long)v);
}

#endif

inline void print(long long v) {
    std::cout << v << std::endl;
}

inline void print(unsigned long long v) {
    std::cout << v << std::endl;
}

inline void print(double v) {
    std::cout << v << std::endl;
}

inline int is_prime(int a) {
    assert(a >= 2);
    for (int i = 2; i * i <= a; i++) {
        if (a % i == 0) return false;
    }
    return true;
}

template <typename T>
inline T hypot2(const T &x, const T &y) {
    return x * x + y * y;
}

inline float pow(const float &a, const float &b) {
    return ::pow(a, b);
}

inline double pow(const double &a, const double &b) {
    return ::pow(a, b);
}

inline real det(const Matrix2 &m) {
    return determinant(m);
}

inline real det(const Matrix3 &m) {
    return determinant(m);
}

inline Vector3 set_up(const Vector3 &a, const Vector3 &y) {
    Vector3 x, z;
    if (std::abs(y.y) > 1.0f - eps) {
        x = Vector3(1, 0, 0);
    } else {
        x = normalize(cross(y, Vector3(0, 1, 0)));
    }
    z = cross(x, y);
    return a.x * x + a.y * y + a.z * z;
}

inline Vector3 multiply_matrix4(Matrix4 m, Vector3 v, real w) {
    Vector4 tmp(v, w);
    tmp = m * tmp;
    return Vector3(tmp.x, tmp.y, tmp.z);
}

inline Vector3 random_diffuse(const Vector3 &normal, real u, real v) {
    if (u > v) {
        std::swap(u, v);
    }
    if (v < eps) {
        v = eps;
    }
    u /= v;
    real xz = v, y = sqrt(1 - v * v);
    real phi = u * pi * 2;
    return set_up(Vector3(xz * cos(phi), y, xz * sin(phi)), normal);
}

inline Vector3 reflect(const Vector3 &d, const Vector3 &n) {
    return d - n * dot(d, n) * 2;
}

inline Vector3 random_diffuse(const Vector3 &normal) {
    return random_diffuse(normal, rand(), rand());
}

inline Vector4 pow(const Vector4 &v, const float &p) {
    return Vector4(
            std::pow(v[0], p),
            std::pow(v[1], p),
            std::pow(v[2], p),
            std::pow(v[3], p)
    );
}

inline Vector3 pow(const Vector3 &v, const float &p) {
    return Vector3(
            std::pow(v[0], p),
            std::pow(v[1], p),
            std::pow(v[2], p)
    );
}


inline real max_component(const Vector3 &v) {
    return std::max(v.x, std::max(v.y, v.z));
}

inline double max_component(const Vector3d &v) {
    return std::max(v.x, std::max(v.y, v.z));
}

#ifdef CV_ON
#define CV(v) if (abnormal(v)) {for (int i = 0; i < 1; i++) printf("Abnormal value %s (Ln %d)\n", #v, __LINE__); taichi::print(v); puts("");}
#else
#define CV(v)
#endif

template <typename T>
inline bool is_normal(T m) {
    return std::isfinite(m);
}

template <typename T>
inline bool abnormal(T m) {
    return !is_normal(m);
}

template <>
inline bool is_normal(Vector2 v) {
    return is_normal(v[0]) && is_normal(v[1]);
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

inline Vector2 clamp(const Vector2 &v) {
    return Vector2(clamp(v[0]), clamp(v[1]));
}

inline float cross(const Vector2 &a, const Vector2 &b) {
    return a.x * b.y - a.y * b.x;
}

inline float matrix_norm_squared(const Matrix2 &a) {
    float sum = 0.0f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sum += a[i][j] * a[i][j];
        }
    }
    return sum;
}

inline float matrix_norm_squared(const Matrix3 &a) {
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sum += a[i][j] * a[i][j];
        }
    }
    return sum;
}

inline double frobenius_norm2(const Matrix2d &a) {
    return a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1];
}

inline double frobenius_norm(const Matrix2d &a) {
    return sqrt(frobenius_norm2(a));
}

inline real frobenius_norm2(const Matrix2 &a) {
    return a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1];
}

inline real frobenius_norm(const Matrix2 &a) {
    return sqrt(frobenius_norm2(a));
}

inline real frobenius_norm2(const Matrix3 &a) {
    return a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1];
}

inline real frobenius_norm(const Matrix3 &a) {
    return sqrt(frobenius_norm2(a));
}


inline bool intersect(const Vector2 &a, const Vector2 &b, const Vector2 &c, const Vector2 &d) {
    if (cross(c - a, b - a) * cross(b - a, d - a) > 0 && cross(a - d, c - d) * cross(c - d, b - d) > 0) {
        return true;
    } else {
        return false;
    }
}

inline float nearest_distance(const Vector2 &p, const Vector2 &a, const Vector2 &b) {
    float ab = length(a - b);
    Vector2 dir = (b - a) / ab;
    float pos = clamp(dot(p - a, dir), 0.0f, ab);
    return length(a + pos * dir - p);
}

inline float nearest_distance(const Vector2 &p, const std::vector<Vector2> &polygon) {
    float dist = std::numeric_limits<float>::infinity();
    for (int i = 0; i < (int)polygon.size(); i++) {
        dist = std::min(dist, nearest_distance(p, polygon[i], polygon[(i + 1) % polygon.size()]));
    }
    return dist;
}

inline bool inside_polygon(const Vector2 &p, const std::vector<Vector2> &polygon) {
    int count = 0;
    static const Vector2 q(123532, 532421123);
    for (int i = 0; i < (int)polygon.size(); i++) {
        count += intersect(p, q, polygon[i], polygon[(i + 1) % polygon.size()]);
    }
    return count % 2 == 1;
}

inline std::vector<Vector2>
points_inside_polygon(std::vector<float> x_range, std::vector<float> y_range, const std::vector<Vector2> &polygon) {
    std::vector<Vector2> ret;
    for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
        for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
            Vector2 p(x, y);
            if (inside_polygon(p, polygon)) {
                ret.push_back(p);
            }
        }
    }
    return ret;
}

inline std::vector<Vector2>
points_inside_sphere(std::vector<float> x_range, std::vector<float> y_range, const Vector2 &center, float radius) {
    std::vector<Vector2> ret;
    for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
        for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
            Vector2 p(x, y);
            if (length(p - center) < radius) {
                ret.push_back(p);
            }
        }
    }
    return ret;
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

MatrixND<2, float32, InstructionSet::AVX> inversed(const MatrixND<2, float32, InstructionSet::AVX> &mat) {
    real det = determinant(mat);
    return 1.0f / det * MatrixND<2, float32, InstructionSet::AVX>(
            VectorND<2, float32, InstructionSet::AVX>(mat[1][1], -mat[0][1]),
            VectorND<2, float32, InstructionSet::AVX>(mat[1][0], mat[0][0]
            )
    );
}

MatrixND<3, float32, InstructionSet::AVX> inversed(const MatrixND<3, float32, InstructionSet::AVX> &mat) {
    real det = determinant(mat);
    return 1.0f / det * MatrixND<3, float32, InstructionSet::AVX>(
            VectorND<3, float32, InstructionSet::AVX>(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2],
                                                      mat[2][1] * mat[0][2] - mat[0][1] * mat[2][2],
                                                      mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]),
            VectorND<3, float32, InstructionSet::AVX>(mat[2][0] * mat[1][2] - mat[1][0] * mat[2][2],
                                                      mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2],
                                                      mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2]),
            VectorND<3, float32, InstructionSet::AVX>(mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1],
                                                      mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1],
                                                      mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1])
    );
}

template<typename T>
MatrixND<4, T, InstructionSet::AVX> inversed(const MatrixND<4, T, InstructionSet::AVX> &m) {
    // This function is copied from GLM
    /*
    ================================================================================
    OpenGL Mathematics (GLM)
    --------------------------------------------------------------------------------
    GLM is licensed under The Happy Bunny License and MIT License

    ================================================================================
    The Happy Bunny License (Modified MIT License)
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    Restrictions:
     By making use of the Software for military purposes, you choose to make a
     Bunny unhappy.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    ================================================================================
    The MIT License
    --------------------------------------------------------------------------------
    Copyright (c) 2005 - 2014 G-Truc Creation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
     */



    T Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    T Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    T Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

    T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

    T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

    T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
    T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

    T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
    T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

    T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

    using Vector = VectorND<4, T, InstructionSet::AVX>;

    Vector Fac0(Coef00, Coef00, Coef02, Coef03);
    Vector Fac1(Coef04, Coef04, Coef06, Coef07);
    Vector Fac2(Coef08, Coef08, Coef10, Coef11);
    Vector Fac3(Coef12, Coef12, Coef14, Coef15);
    Vector Fac4(Coef16, Coef16, Coef18, Coef19);
    Vector Fac5(Coef20, Coef20, Coef22, Coef23);

    Vector Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
    Vector Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
    Vector Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
    Vector Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

    Vector Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
    Vector Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
    Vector Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
    Vector Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

    Vector SignA(+1, -1, +1, -1);
    Vector SignB(-1, +1, -1, +1);
    MatrixND<4, T, InstructionSet::AVX> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

    Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

    Vector Dot0(m[0] * Row0);
    T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

    T OneOverDeterminant = static_cast<T>(1) / Dot1;

    return Inverse * OneOverDeterminant;
}

template<int DIM, typename T, InstructionSet ISA>
MatrixND<DIM, T, ISA> inverse(const MatrixND<DIM, T, ISA> &m) {
    return inversed(m);
};

//#define rand frand

template <int DIM>
class IndexND;

template <int DIM>
class RegionND;

template <int DIM, typename T>
class ArrayND;

TC_NAMESPACE_END

