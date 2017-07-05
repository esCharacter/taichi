/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

#include <taichi/math/math_util.h>

// >= AVX 2
#include <immintrin.h>

#ifdef _WIN64
#define TC_ALIGNED(x) __declspec(align(x))
#else
#define TC_ALIGNED(x) __attribute__((aligned(x)))
#endif

enum class InstructionSet {
    None,
    AVX,
    AVX2
};

template <int DIM, typename T, InstructionSet ISA = InstructionSet::None>
struct Vector {
    T d[DIM];

    Vector(T v) {
        for (int i = 0; i < DIM; i++) {
            d[i] = v;
        }
    }

    // TODO: complete this
};

template <int DIM, typename T, InstructionSet ISA = InstructionSet::None>
struct Matrix {
    T d[DIM][DIM];

    // TODO: complete this
};

// SIMD Vector4
struct TC_ALIGNED(16) Vector4s {
    union {
        __m128 v;
        struct {
            float x, y, z, w;
        };
    };

    Vector4s() : Vector4s(0.0f) {};

    Vector4s(const Vector4 &vec) : Vector4s(vec.x, vec.y, vec.z, vec.w) {}

    Vector4s(const Vector3 &vec, float w = 0.0f) : Vector4s(vec.x, vec.y, vec.z, w) {}

    Vector4s(real x, real y, real z, real w = 0.0f) : v(_mm_set_ps(w, z, y, x)) {}

    Vector4s(real x) : v(_mm_set1_ps(x)) {}

    Vector4s(__m128 v) : v(v) {}

    float &operator[](int i) { return (&x)[i]; }

    const float &operator[](int i) const { return (&x)[i]; }

    operator __m128() const { return v; }

    operator __m128i() const { return _mm_castps_si128(v); }

    operator __m128d() const { return _mm_castps_pd(v); }

    Vector4s &operator=(const Vector4s o) {
        v = o.v;
        return *this;
    }

    Vector4s operator+(const Vector4s &o) const { return _mm_add_ps(v, o.v); }

    Vector4s operator-(const Vector4s &o) const { return _mm_sub_ps(v, o.v); }

    Vector4s operator*(const Vector4s &o) const { return _mm_mul_ps(v, o.v); }

    Vector4s operator/(const Vector4s &o) const { return _mm_div_ps(v, o.v); }

    Vector4s operator-() const { return _mm_sub_ps(Vector4s(0.0f), v); }

    Vector4s &operator+=(const Vector4s &o) {
        (*this) = (*this) + o;
        return *this;
    }

    Vector4s &operator-=(const Vector4s &o) {
        (*this) = (*this) - o;
        return *this;
    }

    Vector4s &operator*=(const Vector4s &o) {
        (*this) = (*this) * o;
        return *this;
    }

    Vector4s &operator/=(const Vector4s &o) {
        (*this) = (*this) / o;
        return *this;
    }

    Vector3 to_vec3() const {
        return Vector3(x, y, z);
    }

    template <int a, int b, int c, int d>
    Vector4s permute() const {
        return _mm_permute_ps(v, _MM_SHUFFLE(a, b, c, d));
    }

    template <int a>
    Vector4s broadcast() const {
        return permute<a, a, a, a>();
    }

    // TODO: vectorize ?
    Vector4s abs() const {
        return Vector4s(
                std::abs(x),
                std::abs(y),
                std::abs(z),
                std::abs(w)
        );
    }

    Vector4s max() const {
        return std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
    }

    float length2() const {
        return _mm_cvtss_f32(_mm_dp_ps(v, v, 0xf1));
    }

    float length() const {
        return std::sqrt(length2());
    }
};

// FMA: a * b + c
inline Vector4s fused_mul_add(const Vector4s &a, const Vector4s &b, const Vector4s &c) {
    return _mm_fmadd_ps(a, b, c);
}

inline Vector4s operator*(float a, const Vector4s &vec) {
    return Vector4s(a) * vec;
}

inline void transpose4x4(const Vector4s m[], Vector4s t[]) {
    Vector4s t0 = _mm_unpacklo_ps(m[0], m[1]);
    Vector4s t2 = _mm_unpacklo_ps(m[2], m[3]);
    Vector4s t1 = _mm_unpackhi_ps(m[0], m[1]);
    Vector4s t3 = _mm_unpackhi_ps(m[2], m[3]);
    t[0] = _mm_movelh_ps(t0, t2);
    t[1] = _mm_movehl_ps(t2, t0);
    t[2] = _mm_movelh_ps(t1, t3);
    t[3] = _mm_movehl_ps(t3, t1);
}

inline void print(const Vector4s v) {
    for (int i = 0; i < 4; i++) {
        printf("%9.4f ", v[i]);
    }
    printf("\n");
}

// SIMD Matrix4
struct TC_ALIGNED(16) Matrix4s {
    union {
        // Four columns, instead of rows!
        Vector4s v[4];
    };

    Matrix4s() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
        v[3] = 0.0f;
    }

    Matrix4s(Vector4s v0, Vector4s v1, Vector4s v2, Vector4s v3) : v{v0, v1, v2, v3} {}

    Matrix4s(const Matrix4 &o) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                v[i][j] = o[i][j];
            }
        }
    }

    Matrix4s(const Matrix3 &o) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                v[i][j] = o[i][j];
            }
            v[i][3] = 0.0f;
        }
        v[3] = 0.0f;
    }

    Vector4s &operator[](int i) { return v[i]; }

    const Vector4s &operator[](int i) const { return v[i]; }

    Vector4s operator*(const Vector4s &o) const {
        Vector4s ret = o.broadcast<3>() * v[3];
        ret = fused_mul_add(v[2], o.broadcast<2>(), ret);
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return ret;
    }

    Vector4s multiply_vec3(const Vector4s &o) const {
        Vector4s ret = o.broadcast<2>() * v[2];
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return ret;
    }
};

inline Matrix4s operator*(const float a, const Matrix4s &M) {
    Matrix4s ret;
    ret[0] = a * M[0];
    ret[1] = a * M[1];
    ret[2] = a * M[2];
    ret[3] = a * M[3];
    return ret;
}

inline Matrix4s operator*(const Matrix4s &m, const float a) {
    return a * m;
}

inline void print(const Matrix4s &v) {
    printf("\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", v[j][i]);
        }
        printf("\n");
    }
}

// SIMD Matrix3
struct TC_ALIGNED(16) Matrix3s {
    union {
        // Four columns, instead of rows!
        Vector4s v[3];
    };

    Matrix3s() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    }

    Matrix3s &operator=(const Matrix3s &m) {
        v[0] = m.v[0];
        v[1] = m.v[1];
        v[2] = m.v[2];
        return *this;
    }

    Matrix3s(float diag) {
        v[0] = Vector4s(diag, 0.0f, 0.0f, 0.0f);
        v[1] = Vector4s(0.0f, diag, 0.0f, 0.0f);
        v[2] = Vector4s(0.0f, 0.0f, diag, 0.0f);
    }

    Matrix3s(Vector4s v0, Vector4s v1, Vector4s v2) : v{v0, v1, v2} {}

    Matrix3s(const Matrix3 &o) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                v[i][j] = o[i][j];
            }
        }
    }

    Vector4s &operator[](int i) { return v[i]; }

    const Vector4s &operator[](int i) const { return v[i]; }

    Vector4s operator*(const Vector4s &o) const {
        Vector4s ret = o.broadcast<2>() * v[2];
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return ret;
    }

    Matrix3s operator*(const Matrix3s &o) const {
        Matrix3s ret;
        ret[0] = (*this) * o.v[0];
        ret[1] = (*this) * o.v[1];
        ret[2] = (*this) * o.v[2];
        return ret;
    }

    static Matrix3s outer_product(Vector4s row, Vector4s column) {
        Matrix3s ret;
        ret[0] = column * row[0];
        ret[1] = column * row[1];
        ret[2] = column * row[2];
        return ret;
    }

    Matrix3s operator+(const Matrix3s &o) const {
        return Matrix3s(v[0] + o[0], v[1] + o[1], v[2] + o[2]);
    }

    Matrix3s &operator+=(const Matrix3s &o) {
        v[0] += o[0];
        v[1] += o[1];
        v[2] += o[2];
        return *this;
    }

    Matrix3s &operator-=(const Matrix3s &o) {
        v[0] -= o[0];
        v[1] -= o[1];
        v[2] -= o[2];
        return *this;
    }

    Matrix3s operator-(const Matrix3s &o) const {
        return Matrix3s(v[0] - o[0], v[1] - o[1], v[2] - o[2]);
    }

    Matrix3 to_mat3() const {
        return Matrix3(v[0][0], v[0][1], v[0][2],
                       v[1][0], v[1][1], v[1][2],
                       v[2][0], v[2][1], v[2][2]
        );
    }

    float frobenius_norm2() const {
        return v[0].length2() + v[1].length2() + v[2].length2();
    }

    float frobenius_norm() const {
        return std::sqrt(frobenius_norm2());
    }

    Matrix3s transposed() const {
        Matrix3s ret;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                ret[i][j] = v[j][i];
                ret[j][i] = v[i][j];
            }
        }
        return ret;
    }
};

inline Matrix3s operator*(const float a, const Matrix3s &M) {
    Matrix3s ret;
    ret[0] = a * M[0];
    ret[1] = a * M[1];
    ret[2] = a * M[2];
    return ret;
}

inline Matrix3s operator*(const Matrix3s &m, const float a) {
    return a * m;
}

inline void print(const Matrix3s &v) {
    printf("\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", v[j][i]);
        }
        printf("\n");
    }
}

TC_NAMESPACE_END
