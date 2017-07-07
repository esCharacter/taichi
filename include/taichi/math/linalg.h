/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>
#include <taichi/common/util.h>

// >= AVX 2
#include <immintrin.h>

TC_NAMESPACE_BEGIN

#ifdef _WIN64
#define TC_ALIGNED(x) __declspec(align(x))
#else
#define TC_ALIGNED(x) __attribute__((aligned(x)))
#endif

enum class InstructionSetExtension {
    None,
    AVX,
    AVX2
};

const InstructionSetExtension default_instruction_set = InstructionSetExtension::AVX;

template <int DIM, typename T, InstructionSetExtension ISE>
struct VectorNDBase {
    T d[DIM];
};

template <typename T, InstructionSetExtension ISE>
struct VectorNDBase<1, T, ISE> {
    union {
        T d[1];
        struct {
            T x;
        };
    };
};

template <typename T, InstructionSetExtension ISE>
struct VectorNDBase<2, T, ISE> {
    union {
        T d[2];
        struct {
            T x, y;
        };
    };
};

template <typename T, InstructionSetExtension ISE>
struct VectorNDBase<3, T, ISE> {
    union {
        T d[3];
        struct {
            T x, y, z;
        };
    };
};

template <typename T, InstructionSetExtension ISE>
struct VectorNDBase<4, T, ISE> {
    union {
        T d[4];
        struct {
            T x, y, z, w;
        };
    };
};

template <int DIM, typename T, InstructionSetExtension ISE = InstructionSetExtension::AVX>
struct VectorND : public VectorNDBase<DIM, T, ISE> {
    VectorND() {
        for (int i = 0; i < DIM; i++) {
            this->d[i] = T(0);
        }
    }

    VectorND(T v) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] = v;
        }
    }

    VectorND(T v0, T v1) {
        static_assert(DIM == 2, "Vector dim must be 2");
        this->d[0] = v0;
        this->d[1] = v1;
    }

    VectorND(T v0, T v1, T v2) {
        static_assert(DIM == 3, "Vector dim must be 3");
        this->d[0] = v0;
        this->d[1] = v1;
        this->d[2] = v2;
    }

    VectorND(T v0, T v1, T v2, T v3) {
        static_assert(DIM == 4, "Vector dim must be 4");
        this->d[0] = v0;
        this->d[1] = v1;
        this->d[2] = v2;
        this->d[3] = v3;
    }

    T &operator[](int i) { return this->d[i]; }

    const T &operator[](int i) const { return this->d[i]; }

    static T dot(VectorND<DIM, T, ISE> row, VectorND<DIM, T, ISE> column) {
        T ret = T(0);
        for (int i = 0; i < DIM; i++)
            ret += row[i] * column[i];
        return ret;
    }

    VectorND &operator=(const VectorND o) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] = o[i];
        }
        return *this;
    }

    VectorND operator+(const VectorND &o) const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = this->d[i] + o[i];
        }
        return ret;
    }

    VectorND operator-(const VectorND &o) const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = this->d[i] - o[i];
        }
        return ret;
    }

    VectorND operator*(const VectorND &o) const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = this->d[i] * o[i];
        }
        return ret;
    }

    VectorND operator/(const VectorND &o) const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = this->d[i] / o[i];
        }
        return ret;
    }

    VectorND operator-() const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = -this->d[i];
        }
        return ret;
    }

    VectorND &operator+=(const VectorND &o) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] += o[i];
        };
        return *this;
    }

    VectorND &operator-=(const VectorND &o) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] -= o[i];
        };
        return *this;
    }

    VectorND &operator*=(const VectorND &o) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] *= o[i];
        };
        return *this;
    }

    VectorND &operator/=(const VectorND &o) {
        for (int i = 0; i < DIM; i++) {
            this->d[i] /= o[i];
        };
        return *this;
    }

    bool operator==(const VectorND &o) const {
        for (int i = 0; i < DIM; i++)
            if (this->d[i] != o[i]) return false;
        return true;
    }

    bool operator!=(const VectorND &o) const {
        for (int i = 0; i < DIM; i++)
            if (this->d[i] != o[i]) return true;
        return false;
    }

    VectorND abs() const {
        VectorND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = std::abs(this->d[i]);
        }
        return ret;
    }

    T max() const {
        T ret = this->d[0];
        for (int i = 1; i < DIM; i++) {
            ret = std::max(ret, this->d[i]);
        }
        return ret;
    }

    T length2() const {
        T ret = 0;
        for (int i = 0; i < DIM; i++) {
            ret += this->d[i] * this->d[i];
        }
        return ret;
    }

    real length() const {
        return std::sqrt(length2());
    }

    template <typename G>
    VectorND<DIM, G, ISE> cast() const {
        VectorND<DIM, G, ISE> ret;
        for (int i = 0; i < DIM; i++)
            ret[i] = static_cast<G>(this->d[i]);
        return ret;
    }
};


template <int DIM, typename T, InstructionSetExtension ISE>
VectorND<DIM, T, ISE> operator*(T a, const VectorND<DIM, T, ISE> &v) {
    return VectorND<DIM, T, ISE>(a) * v;
}

template <int DIM, typename T, InstructionSetExtension ISE>
VectorND<DIM, T, ISE> operator*(const VectorND<DIM, T, ISE> &v, T a) {
    return a * v;
}

using Vector2 = VectorND<2, real, InstructionSetExtension::AVX>;
using Vector3 = VectorND<3, real, InstructionSetExtension::AVX>;
using Vector4 = VectorND<4, real, InstructionSetExtension::AVX>;

template <int DIM, typename T, InstructionSetExtension ISE = InstructionSetExtension::AVX>
struct MatrixND {
    using Vector = VectorND<DIM, T, ISE>;
    Vector d[DIM];

    template<int DIM1, typename T1, InstructionSetExtension ISA1>
    MatrixND(const MatrixND<DIM1, T1, ISA1> &o) {
        (*this) = T(0);
        for (int i = 0; i < DIM1; i++) {
            for (int j = 0; j < DIM1; j++) {
                d[i][j] = o[i][j];
            }
        }
    }

    MatrixND() {
        for (int i = 0; i < DIM; i++) {
            d[i] = VectorND<DIM, T, ISE>();
        }
    }

    MatrixND(T v) {
        for (int i = 0; i < DIM; i++) {
            d[i] = VectorND<DIM, T, ISE>();
        }
        for (int i = 0; i < DIM; i++) {
            d[i][i] = v;
        }
    }

    // Diag
    MatrixND(Vector v) {
        for (int i = 0; i < DIM; i++) {
            this->d[i][i] = v[i];
        }
    }

    MatrixND(Vector v0, Vector v1) {
        static_assert(DIM == 2, "Matrix dim must be 2");
        this->d[0] = v0;
        this->d[1] = v1;
    }

    MatrixND(Vector v0, Vector v1, Vector v2) {
        static_assert(DIM == 3, "Matrix dim must be 3");
        this->d[0] = v0;
        this->d[1] = v1;
        this->d[2] = v2;
    }

    MatrixND(Vector v0, Vector v1, Vector v2, Vector v3) {
        static_assert(DIM == 4, "Matrix dim must be 4");
        this->d[0] = v0;
        this->d[1] = v1;
        this->d[2] = v2;
        this->d[3] = v3;
    }

    MatrixND &operator=(const MatrixND &m) {
        for (int i = 0; i < DIM; i++) {
            d[i] = m[i];
        }
        return *this;
    }

    MatrixND(const MatrixND &o) {
        for (int i = 0; i < DIM; i++) {
            d[i] = o[i];
        }
    }

    VectorND<DIM, T, ISE> &operator[](int i) {
        return d[i];
    }

    const VectorND<DIM, T, ISE> &operator[](int i) const {
        return d[i];
    }

    VectorND<DIM, T, ISE> operator*(const VectorND<DIM, T, ISE> &o) const {
        VectorND<DIM, T, ISE> ret;
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++) {
                ret[i] += d[j][i] * o[j];
            }
        return ret;
    }

    MatrixND operator*(const MatrixND &o) const {
        MatrixND ret;
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                for (int k = 0; k < DIM; k++) {
                    ret[j][i] += d[k][i] * o[j][k];
                }
        return ret;
    }

    static MatrixND outer_product(VectorND<DIM, T, ISE> row, VectorND<DIM, T, ISE> column) {
        MatrixND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = column * row[i];
        }
        return ret;
    }

    MatrixND operator+(const MatrixND &o) const {
        MatrixND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = d[i] + o[i];
        }
        return ret;
    }

    MatrixND &operator+=(const MatrixND &o) {
        for (int i = 0; i < DIM; i++) {
            d[i] += o[i];
        }
        return *this;
    }

    MatrixND &operator-=(const MatrixND &o) {
        for (int i = 0; i < DIM; i++) {
            d[i] -= o[i];
        }
        return *this;
    }

    MatrixND operator-(const MatrixND &o) const {
        MatrixND ret;
        for (int i = 0; i < DIM; i++) {
            ret[i] = -d[i];
        }
        return ret;
    }

    bool operator==(const MatrixND &o) const {
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                if (d[i][j] != o[i][j]) return false;
        return true;
    }

    bool operator!=(const MatrixND &o) const {
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                if (d[i][j] != o[i][j]) return true;
        return false;
    }

    T frobenius_norm2() const {
        return d[0].length2() + d[1].length2() + d[2].length2();
    }

    real frobenius_norm() const {
        return std::sqrt(frobenius_norm2());
    }

    MatrixND transposed() const {
        MatrixND ret;
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++) {
                ret[i][j] = d[j][i];
            }
        return ret;
    }

    template <typename G>
    MatrixND<DIM, G, ISE> cast() const {
        MatrixND <DIM, G, ISE> ret;
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++)
                ret[i][j] = static_cast<G>(d[i][j]);
        return ret;
    }
};

template <int DIM, typename T, InstructionSetExtension ISE>
MatrixND<DIM, T, ISE> operator*(const float a, const MatrixND<DIM, T, ISE> &M) {
    MatrixND <DIM, T, ISE> ret;
    for (int i = 0; i < DIM; i++) {
        ret[i] = a * M[i];
    }
    return ret;
}

template <int DIM, typename T, InstructionSetExtension ISE>
MatrixND<DIM, T, ISE> operator*(const MatrixND<DIM, T, ISE> &M, const float a) {
    return a * M;
}


template <>
struct TC_ALIGNED(16) VectorND<3, float32, InstructionSetExtension::AVX> {
    using Vector3s = VectorND<3, float32, InstructionSetExtension::AVX>;

    union {
        __m128 v;
        struct {
            float x, y, z, w;
        };
    };

    VectorND() : VectorND(0.0f) {};

    VectorND(real x, real y, real z, real w = 0.0f) : v(_mm_set_ps(w, z, y, x)) {}

    VectorND(real x) : VectorND(x, x, x, 0.0f) {}

    VectorND(__m128 v) : v(v) {}

    float &operator[](int i) { return (&x)[i]; }

    const float &operator[](int i) const { return (&x)[i]; }

    operator __m128() const { return v; }

    operator __m128i() const { return _mm_castps_si128(v); }

    operator __m128d() const { return _mm_castps_pd(v); }

    Vector3s &operator=(const Vector3s &o) {
        v = o.v;
        return *this;
    }

    VectorND operator+(const VectorND &o) const { return _mm_add_ps(v, o.v); }

    VectorND operator-(const VectorND &o) const { return _mm_sub_ps(v, o.v); }

    VectorND operator*(const VectorND &o) const { return _mm_mul_ps(v, o.v); }

    VectorND operator/(const VectorND &o) const { return _mm_div_ps(v, o.v); }

    VectorND operator-() const { return _mm_sub_ps(VectorND(0.0f), v); }

    VectorND &operator+=(const VectorND &o) {
        (*this) = (*this) + o;
        return *this;
    }

    VectorND &operator-=(const VectorND &o) {
        (*this) = (*this) - o;
        return *this;
    }

    VectorND &operator*=(const VectorND &o) {
        (*this) = (*this) * o;
        return *this;
    }

    VectorND &operator/=(const VectorND &o) {
        (*this) = (*this) / o;
        return *this;
    }

    bool operator==(const VectorND &o) const {
        for (int i = 0; i < 3; i++)
            if (this->v[i] != o[i]) return false;
        return true;
    }

    bool operator!=(const VectorND &o) const {
        for (int i = 0; i < 3; i++)
            if (this->v[i] != o[i]) return true;
        return false;
    }

    template <int a, int b, int c, int d>
    VectorND permute() const {
        return _mm_permute_ps(v, _MM_SHUFFLE(a, b, c, d));
    }

    template <int a>
    VectorND broadcast() const {
        return permute<a, a, a, a>();
    }

    VectorND abs() const {
        return VectorND(
                std::abs(x),
                std::abs(y),
                std::abs(z),
                std::abs(w)
        );
    }

    real max() const {
        return std::max(std::max(v[0], v[1]), v[2]);
    }

    float length2() const {
        return _mm_cvtss_f32(_mm_dp_ps(v, v, 0x71));
    }

    float length() const {
        return std::sqrt(length2());
    }

    template <typename G>
    VectorND<3, G, InstructionSetExtension::AVX> cast() const {
        VectorND<3, G, InstructionSetExtension::AVX> ret;
        for (int i = 0; i < 3; i++)
            ret[i] = static_cast<G>(this->v[i]);
        return ret;
    }
};

typedef VectorND<3, float32, InstructionSetExtension::AVX> Vector3s;


// SIMD Vector4
template <>
struct TC_ALIGNED(16) VectorND<4, float32, InstructionSetExtension::AVX> {
    using Vector4s = VectorND<4, float32, InstructionSetExtension::AVX>;

    union {
        __m128 v;
        struct {
            float x, y, z, w;
        };
    };

    VectorND() : VectorND(0.0f) {};

    VectorND(const Vector3 &vec, float w = 0.0f) : VectorND(vec[0], vec[1], vec[2], w) {}

    VectorND(real x, real y, real z, real w = 0.0f) : v(_mm_set_ps(w, z, y, x)) {}

    VectorND(real x) : v(_mm_set1_ps(x)) {}

    VectorND(__m128 v) : v(v) {}

    float &operator[](int i) { return (&x)[i]; }

    const float &operator[](int i) const { return (&x)[i]; }

    operator __m128() const { return v; }

    operator __m128i() const { return _mm_castps_si128(v); }

    operator __m128d() const { return _mm_castps_pd(v); }

    Vector4s &operator=(const Vector4s &o) {
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

    real max() const {
        return std::max(std::max(v[0], v[1]), std::max(v[2], v[3]));
    }

    float length2() const {
        return _mm_cvtss_f32(_mm_dp_ps(v, v, 0xf1));
    }

    float length() const {
        return std::sqrt(length2());
    }

    template <typename G>
    VectorND<3, G, InstructionSetExtension::AVX> cast() const {
        VectorND<3, G, InstructionSetExtension::AVX> ret;
        for (int i = 0; i < 3; i++)
            ret[i] = static_cast<G>(this->v[i]);
        return ret;
    }
};

typedef VectorND<4, float32, InstructionSetExtension::AVX> Vector4s;


// FMA: a * b + c
inline Vector4s fused_mul_add(const Vector4s &a, const Vector4s &b, const Vector4s &c) {
    return _mm_fmadd_ps(a, b, c);
}

// FMA: a * b + c
inline Vector3s fused_mul_add(const Vector3s &a, const Vector3s &b, const Vector3s &c) {
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

// SIMD Matrix4
template <>
struct TC_ALIGNED(16) MatrixND<4, float32, InstructionSetExtension::AVX> {
    using Vector = VectorND<4, float32, InstructionSetExtension::AVX>;
    using Matrix4s = MatrixND<4, float32, InstructionSetExtension::AVX>;
    union {
        // Four columns, instead of rows!
        Vector4s v[4];
    };

    MatrixND() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
        v[3] = 0.0f;
    }

    MatrixND(float32 val) {
        for (int i = 0; i < 4; i++) {
            v[i] = Vector();
        }
        for (int i = 0; i < 4; i++) {
            v[i][i] = val;
        }
    }

    // Diag
    MatrixND(Vector diag) {
        for (int i = 0; i < 4; i++) {
            this->v[i][i] = diag[i];
        }
    }

    MatrixND(Vector v0, Vector v1, Vector v2, Vector v3) : v{v0, v1, v2, v3} {}

    template<int DIM1, typename T1, InstructionSetExtension ISA1>
    MatrixND(const MatrixND<DIM1, T1, ISA1> &o) {
        for (int i = 0; i < DIM1; i++) {
            for (int j = 0; j < DIM1; j++) {
                v[i][j] = o[i][j];
            }
            for (int j = DIM1; j < 4; j++) {
                v[i][j] = 0;
            }
        }
        for (int i = DIM1; i < 4; i++) {
            v[i] = 0;
        }
    }

    MatrixND &operator=(const MatrixND &m) {
        for (int i = 0; i < 4; i++) {
            v[i] = m[i];
        }
        return *this;
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

    MatrixND transposed() const {
        MatrixND ret;
        transpose4x4(v, ret.v);
        return ret;
    }
};

using Matrix4s = MatrixND<4, float32, InstructionSetExtension::AVX>;

inline Matrix4s operator*(const float a, const Matrix4s &M) {
    Matrix4s ret;
    ret[0] = a * M[0];
    ret[1] = a * M[1];
    ret[2] = a * M[2];
    ret[3] = a * M[3];
    return ret;
}

inline Matrix4s operator*(const Matrix4s &M, const Matrix4s &N) {
    Matrix4s ret;
    for (int i = 0; i < 4; i++)
        ret[i] = M[i] * N[i];
    return ret;
}

inline Matrix4s operator*(const Matrix4s &m, const float a) {
    return a * m;
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline void print(const VectorND<DIM, T, ISE> &v) {
    std::cout << std::endl;
    for (int i = 0; i < DIM; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline void print(const MatrixND<DIM, T, ISE> &v) {
    std::cout << std::endl;
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            std::cout << v[j][i] << " ";
        }
        std::cout << std::endl;
    }
}

// SIMD Matrix3
template <>
struct TC_ALIGNED(16) MatrixND<3, float32, InstructionSetExtension::AVX> {
    using Matrix3s = MatrixND<3, float32, InstructionSetExtension::AVX>;
    union {
        // Three columns, instead of rows!
        Vector4s v[3];
    };

    MatrixND() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    }

    template<int DIM1, typename T1, InstructionSetExtension ISA1>
    MatrixND(const MatrixND<DIM1, T1, ISA1> &o) {
        for (int i = 0; i < DIM1; i++) {
            for (int j = 0; j < DIM1; j++) {
                v[i][j] = o[i][j];
            }
            for (int j = DIM1; j < 3; j++) {
                v[i][j] = 0;
            }
        }
        for (int i = DIM1; i < 3; i++) {
            v[i] = 0;
        }
    }

    MatrixND &operator=(const Matrix3s &m) {
        v[0] = m.v[0];
        v[1] = m.v[1];
        v[2] = m.v[2];
        return *this;
    }

    MatrixND(float diag) {
        v[0] = Vector4s(diag, 0.0f, 0.0f, 0.0f);
        v[1] = Vector4s(0.0f, diag, 0.0f, 0.0f);
        v[2] = Vector4s(0.0f, 0.0f, diag, 0.0f);
    }

    MatrixND(Vector3 diag) {
        v[0] = Vector4s(diag[0], 0.0f, 0.0f, 0.0f);
        v[1] = Vector4s(0.0f, diag[1], 0.0f, 0.0f);
        v[2] = Vector4s(0.0f, 0.0f, diag[2], 0.0f);
    }

    MatrixND(float32 diag0, float32 diag1, float32 diag2) {
        v[0] = Vector4s(diag0, 0.0f, 0.0f, 0.0f);
        v[1] = Vector4s(0.0f, diag1, 0.0f, 0.0f);
        v[2] = Vector4s(0.0f, 0.0f, diag2, 0.0f);
    }

    MatrixND(Vector4s v0, Vector4s v1, Vector4s v2) : v{v0, v1, v2} {}

    MatrixND(const MatrixND<3, real, InstructionSetExtension::None> &o) {
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

    float32 frobenius_norm2() const {
        return v[0].length2() + v[1].length2() + v[2].length2();
    }

    float32 frobenius_norm() const {
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

    Vector3s operator*(const Vector3s &o) const {
        Vector4s ret = v[2] * o.broadcast<2>();
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return Vector3s(ret[0], ret[1], ret[2]);
    }
};

using Matrix3s = MatrixND<3, float32, InstructionSetExtension::AVX>;

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

template <int dim, typename T>
inline void test_dim_type() {
    T a[dim][dim], b[dim][dim], c[dim][dim];
    T x[dim], y[dim], z[dim];

    MatrixND <dim, T> m_a, m_b, m_c;
    VectorND<dim, T> v_x, v_y, v_z;

    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) {
            m_a[i][j] = a[i][j] = rand();
            m_b[i][j] = b[i][j] = rand();
            m_c[i][j] = c[i][j] = rand();
            v_x[i] = x[i] = rand();
            v_y[i] = y[i] = rand();
            v_z[i] = z[i] = rand();
        }

    auto result_test = [&]() {
        bool same = true;
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) {
                if (std::abs(a[i][j] - m_a[i][j]) > T(1e-6f)) same = false;
                if (std::abs(b[i][j] - m_b[i][j]) > T(1e-6f)) same = false;
                if (std::abs(c[i][j] - m_c[i][j]) > T(1e-6f)) same = false;
            }
        for (int i = 0; i < dim; i++) {
            if (std::abs(x[i] - v_x[i]) > T(1e-6f)) same = false;
            if (std::abs(y[i] - v_y[i]) > T(1e-6f)) same = false;
            if (std::abs(z[i] - v_z[i]) > T(1e-6f)) same = false;
        }
        assert(same);
    };

    m_c += m_a * m_b;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            for (int k = 0; k < dim; k++)
                c[j][i] += a[k][i] * b[j][k];
    result_test();

    v_z = v_x / v_y - v_z;
    for (int i = 0; i < dim; i++)
        z[i] = x[i] / y[i] - z[i];
    result_test();
}

inline void test_vector_and_matrix() {
    test_dim_type<2, float>();
    test_dim_type<3, float>();
    /*
    glm::vec2 a(1, 2);
    glm::vec2 b(3, 4);
    glm::mat2 c = glm::outerProduct(a, b);
    VectorND<2, float> aa(1, 2);
    VectorND<2, float> bb(3, 4);
    MatrixND<2, float> cc = MatrixND<2, float>::outer_product(aa, bb);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            printf("%.4f %.4f\n", c[i][j], cc[i][j]);
        }
      */
    printf("Vector and matrix test passes.\n");
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transpose(const MatrixND<DIM, T, ISE> &mat) {
    return mat.transposed();
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transposed(const MatrixND<DIM, T, ISE> &mat) {
    return transpose(mat);
}

using Vector4 = Vector4s;

using Vector2d = VectorND<2, float64, InstructionSetExtension::AVX>;
using Vector3d = VectorND<3, float64, InstructionSetExtension::AVX>;
using Vector4d = VectorND<4, float64, InstructionSetExtension::AVX>;

using Vector2i = VectorND<2, int, InstructionSetExtension::AVX>;
using Vector3i = VectorND<3, int, InstructionSetExtension::AVX>;
using Vector4i = VectorND<4, int, InstructionSetExtension::AVX>;

using Matrix2 = MatrixND<2, float32, InstructionSetExtension::AVX>;
using Matrix3 = MatrixND<3, float32, InstructionSetExtension::AVX>;
using Matrix4 = MatrixND<4, float32, InstructionSetExtension::AVX>;

using Matrix2d = MatrixND<2, float64, InstructionSetExtension::AVX>;
using Matrix3d = MatrixND<3, float64, InstructionSetExtension::AVX>;
using Matrix4d = MatrixND<4, float64, InstructionSetExtension::AVX>;

inline float32 determinant(const MatrixND<2, float32, InstructionSetExtension::AVX> &mat) {
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

inline float32 determinant(const MatrixND<3, float32, InstructionSetExtension::AVX> &mat) {
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
           - mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2])
           + mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, InstructionSetExtension ISE>
inline VectorND<3, T, ISE> cross(const VectorND<3, T, ISE> &a, const VectorND<3, T, ISE> &b) {
    return VectorND<3, T, ISE>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline T dot(const VectorND<DIM, T, ISE> &a, const VectorND<DIM, T, ISE> &b) {
    T sum = a[0] * b[0];
    for (int i = 1; i < DIM; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline VectorND<DIM, T, ISE> normalize(const VectorND<DIM, T, ISE> &a) {
    return (T(1) / a.length()) * a;
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline VectorND<DIM, T, ISE> normalized(const VectorND<DIM, T, ISE> &a) {
    return normalize(a);
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline float32 length(const VectorND<DIM, T, ISE> &a) {
    return a.length();
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline VectorND<DIM, T, ISE> fract(const VectorND<DIM, T, ISE> &a) {
    VectorND<DIM, T, ISE> ret;
    for (int i = 0; i < DIM; i++) {
        ret[i] = a[i] - (int)floor(a[i]);
    }
    return ret;
}

TC_NAMESPACE_END
