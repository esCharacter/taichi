/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <type_traits>
#include <immintrin.h>
#include <cmath>
#include <vector>

TC_NAMESPACE_BEGIN

#ifdef _WIN64
#define TC_ALIGNED(x) __declspec(align(x))
#else
#define TC_ALIGNED(x) __attribute__((aligned(x)))
#endif

enum class InstructionSetExtension {
    None,
    SSE,
    AVX,
    AVX2
};

constexpr InstructionSetExtension default_instruction_set = InstructionSetExtension::SSE;

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

template <>
struct VectorNDBase<3, float32, InstructionSetExtension::SSE> {
    union {
        __m128 v;
        struct {
            float32 x, y, z, w;
        };
        float32 d[4];
    };

    VectorNDBase() : v(_mm_set_ps1(0.0f)) {}

    VectorNDBase(__m128 v) : v(v) {}
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

template <>
struct VectorNDBase<4, float32, InstructionSetExtension::SSE> {
    union {
        __m128 v;
        struct {
            float32 x, y, z, w;
        };
        float32 d[4];
    };

    VectorNDBase() : v(_mm_set_ps1(0.0f)) {}

    VectorNDBase(__m128 v) : v(v) {}
};

template <int DIM, typename T, InstructionSetExtension ISE = default_instruction_set>
struct VectorND : public VectorNDBase<DIM, T, ISE> {
    template <int DIM_, typename T_, InstructionSetExtension ISE_>
    static constexpr bool SIMD_4_32F = (DIM_ == 3 || DIM_ == 4) &&
                                       std::is_same<T_, float32>::value &&
                                       ISE_ >= InstructionSetExtension::SSE;

    template <int DIM_, typename T_, InstructionSetExtension ISE_>
    static constexpr bool SIMD_NONE = !SIMD_4_32F<DIM_, T_, ISE_>;

    using VectorBase = VectorNDBase<DIM, T, ISE>;
    using VectorBase::d;

    VectorND() {
        for (int i = 0; i < DIM; i++) {
            this->d[i] = T(0);
        }
    }

    // Vector3f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 3, int> = 0>
    VectorND(float32 x) : VectorND(x, x, x, 0.0f) {}

    // Vector3f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 3, int> = 0>
    VectorND(real x, real y, real z, real w = 0.0f) : VectorBase(_mm_set_ps(w, z, y, x)) {}

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_> || DIM_ != 3, int> = 0>
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

    // All except Vector3f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<!(SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 3), int> = 0>
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

    template <typename F>
    VectorND(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
    }

    template <typename F>
    VectorND &set(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
        return *this;
    }

    VectorND &operator=(const VectorND o) {
        return this->set([&](int i) { return o[i]; });
    }

    // SIMD: Vector3f & Vector4f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND operator+(const VectorND &o) const { return _mm_add_ps(this->v, o.v); }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND operator-(const VectorND &o) const { return _mm_sub_ps(this->v, o.v); }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND operator*(const VectorND &o) const { return _mm_mul_ps(this->v, o.v); }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND operator/(const VectorND &o) const { return _mm_div_ps(this->v, o.v); }

    // Non-SIMD cases
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    VectorND operator+(const VectorND o) const {
        return VectorND([=](int i) { return this->d[i] + o[i]; });
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    VectorND operator-(const VectorND o) const {
        return VectorND([=](int i) { return this->d[i] - o[i]; });
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    VectorND operator*(const VectorND o) const {
        return VectorND([=](int i) { return this->d[i] * o[i]; });
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    VectorND operator/(const VectorND o) const {
        return VectorND([=](int i) { return this->d[i] / o[i]; });
    }

    // Inplace operations
    VectorND &operator+=(const VectorND o) {
        return this->set([&](int i) { return this->d[i] + o[i]; });
    }

    VectorND &operator-=(const VectorND o) {
        return this->set([&](int i) { return this->d[i] - o[i]; });
    }

    VectorND &operator*=(const VectorND o) {
        return this->set([&](int i) { return this->d[i] * o[i]; });
    }

    VectorND &operator/=(const VectorND o) {
        return this->set([&](int i) { return this->d[i] / o[i]; });
    }

    bool operator==(const VectorND &o) const {
        for (int i = 0; i < DIM; i++)
            if (this->d[i] != o[i])
                return false;
        return true;
    }

    bool operator==(const std::vector<T> &o) const {
        if (o.size() != DIM)
            return false;
        for (int i = 0; i < DIM; i++)
            if (this->d[i] != o[i])
                return false;
        return true;
    }

    bool operator!=(const VectorND &o) const {
        for (int i = 0; i < DIM; i++)
            if (this->d[i] != o[i])
                return true;
        return false;
    }

    VectorND abs() const {
        return VectorND([&](int i) { return std::abs(d[i]); });
    }

    VectorND floor() const {
        return VectorND([&](int i) { return std::floor(d[i]); });
    }

    VectorND sin() const {
        return VectorND([&](int i) { return std::sin(d[i]); });
    }

    VectorND cos() const {
        return VectorND([&](int i) { return std::cos(d[i]); });
    }

    T max() const {
        T ret = this->d[0];
        for (int i = 1; i < DIM; i++) {
            ret = std::max(ret, this->d[i]);
        }
        return ret;
    }

    template <typename G>
    VectorND<DIM, G, ISE> cast() const {
        return VectorND<DIM, G, ISE>([this](int i) { return static_cast<G>(this->d[i]); });
    }

    void print() const {
        for (int i = 0; i < DIM; i++) {
            std::cout << this->d[i] << " ";
        }
        std::cout << std::endl;
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    operator __m128() const { return this->v; }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    operator __m128i() const { return _mm_castps_si128(this->v); }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    operator __m128d() const { return _mm_castps_pd(this->v); }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND(__m128 v) { this->v = v; }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND operator-() const { return _mm_sub_ps(VectorND(0.0f), this->v); }


    template <int a, int b, int c, int d, int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND permute() const {
        return _mm_permute_ps(this->v, _MM_SHUFFLE(a, b, c, d));
    }

    template <int a, int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND broadcast() const {
        return permute<a, a, a, a>();
    }

    // member function: length
    // Vector3f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 3, int> = 0>
    float32 length2() const {
        return _mm_cvtss_f32(_mm_dp_ps(this->v, this->v, 0x71));
    }

    // Vector4f
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 4, int> = 0>
    float32 length2() const {
        return _mm_cvtss_f32(_mm_dp_ps(this->v, this->v, 0xf1));
    }

    // Others
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    T length2() const {
        T ret = 0;
        for (int i = 0; i < DIM; i++) {
            ret += this->d[i] * this->d[i];
        }
        return ret;
    }

    auto length() const {
        return std::sqrt(length2());
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

/*
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

*/

// SIMD Vector4
template <>
struct TC_ALIGNED(16) VectorND<4, float32, InstructionSetExtension::AVX> {
};

using Vector1 = VectorND<1, float32, default_instruction_set>;
using Vector2 = VectorND<2, float32, default_instruction_set>;
using Vector3 = VectorND<3, float32, default_instruction_set>;
using Vector4 = VectorND<4, float32, default_instruction_set>;

using Vector1d = VectorND<1, float64, default_instruction_set>;
using Vector2d = VectorND<2, float64, default_instruction_set>;
using Vector3d = VectorND<3, float64, default_instruction_set>;
using Vector4d = VectorND<4, float64, default_instruction_set>;

using Vector1i = VectorND<1, int, default_instruction_set>;
using Vector2i = VectorND<2, int, default_instruction_set>;
using Vector3i = VectorND<3, int, default_instruction_set>;
using Vector4i = VectorND<4, int, default_instruction_set>;

/*
// FMA: a * b + c
inline Vector4 fused_mul_add(const Vector4 &a, const Vector4 &b, const Vector4 &c) {
    return _mm_fmadd_ps(a, b, c);
}

// FMA: a * b + c
inline Vector3 fused_mul_add(const Vector3 &a, const Vector3 &b, const Vector3 &c) {
    return _mm_fmadd_ps(a, b, c);
}

inline void transpose4x4(const Vector4 m[], Vector4 t[]) {
    Vector4 t0 = _mm_unpacklo_ps(m[0], m[1]);
    Vector4 t2 = _mm_unpacklo_ps(m[2], m[3]);
    Vector4 t1 = _mm_unpackhi_ps(m[0], m[1]);
    Vector4 t3 = _mm_unpackhi_ps(m[2], m[3]);
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
        Vector4 v[4];
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

    Vector4 &operator[](int i) { return v[i]; }

    const Vector4 &operator[](int i) const { return v[i]; }

    Vector4 operator*(const Vector4 &o) const {
        Vector4 ret = o.broadcast<3>() * v[3];
        ret = fused_mul_add(v[2], o.broadcast<2>(), ret);
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return ret;
    }

    Vector4 multiply_vec3(const Vector4 &o) const {
        Vector4 ret = o.broadcast<2>() * v[2];
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
        Vector4 v[3];
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
        v[0] = Vector4(diag, 0.0f, 0.0f, 0.0f);
        v[1] = Vector4(0.0f, diag, 0.0f, 0.0f);
        v[2] = Vector4(0.0f, 0.0f, diag, 0.0f);
    }

    MatrixND(Vector3 diag) {
        v[0] = Vector4(diag[0], 0.0f, 0.0f, 0.0f);
        v[1] = Vector4(0.0f, diag[1], 0.0f, 0.0f);
        v[2] = Vector4(0.0f, 0.0f, diag[2], 0.0f);
    }

    MatrixND(float32 diag0, float32 diag1, float32 diag2) {
        v[0] = Vector4(diag0, 0.0f, 0.0f, 0.0f);
        v[1] = Vector4(0.0f, diag1, 0.0f, 0.0f);
        v[2] = Vector4(0.0f, 0.0f, diag2, 0.0f);
    }

    MatrixND(Vector4 v0, Vector4 v1, Vector4 v2) : v{v0, v1, v2} {}

    MatrixND(const MatrixND<3, real, InstructionSetExtension::None> &o) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                v[i][j] = o[i][j];
            }
        }
    }

    Vector4 &operator[](int i) { return v[i]; }

    const Vector4 &operator[](int i) const { return v[i]; }

    Vector4 operator*(const Vector4 &o) const {
        Vector4 ret = o.broadcast<2>() * v[2];
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

    static Matrix3s outer_product(Vector4 row, Vector4 column) {
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

    Vector3 operator*(const Vector3 &o) const {
        Vector4 ret = v[2] * o.broadcast<2>();
        ret = fused_mul_add(v[1], o.broadcast<1>(), ret);
        ret = fused_mul_add(v[0], o.broadcast<0>(), ret);
        return Vector3(ret[0], ret[1], ret[2]);
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

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transpose(const MatrixND<DIM, T, ISE> &mat) {
    return mat.transposed();
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transposed(const MatrixND<DIM, T, ISE> &mat) {
    return transpose(mat);
}
*/


/*
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
*/

void test_linalg() {
    Vector3 a(1, 2, 3), b(4, 2, 5);
    assert(a + b == Vector3(5, 4, 8));
    assert(b - a == Vector3(3, 0, 2));
    assert(b * a == Vector3(4, 4, 15));
    assert(b / a == Vector3(4, 1, 5.0f / 3.0f));
    a += b;
    assert(a == Vector3(5, 4, 8));
    a -= b;
    assert(a == Vector3(1, 2, 3));
    a *= b;
    assert(a == Vector3(4, 4, 15));
    a /= b;
    assert(a == Vector3(1, 2, 3));

    auto t = __m128(a);

    Vector2 c(1, 2), d(2, 5);
    assert(c + d == Vector2(3, 7));

    assert(Vector4(1, 2, 3, 1).length2() == 15.0f);
    assert(Vector3(1, 2, 3, 1).length2() == 14.0f);

    std::cout << "Passed." << std::endl;
}

TC_NAMESPACE_END
