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

inline float32 fract(float32 a) {
    return a - (int)floor(a);
}

inline float64 fract(float64 a) {
    return a - (int)floor(a);
}

enum class InstructionSetExtension {
    None,
    SSE,
    AVX,
    AVX2
};

constexpr InstructionSetExtension default_instruction_set = InstructionSetExtension::SSE;


/////////////////////////////////////////////////////////////////
/////              N dimensional Vector
/////////////////////////////////////////////////////////////////


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


    // Vector initialization
    template <typename F, std::enable_if_t<std::is_same<F, VectorND>::value, int> = 0>
    VectorND(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f[i];
    }

    // Scalar initialization
    template <typename F, std::enable_if_t<std::is_same<F, T>::value, int> = 0>
    VectorND(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f;
    }

    // Function intialization
    template <typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value, int> = 0>
    VectorND(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_> || DIM_ != 3, int> = 0>
    VectorND(T v) {
        for (int i = 0; i < std::min(DIM, DIM_); i++) {
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

    // Vector extension
    template <int DIM_ = DIM, std::enable_if_t<(DIM_ > 1), int> = 0>
    VectorND(const VectorND<DIM - 1, T, ISE> &o, T extra) {
        for (int i = 0; i < DIM_; i++) {
            this->d[i] = o[i];
        }
        this->d[DIM - 1] = extra;
    }

    template <typename T_>
    VectorND(const std::vector<T_> &o) {
        if (o.size() != DIM) {
            error("Dimension mismatch: " + std::to_string(DIM) + " v.s. " + std::to_string((int)o.size()));
        }
        for (int i = 0; i < DIM; i++)
            this->d[i] = T(o[i]);
    }

    T &operator[](int i) { return this->d[i]; }

    const T &operator[](int i) const { return this->d[i]; }

    T dot(VectorND<DIM, T, ISE> o) const {
        T ret = T(0);
        for (int i = 0; i < DIM; i++)
            ret += this->d[i] * o[i];
        return ret;
    }


    template <typename F, std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value, int> = 0>
    VectorND &set(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
        return *this;
    }

    VectorND &operator=(const VectorND &o) {
        return this->set([&](int i) { return o[i]; });
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_>, int> = 0>
    VectorND &operator=(__m128 v) {
        this->v = v;
        return *this;
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

    VectorND operator-() const {
        return VectorND([=](int i) { return -this->d[i]; });
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

    VectorND fract() const {
        return VectorND([&](int i) { return taichi::fract(d[i]); });
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
inline void print(const VectorND<DIM, T, ISE> &v) {
    std::cout << std::endl;
    for (int i = 0; i < DIM; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

template <int DIM, typename T, InstructionSetExtension ISE>
VectorND<DIM, T, ISE> operator*(T a, const VectorND<DIM, T, ISE> &v) {
    return VectorND<DIM, T, ISE>(a) * v;
}

template <int DIM, typename T, InstructionSetExtension ISE>
VectorND<DIM, T, ISE> operator*(const VectorND<DIM, T, ISE> &v, T a) {
    return a * v;
}

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


/////////////////////////////////////////////////////////////////
/////              N dimensional Matrix
/////////////////////////////////////////////////////////////////


template <int DIM, typename T, InstructionSetExtension ISE = default_instruction_set>
struct MatrixND {
    template <int DIM_, typename T_, InstructionSetExtension ISE_>
    static constexpr bool SIMD_4_32F = (DIM_ == 3 || DIM_ == 4) &&
                                       std::is_same<T_, float32>::value &&
                                       ISE_ >= InstructionSetExtension::SSE;

    template <int DIM_, typename T_, InstructionSetExtension ISE_>
    static constexpr bool SIMD_NONE = !SIMD_4_32F<DIM_, T_, ISE_>;
    using Vector = VectorND<DIM, T, ISE>;
    Vector d[DIM];

    template <int DIM_, typename T_, InstructionSetExtension ISE_>
    MatrixND(const MatrixND<DIM_, T_, ISE_> &o) : MatrixND() {
        for (int i = 0; i < std::min(DIM_, DIM); i++) {
            for (int j = 0; j < std::min(DIM_, DIM); j++) {
                d[i][j] = o[i][j];
            }
        }
    }

    MatrixND() {
        for (int i = 0; i < DIM; i++) {
            d[i] = VectorND<DIM, T, ISE>();
        }
    }

    MatrixND(T v) : MatrixND() {
        for (int i = 0; i < DIM; i++) {
            d[i][i] = v;
        }
    }

    // Diag
    MatrixND(Vector v) : MatrixND() {
        for (int i = 0; i < DIM; i++)
            this->d[i][i] = v[i];
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

    // Function intialization
    template <typename F, std::enable_if_t<std::is_convertible<F, std::function<Vector(int)>>::value, int> = 0>
    MatrixND(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
    }

    template <typename F, std::enable_if_t<std::is_convertible<F, std::function<Vector(int)>>::value, int> = 0>
    MatrixND &set(const F &f) {
        for (int i = 0; i < DIM; i++)
            this->d[i] = f(i);
        return *this;
    }

    MatrixND &operator=(const MatrixND &o) {
        return this->set([&](int i) { return o[i]; });
    }

    MatrixND(const MatrixND &o) {
        *this = o;
    }

    VectorND<DIM, T, ISE> &operator[](int i) {
        return d[i];
    }

    const VectorND<DIM, T, ISE> &operator[](int i) const {
        return d[i];
    }

    VectorND<DIM, T, ISE> operator*(const VectorND<DIM, T, ISE> &o) const {
        VectorND <DIM, T, ISE> ret = d[0] * o[0];
        for (int i = 1; i < DIM; i++)
            ret += d[i] * o[i];
        return ret;
    }

    // No FMA
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_>, int> = 0>
    MatrixND operator*(const MatrixND &o) const {
        return MatrixND([&](int i) { return (*this) * o[i]; });
    }

    // Matrix3
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 3, int> = 0>
    Vector operator*(const Vector &o) const {
        Vector3 ret = d[2] * o.template broadcast<2>();
        ret = fused_mul_add(d[1], o.template broadcast<1>(), ret);
        ret = fused_mul_add(d[0], o.template broadcast<0>(), ret);
        return ret;
    }

    // Matrix4
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 4, int> = 0>
    Vector4 operator*(const Vector4 &o) const {
        Vector4 ret = o.broadcast<3>() * d[3];
        ret = fused_mul_add(d[2], o.template broadcast<2>(), ret);
        ret = fused_mul_add(d[1], o.template broadcast<1>(), ret);
        ret = fused_mul_add(d[0], o.template broadcast<0>(), ret);
        return ret;
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 4, int> = 0>
    Vector4 multiply_vec3(const Vector4 &o) const {
        Vector4 ret = o.broadcast<2>() * d[2];
        ret = fused_mul_add(d[1], o.template broadcast<1>(), ret);
        ret = fused_mul_add(d[0], o.template broadcast<0>(), ret);
        return ret;
    }

    static MatrixND outer_product(Vector row, Vector column) {
        return MatrixND([&](int i) { return column * row[i]; });
    }


    MatrixND operator+(const MatrixND &o) const {
        return MatrixND([=](int i) { return this->d[i] + o[i]; });
    }

    MatrixND operator-(const MatrixND &o) const {
        return MatrixND([=](int i) { return this->d[i] - o[i]; });
    }

    MatrixND &operator+=(const MatrixND &o) {
        return this->set([&](int i) { return this->d[i] + o[i]; });
    }

    MatrixND &operator-=(const MatrixND &o) {
        return this->set([&](int i) { return this->d[i] - o[i]; });
    }

    MatrixND operator-() const {
        return MatrixND([=](int i) { return -this->d[i]; });
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
        T sum = d[0].length2();
        for (int i = 1; i < DIM; i++) {
            sum += d[i].length2();
        }
        return sum;
    }

    auto frobenius_norm() const {
        return std::sqrt(frobenius_norm2());
    }

    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_NONE<DIM_, T_, ISE_> || DIM_ != 4, int> = 0>
    MatrixND transposed() const {
        MatrixND ret;
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++) {
                ret[i][j] = d[j][i];
            }
        return ret;
    }

    // Matrix4
    template <int DIM_ = DIM, typename T_=T, InstructionSetExtension ISE_ = ISE,
            typename std::enable_if_t<SIMD_4_32F<DIM_, T_, ISE_> && DIM_ == 4, int> = 0>
    MatrixND transposed() const {
        MatrixND ret;
        transpose4x4(d, ret.d);
        return ret;
    }

    template <typename G>
    MatrixND<DIM, G, ISE> cast() const {
        return MatrixND([=](int i) { return d[i].template cast<G>(); });
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

template <int DIM, typename T, InstructionSetExtension ISE>
MatrixND<DIM, T, ISE> operator*(const MatrixND<DIM, T, ISE> &M, const MatrixND<DIM, T, ISE> &N) {
    MatrixND <DIM, T, ISE> ret;
    for (int i = 0; i < DIM; i++) {
        ret[i] = M * N[i];
    }
    return ret;
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


template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transpose(const MatrixND<DIM, T, ISE> &mat) {
    return mat.transposed();
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> transposed(const MatrixND<DIM, T, ISE> &mat) {
    return transpose(mat);
}

using Matrix2 = MatrixND<2, float32, default_instruction_set>;
using Matrix3 = MatrixND<3, float32, default_instruction_set>;
using Matrix4 = MatrixND<4, float32, default_instruction_set>;

using Matrix2d = MatrixND<2, float64, default_instruction_set>;
using Matrix3d = MatrixND<3, float64, default_instruction_set>;
using Matrix4d = MatrixND<4, float64, default_instruction_set>;

template <typename T, InstructionSetExtension ISE>
inline float32 determinant(const MatrixND<2, T, ISE> &mat) {
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

template <typename T, InstructionSetExtension ISE>
inline float32 determinant(const MatrixND<3, T, ISE> &mat) {
    return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
           - mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2])
           + mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, InstructionSetExtension ISE>
inline T cross(const VectorND<2, T, ISE> &a, const VectorND<2, T, ISE> &b) {
    return a.x * b.y - a.y * b.x;
}

template <typename T, InstructionSetExtension ISE>
inline VectorND<3, T, ISE> cross(const VectorND<3, T, ISE> &a, const VectorND<3, T, ISE> &b) {
    return VectorND<3, T, ISE>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


template <int DIM, typename T, InstructionSetExtension ISE>
inline T dot(const VectorND<DIM, T, ISE> &a, const VectorND<DIM, T, ISE> &b) {
    return a.dot(b);
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
    return a.fract();
}

template <InstructionSetExtension ISE>
inline MatrixND<2, float32, ISE> inversed(const MatrixND<2, float32, ISE> &mat) {
    real det = determinant(mat);
    return 1.0f / det * MatrixND<2, float32, ISE>(
            VectorND<2, float32, ISE>(mat[1][1], -mat[0][1]),
            VectorND<2, float32, ISE>(mat[1][0], mat[0][0]
            )
    );
}

template <InstructionSetExtension ISE>
inline MatrixND<3, float32, ISE> inversed(const MatrixND<3, float32, ISE> &mat) {
    real det = determinant(mat);
    return 1.0f / det * MatrixND<3, float32, ISE>(
            VectorND<3, float32, ISE>(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2],
                                      mat[2][1] * mat[0][2] - mat[0][1] * mat[2][2],
                                      mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]),
            VectorND<3, float32, ISE>(mat[2][0] * mat[1][2] - mat[1][0] * mat[2][2],
                                      mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2],
                                      mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2]),
            VectorND<3, float32, ISE>(mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1],
                                      mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1],
                                      mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1])
    );
}

template <typename T, InstructionSetExtension ISE>
inline MatrixND<4, T, ISE> inversed(const MatrixND<4, T, ISE> &m) {
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

    using Vector = VectorND<4, T, ISE>;

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
    MatrixND<4, T, ISE> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

    Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

    Vector Dot0(m[0] * Row0);
    T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

    T OneOverDeterminant = static_cast<T>(1) / Dot1;

    return Inverse * OneOverDeterminant;
}

template <int DIM, typename T, InstructionSetExtension ISE>
inline MatrixND<DIM, T, ISE> inverse(const MatrixND<DIM, T, ISE> &m) {
    return inversed(m);
}

TC_NAMESPACE_END
