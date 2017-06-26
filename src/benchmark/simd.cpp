/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math_simd.h>

TC_NAMESPACE_BEGIN

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

    float frobenius_norm2() const {
        return v[0].length2() + v[1].length2() + v[2].length2();
    }

    float frobenius_norm() const {
        return std::sqrt(frobenius_norm2());
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

class Matrix4sBenchmark : public Benchmark {
private:
    int n;
    bool brute_force;
    std::vector<Vector4> input;
    std::vector<Vector4s> input_s;
    Matrix4 M;
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        brute_force = config.get_bool("brute_force");
        input.resize(workload);
        input_s.resize(workload);
        for (int i = 0; i < workload; i++) {
            input[i] = Vector4(rand(), rand(), rand(), rand());
            input_s[i] = input[i];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                M[i][j] = rand();
            }
        }
    }

protected:

    void iterate() override {
        if (brute_force) {
            Vector4s ret(0.0f);
            Matrix4s Ms(M);
            for (int i = 0; i < workload; i++) {
                ret += Ms * input_s[i];
            }
            dummy = (int)(ret.length());
        } else {
            Vector4 ret(0.0f);
            for (int i = 0; i < workload; i++) {
                ret += M * input[i];
            }
            dummy = (int)(glm::length(ret));
        }
    }

public:
    bool test() const override {
        Matrix4s Ms(M);
        for (int i = 0; i < workload; i++) {
            Vector4s bf_result = M * input[i];
            Vector4s simd_result = Ms * input_s[i];
            if ((bf_result - simd_result).length() > 1e-6) {
                P(M);
                P(Ms);
                P(input[i]);
                P(input_s[i]);
                P(i);
                P(bf_result);
                P(simd_result);
                error("value mismatch");
            }
        }
        return true;
    }
};

TC_IMPLEMENTATION(Benchmark, Matrix4sBenchmark, "matrix4s");

TC_NAMESPACE_END
