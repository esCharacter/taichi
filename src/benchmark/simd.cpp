/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/benchmark.h>
#include <taichi/math/math_simd.h>

TC_NAMESPACE_BEGIN

// SIMD Matrix4
struct TC_ALIGNED(64) Matrix4s {
    union {
        // Four columns, instead of rows!
        Vector4s v[4];
    };

    // without zero-initialization
    Matrix4s(void *) {};

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
};

Matrix4s operator*(const float a, const Matrix4s &M) {
    Matrix4s ret(nullptr);
    ret[0] = a * M[0];
    ret[1] = a * M[1];
    ret[2] = a * M[2];
    ret[3] = a * M[3];
    return ret;
}

Matrix4s operator*(const Matrix4s &m, const float a) {
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
